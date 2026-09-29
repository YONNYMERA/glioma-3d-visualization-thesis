"""Entrenamiento reanudable por epoch e inferencia out-of-fold sin ensemble."""
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from . import manifest as mf


def environment():
    import torch, monai, scipy, sklearn, nibabel
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'monai': monai.__version__, 'numpy': np.__version__,
            'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__,
            'nibabel': nibabel.__version__, 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def preflight(args):
    import nibabel as nib
    import torch
    manifest = mf.load(args.manifest)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; run inside an allocated GPU job and check PyTorch installation')
    checked = []
    for case in manifest['cases']:
        images = [nib.load(str(Path(args.data_root) / p)) for p in case['image'] + [case['label']]]
        ref = images[0]
        for im in images:
            if len(im.shape) != 3 or im.shape != ref.shape or not np.allclose(im.affine, ref.affine, atol=1e-4):
                raise ValueError(f'Modalities/reference not aligned: {case["case_id"]}')
            if np.any(np.asarray(im.header.get_zooms()) <= 0) or not np.isfinite(im.affine).all():
                raise ValueError(f'Invalid physical geometry: {case["case_id"]}')
        if args.deep:
            for im in images:
                if not np.isfinite(np.asarray(im.dataobj)).all():
                    raise ValueError(f'NaN/Inf image: {case["case_id"]}')
            labels = np.unique(np.asarray(images[-1].dataobj))
            if not set(labels).issubset({0, 1, 2, 3, 4}):
                raise ValueError(f'Unexpected segmentation labels {labels}: {case["case_id"]}')
        checked.append(case['case_id'])
    # Ensure persisted file sizes still agree across the entire dataset.
    for role in mf.ROLES:
        mf.samples(manifest, args.data_root, manifest['folds'][0]['fold'], role)
    report = {'environment': environment(), 'n_cases_checked': len(checked),
              'deep_image_check': args.deep, 'manifest_sha256': manifest['manifest_sha256'],
              'split_provenance': manifest['provenance']}
    if args.output:
        mf.write_json(args.output, report)
    print(json.dumps(report, indent=2))


def atomic_torch_save(value, path):
    import torch
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def guard_run(out, metadata, resume):
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'run.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous != metadata:
            raise ValueError(f'Output belongs to a different run: {out}. Choose a new output directory.')
        if not resume:
            raise FileExistsError(f'{out} exists; use --resume to continue this exact run')
    else:
        if any(out.iterdir()):
            raise FileExistsError(f'Output directory is nonempty without run.json: {out}')
        mf.write_json(path, metadata)


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from monai.data import Dataset, list_data_collate
    from monai.losses import DiceCELoss
    from monai.utils import set_determinism
    from .models import model, transforms, infer
    manifest = mf.load(args.manifest)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select --device cpu only for a smoke test')
    config = {'architecture': args.arch, 'epochs': args.epochs, 'batch_size': args.batch_size,
              'lr': args.lr, 'scheduler': args.scheduler, 'patience': args.patience,
              'roi': args.roi, 'workers': args.workers, 'fold': args.fold,
              'manifest_sha256': manifest['manifest_sha256'], 'seed': manifest['seed'],
              'environment': environment(), 'device': str(device),
              'loss': 'DiceCE sigmoid squared_pred smooth1e-5 weights0.7/0.3',
              'epoch_reseeding': True, 'evaluation_grid': 'RAS_1mm_image_foreground_divisible16'}
    out = Path(args.output) / f'fold_{args.fold:02d}'
    guard_run(out, config, args.resume)
    if (out / 'TRAINING_COMPLETED.json').exists():
        print(f'Already completed: {out}')
        return
    set_determinism(seed=manifest['seed'] + args.fold)
    net = model(args.arch).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=5) if args.scheduler == 'plateau' else None
    loss_fn = DiceCELoss(include_background=True, sigmoid=True, to_onehot_y=False,
                        squared_pred=True, smooth_nr=1e-5, smooth_dr=1e-5, lambda_dice=.7, lambda_ce=.3)
    train_files = mf.samples(manifest, args.data_root, args.fold, 'train')
    val_files = mf.samples(manifest, args.data_root, args.fold, 'inner_val')
    train_tf = transforms(True, tuple(args.roi))
    val_ds = Dataset(val_files, transforms(False, tuple(args.roi)))
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.workers, shuffle=False)
    history, start, best, best_epoch, bad = [], 1, -1., 0, 0
    last = out / 'last.pth'
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=True)
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler'])
        history, best, best_epoch, bad = state['history'], state['best'], state['best_epoch'], state['bad_epochs']
        start = state['epoch'] + 1
    for epoch in range(start, args.epochs + 1):
        # Makes epoch-boundary restart reproducible; differs from notebook's continuous random stream.
        seed = manifest['seed'] + args.fold + 10000 * epoch
        set_determinism(seed=seed)
        train_tf.set_random_state(seed=seed)
        train_loader = DataLoader(Dataset(train_files, train_tf), batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.workers, collate_fn=list_data_collate,
                                  pin_memory=device.type == 'cuda')
        started = time.perf_counter()
        net.train()
        total, count = 0., 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(batch['image'].to(device)), batch['label'].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; stop and inspect inputs')
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1
        net.eval()
        vals = []
        with torch.inference_mode():
            for batch in val_loader:
                pred = infer(net, batch['image'].to(device), tuple(args.roi))
                gt = batch['label'].to(device) > 0
                den = pred.sum() + gt.sum()
                vals.append(float((2 * (pred & gt).sum() / den).item()) if den else 1.)
        score = float(np.mean(vals))
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            atomic_torch_save(net.state_dict(), out / 'best.pth')
        else:
            bad += 1
        used_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step(score)
        history.append({'epoch': epoch, 'train_loss': total / count, 'inner_val_dice': score,
                        'learning_rate': used_lr, 'epoch_seconds': time.perf_counter() - started})
        atomic_torch_save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict() if scheduler else None, 'epoch': epoch,
                          'best': best, 'best_epoch': best_epoch, 'bad_epochs': bad, 'history': history}, last)
        pd.DataFrame(history).to_csv(out / 'training_log.csv', index=False)
        print(f'fold={args.fold} epoch={epoch} loss={total/count:.5f} inner_dice={score:.5f}', flush=True)
        if args.patience and bad >= args.patience:
            break
    mf.write_json(out / 'TRAINING_COMPLETED.json', {'best_epoch': best_epoch, 'best_inner_val_dice': best,
                  'epochs_completed': len(history), 'training_seconds': sum(r['epoch_seconds'] for r in history),
                  'checkpoint_sha256': mf.file_sha256(out / 'best.pth')})


def evaluate(args):
    import torch
    import nibabel as nib
    from .models import model, transforms, infer, load_weights
    from .metrics import binary_metrics
    from .figures import panels
    manifest = mf.load(args.manifest)
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    training_meta = checkpoint.parent / 'run.json'
    generated = False
    if training_meta.exists():
        training = json.loads(training_meta.read_text(encoding='utf-8'))
        generated = (training.get('manifest_sha256') == manifest['manifest_sha256'] and
                     training.get('fold') == args.fold and training.get('architecture') == args.arch)
        if not generated:
            raise ValueError('Checkpoint run metadata does not match architecture, fold or manifest')
        completed = checkpoint.parent / 'TRAINING_COMPLETED.json'
        if not completed.exists():
            raise ValueError('Complete training before evaluating the outer held-out fold')
        expected = json.loads(completed.read_text(encoding='utf-8'))['checkpoint_sha256']
        if mf.file_sha256(checkpoint) != expected:
            raise ValueError('Evaluation requires the selected best checkpoint from this completed run')
    elif manifest['provenance'] == 'imported_original_split_csv':
        if args.arch != 'unet' or checkpoint.name != f'best_metric_model_fold_{args.fold:02d}.pth':
            raise ValueError('Legacy weights must be the original UNet checkpoint matching this fold filename')
    trusted = generated or manifest['provenance'] == 'imported_original_split_csv'
    if not trusted and not args.allow_unverified_original_splits:
        raise ValueError('Original checkpoint splits are unverified. Import original CSV; or explicitly use --allow-unverified-original-splits for diagnostic results only.')
    evidence = 'heldout_with_persisted_training_manifest' if generated else ('heldout_with_imported_original_manifest' if trusted else 'UNVERIFIED_original_checkpoint_split')
    config = {'architecture': args.arch, 'model_name': args.name or args.arch, 'fold': args.fold,
              'manifest_sha256': manifest['manifest_sha256'], 'checkpoint_sha256': mf.file_sha256(checkpoint),
              'evidence': evidence, 'roi': args.roi, 'tolerance_mm': args.tolerance_mm,
              'device': str(device), 'environment': environment(),
              'timing_scope': 'MONAI_sliding_window_only_after_transfer_FP32_warmup',
              'evaluation_grid': 'RAS_1mm_image_foreground_divisible16', 'save_predictions': args.save_predictions,
              'figures_per_fold': args.figures_per_fold}
    out = Path(args.output) / f'fold_{args.fold:02d}'
    guard_run(out, config, args.resume)
    (out / 'cases').mkdir(exist_ok=True)
    net = load_weights(model(args.arch).to(device), checkpoint, device)
    tf = transforms(False, tuple(args.roi))
    rows, warmed = [], False
    for index, sample in enumerate(mf.samples(manifest, args.data_root, args.fold, 'heldout')):
        csv_path = out / 'cases' / f'{sample["case_id"]}.csv'
        if csv_path.exists():
            rows.extend(pd.read_csv(csv_path).to_dict('records'))
            continue
        data = tf(sample)
        inputs = data['image'].unsqueeze(0).to(device)
        gt = np.asarray(data['label'][0].cpu(), dtype=bool)
        with torch.inference_mode():
            if not warmed:
                _ = infer(net, inputs, tuple(args.roi))
                warmed = True
                del _
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            output = infer(net, inputs, tuple(args.roi))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else np.nan
        pred = np.asarray(output[0, 0].cpu(), dtype=bool)
        row = {'Case_ID': sample['case_id'], 'Patient_ID': sample['patient_id'], 'Fold': args.fold,
               'Model': args.name or args.arch, 'Evidence': evidence,
               'Inference_seconds': seconds, 'GPU_peak_allocated_MiB': peak,
               'Timing_scope': config['timing_scope'],
               'Manifest_SHA256': manifest['manifest_sha256'],
               **binary_metrics(pred, gt, tolerance_mm=args.tolerance_mm)}
        if args.save_predictions:
            pred_dir = out / 'predictions'
            pred_dir.mkdir(exist_ok=True)
            affine = np.asarray(data['label'].affine.cpu())
            nib.save(nib.Nifti1Image(pred.astype('uint8'), affine), pred_dir / f'{sample["case_id"]}.nii.gz')
        if index < args.figures_per_fold:
            row['Figure_slice_z'] = panels(np.asarray(data['image'].cpu()), gt, pred, out / 'figures', sample['case_id'])
        tmp = csv_path.with_suffix('.csv.tmp')
        pd.DataFrame([row]).to_csv(tmp, index=False)
        tmp.replace(csv_path)
        rows.append(row)
        print(f'{args.fold}: {index+1} {sample["case_id"]} Dice={row["Dice"]:.4f}', flush=True)
        del inputs, output, data
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    mf.write_json(out / 'EVALUATION_COMPLETED.json', {'n_cases': len(rows), 'evidence': evidence,
                  'checkpoint_sha256': config['checkpoint_sha256']})
