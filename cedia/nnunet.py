"""nnU-Net v2: cinco datasets independientes impiden que el planner vea el outer test."""
import json
import os
import subprocess
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
from . import manifest as mf


def export(args):
    import nibabel as nib
    manifest = mf.load(args.manifest)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for fold in manifest['folds']:
        number = fold['fold']
        name = f'Dataset{args.dataset_base + number:03d}_GliomaOuter{number}'
        dest = out / 'nnUNet_raw' / name
        metadata = {'manifest_sha256': manifest['manifest_sha256'], 'outer_fold': number,
                    'dataset_name': name, 'train': fold['train'], 'inner_val': fold['inner_val'],
                    'heldout': fold['heldout'], 'link_mode': args.link_mode,
                    'target': 'whole_tumor_binary_label_gt0'}
        audit = dest / 'outer_protocol.json'
        if audit.exists() and json.loads(audit.read_text(encoding='utf-8')) != metadata:
            raise ValueError(f'Different experiment already exported at {dest}')
        for sub in ('imagesTr','labelsTr','imagesTs'):
            (dest / sub).mkdir(parents=True, exist_ok=True)
        for role in mf.ROLES:
            for case in mf.samples(manifest, args.data_root, number, role):
                for i, path in enumerate(case['image']):
                    target = dest / ('imagesTs' if role == 'heldout' else 'imagesTr') / f'{case["case_id"]}_{i:04d}.nii.gz'
                    if not target.exists():
                        if args.link_mode == 'symlink':
                            target.symlink_to(Path(path).resolve())
                        else:
                            shutil.copy2(path, target)
                if role != 'heldout':
                    label = dest / 'labelsTr' / f'{case["case_id"]}.nii.gz'
                    if not label.exists():
                        im = nib.load(case['label'])
                        nib.save(nib.Nifti1Image((np.asarray(im.dataobj)>0).astype('uint8'),im.affine),label)
        mf.write_json(dest / 'dataset.json', {'channel_names':{'0':'T1','1':'T1c','2':'T2','3':'FLAIR'},
                      'labels':{'background':0,'whole_tumor':1},
                      'numTraining':len(fold['train'])+len(fold['inner_val']), 'file_ending':'.nii.gz'})
        # Installation AFTER planning is explicit; no automatic random folds.
        split = [{'train':fold['train'],'val':fold['inner_val']}]
        mf.write_json(dest / 'splits_final_required.json',split)
        mf.write_json(audit,metadata)
    mf.write_json(out / 'export.json', {'manifest_sha256':manifest['manifest_sha256'],
                  'dataset_base':args.dataset_base,'folds':[f['fold'] for f in manifest['folds']]})
    print(f'Export ready: {out}. Follow README planning -> install split -> train fold 0 for EACH outer dataset.')


def install_split(args):
    source = Path(args.dataset_dir)
    preprocessed = Path(args.preprocessed_dir)
    if not (preprocessed / 'nnUNetPlans.json').is_file():
        raise ValueError('Run nnUNetv2_plan_and_preprocess first; nnUNetPlans.json is absent')
    if source.name != preprocessed.name:
        raise ValueError('Raw and preprocessed dataset names differ')
    target = preprocessed / 'splits_final.json'
    required = json.loads((source/'splits_final_required.json').read_text(encoding='utf-8'))
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != required:
        raise FileExistsError('Existing splits_final.json differs; use a fresh nnUNet_preprocessed tree')
    mf.write_json(target,required)
    print(f'Custom split installed: {target}; use nnUNet inner fold 0 only')


def train(args):
    """Train only after checking the persisted inner split; bind weights to this protocol."""
    from .runner import environment
    manifest = mf.load(args.manifest)
    raw = Path(args.dataset_dir).resolve()
    exported = json.loads((raw / 'outer_protocol.json').read_text(encoding='utf-8'))
    if exported['manifest_sha256'] != manifest['manifest_sha256'] or exported['outer_fold'] != args.fold:
        raise ValueError('Export does not match manifest/fold')
    for variable in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        if not os.environ.get(variable):
            raise ValueError(f'Set {variable} before training')
    if Path(os.environ['nnUNet_raw']).resolve() != raw.parent:
        raise ValueError('nnUNet_raw does not point to this export')
    pre = Path(os.environ['nnUNet_preprocessed']) / raw.name
    required = json.loads((raw / 'splits_final_required.json').read_text(encoding='utf-8'))
    actual = json.loads((pre / 'splits_final.json').read_text(encoding='utf-8'))
    if actual != required:
        raise ValueError('Install the required inner split before training')
    selected = next(f for f in manifest['folds'] if f['fold'] == args.fold)
    if required != [{'train': selected['train'], 'val': selected['inner_val']}]:
        raise ValueError('Export split has changed')
    for role in mf.ROLES:
        mf.samples(manifest, args.data_root, args.fold, role)
    if args.trainer not in ('nnUNetTrainer_20epochs', 'nnUNetTrainer_100epochs', 'nnUNetTrainer'):
        raise ValueError('Supported trainer: nnUNetTrainer_20epochs, nnUNetTrainer_100epochs, nnUNetTrainer (1000 epochs)')
    trained = Path(os.environ['nnUNet_results']) / raw.name / f'{args.trainer}__nnUNetPlans__3d_fullres'
    audit_path = trained / 'cedia_protocol.json'
    record = {'manifest_sha256': manifest['manifest_sha256'], 'outer_fold': args.fold,
              'trainer': args.trainer, 'device': args.device, 'inner_fold': 0, 'split': actual,
              'plans_sha256': mf.file_sha256(pre / 'nnUNetPlans.json'), 'environment': environment()}
    if audit_path.exists():
        old = json.loads(audit_path.read_text(encoding='utf-8'))
        if {k: old.get(k) for k in record} != record or not args.resume:
            raise ValueError('Existing nnUNet run requires identical protocol and --resume')
        if old.get('completed'):
            if mf.file_sha256(trained/'fold_0'/'checkpoint_best.pth') != old['checkpoint_sha256']:
                raise ValueError('Completed checkpoint was modified')
            print(f'Already completed: {trained}')
            return
    elif trained.exists() and any(trained.iterdir()):
        raise FileExistsError('Use a fresh nnUNet_results directory; untracked model data exist')
    mf.write_json(audit_path, {**record, 'completed': False})
    command = ['nnUNetv2_train', raw.name, '3d_fullres', '0', '-tr', args.trainer, '-device', args.device]
    if args.resume and (trained/'fold_0'/'checkpoint_final.pth').exists():
        command.append('--val')
    elif args.resume and (trained/'fold_0'/'checkpoint_latest.pth').exists():
        command.append('--c')
    subprocess.run(command, check=True)
    mf.write_json(audit_path, {**record, 'completed': True,
                  'checkpoint_sha256': mf.file_sha256(trained/'fold_0'/'checkpoint_best.pth')})


def predict(args):
    """Uses the documented reader and public single-case API; records its wider timing scope."""
    manifest = mf.load(args.manifest)
    trained = Path(args.trained_model)
    raw_dir = Path(args.dataset_dir)
    exported = json.loads((raw_dir / 'outer_protocol.json').read_text(encoding='utf-8'))
    if exported['manifest_sha256'] != manifest['manifest_sha256'] or exported['outer_fold'] != args.fold:
        raise ValueError('Export does not match manifest/fold')
    if trained.parent.name != raw_dir.name:
        raise ValueError('nnUNet trained model directory belongs to another dataset')
    audit_path = trained / 'cedia_protocol.json'
    if not audit_path.is_file():
        raise ValueError('Use nnunet-train first: training partition provenance is absent')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if (audit['manifest_sha256'] != manifest['manifest_sha256'] or
            audit['outer_fold'] != args.fold or audit.get('completed') is not True):
        raise ValueError('Use nnunet-train to obtain a completed model with verified partition provenance')
    checkpoint = trained / 'fold_0' / 'checkpoint_best.pth'
    if mf.file_sha256(checkpoint) != audit['checkpoint_sha256']:
        raise ValueError('nnUNet checkpoint changed since training audit')
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from .runner import guard_run, environment
    device = torch.device(args.device)
    out = Path(args.output) / f'fold_{args.fold:02d}'
    config = {'architecture':'nnunet','fold':args.fold,'manifest_sha256':manifest['manifest_sha256'],
              'checkpoint_sha256':mf.file_sha256(checkpoint),'environment':environment(),'device':str(device),
              'timing_scope':'nnUNet_preprocessing_inference_resampling_export_after_read_warmup',
              'TTA':False,'checkpoint':'checkpoint_best.pth','inner_fold':0}
    guard_run(out,config,args.resume)
    predictor = nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,
                               perform_everything_on_device=device.type=='cuda',device=device,allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(trained),use_folds=(0,),checkpoint_name='checkpoint_best.pth')
    reader = predictor.plans_manager.image_reader_writer_class()
    (out / 'predictions').mkdir(exist_ok=True)
    (out / 'timing').mkdir(exist_ok=True)
    warmed = False
    for sample in mf.samples(manifest,args.data_root,args.fold,'heldout'):
        cid = sample['case_id']
        timing_path = out / 'timing' / f'{cid}.csv'
        dest = out / 'predictions' / cid
        if timing_path.exists() and Path(str(dest)+'.nii.gz').exists():
            continue
        array, props = reader.read_images(sample['image'])
        if not warmed:
            with torch.inference_mode():
                predictor.predict_single_npy_array(array,props)
            warmed = True
        if device.type=='cuda':
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.inference_mode():
            predictor.predict_single_npy_array(array,props,output_file_truncated=str(dest))
        if device.type=='cuda':
            torch.cuda.synchronize(device)
        seconds = time.perf_counter()-started
        peak = torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else np.nan
        pd.DataFrame([{'Case_ID':cid,'Inference_seconds':seconds,'GPU_peak_allocated_MiB':peak,
                       'Timing_scope':config['timing_scope']}]).to_csv(timing_path,index=False)
    print(f'Native-grid predictions and per-case timing: {out}')


def evaluate(args):
    import nibabel as nib
    from .models import transforms
    from .metrics import binary_metrics
    from .figures import panels
    from .runner import guard_run
    manifest = mf.load(args.manifest)
    pred_dir = Path(args.prediction_root) / f'fold_{args.fold:02d}'
    pred_meta = json.loads((pred_dir/'run.json').read_text(encoding='utf-8'))
    if pred_meta['manifest_sha256'] != manifest['manifest_sha256'] or pred_meta['fold'] != args.fold:
        raise ValueError('Prediction metadata mismatch')
    out = Path(args.output) / f'fold_{args.fold:02d}'
    metadata = {**pred_meta,'evaluation_grid':'RAS_1mm_image_foreground_divisible16',
                'tolerance_mm':args.tolerance_mm,'figures_per_fold':args.figures_per_fold}
    guard_run(out,metadata,args.resume)
    (out/'cases').mkdir(exist_ok=True)
    tf = transforms(with_prediction=True)
    rows = []
    for index, sample in enumerate(mf.samples(manifest,args.data_root,args.fold,'heldout')):
        cid = sample['case_id']
        case_csv = out/'cases'/f'{cid}.csv'
        if case_csv.exists():
            rows.extend(pd.read_csv(case_csv).to_dict('records')); continue
        prediction = pred_dir/'predictions'/f'{cid}.nii.gz'
        original, pred_im = nib.load(sample['label']),nib.load(prediction)
        if pred_im.shape!=original.shape or not np.allclose(pred_im.affine,original.affine,atol=1e-4):
            raise ValueError(f'Native prediction geometry mismatch: {cid}')
        data = tf({**sample,'prediction':str(prediction)})
        gt = np.asarray(data['label'][0],bool); pred = np.asarray(data['prediction'][0],bool)
        timing = pd.read_csv(pred_dir/'timing'/f'{cid}.csv').iloc[0].to_dict()
        row = {'Case_ID':cid,'Patient_ID':sample['patient_id'],'Fold':args.fold,'Model':'nnunet',
               'Evidence':'heldout_with_persisted_training_manifest','Manifest_SHA256':manifest['manifest_sha256'],
               **timing,**binary_metrics(pred,gt,tolerance_mm=args.tolerance_mm)}
        if index<args.figures_per_fold:
            row['Figure_slice_z'] = panels(np.asarray(data['image']),gt,pred,out/'figures',cid)
        pd.DataFrame([row]).to_csv(case_csv,index=False); rows.append(row)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    mf.write_json(out/'EVALUATION_COMPLETED.json',{'n_cases':len(rows),'manifest_sha256':manifest['manifest_sha256']})
