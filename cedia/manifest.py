"""Inventario portable y particiones persistidas, con control de fuga por paciente."""
import csv
import hashlib
import json
import re
from pathlib import Path

MODALITIES = ('t1n', 't1c', 't2w', 't2f')
ROLES = ('train', 'inner_val', 'heldout')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def inventory(root, patient_map=None, required_case_ids=None):
    root = Path(root).resolve()
    mapping = {}
    if patient_map:
        with open(patient_map, encoding='utf-8-sig', newline='') as f:
            mapping = {r['case_id']: r['patient_id'] for r in csv.DictReader(f)}
    cases, skipped = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if required_case_ids is not None and folder.name not in required_case_ids:
            skipped.append({'directory': folder.name, 'reason': 'Not in original split CSV'})
            continue
        matches = sorted(folder.glob('*-t1n.nii.gz'))
        if len(matches) != 1:
            skipped.append({'directory': folder.name, 'reason': 'Expected exactly one *-t1n.nii.gz'})
            continue
        prefix = matches[0].name[:-len('-t1n.nii.gz')]
        paths = [folder / f'{prefix}-{m}.nii.gz' for m in (*MODALITIES, 'seg')]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise ValueError(f'Incomplete case {folder.name}: {missing}')
        inferred_patient = re.sub(r'^(BraTS-GLI-\d+)-\d+$', r'\1', prefix)
        pid = mapping.get(folder.name, inferred_patient)
        cases.append({'case_id': folder.name, 'patient_id': pid,
                      'image': [p.relative_to(root).as_posix() for p in paths[:4]],
                      'label': paths[4].relative_to(root).as_posix(),
                      'sizes_bytes': [p.stat().st_size for p in paths]})
    if required_case_ids is not None:
        absent = set(required_case_ids) - {c['case_id'] for c in cases}
        if absent:
            raise ValueError(f'Original split cases missing from inventory: {sorted(absent)}')
    if not cases:
        raise ValueError('No complete BraTS cases found')
    if mapping and set(mapping) != {c['case_id'] for c in cases}:
        raise ValueError('patient-map must contain exactly all case IDs')
    return cases, skipped


def validate(manifest):
    cases = manifest['cases']
    ids = {c['case_id'] for c in cases}
    if len(ids) != len(cases):
        raise ValueError('Duplicate case IDs')
    patients = {c['case_id']: c['patient_id'] for c in cases}
    held = []
    for fold in manifest['folds']:
        sets = [set(fold[r]) for r in ROLES]
        if any(len(s) != len(fold[r]) for s, r in zip(sets, ROLES)):
            raise ValueError('Duplicate case within a role')
        if not all(sets) or set.union(*sets) != ids:
            raise ValueError('Every fold must assign each case exactly one nonempty role')
        for i in range(3):
            for j in range(i + 1, 3):
                if sets[i] & sets[j] or {patients[c] for c in sets[i]} & {patients[c] for c in sets[j]}:
                    raise ValueError('Patient/case leakage across roles')
        held.extend(fold['heldout'])
    if len(held) != len(ids) or set(held) != ids:
        raise ValueError('Every case must be held out exactly once')
    if manifest.get('manifest_sha256') and manifest['manifest_sha256'] != digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'}):
        raise ValueError('Manifest fingerprint mismatch; do not edit persisted splits')


def prepare(root, output, nfolds=5, seed=0, inner_fraction=.15, original_csv=None, patient_map=None):
    from sklearn.model_selection import KFold, train_test_split
    original_rows = None
    if original_csv:
        with open(original_csv, encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not {'Fold', 'Role', 'Case_ID'}.issubset(reader.fieldnames or []):
                raise ValueError('Original CSV requires Fold,Role,Case_ID columns')
            original_rows = list(reader)
        if not original_rows or any(not r.get('Case_ID') for r in original_rows):
            raise ValueError('Original CSV must contain nonempty case IDs')
    required_case_ids = {r['Case_ID'] for r in original_rows} if original_rows is not None else None
    cases, skipped = inventory(root, patient_map, required_case_ids)
    unique = sorted({c['patient_id'] for c in cases})
    # Original notebook split sorted case directories; preserve this ordering when one scan/patient.
    one_scan = len(unique) == len(cases)
    units = [c['patient_id'] for c in cases] if one_scan else unique
    if len(units) < nfolds or not 0 < inner_fraction < 1:
        raise ValueError('Insufficient patients or invalid inner fraction')
    folds = []
    provenance = 'reconstructed_from_current_inventory_unverified_for_original_checkpoints'
    if original_csv:
        rows = original_rows
        rolemap = {'train': 'train', 'inner_val_checkpoint': 'inner_val', 'held_out_evaluation': 'heldout'}
        for number in sorted({int(r['Fold']) for r in rows}):
            fold = {'fold': number, **{role: [] for role in ROLES}}
            for row in rows:
                if int(row['Fold']) == number:
                    fold[rolemap[row['Role']]].append(row['Case_ID'])
            folds.append(fold)
        provenance = 'imported_original_split_csv'
    else:
        for number, (pool, held) in enumerate(KFold(nfolds, shuffle=True, random_state=seed).split(units), 1):
            count = max(1, min(len(pool) - 1, round(len(pool) * inner_fraction)))
            train, val = train_test_split(pool, test_size=count, random_state=seed + number, shuffle=True)
            fold = {'fold': number}
            for role, idx in zip(ROLES, (train, val, held)):
                selected = {units[i] for i in idx}
                # Match train_test_split ordering from notebook for single-scan patients.
                fold[role] = [c['case_id'] for i in idx for c in cases if c['patient_id'] == units[i]]
            folds.append(fold)
    manifest = {'schema': 1, 'seed': seed, 'inner_fraction': inner_fraction,
                'provenance': provenance, 'patient_grouping': 'one_scan_per_patient' if one_scan else 'grouped_multiple_scans',
                'original_csv_sha256': file_sha256(original_csv) if original_csv else None,
                'cases': cases, 'folds': folds, 'ignored_directories': skipped}
    validate(manifest)
    manifest['manifest_sha256'] = digest(manifest)
    path = Path(output)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise FileExistsError(f'{path} already contains a different experiment. Use another path.')
    else:
        write_json(path, manifest)
    return manifest


def load(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    validate(manifest)
    return manifest


def samples(manifest, root, fold, role):
    selected = next(f for f in manifest['folds'] if f['fold'] == fold)[role]
    index = {c['case_id']: c for c in manifest['cases']}
    result = []
    for cid in selected:
        c = index[cid]
        image = [str(Path(root) / p) for p in c['image']]
        label = str(Path(root) / c['label'])
        if [Path(p).stat().st_size for p in image + [label]] != c['sizes_bytes']:
            raise ValueError(f'Dataset changed after manifest creation: {cid}')
        result.append({'case_id': cid, 'patient_id': c['patient_id'], 'image': image, 'label': label})
    return result
