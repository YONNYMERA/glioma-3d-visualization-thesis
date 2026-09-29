#!/usr/bin/env python3
"""Verify the published cohort, evaluation evidence and checkpoint identities (stdlib)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cedia import manifest as mf


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checksums', action='store_true', help='Also verify every file in the release SHA256SUMS.txt')
    args = parser.parse_args()
    manifest = mf.load(ROOT/'provenance/manifest_por_paciente.json')
    cases = {r['case_id']: r for r in manifest['cases']}
    require(len(cases) == 1250 and len({r['patient_id'] for r in cases.values()}) == 1132, 'Unexpected cohort')
    weights = read(ROOT/'provenance/weights.json')['weights']
    index = {r['path']: r for r in weights}
    require(len(weights) == 20 and len(index) == 20, 'Expected 15 benchmark and five historical weights')
    pooled = list(csv.DictReader((ROOT/'results/comparacion_tres_modelos/all_cases.csv').open(encoding='utf-8')))
    require(len(pooled) == 3750, 'Expected 3750 case-model rows')
    require(len({(r['Model'], r['Case_ID']) for r in pooled}) == 3750, 'Duplicate result')
    for model in ('unet_baseline', 'segresnet_baseline', 'nnunet'):
        actual = [r for r in pooled if r['Model'] == model]
        require({r['Case_ID'] for r in actual} == set(cases), f'Incomplete model: {model}')
        for fold in manifest['folds']:
            n = fold['fold']
            folder = ROOT/f'results/eval/{model}/fold_{n:02d}'
            run = read(folder/'run.json')
            completed = read(folder/'EVALUATION_COMPLETED.json')
            require(run['manifest_sha256'] == manifest['manifest_sha256'], 'Wrong evaluation manifest')
            require(completed['n_cases'] == len(fold['heldout']), 'Invalid evaluation completion count')
            if model == 'nnunet':
                require(completed['manifest_sha256'] == manifest['manifest_sha256'], 'Invalid evaluation completion manifest')
            else:
                require(completed['checkpoint_sha256'] == run['checkpoint_sha256']
                        and completed['evidence'] == 'heldout_with_persisted_training_manifest',
                        'Invalid evaluation completion checkpoint/evidence')
            rows = list(csv.DictReader((folder/'metrics.csv').open(encoding='utf-8')))
            require(len(rows) == len(fold['heldout']) and {r['Case_ID'] for r in rows} == set(fold['heldout']), 'Incomplete heldout fold')
            for r in rows:
                require(r['Patient_ID'] == cases[r['Case_ID']]['patient_id'] and int(r['Fold']) == n, 'Patient/fold mismatch')
            subset = [r for r in actual if int(r['Fold']) == n]
            by_case = {r['Case_ID']: r for r in subset}
            for row in rows:
                other = by_case[row['Case_ID']]
                for key, value in row.items():
                    if value == other.get(key):
                        continue
                    try:
                        a, b = float(value), float(other[key])
                        require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                                or (math.isnan(a) and math.isnan(b)), f'Pooled metric differs: {key}')
                    except (ValueError, TypeError, KeyError):
                        raise ValueError(f'Pooled value differs: {row["Case_ID"]} {key}')
            if model != 'nnunet':
                key = f'train/{model}/fold_{n:02d}/best.pth'
                training = read(ROOT/'results'/Path(key).parent/'TRAINING_COMPLETED.json')
                require(training['epochs_completed'] == 20, 'Unexpected MONAI epochs')
            else:
                key = f'nnunet/nnUNet_results/Dataset{700+n}_GliomaOuter{n}/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth'
                training = read(ROOT/'results'/Path(key).parent.parent/'cedia_protocol.json')
                require(training['completed'] is True, 'Incomplete nnU-Net training')
            require(run['checkpoint_sha256'] == index[key]['sha256'] == training['checkpoint_sha256'], 'Checkpoint identity mismatch')
    protocol = read(ROOT/'results/comparacion_tres_modelos/analysis_protocol.json')
    require(protocol['complete_cohort_verified'] and protocol['manifest_sha256'] == manifest['manifest_sha256'], 'Invalid final analysis protocol')
    checked = 0
    if args.checksums:
        for line in (ROOT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
            expected, name = line.split('  ', 1)
            path = (ROOT/name).resolve()
            require(path.is_relative_to(ROOT), 'Unsafe checksum path')
            require(mf.file_sha256(path) == expected, f'File changed: {name}')
            checked += 1
    print(json.dumps({'status':'OK', 'studies':1250, 'patients':1132, 'benchmark_models':3,
                      'heldout_evaluations':15, 'rows':3750, 'checkpoint_identities':15,
                      'release_files_checked':checked}, indent=2))


if __name__ == '__main__':
    main()
