import csv
import tempfile
import unittest
from pathlib import Path
from cedia import manifest as mf


class OriginalInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        for i in range(10):
            self.add_case(f'BraTS-GLI-{i:05d}-000')
        self.original = mf.prepare(self.data, self.root / 'initial.json')
        self.csv = self.root / 'splits.csv'
        with self.csv.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Fold', 'Role', 'Case_ID'])
            for fold in self.original['folds']:
                for role, old in zip(mf.ROLES, ['train', 'inner_val_checkpoint', 'held_out_evaluation']):
                    for cid in fold[role]:
                        writer.writerow([fold['fold'], old, cid])

    def add_case(self, cid, complete=True):
        folder = self.data / cid
        folder.mkdir(parents=True)
        for modality in (*mf.MODALITIES, 'seg') if complete else mf.MODALITIES:
            (folder / f'{cid}-{modality}.nii.gz').write_bytes(b'inventory-only fixture')

    def test_extra_cases_are_ignored_only_when_outside_original_csv(self):
        self.add_case('BraTS-GLI-99998-000', complete=False)
        self.add_case('BraTS-GLI-99999-000')
        imported = mf.prepare(self.data, self.root / 'imported.json', original_csv=self.csv)
        self.assertEqual(imported['folds'], self.original['folds'])
        self.assertEqual(imported['cases'], self.original['cases'])
        self.assertEqual({r['directory'] for r in imported['ignored_directories']},
                         {'BraTS-GLI-99998-000', 'BraTS-GLI-99999-000'})
        with self.assertRaisesRegex(ValueError, 'Incomplete case'):
            mf.prepare(self.data, self.root / 'unverified.json')

    def test_missing_required_mask_still_fails(self):
        cid = 'BraTS-GLI-00000-000'
        (self.data / cid / f'{cid}-seg.nii.gz').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete case'):
            mf.prepare(self.data, self.root / 'bad.json', original_csv=self.csv)
        self.assertFalse((self.root / 'bad.json').exists())

    def test_required_case_without_t1n_cannot_be_silently_skipped(self):
        cid = 'BraTS-GLI-00000-000'
        (self.data / cid / f'{cid}-t1n.nii.gz').unlink()
        with self.assertRaisesRegex(ValueError, 'Original split cases missing'):
            mf.prepare(self.data, self.root / 'bad.json', original_csv=self.csv)

    def test_empty_original_csv_rejected(self):
        self.csv.write_text('Fold,Role,Case_ID\n')
        with self.assertRaisesRegex(ValueError, 'nonempty case IDs'):
            mf.prepare(self.data, self.root / 'bad.json', original_csv=self.csv)
