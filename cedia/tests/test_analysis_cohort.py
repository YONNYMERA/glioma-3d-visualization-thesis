"""A matched partial cohort must not be certified as a complete manifest analysis."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from cedia import manifest as mf
from cedia.cli import main


class CompleteCohortTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        ids=[f'{i:03d}' for i in range(1,11)]
        cases=[{'case_id':cid,'patient_id':f'patient-{cid}'} for cid in ids]
        folds=[]
        for i in range(5):
            held=ids[2*i:2*i+2];pool=[cid for cid in ids if cid not in held]
            folds.append({'fold':i+1,'train':pool[:-1],'inner_val':pool[-1:],'heldout':held})
        manifest={'schema':1,'cases':cases,'folds':folds}
        mf.validate(manifest);manifest['manifest_sha256']=mf.digest(manifest)
        self.manifest=self.root/'manifest.json';mf.write_json(self.manifest,manifest)
        heldfold={cid:f['fold'] for f in folds for cid in f['heldout']}
        rows=[]
        for model in ['unet','segresnet']:
            for cid in ids:
                rows.append({'Case_ID':cid,'Patient_ID':f'patient-{cid}','Fold':heldfold[cid],
                    'Model':model,'Manifest_SHA256':manifest['manifest_sha256'],
                    'Evidence':'heldout_with_persisted_training_manifest','Timing_scope':'same',
                    'Vol_GT_mL':float(int(cid)),'Vol_Pred_mL':float(int(cid)),
                    'Dice':1.,'HD95_mm':0.,'ASSD_mm':0.,'SurfaceDice':1.,'AVD_mL':0.,
                    'surface_tolerance_mm':1.,'Inference_seconds':1.,'GPU_peak_allocated_MiB':np.nan})
        self.frame=pd.DataFrame(rows)

    def analyze(self,frame,name,with_manifest=True):
        source=self.root/f'{name}.csv';frame.to_csv(source,index=False)
        output=self.root/name
        command=['analyze','--inputs',str(source),'--output',str(output),'--bootstrap','100']
        if with_manifest:
            command.extend(['--manifest',str(self.manifest)])
        main(command)
        return json.loads((output/'analysis_protocol.json').read_text())

    def test_complete_manifest_cohort_passes_and_partial_preview_is_not_verified(self):
        protocol=self.analyze(self.frame,'complete')
        self.assertTrue(protocol['complete_cohort_verified'])
        self.assertEqual(protocol['expected_cases_per_model'],10)
        self.assertEqual(protocol['expected_heldout_folds'],[1,2,3,4,5])
        preview=self.frame[(self.frame.Model=='unet') & (self.frame.Fold==1)]
        protocol=self.analyze(preview,'preview',with_manifest=False)
        self.assertFalse(protocol['complete_cohort_verified'])
        self.assertIsNone(protocol['expected_cases_per_model'])
        self.assertIsNone(protocol['expected_heldout_folds'])

    def test_same_missing_case_or_fold_for_all_models_is_rejected(self):
        for name,frame in [('missing_case',self.frame[self.frame.Case_ID!='010']),
                           ('missing_fold',self.frame[self.frame.Fold!=5])]:
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'Complete manifest cohort'):
                self.analyze(frame,name)
            self.assertFalse((self.root/name).exists())

    def test_manifest_hash_and_patient_fold_assignments_are_checked(self):
        for column,value in [('Manifest_SHA256','wrong-sha'),('Patient_ID','different-patient'),('Fold',2)]:
            frame=self.frame.copy()
            if column=='Manifest_SHA256':
                frame[column]=value
            else:
                # Both models agree with each other but disagree with the persisted manifest.
                frame.loc[frame.Case_ID=='001',column]=value
            name=f'bad_{column}'
            with self.subTest(column=column),self.assertRaisesRegex(ValueError,'supplied manifest'):
                self.analyze(frame,name)
            self.assertFalse((self.root/name).exists())


if __name__=='__main__':
    unittest.main()
