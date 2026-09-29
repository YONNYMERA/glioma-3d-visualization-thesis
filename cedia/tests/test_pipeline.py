"""Synthetic-only checks. Run: python -m unittest discover -s cedia/tests -v"""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import nibabel as nib
import torch
torch.set_num_threads(1)
from cedia import manifest as mf
from cedia.metrics import binary_metrics
from cedia.cli import main


def synthetic_dataset(root, n=10):
    rng = np.random.default_rng(99)
    for i in range(n):
        cid = f'BraTS-GLI-{i:05d}-000'
        folder = root / cid; folder.mkdir(parents=True)
        affine = np.diag([1.,1.,1.,1.]); affine[:3,3]=[-16,-16,-16]
        label = np.zeros((32,32,32),np.uint8); label[10:18,11:19,12:20]=2
        for m in mf.MODALITIES:
            image = rng.normal(10,2,label.shape).astype('float32')
            image[label>0]+=8
            nib.save(nib.Nifti1Image(image,affine),folder/f'{cid}-{m}.nii.gz')
        nib.save(nib.Nifti1Image(label,affine),folder/f'{cid}-seg.nii.gz')


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.data=cls.root/'data'; synthetic_dataset(cls.data)
        cls.path=cls.root/'manifest.json'
        cls.manifest=mf.prepare(cls.data,cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_partition_reproducible_and_original_import(self):
        m=mf.prepare(self.data,self.root/'second.json')
        self.assertEqual(m,self.manifest)
        held=[cid for f in m['folds'] for cid in f['heldout']]
        self.assertEqual(len(held),len(set(held)))
        from sklearn.model_selection import KFold,train_test_split
        ids=[c['case_id'] for c in m['cases']]
        for number,(pool,test) in enumerate(KFold(5,shuffle=True,random_state=0).split(ids),1):
            tr,va=train_test_split(pool,test_size=max(1,round(len(pool)*.15)),random_state=number)
            actual=m['folds'][number-1]
            self.assertEqual(actual['train'],[ids[i] for i in tr])
            self.assertEqual(actual['inner_val'],[ids[i] for i in va])
            self.assertEqual(actual['heldout'],[ids[i] for i in test])
        csvfile=self.root/'original.csv'
        with csvfile.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['Fold','Role','Case_ID']);writer.writeheader()
            names=['train','inner_val_checkpoint','held_out_evaluation']
            for fold in m['folds']:
                for role,old in zip(mf.ROLES,names):
                    for cid in fold[role]:
                        writer.writerow({'Fold':fold['fold'],'Role':old,'Case_ID':cid})
        imported=mf.prepare(self.data,self.root/'imported.json',original_csv=csvfile)
        self.assertEqual(imported['folds'],m['folds'])
        self.assertEqual(imported['provenance'],'imported_original_split_csv')

    def test_leakage_and_manifest_tampering_rejected(self):
        m=copy.deepcopy(self.manifest);m.pop('manifest_sha256')
        a,b=m['folds'][0]['train'][0],m['folds'][0]['heldout'][0]
        index={c['case_id']:c for c in m['cases']}
        index[a]['patient_id']=index[b]['patient_id']
        with self.assertRaisesRegex(ValueError,'leakage'):
            mf.validate(m)
        m=copy.deepcopy(self.manifest);m['seed']=4
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            mf.validate(m)

    def test_metrics_physical_spacing_empty_masks(self):
        gt=np.zeros((9,9,9),bool);gt[3:6,3:6,3:6]=True
        result=binary_metrics(gt,gt,spacing=(2,1,3))
        self.assertEqual(result['Dice'],1)
        self.assertEqual(result['SurfaceDice'],1)
        self.assertEqual(result['ASSD_mm'],0)
        self.assertAlmostEqual(result['Vol_GT_mL'],27*6/1000)
        one=np.zeros((9,9,9),bool);two=one.copy();one[3,3,3]=1;two[4,3,3]=1
        result=binary_metrics(one,two,spacing=(2,1,1))
        self.assertEqual(result['HD95_mm'],2)
        self.assertEqual(result['ASSD_mm'],2)
        result=binary_metrics(np.zeros_like(gt),gt)
        self.assertTrue(np.isinf(result['HD95_mm']))
        self.assertEqual(result['RVE_percent'],-100)
        self.assertTrue(np.isnan(result['Precision']))
        self.assertEqual(result['SurfaceDice'],0)
        result=binary_metrics(np.zeros_like(gt),np.zeros_like(gt))
        self.assertEqual(result['Dice'],1)
        self.assertTrue(np.isnan(result['RVE_percent']))

    def test_geometry_and_nnunet_no_outer_training_leakage(self):
        from cedia.models import transforms
        sample=mf.samples(self.manifest,self.data,1,'heldout')[0]
        transformed=transforms(with_prediction=True)({**sample,'prediction':sample['label']})
        self.assertEqual(transformed['image'].shape[0],4)
        self.assertTrue(torch.equal(transformed['prediction'],transformed['label']))
        self.assertTrue(set(np.unique(transformed['label'])).issubset({0.,1.}))
        self.assertTrue(np.allclose(np.linalg.norm(transformed['label'].affine[:3,:3],axis=0),1))
        from cedia.nnunet import export,install_split
        out=self.root/'nnunet'
        export(SimpleNamespace(manifest=self.path,data_root=self.data,output=out,dataset_base=700,link_mode='copy'))
        raw=out/'nnUNet_raw'/'Dataset701_GliomaOuter1'
        outer=set(self.manifest['folds'][0]['heldout'])
        training={p.name[:-len('_0000.nii.gz')] for p in (raw/'imagesTr').glob('*_0000.nii.gz')}
        self.assertFalse(outer & training)
        self.assertEqual(len(list((raw/'labelsTr').glob('*.nii.gz'))),8)
        pre=out/'preprocessed'/raw.name;pre.mkdir(parents=True)
        mf.write_json(pre/'nnUNetPlans.json',{})
        install_split(SimpleNamespace(dataset_dir=raw,preprocessed_dir=pre))
        split=json.loads((pre/'splits_final.json').read_text())[0]
        self.assertEqual(set(split['train']),set(self.manifest['folds'][0]['train']))
        self.assertEqual(set(split['val']),set(self.manifest['folds'][0]['inner_val']))

    def test_bootstrap_patient_unit(self):
        import pandas as pd
        from cedia.analysis import patient_bootstrap
        df=pd.DataFrame({'Patient_ID':['a','a','b'],'Fold':[1,1,1],'Dice':[0.,0.,1.]})
        s=patient_bootstrap(df,'Dice',100,1)
        self.assertEqual(s['n_patients_finite'],2)
        self.assertEqual(s['mean_finite_patient'],.5)

    def test_invalid_metric_geometry_and_surface_tolerance(self):
        gt=np.zeros((5,5,5),bool);gt[2,2,2]=True
        for spacing in [(0,1,1),(np.nan,1,1),(np.inf,1,1)]:
            with self.subTest(spacing=spacing),self.assertRaises(ValueError):
                binary_metrics(gt,gt,spacing=spacing)
        for tolerance in [-1,np.nan,np.inf]:
            with self.subTest(tolerance=tolerance),self.assertRaises(ValueError):
                binary_metrics(gt,gt,tolerance_mm=tolerance)
        corrupted=gt.astype(float);corrupted[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            binary_metrics(corrupted,gt)
        shifted=np.roll(gt,1,axis=0)
        zero=binary_metrics(shifted,gt,spacing=(2,1,1),tolerance_mm=0)
        wide=binary_metrics(shifted,gt,spacing=(2,1,1),tolerance_mm=10)
        self.assertEqual(zero['ASSD_mm'],2)
        self.assertLess(zero['SurfaceDice'],wide['SurfaceDice'])
        self.assertEqual(wide['SurfaceDice'],1)
        # A nonempty prediction against an empty reference is an explicit failure.
        empty=binary_metrics(gt,np.zeros_like(gt))
        self.assertEqual(empty['Dice'],0)
        self.assertTrue(np.isnan(empty['Sensitivity']))
        self.assertTrue(np.isnan(empty['RVE_percent']))
        self.assertTrue(np.isinf(empty['ASSD_mm']))

    def test_bootstrap_failures_and_cross_fold_patient_rejected(self):
        import pandas as pd
        from cedia.analysis import patient_bootstrap
        df=pd.DataFrame({'Patient_ID':['a','b'],'Fold':[1,1],'HD95_mm':[np.inf,np.nan]})
        result=patient_bootstrap(df,'HD95_mm',100,1)
        self.assertEqual(result['n_infinite'],1)
        self.assertEqual(result['n_undefined'],1)
        self.assertEqual(result['n_finite_cases'],0)
        self.assertTrue(np.isnan(result['mean_finite_patient']))
        df=pd.DataFrame({'Patient_ID':['a','a'],'Fold':[1,2],'Dice':[0.,1.]})
        with self.assertRaisesRegex(ValueError,'multiple held-out folds'):
            patient_bootstrap(df,'Dice',100,1)
        df=pd.DataFrame({'Patient_ID':['a','a','b','c'],'Fold':[1,1,1,2],'Dice':[0.,0.,1.,.5]})
        self.assertEqual(patient_bootstrap(df,'Dice',100,4),patient_bootstrap(df,'Dice',100,4))

    def test_analysis_pairing_guards_and_identifier_preservation(self):
        import pandas as pd
        from cedia.analysis import read_results,validate_pairs
        mask=np.zeros((5,5,5),bool);mask[2,2,2]=True
        rows=[]
        for model in ['a','b']:
            for cid in ['001','002']:
                rows.append({'Model':model,'Case_ID':cid,'Patient_ID':cid,'Fold':1,
                    'Manifest_SHA256':'sha','Evidence':'heldout_with_persisted_training_manifest',
                    'Timing_scope':'same','Inference_seconds':1.,'GPU_peak_allocated_MiB':np.nan,
                    **binary_metrics(mask,mask)})
        df=pd.DataFrame(rows);path=self.root/'pair_guard.csv';df.to_csv(path,index=False)
        loaded=read_results([path]);self.assertEqual(loaded.Case_ID.iloc[0],'001')
        validate_pairs(loaded)
        for column,value in [('Patient_ID','other'),('Fold',2),('Vol_GT_mL',8)]:
            changed=loaded.copy();changed.loc[changed.Model=='b',column]=value
            with self.subTest(column=column),self.assertRaises(ValueError):
                validate_pairs(changed)
        with self.assertRaisesRegex(ValueError,'same evaluated cases'):
            validate_pairs(loaded.iloc[:-1])
        # Mismatched pairs must fail before any new analysis artifacts are written.
        changed=loaded.copy();changed.loc[changed.Model=='b','Fold']=2;changed.to_csv(path,index=False)
        output=self.root/'invalid_pair_output'
        with self.assertRaisesRegex(ValueError,'identical patient and fold'):
            main(['analyze','--inputs',str(path),'--output',str(output),'--bootstrap','100'])
        self.assertFalse(output.exists())
        for column,value in [('surface_tolerance_mm',2.),('Timing_scope','different')]:
            changed=loaded.copy();changed.loc[0,column]=value;changed.to_csv(path,index=False)
            with self.subTest(column=column),self.assertRaises(ValueError):
                read_results([path])

    def test_analysis_empty_prediction_failures_are_reported(self):
        import pandas as pd
        mask=np.zeros((5,5,5),bool);mask[2,2,2]=True
        rows=[]
        for model in ['a','b']:
            for cid in ['001','002']:
                rows.append({'Model':model,'Case_ID':cid,'Patient_ID':cid,'Fold':1,
                    'Manifest_SHA256':'sha','Evidence':'heldout_with_persisted_training_manifest',
                    'Timing_scope':'same','Inference_seconds':1.,'GPU_peak_allocated_MiB':np.nan,
                    **binary_metrics(np.zeros_like(mask),mask)})
        source=self.root/'empty_pred.csv';pd.DataFrame(rows).to_csv(source,index=False)
        output=self.root/'empty_pred_analysis'
        main(['analyze','--inputs',str(source),'--output',str(output),'--bootstrap','100'])
        summary=pd.read_csv(output/'summary_ci_and_size.csv')
        hd=summary[(summary.Metric=='HD95_mm') & (summary.Size_group=='all')]
        self.assertTrue((hd.n_infinite==2).all())
        self.assertTrue(hd.mean_finite_patient.isna().all())
        self.assertEqual(len(pd.read_csv(output/'outliers_for_review.csv')),4)
        paired=pd.read_csv(output/'paired_patient_differences.csv')
        hd=paired[paired.Metric=='HD95_mm'].iloc[0]
        self.assertEqual(hd.n_nonfinite_left_cases,2)
        self.assertEqual(hd.n_nonfinite_right_cases,2)
        self.assertEqual(hd.n_both_finite_cases,0)

    def test_figure_input_geometry_rejected(self):
        from cedia.figures import panels
        image=np.zeros((4,5,5,5));mask=np.zeros((5,5,5),bool)
        with self.assertRaisesRegex(ValueError,'aligned'):
            panels(image[:3],mask,mask,self.root/'bad_panels','case')
        with self.assertRaisesRegex(ValueError,'aligned'):
            panels(image,mask,mask[:2],self.root/'bad_panels','case')
        image[0,0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):
            panels(image,mask,mask,self.root/'bad_panels','case')
        self.assertFalse((self.root/'bad_panels').exists())

    def test_synthetic_train_resume_evaluate_analysis(self):
        train=self.root/'training';evaluation=self.root/'evaluation';analysis=self.root/'analysis'
        common=['--data-root',str(self.data),'--manifest',str(self.path),'--fold','1']
        command=['train',*common,'--output',str(train),'--epochs','1','--batch-size','1','--workers','0','--roi','16','16','16','--device','cpu']
        main(command)
        checkpoint=train/'fold_01'/'best.pth';before=mf.file_sha256(checkpoint)
        main(command+['--resume'])
        self.assertEqual(before,mf.file_sha256(checkpoint))
        main(['evaluate',*common,'--output',str(evaluation),'--checkpoint',str(checkpoint),'--roi','32','32','32',
              '--device','cpu','--figures-per-fold','1','--save-predictions'])
        main(['analyze','--inputs',str(evaluation),'--output',str(analysis),'--bootstrap','100'])
        self.assertTrue((analysis/'summary_ci_and_size.csv').is_file())
        self.assertTrue((analysis/'architecture_comparison.pdf').is_file())
        self.assertTrue(list((evaluation/'fold_01'/'figures').glob('*comparison.png')))
        # A legacy checkpoint without its training manifest cannot silently become validated heldout.
        import shutil
        legacy=self.root/'legacy.pth';shutil.copyfile(checkpoint,legacy)
        with self.assertRaisesRegex(ValueError,'unverified'):
            main(['evaluate',*common,'--output',str(self.root/'blocked'),'--checkpoint',str(legacy),'--device','cpu'])


if __name__=='__main__':
    unittest.main()
