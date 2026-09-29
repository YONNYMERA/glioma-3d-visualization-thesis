"""Protocol checks with mocked nnU-Net subprocesses; no network training or inference."""
import builtins
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cedia import manifest as mf
from cedia import nnunet


class NNUNetProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root/'raw'/'Dataset701_GliomaOuter1'
        self.pre = self.root/'pre' / self.raw.name
        self.results = self.root/'results'
        self.trained = self.results/self.raw.name/'nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres'
        self.audit_path = self.trained/'cedia_protocol.json'
        self.best = self.trained/'fold_0'/'checkpoint_best.pth'
        self.manifest = {'manifest_sha256':'a'*64,
                         'folds':[{'fold':1,'train':['A'],'inner_val':['B'],'heldout':['C']}]}
        self.required = [{'train':['A'],'val':['B']}]
        mf.write_json(self.raw/'outer_protocol.json', {'manifest_sha256':'a'*64,'outer_fold':1})
        mf.write_json(self.raw/'splits_final_required.json', self.required)
        mf.write_json(self.pre/'splits_final.json', self.required)
        mf.write_json(self.pre/'nnUNetPlans.json', {'test':'plan'})
        self.args = SimpleNamespace(manifest=self.root/'manifest.json',data_root=self.root/'data',
            dataset_dir=self.raw,fold=1,trainer='nnUNetTrainer_100epochs',device='cpu',resume=False,
            trained_model=self.trained,output=self.root/'predictions')
        for context in (
            patch.dict(os.environ, {'nnUNet_raw':str(self.raw.parent),
                'nnUNet_preprocessed':str(self.pre.parent),'nnUNet_results':str(self.results)}),
            patch.object(mf,'load',return_value=self.manifest),
            patch.object(mf,'samples',return_value=[]),
            patch('cedia.runner.environment',return_value={'synthetic_test':True}),
        ):
            context.start()
            self.addCleanup(context.stop)

    def fake_success(self, command, check):
        self.assertTrue(check)
        # Before launching external training, provenance must say incomplete and contain no best hash.
        initial=json.loads(self.audit_path.read_text())
        self.assertFalse(initial['completed'])
        self.assertNotIn('checkpoint_sha256',initial)
        self.best.parent.mkdir(parents=True,exist_ok=True)
        self.best.write_bytes(b'synthetic checkpoint, not torch weights')
        return subprocess.CompletedProcess(command,0)

    def complete_training(self):
        with patch.object(nnunet.subprocess,'run',side_effect=self.fake_success) as run:
            nnunet.train(self.args)
        return run

    def test_bad_manifest_or_export_never_launches_subprocess(self):
        with patch.object(mf,'load',side_effect=ValueError('manifest fingerprint mismatch')):
            with patch.object(nnunet.subprocess,'run') as run:
                with self.assertRaisesRegex(ValueError,'fingerprint'):
                    nnunet.train(self.args)
                run.assert_not_called()
        mf.write_json(self.raw/'outer_protocol.json',{'manifest_sha256':'wrong','outer_fold':1})
        with patch.object(nnunet.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'manifest/fold'):
                nnunet.train(self.args)
            run.assert_not_called()
        self.assertFalse(self.audit_path.exists())

    def test_bad_inner_split_never_launches_subprocess(self):
        changed=[{'train':['A','C'],'val':['B']}]
        mf.write_json(self.pre/'splits_final.json',changed)
        with patch.object(nnunet.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'required inner split'):
                nnunet.train(self.args)
            run.assert_not_called()
        # Altering BOTH split files still cannot override the manifest's reserved partition.
        mf.write_json(self.raw/'splits_final_required.json',changed)
        with patch.object(nnunet.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'split has changed'):
                nnunet.train(self.args)
            run.assert_not_called()
        self.assertFalse(self.audit_path.exists())

    def test_success_records_checkpoint_hash_and_completed_resume_skips(self):
        run=self.complete_training()
        run.assert_called_once_with(['nnUNetv2_train',self.raw.name,'3d_fullres','0',
            '-tr','nnUNetTrainer_100epochs','-device','cpu'],check=True)
        audit=json.loads(self.audit_path.read_text())
        self.assertTrue(audit['completed'])
        self.assertEqual(audit['checkpoint_sha256'],mf.file_sha256(self.best))
        self.assertEqual(audit['plans_sha256'],mf.file_sha256(self.pre/'nnUNetPlans.json'))
        self.assertEqual(audit['split'],self.required)
        self.args.resume=True
        with patch.object(nnunet.subprocess,'run') as run:
            nnunet.train(self.args)
            run.assert_not_called()

    def test_failed_training_does_not_claim_completion(self):
        with patch.object(nnunet.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'mock')):
            with self.assertRaises(subprocess.CalledProcessError):
                nnunet.train(self.args)
        audit=json.loads(self.audit_path.read_text())
        self.assertFalse(audit['completed'])
        self.assertNotIn('checkpoint_sha256',audit)
        # Even a subprocess success is insufficient if it produced no selected checkpoint.
        self.args.resume=True
        with patch.object(nnunet.subprocess,'run',return_value=subprocess.CompletedProcess('mock',0)):
            with self.assertRaises(FileNotFoundError):
                nnunet.train(self.args)
        self.assertFalse(json.loads(self.audit_path.read_text())['completed'])

    def test_changed_checkpoint_rejects_resume(self):
        self.complete_training();self.args.resume=True
        self.best.write_bytes(b'tampered')
        with patch.object(nnunet.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'checkpoint was modified'):
                nnunet.train(self.args)
            run.assert_not_called()

    def test_changed_device_and_plans_reject_resume(self):
        self.complete_training();self.args.resume=True;self.args.device='cuda'
        with patch.object(nnunet.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'identical protocol'):
                nnunet.train(self.args)
            run.assert_not_called()
        self.args.device='cpu'
        mf.write_json(self.pre/'nnUNetPlans.json',{'test':'changed'})
        with patch.object(nnunet.subprocess,'run') as run:
            with self.assertRaisesRegex(ValueError,'identical protocol'):
                nnunet.train(self.args)
            run.assert_not_called()

    def test_incomplete_resume_uses_latest_checkpoint(self):
        with patch.object(nnunet.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'mock')):
            with self.assertRaises(subprocess.CalledProcessError):
                nnunet.train(self.args)
        latest=self.best.parent/'checkpoint_latest.pth'
        latest.parent.mkdir(parents=True);latest.write_bytes(b'latest')
        self.args.resume=True
        run=self.complete_training()
        self.assertIn('--c',run.call_args.args[0])

    def test_predictor_rejects_missing_incomplete_and_changed_weights_before_heavy_import(self):
        real_import=builtins.__import__
        def checked_import(name,*args,**kwargs):
            if name=='torch' or name.startswith('nnunetv2'):
                raise AssertionError(f'Heavy import before provenance rejection: {name}')
            return real_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=checked_import):
            with self.assertRaisesRegex(ValueError,'provenance is absent'):
                nnunet.predict(self.args)
        mf.write_json(self.audit_path,{'manifest_sha256':'a'*64,'outer_fold':1,'completed':False})
        with patch('builtins.__import__',side_effect=checked_import):
            with self.assertRaisesRegex(ValueError,'completed model'):
                nnunet.predict(self.args)
        self.best.parent.mkdir(parents=True,exist_ok=True);self.best.write_bytes(b'weights')
        mf.write_json(self.audit_path,{'manifest_sha256':'a'*64,'outer_fold':1,'completed':True,
                                     'checkpoint_sha256':'wrong'})
        with patch('builtins.__import__',side_effect=checked_import):
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                nnunet.predict(self.args)
        self.assertFalse(self.args.output.exists())

    def test_monai_outer_evaluation_requires_completed_selected_checkpoint(self):
        from cedia import runner
        checkpoint=self.root/'monai'/'fold_01'/'best.pth'
        checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'selected weights')
        mf.write_json(checkpoint.parent/'run.json',{'manifest_sha256':'a'*64,'fold':1,'architecture':'unet'})
        args=SimpleNamespace(manifest=self.args.manifest,device='cpu',checkpoint=checkpoint,
                             fold=1,arch='unet')
        with patch('cedia.models.load_weights') as load:
            with self.assertRaisesRegex(ValueError,'Complete training'):
                runner.evaluate(args)
            load.assert_not_called()
        # The saved best-checkpoint hash is mandatory, not merely an existing completion file.
        mf.write_json(checkpoint.parent/'TRAINING_COMPLETED.json',{'checkpoint_sha256':'wrong'})
        with patch('cedia.models.load_weights') as load:
            with self.assertRaisesRegex(ValueError,'selected best checkpoint'):
                runner.evaluate(args)
            load.assert_not_called()


if __name__=='__main__':
    unittest.main()
