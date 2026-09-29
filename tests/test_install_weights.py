"""Safety and reproducibility checks with tiny synthetic checkpoint payloads."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_weights.py"
SPEC = importlib.util.spec_from_file_location("install_weights", SCRIPT)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class InstallWeightsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "unet_weights.tar.gz"
        self.output = self.root / "installed"
        self.metadata = self.root / "results"
        self.index = self.root / "weights.json"
        self.relative = "train/unet_baseline/fold_01/best.pth"
        self.payload = b"synthetic checkpoint: never deserialize"

    def package(self, members=None, registered=None, model="unet_baseline"):
        members = [(self.relative, self.payload, tarfile.REGTYPE)] if members is None else members
        with tarfile.open(self.archive, "w:gz") as bundle:
            for name, content, kind in members:
                entry = tarfile.TarInfo(name)
                entry.type = kind
                entry.size = len(content) if kind == tarfile.REGTYPE else 0
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    entry.linkname = "../../outside.pth"
                bundle.addfile(entry, io.BytesIO(content) if entry.isreg() else None)
        registered = {self.relative: self.payload} if registered is None else registered
        inventory = {
            "schema": 1,
            "archives": [{"name": self.archive.name, "sha256": installer.sha256_file(self.archive), "model": model}],
            "weights": [{"model": model, "path": name, "bytes": len(content), "sha256": digest(content)}
                        for name, content in registered.items()],
        }
        self.index.write_text(json.dumps(inventory), encoding="utf-8")

    def install(self):
        return installer.install_archive(self.archive, self.output, self.index, self.metadata)

    def test_valid_metadata_and_idempotence(self):
        self.package()
        run = self.metadata / "train/unet_baseline/fold_01/run.json"
        run.parent.mkdir(parents=True)
        run.write_text('{"fold": 1}', encoding="utf-8")
        completed = run.with_name("TRAINING_COMPLETED.json")
        completed.write_text('{"completed": true}', encoding="utf-8")
        report = self.install()
        self.assertEqual(len(report["installed"]), 3)
        self.assertEqual((self.output / self.relative).read_bytes(), self.payload)
        self.assertEqual((self.output / run.relative_to(self.metadata)).read_bytes(), run.read_bytes())
        original_mtime = (self.output / self.relative).stat().st_mtime_ns
        second = self.install()
        self.assertEqual(second["installed"], [])
        self.assertEqual(len(second["already_present"]), 3)
        self.assertEqual((self.output / self.relative).stat().st_mtime_ns, original_mtime)

    def test_archive_hash_mismatch_writes_nothing(self):
        self.package()
        with self.archive.open("ab") as stream:
            stream.write(b"tampering")
        with self.assertRaisesRegex(installer.InstallError, "Archive SHA-256"):
            self.install()
        self.assertFalse(self.output.exists())

    def test_missing_metadata_writes_nothing(self):
        self.package()
        with self.assertRaisesRegex(installer.InstallError, "Required metadata missing"):
            self.install()
        self.assertFalse(self.output.exists())

    def test_checkpoint_hash_mismatch_writes_nothing(self):
        self.package(members=[(self.relative, b"X" * len(self.payload), tarfile.REGTYPE)])
        with self.assertRaisesRegex(installer.InstallError, "Checkpoint SHA-256"):
            self.install()
        self.assertFalse(self.output.exists())

    def test_malicious_paths_and_links_write_nothing(self):
        for name, kind in (("../outside.pth", tarfile.REGTYPE), ("/outside.pth", tarfile.REGTYPE),
                           ("C:/outside.pth", tarfile.REGTYPE), ("train\\outside.pth", tarfile.REGTYPE),
                           (self.relative, tarfile.SYMTYPE), (self.relative, tarfile.LNKTYPE)):
            with self.subTest(name=name, kind=kind):
                self.package(members=[(name, self.payload, kind)])
                with self.assertRaises(installer.InstallError):
                    self.install()
                self.assertFalse(self.output.exists())
                self.assertFalse((self.root / "outside.pth").exists())

    def test_extra_missing_and_duplicate_members_rejected(self):
        valid = (self.relative, self.payload, tarfile.REGTYPE)
        for members in ([valid, ("unknown.pth", b"x", tarfile.REGTYPE)], [], [valid, valid]):
            with self.subTest(members=members):
                self.package(members=members)
                with self.assertRaises(installer.InstallError):
                    self.install()
                self.assertFalse(self.output.exists())

    def test_conflicting_existing_file_preserved_before_any_write(self):
        second = "train/unet_baseline/fold_02/best.pth"
        self.package(members=[(self.relative, self.payload, tarfile.REGTYPE), (second, b"second", tarfile.REGTYPE)],
                     registered={self.relative: self.payload, second: b"second"})
        existing = self.output / second
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"valuable existing model")
        with self.assertRaisesRegex(installer.InstallError, "different content"):
            self.install()
        self.assertEqual(existing.read_bytes(), b"valuable existing model")
        self.assertFalse((self.output / self.relative).exists())

    def test_conflicting_metadata_preserved_before_any_write(self):
        self.package()
        relative = "train/unet_baseline/fold_01/run.json"
        source, existing = self.metadata / relative, self.output / relative
        source.parent.mkdir(parents=True)
        existing.parent.mkdir(parents=True)
        source.write_text('{"fold": 1}', encoding="utf-8")
        existing.write_text('{"different": true}', encoding="utf-8")
        with self.assertRaisesRegex(installer.InstallError, "different content"):
            self.install()
        self.assertEqual(existing.read_text(encoding="utf-8"), '{"different": true}')
        self.assertFalse((self.output / self.relative).exists())

    def test_nnunet_metadata_location(self):
        relative = "nnunet/nnUNet_results/Dataset701_GliomaOuter1/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth"
        self.package(members=[(relative, self.payload, tarfile.REGTYPE)], registered={relative: self.payload}, model="nnunet")
        model_directory = (self.metadata / relative).parent.parent
        model_directory.mkdir(parents=True)
        for filename in ("plans.json", "dataset.json", "cedia_protocol.json"):
            (model_directory / filename).write_text("{}", encoding="utf-8")
        self.assertEqual(len(self.install()["installed"]), 4)

    def test_historical_weights_only(self):
        relative = "MODELS/best_metric_model_fold_01.pth"
        self.package(members=[(relative, self.payload, tarfile.REGTYPE)], registered={relative: self.payload}, model="historical_app")
        report = self.install()
        self.assertEqual(report["installed"], [relative])
        self.assertEqual(report["metadata_not_available"], [])


if __name__ == "__main__":
    unittest.main()
