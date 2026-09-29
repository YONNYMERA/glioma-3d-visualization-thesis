"""Synthetic tests; no BraTS images or checkpoint deserialization required."""
import json
from pathlib import Path
import tempfile
import unittest

from cedia import manifest as mf
from scripts.prepare_cohort import prepare_cohort


class PrepareCohortTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "raw"
        self.output = self.root / "cohort"
        self.manifest = self.root / "manifest.json"
        cases = []
        for i in range(3):
            cid = f"BraTS-GLI-{i:05d}-000"
            folder = self.source / cid
            folder.mkdir(parents=True)
            names = [f"{cid}/{cid}-{m}.nii.gz" for m in (*mf.MODALITIES, "seg")]
            for j, name in enumerate(names):
                target = self.source / name
                if j == 4:
                    target = target.with_name(target.name.replace("-seg.", "-segs."))
                target.write_bytes(f"case{i} modality{j}".encode())
            cases.append({"case_id": cid, "patient_id": f"patient{i}",
                          "image": names[:4], "label": names[4], "sizes_bytes": [15] * 5})
        self.data = {"cases": cases, "folds": [
            {"fold": i + 1, "train": [cases[i]["case_id"]],
             "inner_val": [cases[(i + 1) % 3]["case_id"]],
             "heldout": [cases[(i + 2) % 3]["case_id"]]} for i in range(3)]}
        self.save_manifest()

    def save_manifest(self):
        self.data.pop("manifest_sha256", None)
        self.data["manifest_sha256"] = mf.digest(self.data)
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")

    def run_copy(self):
        return prepare_cohort(self.source, self.manifest, self.output, copy=True)

    def test_selects_manifest_cases_aliases_masks_and_is_idempotent(self):
        extra = self.source / "unrelated"
        extra.mkdir()
        (extra / "unused.txt").write_text("do not copy")
        before = self.manifest.read_bytes()
        result = self.run_copy()
        self.assertEqual((result["cases"], result["files"], result["label_aliases"]), (3, 15, 3))
        label = self.output / self.data["cases"][0]["label"]
        self.assertEqual(label.read_bytes(), b"case0 modality4")
        self.assertFalse((self.output / "unrelated").exists())
        mtime = label.stat().st_mtime_ns
        self.assertEqual(self.run_copy()["created"], 0)
        self.assertEqual(label.stat().st_mtime_ns, mtime)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_missing_or_changed_last_input_creates_no_output(self):
        last = self.source / self.data["cases"][-1]["image"][-1]
        last.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "File size"):
            self.run_copy()
        self.assertFalse(self.output.exists())
        last.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_copy()
        self.assertFalse(self.output.exists())

    def test_unrelated_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        existing = self.output / "important.txt"
        existing.write_bytes(b"preserve me")
        with self.assertRaisesRegex(ValueError, "Unexpected file"):
            self.run_copy()
        self.assertEqual(existing.read_bytes(), b"preserve me")
        self.assertEqual(len(list(self.output.iterdir())), 1)

    def test_modified_existing_copy_is_rejected_even_with_same_size(self):
        self.run_copy()
        target = self.output / self.data["cases"][0]["image"][0]
        target.write_bytes(b"x" * 15)
        with self.assertRaisesRegex(ValueError, "content differs"):
            self.run_copy()
        self.assertEqual(target.read_bytes(), b"x" * 15)

    def test_unsafe_manifest_paths_and_nested_outputs_are_rejected(self):
        original = self.data["cases"][0]["image"][0]
        for invalid in ("../outside", "/absolute/file", "C:/private/file", "case/../outside", "case\\file"):
            self.data["cases"][0]["image"][0] = invalid
            self.save_manifest()
            with self.subTest(path=invalid), self.assertRaisesRegex(ValueError, "Unsafe manifest path"):
                self.run_copy()
            self.assertFalse(self.output.exists())
        self.data["cases"][0]["image"][0] = original
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "non-nested"):
            prepare_cohort(self.source, self.manifest, self.source / "bad", copy=True)

    def test_symlink_mode_and_existing_foreign_link(self):
        probe = self.root / "probe"
        try:
            probe.symlink_to(self.manifest)
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        probe.unlink()
        result = prepare_cohort(self.source, self.manifest, self.output)
        self.assertEqual(result["created"], 15)
        self.assertEqual(prepare_cohort(self.source, self.manifest, self.output)["created"], 0)
        target = self.output / self.data["cases"][0]["image"][0]
        target.unlink()
        target.symlink_to(self.manifest)
        with self.assertRaisesRegex(ValueError, "Existing link"):
            prepare_cohort(self.source, self.manifest, self.output)


if __name__ == "__main__":
    unittest.main()
