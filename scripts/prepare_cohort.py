#!/usr/bin/env python3
"""Build a local data view for the published, unchanged patient manifest.

Only cases/files already listed in the manifest are selected. The original data
are never renamed or rewritten. A source label ending in ``-segs.nii.gz`` may
supply a manifest path ending in ``-seg.nii.gz``; no image values are changed.
"""
import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cedia import manifest as mf


def _inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _relative(value, case_id):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Unsafe manifest path: {value!r}")
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive or ":" in value
            or any(p in ("", ".", "..") for p in value.split("/"))
            or len(path.parts) < 2 or path.parts[0] != case_id):
        raise ValueError(f"Unsafe manifest path: {value!r}")
    return Path(*path.parts)


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.digest()


def prepare_cohort(source, manifest, output, copy=False):
    """Validate every input and existing output before creating any cohort files."""
    source = Path(source).expanduser().resolve(strict=True)
    manifest_path = Path(manifest).expanduser().resolve(strict=True)
    requested_output = Path(output).expanduser().absolute()
    if requested_output.is_symlink():
        raise ValueError("Output must be a real directory, not a directory symlink")
    output = requested_output.resolve()
    if not source.is_dir():
        raise ValueError("Source must be a directory")
    if _inside(output, source) or _inside(source, output):
        raise ValueError("Source and output must be separate, non-nested directories")
    if _inside(manifest_path, output):
        raise ValueError("Keep the manifest outside the output data directory")
    if output.exists() and not output.is_dir():
        raise ValueError("Output is not a directory")

    data = mf.load(manifest_path)
    planned = {}
    allowed_dirs = set()
    aliases = 0
    for case in data["cases"]:
        images = case["image"]
        sizes = case["sizes_bytes"]
        if len(images) != 4 or len(sizes) != 5:
            raise ValueError("Each case must contain four modalities and one label")
        for index, (name, size) in enumerate(zip(images + [case["label"]], sizes)):
            if type(size) is not int or size < 0:
                raise ValueError(f"Invalid size for {name}")
            relative = _relative(name, case["case_id"])
            if relative in planned:
                raise ValueError(f"Duplicate manifest path: {relative}")
            candidate = source / relative
            if not candidate.exists() and index == 4 and candidate.name.endswith("-seg.nii.gz"):
                candidate = candidate.with_name(candidate.name[:-len("-seg.nii.gz")] + "-segs.nii.gz")
                aliases += 1
            resolved = candidate.resolve(strict=True)
            if not _inside(resolved, source) or not resolved.is_file():
                raise ValueError(f"Source file must remain within source directory: {candidate}")
            stat = resolved.stat()
            if stat.st_size != size:
                raise ValueError(f"File size differs from persisted manifest: {candidate}")
            planned[relative] = (resolved, size, stat.st_mtime_ns)
            allowed_dirs.update(p for p in relative.parents if p != Path("."))

    # Reject unrelated content and directory links before adding anything.
    if output.exists():
        for directory, dirs, files in os.walk(output, followlinks=False):
            directory = Path(directory)
            for name in dirs:
                path = directory / name
                if path.is_symlink() or path.relative_to(output) not in allowed_dirs:
                    raise ValueError(f"Unexpected directory in output: {path}")
            for name in files:
                path = directory / name
                relative = path.relative_to(output)
                if relative not in planned:
                    raise ValueError(f"Unexpected file in output: {path}")
                original, size, _ = planned[relative]
                if copy:
                    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
                        raise ValueError(f"Existing copy does not match: {path}")
                    if _sha256(path) != _sha256(original):
                        raise ValueError(f"Existing copy content differs: {path}")
                elif not path.is_symlink() or path.resolve(strict=True) != original:
                    raise ValueError(f"Existing link does not match: {path}")

    # All validations above are read-only. Never overwrite an existing path.
    output.mkdir(parents=True, exist_ok=True)
    created = 0
    for relative, (original, size, mtime) in planned.items():
        target = output / relative
        if os.path.lexists(target):
            continue
        current = original.stat()
        if (current.st_size, current.st_mtime_ns) != (size, mtime):
            raise ValueError(f"Source changed during preparation: {original}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            with original.open("rb") as src, target.open("xb") as dest:
                shutil.copyfileobj(src, dest, 2**20)
            current = original.stat()
            if target.stat().st_size != size or (current.st_size, current.st_mtime_ns) != (size, mtime):
                raise ValueError(f"Source changed while copying: {original}")
        else:
            target.symlink_to(original)
        created += 1
    return {"cases": len(data["cases"]), "files": len(planned), "created": created,
            "label_aliases": aliases, "output": str(output), "mode": "copy" if copy else "symlink",
            "manifest_sha256": data["manifest_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Original BraTS case directories")
    parser.add_argument("--manifest", required=True, type=Path, help="Published manifest; never regenerated or modified")
    parser.add_argument("--output", required=True, type=Path, help="New/empty directory or the identical previously created view")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of absolute symlinks (portable on Windows; uses dataset-sized disk space)")
    args = parser.parse_args(argv)
    try:
        result = prepare_cohort(args.source, args.manifest, args.output, args.copy)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cohort preparation failed: {error}\n")
    print(f"Cohort ready: {result['cases']} cases, {result['files']} files, "
          f"{result['created']} created, {result['label_aliases']} label aliases ({result['mode']})")
    print(f"DATA_ROOT={result['output']}")
    print(f"Manifest unchanged: {result['manifest_sha256']}")


if __name__ == "__main__":
    main()
