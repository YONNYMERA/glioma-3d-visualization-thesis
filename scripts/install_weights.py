#!/usr/bin/env python3
"""Install verified, separately distributed thesis weights using only stdlib.

This command never downloads files and never deserializes a PyTorch checkpoint.
It validates the archive and every payload before creating destination files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tarfile
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 1024 * 1024


class InstallError(ValueError):
    """The package or destination does not satisfy the verified inventory."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(name: str) -> PurePosixPath:
    """Reject ambiguous paths on both POSIX and Windows."""
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise InstallError(f"Unsafe archive path: {name!r}")
    pieces = name.split("/")
    if ntpath.splitdrive(name)[0] or any(
        part in ("", ".", "..") or ":" in part or part.rstrip(" .") != part
        for part in pieces
    ):
        raise InstallError(f"Unsafe archive path: {name!r}")
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
    if any(part.split(".")[0].upper() in reserved for part in pieces):
        raise InstallError(f"Reserved filename in archive: {name!r}")
    return PurePosixPath(name)


def _is_link(path: Path) -> bool:
    try:
        information = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(information.st_mode) or bool(
        getattr(information, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _check_parent_directories(path: Path) -> None:
    for parent in reversed(path.parents):
        if _is_link(parent):
            raise InstallError(f"Refusing a linked destination directory: {parent}")
        if parent.exists() and not parent.is_dir():
            raise InstallError(f"Destination parent is not a directory: {parent}")


def _destination(output: Path, relative: str, expected_hash: str) -> tuple[Path, bool]:
    destination = output.joinpath(*safe_relative_path(relative).parts)
    _check_parent_directories(destination)
    if _is_link(destination):
        raise InstallError(f"Refusing a linked destination: {destination}")
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != expected_hash:
            raise InstallError(f"Existing destination has different content; nothing overwritten: {destination}")
        return destination, True
    return destination, False


def _validate_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _select_inventory(index: Path, archive: Path) -> tuple[str, str, dict[str, dict]]:
    inventory = json.loads(index.read_text(encoding="utf-8"))
    if inventory.get("schema") != 1:
        raise InstallError("Unsupported weights inventory schema.")
    matches = [entry for entry in inventory.get("archives", []) if entry.get("name") == archive.name]
    if len(matches) != 1:
        raise InstallError(f"Archive name is not uniquely registered in the inventory: {archive.name}")
    archive_record = matches[0]
    archive_hash = archive_record.get("sha256")
    model = archive_record.get("model")
    if not _validate_hash(archive_hash) or not isinstance(model, str) or not model:
        raise InstallError("Invalid archive record in the weights inventory.")
    weights = {}
    portable_names = set()
    for record in inventory.get("weights", []):
        if record.get("model") != model:
            continue
        relative = record.get("path")
        safe_relative_path(relative)
        if not relative.endswith(".pth"):
            raise InstallError(f"Expected a .pth checkpoint in the inventory: {relative}")
        if relative.casefold() in portable_names:
            raise InstallError(f"Duplicate checkpoint path in the inventory: {relative}")
        if not _validate_hash(record.get("sha256")) or type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise InstallError(f"Invalid checkpoint record: {relative}")
        portable_names.add(relative.casefold())
        weights[relative] = record
    if not weights:
        raise InstallError(f"No checkpoints are registered for model {model!r}.")
    return model, archive_hash, weights


def _metadata_paths(weight_paths: dict[str, dict]) -> list[str]:
    candidates = set()
    for name in weight_paths:
        path = PurePosixPath(name)
        if path.parts[0] == "train":
            for filename in ("run.json", "TRAINING_COMPLETED.json"):
                candidates.add(str(path.parent / filename))
        elif path.parts[:2] == ("nnunet", "nnUNet_results"):
            for filename in ("plans.json", "dataset.json", "cedia_protocol.json"):
                candidates.add(str(path.parent.parent / filename))
    return sorted(candidates)


def install_archive(archive: Path, output: Path, index: Path, metadata_root: Path) -> dict:
    """Validate and install one registered package; identical files are skipped."""
    archive, index, metadata_root = Path(archive), Path(index), Path(metadata_root)
    output = Path(os.path.abspath(output))
    if _is_link(archive) or not archive.is_file():
        raise InstallError(f"Archive must be a regular file, not a link: {archive}")
    model, archive_hash, expected = _select_inventory(index, archive)
    if sha256_file(archive) != archive_hash:
        raise InstallError("Archive SHA-256 does not match provenance/weights.json. Nothing installed.")
    initial_archive_stat = archive.stat()
    staged = {}
    missing_metadata = []
    with tempfile.TemporaryDirectory(prefix="glioma_verified_weights_") as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, mode="r:*") as bundle:
            for member in bundle:
                safe_relative_path(member.name)
                if not member.isfile() or member.issparse():
                    raise InstallError(f"Only ordinary checkpoint files are accepted: {member.name}")
                if member.name not in expected or member.name in staged:
                    raise InstallError(f"Unknown or duplicate archive member: {member.name}")
                record = expected[member.name]
                if member.size != record["bytes"]:
                    raise InstallError(f"Checkpoint size differs from inventory: {member.name}")
                source = bundle.extractfile(member)
                if source is None:
                    raise InstallError(f"Cannot read checkpoint: {member.name}")
                staged_path = staging / str(len(staged))
                digest = hashlib.sha256()
                with source, staged_path.open("xb") as destination:
                    for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                        digest.update(chunk)
                        destination.write(chunk)
                if staged_path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                    raise InstallError(f"Checkpoint SHA-256 differs from inventory: {member.name}")
                staged[member.name] = (staged_path, record["sha256"])
        if set(staged) != set(expected):
            missing = sorted(set(expected) - set(staged))
            raise InstallError(f"Archive is missing registered checkpoints: {', '.join(missing)}")
        final_archive_stat = archive.stat()
        if (initial_archive_stat.st_size, initial_archive_stat.st_mtime_ns) != (
            final_archive_stat.st_size, final_archive_stat.st_mtime_ns
        ):
            raise InstallError("Archive changed during verification. Nothing installed.")
        if model != "historical_app":
            for relative in _metadata_paths(expected):
                source = metadata_root.joinpath(*safe_relative_path(relative).parts)
                _check_parent_directories(source.absolute())
                if _is_link(source):
                    raise InstallError(f"Refusing linked metadata: {source}")
                if not source.exists():
                    missing_metadata.append(relative)
                    continue
                if not source.is_file():
                    raise InstallError(f"Metadata must be a regular file: {source}")
                content = source.read_bytes()
                json.loads(content)
                staged_path = staging / str(len(staged))
                staged_path.write_bytes(content)
                staged[relative] = (staged_path, hashlib.sha256(content).hexdigest())
        # Check *all* targets before writing the first destination file.
        for relative, (_, digest) in staged.items():
            _destination(output, relative, digest)
        if missing_metadata:
            raise InstallError("Required metadata missing from repository: " + ", ".join(missing_metadata))
        installed, skipped = [], []
        created = []
        try:
            for relative, (source, digest) in staged.items():
                destination, exists = _destination(output, relative, digest)
                if exists:
                    skipped.append(relative)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    created.append(destination)
                    with source.open("rb") as origin:
                        shutil.copyfileobj(origin, stream, CHUNK_SIZE)
                installed.append(relative)
        except BaseException:
            # These are exclusively files created by this invocation, never inputs.
            for destination in reversed(created):
                destination.unlink(missing_ok=True)
            raise
    return {"model": model, "output": str(output), "installed": installed,
            "already_present": skipped, "metadata_not_available": missing_metadata}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Local weight tar archive registered in provenance/weights.json")
    parser.add_argument("--output", type=Path, help="Destination root (default: repository/results_reproduction)")
    parser.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "results", help="Root of repository result metadata")
    arguments = parser.parse_args(argv)
    index = REPO_ROOT / "provenance" / "weights.json"
    try:
        model, _, _ = _select_inventory(index, arguments.archive)
        if model == "historical_app" and arguments.output is None:
            raise InstallError("Historical app weights require explicit --output; use --output . from the repository root.")
        output = arguments.output if arguments.output is not None else REPO_ROOT / "results_reproduction"
        report = install_archive(arguments.archive, output, index, arguments.metadata_root)
    except (InstallError, OSError, tarfile.TarError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Verified installation complete. No checkpoint was deserialized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
