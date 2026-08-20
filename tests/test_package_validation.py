import hashlib

from hpid_split.validation import _validate_manifest_files


def _entry(path) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def test_manifest_file_validation_detects_tampering(tmp_path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"auditable")
    manifest = {"files": [_entry(payload)]}

    assert _validate_manifest_files(tmp_path, manifest) == []

    payload.write_bytes(b"changed!!")
    errors = _validate_manifest_files(tmp_path, manifest)

    assert "manifest SHA-256 differs: payload.bin" in errors


def test_manifest_file_validation_detects_unlisted_and_escaping_files(
    tmp_path,
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"listed")
    unlisted = tmp_path / "unlisted.bin"
    unlisted.write_bytes(b"unlisted")
    manifest = {
        "files": [
            _entry(payload),
            {"path": "../outside.bin", "sha256": "0" * 64, "bytes": 0},
        ]
    }

    errors = _validate_manifest_files(tmp_path, manifest)

    assert "manifest payload path escapes package: ../outside.bin" in errors
    assert "payload file is absent from manifest: unlisted.bin" in errors
