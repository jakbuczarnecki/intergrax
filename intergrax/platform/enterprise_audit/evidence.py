# © Artur Czarnecki. All rights reserved.

"""Evidence bundle builder for ENT-AUDIT-X qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_evidence_manifest(output_dir: Path, *, run_id: str, qualification_sha: str) -> Path:
    artifacts: list[dict[str, str]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        if rel == "evidence_manifest.v1.json":
            continue
        artifacts.append({"path": rel, "sha256": sha256_file(path)})
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "qualification_sha": qualification_sha,
        "artifacts": artifacts,
    }
    out = output_dir / "evidence_manifest.v1.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def verify_evidence_manifest(output_dir: Path) -> bool:
    manifest_path = output_dir / "evidence_manifest.v1.json"
    if not manifest_path.is_file():
        return False
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in raw.get("artifacts", []):
        path = output_dir / item["path"]
        if not path.is_file():
            return False
        if sha256_file(path) != item["sha256"]:
            return False
    return True
