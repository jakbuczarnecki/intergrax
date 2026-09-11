# © Artur Czarnecki. All rights reserved.

"""Semantic source fingerprinting (orthogonal to repository HEAD drift)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from testing_support.decision_e2e.local_qualification_session.contracts import (
    SourceBlobFingerprint,
    SourceDriftReport,
    SourceFingerprintSnapshot,
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_source_fingerprint(
    repo_root: Path,
    *,
    blob_paths: tuple[str, ...],
    semantic_group: str,
    repository_head_sha: str,
) -> SourceFingerprintSnapshot:
    blobs: list[SourceBlobFingerprint] = []
    for relative in blob_paths:
        absolute = repo_root / relative
        if not absolute.is_file():
            raise FileNotFoundError(f"qualification source blob missing: {relative}")
        blobs.append(
            SourceBlobFingerprint(
                path=relative.replace("\\", "/"),
                content_hash=_hash_file(absolute),
                semantic_group=semantic_group,
            )
        )
    return SourceFingerprintSnapshot(
        repository_head_sha=repository_head_sha,
        blobs=tuple(blobs),
    )


def compare_source_snapshots(
    frozen: SourceFingerprintSnapshot,
    current: SourceFingerprintSnapshot,
) -> SourceDriftReport:
    frozen_map = {item.path: item.content_hash for item in frozen.blobs}
    current_map = {item.path: item.content_hash for item in current.blobs}
    changed: list[str] = []
    for path, frozen_hash in frozen_map.items():
        current_hash = current_map.get(path)
        if current_hash is None or current_hash != frozen_hash:
            changed.append(path)
    return SourceDriftReport(
        repository_head_drift=frozen.repository_head_sha != current.repository_head_sha,
        qualification_semantic_source_drift=bool(changed),
        frozen_head_sha=frozen.repository_head_sha,
        current_head_sha=current.repository_head_sha,
        changed_blobs=tuple(sorted(changed)),
    )
