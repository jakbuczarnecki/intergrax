# © Artur Czarnecki. All rights reserved.

"""Idempotency helpers for managed workspace sync (LKW-PRODUCT-1)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def normalize_source_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    return resolved.as_posix()


def content_hash_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def logical_document_id(
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    normalized_source_path: str,
    content_hash: str,
    materialization_scope: str | None = None,
) -> str:
    payload = "|".join(
        (
            tenant_id.strip(),
            workspace_id.strip(),
            source_id.strip(),
            normalized_source_path.strip(),
            content_hash.strip(),
            (materialization_scope or "").strip(),
        )
    )
    return f"lkwdoc:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"


def path_identity_key(
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    normalized_source_path: str,
) -> str:
    payload = "|".join(
        (
            tenant_id.strip(),
            workspace_id.strip(),
            source_id.strip(),
            normalized_source_path.strip(),
        )
    )
    return f"lkwpath:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"
