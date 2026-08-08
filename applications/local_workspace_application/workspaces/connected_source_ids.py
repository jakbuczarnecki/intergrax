# © Artur Czarnecki. All rights reserved.

"""Deterministic identifiers for connected workspace indexed sources."""

from __future__ import annotations

import hashlib
import json
import re

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(data: dict[str, object]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def connected_source_id(
    tenant_id: str,
    workspace_id: str,
    knowledge_source_binding_ref: str,
) -> str:
    payload = _canonical_json(
        {
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "knowledge_source_binding_ref": knowledge_source_binding_ref.strip(),
        }
    )
    return f"src:connected:{_sha256_hex(payload)[:32]}"


def indexed_source_binding_id_from_semantic_hash(value: str) -> str:
    digest = value.strip()
    if _SHA256_HEX_RE.fullmatch(digest) is None:
        raise ValueError("semantic_identity_hash_invalid")
    return f"idx:{digest[:32]}"


def connected_source_id_from_semantic_hash(value: str) -> str:
    digest = value.strip()
    if _SHA256_HEX_RE.fullmatch(digest) is None:
        raise ValueError("semantic_identity_hash_invalid")
    return f"src:connected:{digest[:32]}"


def indexed_source_binding_id(
    tenant_id: str,
    workspace_id: str,
    knowledge_source_binding_ref: str,
) -> str:
    payload = _canonical_json(
        {
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "knowledge_source_binding_ref": knowledge_source_binding_ref.strip(),
        }
    )
    return f"idx:{_sha256_hex(payload)[:32]}"


def tenant_binding_id(
    *,
    tenant_id: str,
    connection_ref: str,
    provider_id: str,
    integration_kind: str,
    source_kind: str,
    encoded_scope: str,
) -> str:
    payload = _canonical_json(
        {
            "tenant_id": tenant_id.strip(),
            "connection_ref": connection_ref.strip(),
            "provider_id": provider_id.strip(),
            "integration_kind": integration_kind.strip(),
            "source_kind": source_kind.strip(),
            "encoded_scope": encoded_scope.strip(),
        }
    )
    return f"ksb:{_sha256_hex(payload)[:32]}"


def workspace_indexed_source_semantic_hash(
    tenant_id: str,
    workspace_id: str,
    knowledge_source_binding_ref: str,
) -> str:
    payload = _canonical_json(
        {
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "knowledge_source_binding_ref": knowledge_source_binding_ref.strip(),
        }
    )
    digest = _sha256_hex(payload)
    if _SHA256_HEX_RE.fullmatch(digest) is None:
        raise ValueError("semantic_identity_hash_invalid")
    return digest


def connected_logical_path(
    *,
    source_id: str,
    remote_id: str,
    source_kind: str = "slack_conversation",
) -> str:
    payload = _canonical_json(
        {
            "source_id": source_id.strip(),
            "remote_id": remote_id.strip(),
        }
    )
    safe_kind = re.sub(r"[^a-z0-9_-]+", "-", source_kind.strip().lower()).strip("-")
    if not safe_kind:
        raise ValueError("source_kind_required")
    return f"connected/{safe_kind}-message/{_sha256_hex(payload)}.md"


def connected_document_id(
    *,
    tenant_id: str,
    workspace_id: str,
    provider_id: str,
    integration_kind: str,
    source_kind: str,
    binding_id: str,
    remote_id: str,
) -> str:
    """Return a stable indexed document identity for one remote item."""
    payload = _canonical_json(
        {
            "tenant_id": tenant_id.strip(),
            "workspace_id": workspace_id.strip(),
            "provider_id": provider_id.strip(),
            "integration_kind": integration_kind.strip(),
            "source_kind": source_kind.strip(),
            "binding_id": binding_id.strip(),
            "remote_id": remote_id.strip(),
        }
    )
    return f"lkwdoc:{_sha256_hex(payload)[:32]}"


# Backward-compatible aliases for earlier draft names.
connected_workspace_source_id = connected_source_id
workspace_indexed_source_binding_id = indexed_source_binding_id
tenant_knowledge_source_binding_id = tenant_binding_id
workspace_indexed_source_semantic_identity_hash = workspace_indexed_source_semantic_hash
connected_logical_document_path = connected_logical_path
