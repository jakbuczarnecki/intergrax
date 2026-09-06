# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic identity for immutable resolved skill composition snapshots."""

from __future__ import annotations

import hashlib

from intergrax.skills.core.version_binding import ResolvedSkillRef

_SCHEMA_PREFIX = "resolved_skill_pack.v1"
_UINT32_MAX = (1 << 32) - 1


def _encode_text(value: str) -> bytes:
    payload = value.encode("utf-8")
    length = len(payload)
    if length > _UINT32_MAX:
        raise ValueError(f"UTF-8 field length {length} exceeds uint32 maximum")
    return length.to_bytes(4, "big") + payload


def _encode_resolved_skill_ref(ref: ResolvedSkillRef) -> bytes:
    return (
        _encode_text(ref.skill_id)
        + _encode_text(ref.version)
        + _encode_text(ref.resolution_mode.value)
        + _encode_text(ref.role.value)
    )


def _encode_canonical_payload(resolved_skills: tuple[ResolvedSkillRef, ...]) -> bytes:
    count = len(resolved_skills)
    if count > _UINT32_MAX:
        raise ValueError(f"resolved skill count {count} exceeds uint32 maximum")
    parts = [_encode_text(_SCHEMA_PREFIX), count.to_bytes(4, "big")]
    for ref in resolved_skills:
        parts.append(_encode_resolved_skill_ref(ref))
    return b"".join(parts)


def compute_resolved_skill_pack_digest(
    resolved_skills: tuple[ResolvedSkillRef, ...],
) -> str:
    """Return a stable SHA-256 digest for resolved skill evidence in topological order."""
    canonical = _encode_canonical_payload(resolved_skills)
    digest = hashlib.sha256(canonical).hexdigest()
    return f"sha256:{digest}"
