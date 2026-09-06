# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic identity for immutable resolved skill composition snapshots."""

from __future__ import annotations

import hashlib

from intergrax.skills.core.version_binding import ResolvedSkillRef

_SCHEMA_PREFIX = "resolved_skill_pack.v1"


def compute_resolved_skill_pack_digest(
    resolved_skills: tuple[ResolvedSkillRef, ...],
) -> str:
    """Return a stable SHA-256 digest for resolved skill evidence in topological order."""
    parts: list[str] = []
    for ref in resolved_skills:
        parts.append(
            "|".join(
                (
                    ref.skill_id,
                    ref.version,
                    ref.resolution_mode.value,
                    ref.role.value,
                )
            )
        )
    canonical = f"{_SCHEMA_PREFIX}:" + ";".join(parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
