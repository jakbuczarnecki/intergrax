# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-path allowlist helpers for LKW Tier-2 agents (no intergrax.tools imports)."""

from __future__ import annotations

import os
from pathlib import Path


def parse_read_allowlist_roots(raw: str | None) -> frozenset[str]:
    if not raw or not raw.strip():
        return frozenset()
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


def read_allowlist_roots_from_env() -> frozenset[str]:
    return parse_read_allowlist_roots(os.environ.get("INTERGRAX_ALLOWED_READ_ROOTS"))


def require_read_allowlist_roots(roots: frozenset[str] | None) -> frozenset[str]:
    if not roots:
        raise RuntimeError("read_allowlist_not_configured")
    return roots


def resolve_allowed_path(path: str, roots: frozenset[str]) -> Path:
    candidate = Path(path.strip()).expanduser()
    if not candidate.is_absolute():
        raise RuntimeError("path_must_be_absolute")
    resolved = candidate.resolve()
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        try:
            resolved.relative_to(root_path)
            return resolved
        except ValueError:
            continue
    raise RuntimeError("path_not_in_allowlist")
