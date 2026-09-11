# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R3 Final H1 — post-R3 upstream event-surface drift classification (qualification only)."""

from __future__ import annotations

from testing_support.npsc5f_r1_protected_drift import (
    R1_POST_R2_QUALIFIED_BASELINE_SHA,
    git_changed_paths,
)
from testing_support.npsc5f_r3_protected_drift import R3_IMPLEMENTATION_SHA

# Integrated ``development`` HEAD qualified by H1 behavioral gates (immutable provenance pin).
NPSC5F_R3_H1_QUALIFIED_INTEGRATED_SHA = "2965f2fcfed27162f06625c4b8bcd84b18d27704"

# Commit that introduced qualified ``RuntimeEventType.EXECUTION_FAILED`` on the R1-owned enum surface.
EXECUTION_FAILED_RUNTIME_EVENT_QUALIFIED_SHA = R1_POST_R2_QUALIFIED_BASELINE_SHA

_POST_R3_EVENT_SURFACE_PREFIX = "intergrax/runtime/events/"


def _normalize(path: str) -> str:
    return path.strip().replace("\\", "/")


def collect_post_r3_event_surface_paths(
    repo_root,
    *,
    to_ref: str = "HEAD",
) -> list[str]:
    """Production paths under ``intergrax/runtime/events/`` changed since R3 implementation."""
    from pathlib import Path

    root = Path(repo_root)
    return sorted(
        {
            _normalize(path)
            for path in git_changed_paths(root, from_sha=R3_IMPLEMENTATION_SHA, to_ref=to_ref)
            if _normalize(path).startswith(_POST_R3_EVENT_SURFACE_PREFIX)
        },
    )


def classify_post_r3_event_surface_change(path: str) -> str:
    """Single-letter H1 taxonomy bucket for one changed events-module path."""
    normalized = _normalize(path)
    name = normalized.split("/")[-1]
    if name == "runtime_event.py":
        return "A"
    if name == "event_catalog.py":
        return "G"
    if name == "payload_registry.py":
        return "H"
    if normalized.endswith("payloads/canonical.py"):
        return "I"
    if name == "spine_consolidation.py":
        return "J"
    if normalized.endswith("payloads/__init__.py"):
        return "K"
    return "K"
