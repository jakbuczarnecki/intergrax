# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R1 — post-H1 event spine drift classification (qualification only)."""

from __future__ import annotations

from pathlib import Path

from testing_support.npsc5f_r1_protected_drift import git_changed_paths

# R1 sentinel immediately before event-spine qualification (EXECUTION_FAILED enum baseline).
R1_EVENT_SPINE_PRE_QUALIFICATION_BASELINE_SHA = "40cc8c11e0b57ed4cf0d99ed1b9b297820c6eaa8"

_EVENT_SPINE_R1_PROTECTED = frozenset(
    {
        "intergrax/runtime/events/event_bus.py",
        "intergrax/runtime/events/runtime_event.py",
    },
)


def _normalize(path: str) -> str:
    return path.strip().replace("\\", "/")


def collect_event_spine_r1_protected_paths(
    repo_root: Path,
    *,
    from_sha: str = R1_EVENT_SPINE_PRE_QUALIFICATION_BASELINE_SHA,
    to_ref: str = "HEAD",
) -> list[str]:
    """R1-protected event spine paths changed between ``from_sha`` and ``to_ref``."""
    return sorted(
        {
            _normalize(path)
            for path in git_changed_paths(repo_root, from_sha=from_sha, to_ref=to_ref)
            if _normalize(path) in _EVENT_SPINE_R1_PROTECTED
        },
    )


def classify_event_spine_r1_protected_change(path: str) -> str:
    """Single-letter taxonomy for qualified event-spine drift on R1-owned surfaces."""
    normalized = _normalize(path)
    name = normalized.split("/")[-1]
    if name == "runtime_event.py":
        return "A"
    if name == "event_bus.py":
        return "B"
    raise ValueError(f"not an event-spine R1 protected path: {path!r}")
