# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R3 Final H1 — post-R3 upstream event-surface drift classification (qualification only)."""

from __future__ import annotations

from pathlib import Path

from testing_support.frozen_baseline_provenance import (
    assert_frozen_baseline_reachable,
)
from testing_support.npsc5f_r1_protected_drift import (
    collect_r1_protected_production_drift,
    git_changed_paths,
)
from testing_support.npsc5f_r2_protected_drift import (
    collect_r2_protected_production_drift,
)
from testing_support.npsc5f_r3_protected_drift import (
    R3_IMPLEMENTATION_SHA,
    collect_r3_protected_production_drift,
)

# Code-under-test baseline for H1 integrated qualification (immutable; not branch HEAD).
NPSC5F_R3_H1_QUALIFIED_BASELINE_SHA = "ad1a1e57fc70529aedcbfa27808fffdbfe5fdd14"

# Qualification-record commit (proof artifacts); may trail ``origin/development`` after push.
NPSC5F_R3_H1_QUALIFICATION_RECORD_SHA = "145bbd74e6d4175a5b868ed71ed1a6353b2c2c3b"

_H1_EVENT_SURFACE_CLASSIFICATION_BUCKETS = frozenset("ABCDEFGHIJK")

# Commit that introduced qualified ``RuntimeEventType.EXECUTION_FAILED`` on the R1-owned enum surface.
EXECUTION_FAILED_RUNTIME_EVENT_QUALIFIED_SHA = (
    "40cc8c11e0b57ed4cf0d99ed1b9b297820c6eaa8"
)

_POST_R3_EVENT_SURFACE_PREFIX = "intergrax/runtime/events/"


def _normalize(path: str) -> str:
    return path.strip().replace("\\", "/")


class H1IntegratedQualificationError(RuntimeError):
    """Raised when H1 integrated baseline provenance or post-baseline drift rules fail."""


def collect_post_h1_baseline_event_surface_paths(
    repo_root: Path,
    *,
    from_sha: str = NPSC5F_R3_H1_QUALIFIED_BASELINE_SHA,
    to_ref: str = "origin/development",
) -> list[str]:
    """Event-module paths changed since the H1 qualified baseline (not since R3 implementation)."""
    return sorted(
        {
            _normalize(path)
            for path in git_changed_paths(repo_root, from_sha=from_sha, to_ref=to_ref)
            if _normalize(path).startswith(_POST_R3_EVENT_SURFACE_PREFIX)
        },
    )


def classify_h1_post_baseline_event_surface_drift(
    repo_root: Path,
    *,
    from_sha: str = NPSC5F_R3_H1_QUALIFIED_BASELINE_SHA,
    to_ref: str = "origin/development",
) -> dict[str, str]:
    """Map each post-baseline events-module path to an H1 taxonomy bucket letter."""
    changed = collect_post_h1_baseline_event_surface_paths(
        repo_root,
        from_sha=from_sha,
        to_ref=to_ref,
    )
    return {path: classify_post_r3_event_surface_change(path) for path in changed}


def assert_h1_integrated_qualification_contract(
    repo_root: Path,
    *,
    remote_ref: str = "origin/development",
    qualified_baseline_sha: str = NPSC5F_R3_H1_QUALIFIED_BASELINE_SHA,
) -> None:
    """Prove qualified baseline reachability and absence of unqualified protected drift after baseline."""
    assert_frozen_baseline_reachable(
        repo_root=repo_root,
        baseline_sha=qualified_baseline_sha,
        remote_ref=remote_ref,
    )
    assert_frozen_baseline_reachable(
        repo_root=repo_root,
        baseline_sha=NPSC5F_R3_H1_QUALIFICATION_RECORD_SHA,
        remote_ref=remote_ref,
    )
    buckets = classify_h1_post_baseline_event_surface_drift(
        repo_root,
        from_sha=qualified_baseline_sha,
        to_ref=remote_ref,
    )
    unknown_buckets = {
        path: bucket
        for path, bucket in buckets.items()
        if bucket not in _H1_EVENT_SURFACE_CLASSIFICATION_BUCKETS
    }
    if unknown_buckets:
        raise H1IntegratedQualificationError(
            "post-baseline event-surface drift has unclassified buckets: "
            f"{unknown_buckets}",
        )
    protected_drift = {
        "r1": collect_r1_protected_production_drift(
            repo_root,
            from_sha=qualified_baseline_sha,
            to_ref=remote_ref,
        ),
        "r2": collect_r2_protected_production_drift(
            repo_root,
            from_sha=qualified_baseline_sha,
            to_ref=remote_ref,
        ),
        "r3": collect_r3_protected_production_drift(
            repo_root,
            from_sha=qualified_baseline_sha,
            to_ref=remote_ref,
        ),
    }
    violations = {key: paths for key, paths in protected_drift.items() if paths}
    if violations:
        raise H1IntegratedQualificationError(
            "post-baseline protected production drift without requalification: "
            f"{violations}",
        )


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
            for path in git_changed_paths(
                root, from_sha=R3_IMPLEMENTATION_SHA, to_ref=to_ref
            )
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
