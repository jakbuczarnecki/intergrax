# © Artur Czarnecki. All rights reserved.

"""NPSC-5F Final — Evidence Plane production drift tri-classifier (qualification only)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from testing_support.npsc5f_r1_protected_drift import git_changed_paths

# Evidence Plane freeze baseline: NPSC-5F/R4 implementation sign-off (no production drift since).
NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA = "3bec620ab56417a469487347f68045bf3dec6bd5"

NPSC_5E_FINAL_SHA = "fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7"
NPSC_5F_R1_FINAL_SHA = "455c09f342f995ac0a6fcb03ffef2f4d3e36a447"
NPSC_5F_R2_FINAL_SHA = "76c92847f67da22d97943b55896a88c814d7e39d"
NPSC_5F_R3_FINAL_SHA = "0346face3ef68d8f21504822a26f8f45f2384cf9"
NPSC_5F_R4_FINAL_SHA = "37fb051c7f164d705f628760436b8ea10ee0289f"

_PROTECTED_PATH_PREFIXES: tuple[str, ...] = (
    "intergrax/runtime/events/",
    "intergrax/runtime/observability/",
)

_PROTECTED_EXACT_PATHS: frozenset[str] = frozenset(
    {
        "intergrax/contracts/historical_reconstruction.py",
        "intergrax/contracts/runtime_event.py",
    },
)

# Parallel work allowed without Evidence Plane final BREAKING classification.
_EXPLICITLY_UNRELATED_PREFIXES: tuple[str, ...] = (
    "intergrax/runtime/diagnostics/",
    "intergrax/runtime/execution/",
    "intergrax/runtime/long_running/",
    "intergrax/runtime/nexus/",
    "intergrax/runtime/cancellation/",
    "intergrax/runtime/resilience/",
    "intergrax/runtime/hooks/",
    "intergrax/runtime/vendor_knowledge/",
    "intergrax/integrations/",
    "applications/",
    "agents/",
    "intergrax/core/",
)

_QUALIFIED_COMPATIBLE_PREFIXES: tuple[str, ...] = (
    "docs/project/maintainers/qualification/NPSC_5F_FINAL",
    "docs/project/maintainers/qualification/NPSC_5F_R",
    "testing_support/npsc5f_final_",
    "testing_support/npsc5f_r",
    "tests/unit/runtime/architecture/test_npsc5f_final_",
    "tests/unit/testing_support/test_npsc5f_final_",
)


class EvidencePlaneDriftClass(StrEnum):
    QUALIFIED_COMPATIBLE = "QUALIFIED_COMPATIBLE"
    UNRELATED = "UNRELATED"
    BREAKING = "BREAKING"


@dataclass(frozen=True, slots=True)
class ClassifiedEvidencePlaneDrift:
    path: str
    classification: EvidencePlaneDriftClass


def _normalize_repo_path(path: str) -> str:
    return path.strip().replace("\\", "/")


def is_evidence_plane_protected_production_path(path: str) -> bool:
    """True when ``path`` is on a frozen Evidence Plane contract surface."""
    normalized = _normalize_repo_path(path)
    if not normalized:
        return False
    if normalized in _PROTECTED_EXACT_PATHS:
        return True
    for prefix in _PROTECTED_PATH_PREFIXES:
        if normalized.startswith(prefix):
            return True
    return False


def is_qualified_compatible_qualification_path(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized:
        return False
    return any(normalized.startswith(prefix) for prefix in _QUALIFIED_COMPATIBLE_PREFIXES)


def is_explicitly_unrelated_path(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized:
        return True
    return any(normalized.startswith(prefix) for prefix in _EXPLICITLY_UNRELATED_PREFIXES)


def classify_evidence_plane_drift_path(path: str) -> EvidencePlaneDriftClass:
    """Classify one changed path — never treats whole ``intergrax/runtime`` as sentinel."""
    normalized = _normalize_repo_path(path)
    if not normalized:
        return EvidencePlaneDriftClass.UNRELATED
    if is_qualified_compatible_qualification_path(normalized):
        return EvidencePlaneDriftClass.QUALIFIED_COMPATIBLE
    if not is_evidence_plane_protected_production_path(normalized):
        return EvidencePlaneDriftClass.UNRELATED
    if is_explicitly_unrelated_path(normalized):
        return EvidencePlaneDriftClass.UNRELATED
    return EvidencePlaneDriftClass.BREAKING


def classify_evidence_plane_drift(changed_paths: Iterable[str]) -> list[ClassifiedEvidencePlaneDrift]:
    ordered = sorted({_normalize_repo_path(path) for path in changed_paths if path.strip()})
    return [
        ClassifiedEvidencePlaneDrift(path=path, classification=classify_evidence_plane_drift_path(path))
        for path in ordered
    ]


def breaking_evidence_plane_drift(changed_paths: Iterable[str]) -> list[str]:
    return sorted(
        item.path
        for item in classify_evidence_plane_drift(changed_paths)
        if item.classification is EvidencePlaneDriftClass.BREAKING
    )


def collect_breaking_evidence_plane_production_drift(
    repo_root: Path,
    *,
    from_sha: str = NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA,
    to_ref: str = "origin/development",
) -> list[str]:
    return breaking_evidence_plane_drift(
        git_changed_paths(repo_root, from_sha=from_sha, to_ref=to_ref),
    )


def git_head_sha(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git rev-parse HEAD failed")
    return completed.stdout.strip()
