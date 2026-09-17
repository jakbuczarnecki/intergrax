# © Artur Czarnecki. All rights reserved.

"""NPSC-5F Final — Evidence Plane production drift tri-classifier (qualification only)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from testing_support.npsc5f_r1_protected_drift import git_changed_paths

# Evidence Plane scoped re-freeze: NPSC-5F R3+Final requalification after Class C v1→v2.
# Prior EE-FINAL-02 baseline: ``7a3569c64e892588992635c9cee10c264a9fc200``.
# H9 pre-gate: qualified compatible evolution (W5 event-delivery, GR-3-R2 adjacent) — sentinel advance.
# Prior H9 interim baseline: ``48a33db23fafab89b5fdb4ff217dfcb113dd6cc5``.
# OBS-RECONSTRUCTION-1: factual reconstruction relocation to Evidence Plane (read-only, contract-first).
# OBS-ASOF-REBASE-R1: E-scoped lineage port + fail-closed disable of current lineage at historical E.
# HARDENING_9_NPSC5F: OBS-CONTRACT-BOUNDARY-1 / R1 — contract-owned runtime_event, reconstruction DTOs,
# positioned evidence boundary; qualified re-freeze @ ``a2b33ba965c57cd3c812720f7b5f84b40b2b32f1``.
# HARDENING_9_NPSC5F V2: OBS-CONTRACT-BOUNDARY-2, persistence port type graph, runtime history bounds
# (78f4350c8, 1d2936c4c, 550227883); qualified re-freeze @ ``22c4793da4ba751fff6c93f780a7f6848650a5d9``.
# HARDENING_9_NPSC5F V3: OBS-R1 history bounds + probe identity remediation (``36de9ed76``, ``4c809e84a``).
# Prior OBS-ASOF-REBASE-R1 baseline: ``52b9dc41ed7dd83e5516d852f1ef7295cc0b10af``.
# HARDENING_9_NPSC5F V4: OBS-RUNTIME-HISTORY-BOUNDS-R3 (``39265e263``) + platform-owned validation
# (``82d7989bf``); Nexus metric-scope wiring classified unrelated (``intergrax/runtime/nexus/``).
NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA = "33576b80521dda7dfc0e5895c943f91dfebffa94"

NPSC_5E_FINAL_SHA = "fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7"
NPSC_5F_R1_FINAL_SHA = "455c09f342f995ac0a6fcb03ffef2f4d3e36a447"
NPSC_5F_R2_FINAL_SHA = "76c92847f67da22d97943b55896a88c814d7e39d"
NPSC_5F_R3_FINAL_SHA = "aa3b43456a530e1e2f50b81cab486874fe06e3b1"
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
    "docs/project/maintainers/qualification/EE_FINAL_02",
    "docs/project/maintainers/qualification/NPSC_5F_FINAL",
    "docs/project/maintainers/qualification/NPSC_5F_R",
    "docs/project/maintainers/qualification/INTEGRAX_NPSC_5F_",
    "docs/project/maintainers/qualification/W5_H1_OTLP_DEPENDENCY",
    "docs/project/maintainers/qualification/W5_H1_FIX1_NPSC5F_FREEZE_BASELINE_PROVENANCE_REPAIR",
    "docs/project/maintainers/architecture/ENTERPRISE_EXECUTION_SCALE_RESILIENCE_ARCHITECTURE.md",
    "intergrax/applications/_shared/runtime_event_delivery_wiring.py",
    "intergrax/runtime/observability/exporters/",
    "intergrax/runtime/observability/functional_evidence/",
    "intergrax/runtime/observability/functional_evidence",
    "intergrax/runtime/observability/application_execution_stage_signal",
    "testing_support/npsc5f_final_evidence_plane_drift.py",
    "testing_support/npsc5f_final_",
    "testing_support/npsc5f_r",
    "tests/unit/runtime/architecture/test_npsc5f_final_",
    "tests/unit/runtime/observability/test_w5_h1_otlp_dependency_contract.py",
    "tests/unit/runtime/observability/test_enterprise_scale_resilience_w5_g_profile_activation.py",
    "tests/unit/runtime/observability/exporters/test_otlp_transport_adapter.py",
    "tests/unit/runtime/observability/exporters/test_distributed_observability_transport.py",
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
    return any(
        normalized.startswith(prefix) for prefix in _QUALIFIED_COMPATIBLE_PREFIXES
    )


def is_explicitly_unrelated_path(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized:
        return True
    return any(
        normalized.startswith(prefix) for prefix in _EXPLICITLY_UNRELATED_PREFIXES
    )


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


def classify_evidence_plane_drift(
    changed_paths: Iterable[str],
) -> list[ClassifiedEvidencePlaneDrift]:
    ordered = sorted(
        {_normalize_repo_path(path) for path in changed_paths if path.strip()}
    )
    return [
        ClassifiedEvidencePlaneDrift(
            path=path, classification=classify_evidence_plane_drift_path(path)
        )
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
