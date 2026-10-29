# © Artur Czarnecki. All rights reserved.

"""DUP-X B4 — adversarial negative boundary for NPSC-4.2 ten-module exclusion.

Proves prohibited behaviors (second execution ingress, owner-zone escape, lab bypass)
without weakening ``test_npsc42_orchestration_backend_access_confined_to_allowlist``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.dup_x.test_dup_x_npsc_boundary_gates import (
    _NPSC_EXCLUSION_TEN_MODULES,
)

_FORBIDDEN_LAB_IMPORT_MARKERS = (
    "debug_lab_nexus_loop",
    "runtime.transport.bootstrap",
    "build_lab_organization_nexus_loop",
)

_SHADOW_SLUG_MARKERS = (
    "read_integration_slug_from_env",
    "read_integration_slug(",
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_ACCESS_SYMBOL = "orchestration_backend_for_execution_engine"
_ACCESS_MODULE_SUFFIX = "intergrax.runtime.execution._orchestration_backend_access"

_MATERIALIZE_SYMBOL = "materialize_host_orchestration_backend"

_ALLOWED_MATERIALIZE_CALLERS: frozenset[str] = frozenset(
    {
        "intergrax/runtime/execution/environment_orchestration_materialization.py",
        "intergrax/runtime/execution/application_host_orchestration_composition.py",
        "intergrax/runtime/execution/worker_host_task_execution_composition.py",
        "intergrax/applications/_shared/harness_host_runtime.py",
    }
)

_PRODUCTION_SCAN_ROOTS = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "applications",
)


def _relative(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _is_dup_x_production_effective(rel: str) -> bool:
    if "/docker/runtime-context/" in rel:
        return False
    if rel.startswith("agents/") and "/tests/" in rel:
        return False
    return True


def _python_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*.py"):
        if not p.is_file():
            continue
        rel = _relative(p)
        if "/tests/" in rel or p.name.startswith("test_"):
            continue
        if not _is_dup_x_production_effective(rel):
            continue
        out.append(p)
    return out


def _imports_orchestration_backend_access(path: Path) -> bool:
    text = path.read_text(encoding="utf-8-sig")
    return _ACCESS_MODULE_SUFFIX in text or ("_orchestration_backend_access" in text and _ACCESS_SYMBOL in text)


def test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone() -> None:
    """Prohibited: Tier-3 or agents importing EE private backend accessor."""
    violations: list[str] = []
    for root in _PRODUCTION_SCAN_ROOTS:
        for path in _python_files(root):
            rel = _relative(path)
            if rel.startswith("intergrax/runtime/execution/"):
                continue
            if path.name.startswith("test_"):
                continue
            if _imports_orchestration_backend_access(path):
                violations.append(rel)
    assert violations == [], (
        "orchestration_backend_for_execution_engine imported outside runtime/execution:\n" + "\n".join(violations)
    )


def test_b4_materialize_host_orchestration_backend_callers_allowlisted() -> None:
    """Prohibited: ad-hoc Nexus materialization outside sanctioned composition roots."""
    violations: list[str] = []
    for root in _PRODUCTION_SCAN_ROOTS:
        for path in _python_files(root):
            rel = _relative(path)
            if path.name.startswith("test_"):
                continue
            text = path.read_text(encoding="utf-8-sig")
            if f"{_MATERIALIZE_SYMBOL}(" not in text:
                continue
            if rel not in _ALLOWED_MATERIALIZE_CALLERS:
                violations.append(rel)
    assert violations == [], "unexpected materialize_host_orchestration_backend callers:\n" + "\n".join(violations)


def test_b4_ten_modules_exclude_lab_bypass_and_shadow_slug_selection() -> None:
    """Prohibited within NPSC ten-module surface: COMPAT-R3 lab ingress or Class-D shadow reads."""
    for rel in sorted(_NPSC_EXCLUSION_TEN_MODULES):
        path = _REPO_ROOT / rel.replace("/", "\\")
        text = path.read_text(encoding="utf-8")
        for marker in _FORBIDDEN_LAB_IMPORT_MARKERS:
            assert marker not in text, f"{rel}: lab marker {marker!r}"
        for marker in _SHADOW_SLUG_MARKERS:
            assert marker not in text, f"{rel}: shadow slug marker {marker!r}"


def test_b4_ten_modules_execution_wiring_requires_governance_ports_in_call_chain() -> None:
    """Positive chain: scenario/harness paths wire RootExecutionAuthorityAdmissionPort (no naked Nexus run)."""
    scenario = (_REPO_ROOT / "intergrax/applications/_shared/scenario_runtime_baseline.py").read_text(encoding="utf-8")
    assert "compose_application_host_orchestration_session" in scenario
    assert "root_authority_admission" in scenario
    assert "admit_root_governance_identity" in scenario

    app_comp = (_REPO_ROOT / "intergrax/runtime/execution/application_host_orchestration_composition.py").read_text(
        encoding="utf-8"
    )
    assert "RootExecutionAuthorityAdmissionPort" in app_comp
    assert "build_environment_host_task_execution" in app_comp

    worker = (_REPO_ROOT / "intergrax/runtime/execution/worker_host_task_execution_composition.py").read_text(
        encoding="utf-8"
    )
    assert "root_authority_admission" in worker
    assert "admit_root_governance_identity" in worker
