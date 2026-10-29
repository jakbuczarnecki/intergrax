# © Artur Czarnecki. All rights reserved.

"""DUP-X — NPSC-4.2 exclusion drift detection (does not modify NPSC allowlist)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_NPSC_EXCLUSION_TEN_MODULES: frozenset[str] = frozenset(
    {
        "intergrax/applications/_shared/harness_host_runtime.py",
        "intergrax/applications/_shared/scenario_runtime_baseline.py",
        "intergrax/runtime/execution/application_host_orchestration_composition.py",
        "intergrax/runtime/execution/environment_host_task_execution.py",
        "intergrax/runtime/execution/environment_orchestration_materialization.py",
        "intergrax/runtime/execution/harness_host_internal_composition.py",
        "intergrax/runtime/execution/harness_host_orchestration_wiring.py",
        "intergrax/runtime/execution/harness_host_topology_submission_wiring.py",
        "intergrax/runtime/execution/host_orchestration_assembly_validation.py",
        "intergrax/runtime/execution/worker_host_task_execution_composition.py",
    }
)


_NPSC_ALLOWLIST: frozenset[str] = frozenset(
    {
        "intergrax/applications/_shared/harness_host_composition.py",
        "applications/local_workspace_application/model_runtime_proof/runtime.py",
        "scripts/maintenance/check_harness_security_wiring.py",
        "scripts/maintenance/check_harness_reliability_wiring.py",
    }
)


def test_dup_x_npsc_exclusion_record_covers_ten_modules() -> None:
    """Each documented module must exist and reference orchestration backend wiring."""
    for rel in sorted(_NPSC_EXCLUSION_TEN_MODULES):
        path = _REPO_ROOT / rel.replace("/", "\\")
        assert path.is_file(), rel
        assert "_orchestration_backend" in path.read_text(encoding="utf-8-sig")
        assert rel not in _NPSC_ALLOWLIST


def test_dup_x_npsc_exclusion_record_md_lists_ten_modules() -> None:
    text = (_REPO_ROOT / "docs/project/maintainers/qualification/DUP_X_NPSC_4_2_DUPX_EXCLUSION_RECORD.md").read_text(
        encoding="utf-8"
    )
    for rel in sorted(_NPSC_EXCLUSION_TEN_MODULES):
        assert rel in text, rel
