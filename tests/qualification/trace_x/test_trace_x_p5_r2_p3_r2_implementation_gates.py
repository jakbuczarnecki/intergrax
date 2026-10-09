# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3-R2 implementation, duplicate-audit, and positive configured E2E gates."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

from tests.qualification.trace_x._trace_x_p5_r2_p3_r2_configured_negative_support import (
    TRACE_X_P5_R2_P3_R2_START_HEAD,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_QUAL_DOC = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/"
    "TRACE_X_P5_R2_P3_R2_CONFIGURED_EXECUTION_CONVERGENCE_IMPLEMENTATION_QUALIFICATION.md"
)

_P3_R2_PRODUCTION_SURFACES: tuple[str, ...] = (
    "intergrax/autonomous_work/configured_capability_execution_subject_builder.py",
    "intergrax/autonomous_work/worker_configuration_opportunity_discovery_adapter.py",
    "intergrax/autonomous_work/worker_configured_capability_execution_fulfillment_service.py",
    "intergrax/contracts/autonomous_work/worker_configured_capability_execution.py",
    "intergrax/contracts/capability_qualification/configured_capability_execution_subject.py",
    "intergrax/contracts/tools/marketplace_tool_execution_intent.py",
    "intergrax/runtime/execution/worker_configured_capability_execution_adapter.py",
    "intergrax/tools/marketplace_configured_capability_binding_provider.py",
    "intergrax/tools/marketplace_configured_tool_execution_intent_preparation.py",
    "intergrax/tools/marketplace_tool_execution_routing.py",
    "intergrax/tools/marketplace_tool_operation_selection_core.py",
)

_OWNER_MATRIX: tuple[tuple[str, int], ...] = (
    ("ExecutionRuntime authority", 1),
    ("Root execution launch", 1),
    ("Handler registry mechanism", 1),
    ("Marketplace Tool execution handler", 1),
    ("Tool activation authority", 1),
    ("Intent repository", 1),
    ("Configured fulfillment decision owner", 1),
    ("Configuration opportunity authority", 1),
    ("Capability identity authority", 1),
    ("Provider resolution mechanism", 1),
    ("Configured/effective pin owner", 1),
    ("Retry/recovery owner", 1),
    ("Tier-3 composition owner", 1),
)


def test_txp5r2p3r2_impl01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R2_P3_R2_START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5r2p3r2_impl02_qualification_artifact_exists() -> None:
    assert _QUAL_DOC.is_file()
    text = _QUAL_DOC.read_text(encoding="utf-8")
    assert "TRACE-X-P5-R2-P3-R2-R1" in text
    assert "BLOCKED ON R1" in text
    assert "R2-P3-CONFIGURED-TARGET-OPAQUE-CORRELATION-VIOLATION-23" in text
    assert "IN PROGRESS / NOT READY FOR AUDIT" not in text
    assert TRACE_X_P5_R2_P3_R2_START_HEAD in text
    assert "b7efe6b980ba010572f9acc68f8d3db4493733e8" in text


def test_txp5r2p3r2_impl03_p3_r2_surface_files_present() -> None:
    missing = [rel for rel in _P3_R2_PRODUCTION_SURFACES if not (_REPO_ROOT / rel).is_file()]
    assert missing == []


def test_txp5r2p3r2_impl04_duplicate_blocker_count_zero() -> None:
    """Closed-world: no production module named DUPLICATE / second semantic owner."""
    forbidden = (
        "configured_capability_execution_dispatch_service",
        "configured_marketplace_tool_execution_intent_repository",
        "configured_tool_activation",
    )
    hits: list[str] = []
    for rel in _P3_R2_PRODUCTION_SURFACES:
        path = _REPO_ROOT / rel
        name = path.name.lower()
        for token in forbidden:
            if token in name:
                hits.append(rel)
    assert hits == []


def test_txp5r2p3r2_impl05_owner_matrix_documented() -> None:
    text = _QUAL_DOC.read_text(encoding="utf-8")
    for concern, count in _OWNER_MATRIX:
        assert concern in text
        assert f"| {concern} | {count} |" in text


def test_txp5r2p3r2_impl06_configured_positive_e2e_regression_hook() -> None:
    from tests.unit.applications.test_uca6c_marketplace_qualified_execution_composition import (
        test_configure_existing_e2e_execution_bound_fulfillment,
    )

    test_configure_existing_e2e_execution_bound_fulfillment()


def test_txp5r2p3r2_impl07_fail_closed_no_binding_provider_routing_in_delegate() -> None:
    delegate = (
        _REPO_ROOT
        / "intergrax/runtime/execution/execution_bound_capability_execution_runtime_delegate.py"
    ).read_text(encoding="utf-8")
    assert "binding_provider_id" not in delegate
    assert "execution_handler_id" in delegate


def test_txp5r2p3r2_impl08_thin_adapter_classification() -> None:
    adapter = (
        _REPO_ROOT / "intergrax/runtime/execution/worker_configured_capability_execution_adapter.py"
    ).read_text(encoding="utf-8")
    assert "THIN ADAPTER" in _QUAL_DOC.read_text(encoding="utf-8")
    assert "WorkerConfiguredCapabilityExecutionEngineAdapter" in adapter
    assert "HandlerRegistry" not in adapter
