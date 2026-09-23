# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6 — permanent negative gates for legacy TIGAE / uca6c-scope removal."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from intergrax.contracts.autonomous_work import worker_qualified_capability_resume as aw_resume
from intergrax.contracts.execution import qualified_capability_execution_dispatch as qce_dispatch
from intergrax.contracts.execution import qualified_capability_execution_intake as qce_intake
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
UCA_COMPOSITION = (
    REPO
    / "intergrax"
    / "applications"
    / "_shared"
    / "uca6c_codecraft_qualified_execution_composition.py"
)
AW_RUNTIME = REPO / "intergrax" / "autonomous_work"
QCE_RUNTIME = REPO / "intergrax" / "runtime" / "execution"


def test_worker_resume_contract_has_no_governance_approval_evidence_field() -> None:
    for cls in (
        aw_resume.WorkerQualifiedCapabilityResumeRequest,
        aw_resume.WorkerQualifiedCapabilityExecutionRequest,
    ):
        names = {f.name for f in dataclasses.fields(cls)}
        assert "governance_approval_evidence" not in names


def test_qce_dispatch_and_intake_have_no_governance_approval_evidence_field() -> None:
    for cls in (
        qce_dispatch.QualifiedCapabilityExecutionDispatchRequest,
        qce_intake.QualifiedCapabilityExecutionIntakePayload,
    ):
        names = {f.name for f in dataclasses.fields(cls)}
        assert "governance_approval_evidence" not in names


def test_catalog_invoke_request_has_no_governance_approval_evidence_field() -> None:
    names = {f.name for f in dataclasses.fields(ExecutionBoundCatalogToolInvokeRequest)}
    assert "governance_approval_evidence" not in names


def test_production_uca_graph_has_zero_uca6c_scope_literals() -> None:
    roots = (
        REPO / "intergrax" / "applications",
        REPO / "intergrax" / "autonomous_work",
        REPO / "intergrax" / "runtime" / "codecraft",
        REPO / "intergrax" / "runtime" / "execution",
        REPO / "intergrax" / "contracts" / "autonomous_work",
        REPO / "intergrax" / "contracts" / "execution",
    )
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "uca6c-scope:" not in text, f"{path} still references uca6c-scope"


def test_uca_composition_has_no_extras_sandbox_session_manager_lookup() -> None:
    text = UCA_COMPOSITION.read_text(encoding="utf-8")
    assert "sandbox_session_manager" not in text
    assert "wire_sandbox_sessions" not in text
    assert 'extras.get("sandbox_session_manager")' not in text
    assert 'extras["sandbox_session_manager"]' not in text


def test_worker_and_dispatch_runtime_modules_avoid_governance_approval_evidence_field() -> None:
    paths = (
        QCE_RUNTIME / "qualified_capability_execution_dispatch_service.py",
        QCE_RUNTIME / "qualified_capability_execution_runtime_delegate.py",
        QCE_RUNTIME / "worker_qualified_capability_execution_adapter.py",
        AW_RUNTIME / "worker_qualified_capability_resume_coordinator.py",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "governance_approval_evidence" not in text, path.name
