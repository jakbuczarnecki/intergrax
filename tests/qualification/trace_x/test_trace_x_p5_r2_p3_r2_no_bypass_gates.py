# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3-R2 mechanical no-bypass gates for configured execution convergence."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

from tests.qualification.trace_x._trace_x_p5_r2_p3_r2_configured_negative_support import (
    TRACE_X_P5_R2_P3_R2_START_HEAD,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_EXEC = _REPO_ROOT / "intergrax/runtime/execution"
_TOOLS = _REPO_ROOT / "intergrax/tools"
_AW = _REPO_ROOT / "intergrax/autonomous_work"


def test_txp5r2p3r2_nb01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R2_P3_R2_START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5r2p3r2_nb02_no_configured_dispatch_service() -> None:
    assert list(_RUNTIME_EXEC.glob("configured_capability_execution_dispatch*.py")) == []


def test_txp5r2p3r2_nb03_no_configured_runtime_delegate() -> None:
    assert list(_RUNTIME_EXEC.glob("configured_capability_execution_runtime_delegate*.py")) == []
    delegate = (_RUNTIME_EXEC / "execution_bound_capability_execution_runtime_delegate.py").read_text(
        encoding="utf-8",
    )
    assert "ConfiguredCapabilityExecutionRuntimeDelegate" not in delegate


def test_txp5r2p3r2_nb04_single_handler_registry_class() -> None:
    registry_files = list(_RUNTIME_EXEC.glob("*handler_registry*.py"))
    configured_registry = [
        p for p in registry_files if "configured" in p.name.lower()
    ]
    assert configured_registry == []


def test_txp5r2p3r2_nb05_configured_adapter_uses_execution_bound_dispatch_only() -> None:
    adapter = (_RUNTIME_EXEC / "worker_configured_capability_execution_adapter.py").read_text(
        encoding="utf-8",
    )
    assert "ExecutionBoundCapabilityExecutionDispatchPort" in adapter
    assert "RuntimeDelegate" not in adapter
    assert "HandlerRegistry" not in adapter


def test_txp5r2p3r2_nb06_handler_not_invoked_without_execution_bound_ingress() -> None:
    handler = (_TOOLS / "marketplace_qualified_capability_execution_handler.py").read_text(
        encoding="utf-8",
    )
    assert "RootExecutionLaunchPort" not in handler
    assert "dispatch_once" in handler


def test_txp5r2p3r2_nb07_binding_provider_does_not_activate_tool() -> None:
    binding = (_TOOLS / "marketplace_configured_capability_binding_provider.py").read_text(
        encoding="utf-8",
    )
    assert "ensure_exact_active" not in binding
    assert "ToolHostLifecycle" not in binding
    assert "catalog_tool_invoker" not in binding


def test_txp5r2p3r2_nb08_configured_fulfillment_no_tool_runtime_governance_bypass() -> None:
    service = (
        _AW / "worker_configured_capability_execution_fulfillment_service.py"
    ).read_text(encoding="utf-8")
    assert "ToolRuntime" not in service
    assert "catalog_tool_invoker" not in service


def test_txp5r2p3r2_nb09_provider_materialization_only_via_execution_bound_resolution() -> None:
    fulfillment = (
        _AW / "worker_configured_capability_execution_fulfillment_service.py"
    ).read_text(encoding="utf-8")
    assert "ExecutionBoundIntegrationResolution" not in fulfillment
    resolution = (
        _REPO_ROOT / "intergrax/integrations/execution_bound_integration_resolution.py"
    ).read_text(encoding="utf-8")
    assert "class ExecutionBoundIntegrationResolution" in resolution
    projection = (
        _TOOLS / "configured_integration_tool_invocation_projection.py"
    ).read_text(encoding="utf-8")
    assert "InvocationBoundConfiguredRelationalStoreWiringResolver" in projection


def test_txp5r2p3r2_nb10_subject_builder_no_string_identity_reconstruction() -> None:
    builder = (_AW / "configured_capability_execution_subject_builder.py").read_text(
        encoding="utf-8",
    )
    assert "catalog-tool-capability" not in builder
    assert "parse_" not in builder
    assert "CapabilityIdentityKey" in builder


def test_txp5r2p3r2_nb11_configured_path_no_uca_handoff_fields() -> None:
    intent_prep = (_TOOLS / "marketplace_configured_tool_execution_intent_preparation.py").read_text(
        encoding="utf-8",
    )
    assert "handoff_id" not in intent_prep
    assert "uca_handoff" not in intent_prep.lower()


def test_txp5r2p3r2_nb12_adoption_not_from_global_registry() -> None:
    fulfillment = (_AW / "worker_configured_capability_fulfillment_service.py").read_text(
        encoding="utf-8",
    )
    assert "get_global" not in fulfillment
    assert "global_registry" not in fulfillment.lower()
    assert "opportunity_read" in fulfillment


def test_txp5r2p3r2_nb13_forbidden_duplicate_class_names_absent() -> None:
    patterns = (
        "ConfiguredCapabilityExecutionDispatchService",
        "ConfiguredCapabilityExecutionRuntimeDelegate",
        "ConfiguredCapabilityExecutionHandlerRegistry",
        "ConfiguredToolRegistry",
        "ConfiguredMarketplaceToolExecutionHandler",
        "ConfiguredMarketplaceToolExecutionIntentRepository",
    )
    for root in (_RUNTIME_EXEC, _TOOLS, _AW):
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for pattern in patterns:
                assert pattern not in text, f"{pattern} found in {path.relative_to(_REPO_ROOT)}"
