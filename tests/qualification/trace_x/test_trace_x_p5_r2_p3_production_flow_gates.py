# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3 mechanical production-flow qualification gates."""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

TRACE_X_P5_R2_P3_START_HEAD = "fa08dfce43b6bae0918bdda6cad247da492a2143"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_txp5r2p3_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R2_P3_START_HEAD, "HEAD"],
        cwd=_REPO_ROOT,
    )


def test_txp5r2p3_q02_configured_fulfillment_port_exists() -> None:
    mod = importlib.import_module(
        "intergrax.contracts.autonomous_work.worker_configured_capability_fulfillment",
    )
    assert hasattr(mod, "WorkerConfiguredCapabilityFulfillmentPort")


def test_txp5r2p3_q03_configured_service_depends_on_opportunity_read() -> None:
    from intergrax.autonomous_work.worker_configured_capability_fulfillment_service import (
        WorkerConfiguredCapabilityFulfillmentService,
    )

    params = inspect.signature(WorkerConfiguredCapabilityFulfillmentService.__init__).parameters
    assert "opportunity_read" in params


def test_txp5r2p3_q04_configured_service_depends_on_realization_port() -> None:
    from intergrax.autonomous_work.worker_configured_capability_fulfillment_service import (
        WorkerConfiguredCapabilityFulfillmentService,
    )

    params = inspect.signature(WorkerConfiguredCapabilityFulfillmentService.__init__).parameters
    assert "realization" in params


def test_txp5r2p3_q05_aw_service_has_no_governance_port_import() -> None:
    source = (
        _REPO_ROOT
        / "intergrax/autonomous_work/worker_configured_capability_fulfillment_service.py"
    ).read_text(encoding="utf-8")
    assert "ControlPlaneMutationAuthorizationPort" not in source


def test_txp5r2p3_q06_execution_bound_resolution_exists() -> None:
    mod = importlib.import_module("intergrax.integrations.execution_bound_integration_resolution")
    assert hasattr(mod, "ExecutionBoundIntegrationResolution")


def test_txp5r2p3_q07_pinning_store_protocol_dependency() -> None:
    from intergrax.integrations.execution_bound_integration_resolution import (
        ExecutionBoundIntegrationResolution,
    )

    params = inspect.signature(ExecutionBoundIntegrationResolution.__init__).parameters
    assert "pinning_store" in params


def test_txp5r2p3_q08_coordinator_configured_routing() -> None:
    source = (
        _REPO_ROOT / "intergrax/autonomous_work/worker_capability_fulfillment_coordinator.py"
    ).read_text(encoding="utf-8")
    assert "_fulfill_configure_existing" in source
    assert "CONFIGURE_EXISTING_REQUIRED" in source


def test_txp5r2p3_q09_no_category_value_provider_fallback_in_resolution() -> None:
    source = (
        _REPO_ROOT / "intergrax/integrations/execution_bound_integration_resolution.py"
    ).read_text(encoding="utf-8")
    assert "category.value" not in source


def test_txp5r2p3_q10_delegate_does_not_pre_pin_configured_adoption() -> None:
    source = (
        _REPO_ROOT
        / "intergrax/runtime/execution/qualified_capability_execution_runtime_delegate.py"
    ).read_text(encoding="utf-8")
    assert "pin_configured_adoption_for_execution" not in source
    assert "integration_configuration_adoption=" in source


def test_txp5r2p3_q11_handler_protocol_propagates_adoption_kwarg() -> None:
    from intergrax.runtime.execution.qualified_capability_execution_handlers import (
        QualifiedCapabilityExecutionBindingHandler,
    )

    params = inspect.signature(QualifiedCapabilityExecutionBindingHandler.dispatch_once).parameters
    assert "integration_configuration_adoption" in params


def test_txp5r2p3_q12_resolve_config_removed_from_execution_bound_request() -> None:
    from intergrax.integrations.execution_bound_integration_resolution import (
        ExecutionBoundIntegrationResolutionRequest,
    )

    assert "resolve_config" not in ExecutionBoundIntegrationResolutionRequest.__annotations__


def test_txp5r2p3_q13_materialization_port_returns_category_instance() -> None:
    from intergrax.integrations.execution_bound_integration_resolution import (
        ExecutionBoundIntegrationMaterializationPort,
    )

    resolve_from_profile = ExecutionBoundIntegrationMaterializationPort.resolve_from_profile
    hints = resolve_from_profile.__annotations__
    assert hints.get("return") != "object"


def test_txp5r2p3_q14_configured_relational_execution_port_exists() -> None:
    mod = importlib.import_module(
        "intergrax.integrations.contracts.configured_relational_store_execution",
    )
    assert hasattr(mod, "ConfiguredRelationalStoreExecutionPort")


def test_txp5r2p3_q15_tool_invocation_wiring_has_configured_relational_slot() -> None:
    from intergrax.tools.invocation_wiring import ToolInvocationWiring

    assert "configured_relational_store_execution" in ToolInvocationWiring.__annotations__


def test_txp5r2p3_q16_tool_wiring_context_has_relational_store_execution() -> None:
    from intergrax.tools.registry.wiring import ToolWiringContext

    assert "relational_store_execution" in ToolWiringContext.__annotations__


def test_txp5r2p3_q17_database_query_uses_relational_store_execution_only() -> None:
    source = (
        _REPO_ROOT / "intergrax/tools/providers/database/service.py"
    ).read_text(encoding="utf-8")
    query_block = source.split("def database_query", 1)[1].split("def database_execute", 1)[0]
    assert "relational_store_execution" in query_block
    assert "relational_store" not in query_block.replace("relational_store_execution", "")


def test_txp5r2p3_q18_marketplace_handler_requires_projection_when_adoption_present() -> None:
    source = (
        _REPO_ROOT
        / "intergrax/tools/marketplace_qualified_capability_execution_handler.py"
    ).read_text(encoding="utf-8")
    assert "configured_invocation_projection" in source
    assert "configured_invocation_projection_unavailable" in source


def test_txp5r2p3_q19_no_sqlite_import_in_projection_module() -> None:
    source = (
        _REPO_ROOT
        / "intergrax/tools/configured_integration_tool_invocation_projection.py"
    ).read_text(encoding="utf-8")
    assert "sqlite" not in source.lower()


def test_txp5r2p3_q20_no_execution_id_provider_cache() -> None:
    tree = ast.parse(
        (
            _REPO_ROOT
            / "intergrax/integrations/execution_bound_configured_relational_store_port.py"
        ).read_text(encoding="utf-8"),
    )
    source = ast.unparse(tree)
    assert "dict[" not in source or "ExecutionId" not in source


def test_txp5r2p3_q21_database_tool_abi_sql_scalar_typed() -> None:
    from intergrax.tools.providers.database.contracts import (
        DatabaseExecuteInput,
        DatabaseQueryInput,
        DatabaseQueryOutput,
    )

    assert DatabaseQueryInput.model_fields["params"].annotation is not None
    assert "Any" not in str(DatabaseQueryInput.model_fields["params"].annotation)
    assert "Any" not in str(DatabaseExecuteInput.model_fields["params"].annotation)
    assert "Any" not in str(DatabaseQueryOutput.model_fields["rows"].annotation)


def test_txp5r2p3_q22_database_query_output_json_serializes_bytes_scalar() -> None:
    from intergrax.tools.providers.database.contracts import DatabaseQueryOutput

    payload = DatabaseQueryOutput(
        rows=[{"b": b"\x01\x02", "s": "x", "n": 1, "f": 1.5, "t": True, "z": None}],
        row_count=1,
    )
    json_text = payload.model_dump_json()
    assert "b" in json_text
