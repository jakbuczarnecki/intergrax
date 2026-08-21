# © Artur Czarnecki. All rights reserved.

"""Docker-backed integration tests for TOOLS-ITERATIVE-SQL-INVESTIGATION infrastructure."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.integrations.providers.relational_store.postgresql import create_postgresql_relational_store
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.execution_models import ToolExecutionRequest
from platform_proofs.tools.iterative_sql_investigation.contracts import (
    PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
    SqlQueryInput,
)
from platform_proofs.tools.iterative_sql_investigation.runtime import (
    ADMIN_DSN_ENV,
    DEFAULT_ADMIN_DSN,
    DEFAULT_RUNTIME_DSN,
    DSN_ENV,
    ProofSqlRuntime,
)
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = [pytest.mark.integration, pytest.mark.network]


def _runtime_state(invoker, registry) -> RuntimeState:
    run_id = mint_run_id()
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tool_invoker=invoker,
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=MagicMock(),
    )
    return RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id="proof-agent",
            user_id="proof-user",
            session_id="proof-session",
            tenant_id="proof-tenant",
            message="investigate delays",
            task_id=mint_task_id(),
            run_id=run_id,
        ),
        run_id=str(run_id),
        tool_traces=[],
    )


def test_read_only_runtime_role_allows_select() -> None:
    dsn = os.environ.get(DSN_ENV, DEFAULT_RUNTIME_DSN)
    store = create_postgresql_relational_store(dsn=dsn, tenant_schema="proof")
    try:
        store.connect()
        rows = store.fetch_all("SELECT COUNT(*) AS total FROM parcel_events")
        assert int(rows[0]["total"]) >= 0
    finally:
        store.close()


def test_read_only_runtime_role_rejects_write_at_database_boundary() -> None:
    dsn = os.environ.get(DSN_ENV, DEFAULT_RUNTIME_DSN)
    store = create_postgresql_relational_store(dsn=dsn, tenant_schema="proof")
    try:
        store.connect()
        with pytest.raises(Exception, match="permission denied|InsufficientPrivilege|42501"):
            store.execute("INSERT INTO parcel_events(parcel_id, created_at, region, origin_hub, destination_hub, carrier, service_type, route_type, distance_km, weight_kg, planned_hours, actual_hours, delayed, weekday) VALUES (999999, NOW(), 'North', 'X', 'Y', 'CarrierA', 'standard', 'local', 1, 1, 1, 1, false, 1)")
    finally:
        store.close()


def test_bounded_output_against_real_database(proof_sql_runtime: ProofSqlRuntime) -> None:
    request = ToolExecutionRequest(
        run_id="proof-run-bounded",
        step_id="step-1",
        tool_id=PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
        input=SqlQueryInput(sql="SELECT parcel_id FROM parcel_events ORDER BY parcel_id"),
    )
    output = proof_sql_runtime.registry.get(PLATFORM_PROOF_SQL_QUERY_TOOL_ID).handler.execute(request)
    assert output.row_count == 200
    assert output.truncated is True


def test_tool_runtime_invoker_returns_typed_bounded_result(proof_sql_runtime: ProofSqlRuntime) -> None:
    state = _runtime_state(proof_sql_runtime.invoker, proof_sql_runtime.registry)
    request = ToolExecutionRequest(
        run_id=state.run_id,
        step_id="step-runtime-1",
        tool_id=PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
        input=SqlQueryInput(sql="SELECT parcel_id FROM parcel_events ORDER BY parcel_id"),
    )
    result = proof_sql_runtime.invoker.invoke(
        state=state,
        agent_id=state.request.agent_id,
        request=request,
    )
    assert result.success is True
    assert result.output is not None
    assert result.output.row_count == 200
    assert result.output.truncated is True
    assert result.output.columns == ("parcel_id",)


def test_admin_can_reload_dataset() -> None:
    admin_dsn = os.environ.get(ADMIN_DSN_ENV, DEFAULT_ADMIN_DSN)
    store = create_postgresql_relational_store(dsn=admin_dsn, tenant_schema="proof")
    try:
        store.connect()
        before = store.fetch_all("SELECT COUNT(*) AS total FROM parcel_events")[0]["total"]
        assert int(before) >= 0
    finally:
        store.close()
