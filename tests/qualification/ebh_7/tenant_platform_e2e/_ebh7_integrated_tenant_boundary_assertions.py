# © Artur Czarnecki. All rights reserved.

"""EBH-7 gap-closure integrated tenant boundary assertions (inspected @ disposition)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from intergrax.contracts.tools.tool_invocation_wiring_resolution import ToolInvocationContext
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
)
from intergrax.runtime.governance.governance_identity_projection import (
    GovernanceIdentityProjectionMismatchError,
    validate_governance_identity_projection,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from pydantic import BaseModel
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
    tools_agent_make_contract,
)
from tests.qualification.ebh_7._proof_execution import execute_proof
from tests.qualification.tenant_x._r2_adversarial import (
    assert_reference_executor_rejects_evidence_tenant_mismatch,
)


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _EchoHandler(ToolHandler[_In, _Out]):
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        _ = request
        return _Out(result=1)


def assert_gov_exec_cross_tenant_projection_denied() -> None:
    import asyncio

    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_w1w2_reconciliation_probes import (
        probe_governed_root_launch_denied_before_intake,
    )

    identity = ActiveExecutionGovernanceIdentity(
        tenant_id="tenant-ebh7-a",
        workspace_id="workspace-ebh7",
        principal_id="principal-ebh7",
    )
    with pytest.raises(GovernanceIdentityProjectionMismatchError, match="tenant"):
        validate_governance_identity_projection(identity, tenant_id="tenant-ebh7-b")
    asyncio.run(probe_governed_root_launch_denied_before_intake())


async def assert_exec_tool_cross_tenant_via_tool_runtime() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_w1w2_reconciliation_probes import (
        probe_tool_runtime_invoke_chain_cross_tenant_denied,
    )

    await probe_tool_runtime_invoke_chain_cross_tenant_denied()


def assert_exec_tool_cross_tenant_invocation_denied() -> None:
    registry = ToolRegistry()
    tool_id = "ebh7.integrated.echo"
    registry.register(
        tools_agent_make_contract(tool_id, _In, _Out),
        _EchoHandler(),
    )
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    run_id = canonical_run_id_for_tests("ebh7-ten-e2e-04")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        tool_invoker=invoker,
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=MagicMock(),
        context_builder=MagicMock(),
        rag_prompt_builder=MagicMock(),
        websearch_executor=MagicMock(),
        websearch_prompt_builder=MagicMock(),
    )
    request = RuntimeRequest(
        agent_id="agent-ebh7",
        user_id="user-ebh7",
        session_id="session-ebh7",
        tenant_id="tenant-ebh7-a",
        message="invoke",
        task_id="task_00000000000000000000000000000001",
        run_id=run_id,
    )
    state = RuntimeState(
        context=ctx,
        request=request,
        run_id=run_id,
        session=ChatSession(
            id="session-ebh7",
            tenant_id="tenant-ebh7-a",
            user_id="user-ebh7",
        ),
        tool_traces=[],
    )
    exec_request = ToolExecutionRequest(
        run_id=str(run_id),
        step_id="step-ebh7-04",
        tool_id=tool_id,
        input=_In(),
        invocation_context=ToolInvocationContext(
            run_id=str(run_id),
            step_id="step-ebh7-04",
            tool_id=tool_id,
            agent_id="agent-ebh7",
            tenant_id="tenant-ebh7-b",
        ),
    )
    with (
        canonical_execution_identity_scope(run_id),
        pytest.raises(ToolScopeViolationError),
    ):
        invoker.invoke(state=state, agent_id="agent-ebh7", request=exec_request)


def assert_config_provider_cross_tenant_rejected() -> None:
    assert_reference_executor_rejects_evidence_tenant_mismatch()


def assert_backend_switch_preserves_tenant_binding(tmp_path: Path) -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_w1w2_reconciliation_probes import (
        probe_relational_store_sqlite_duckdb_dual_slug,
    )
    from tests.unit.integrations.providers.relational_store.sqlite.test_configuration_realization import (
        test_p2_16_same_config_distinct_tenant_binding_identity,
    )

    probe_relational_store_sqlite_duckdb_dual_slug(tmp_path)
    test_p2_16_same_config_distinct_tenant_binding_identity()


def assert_obs_tenant_configuration_isolated() -> None:
    from tests.qualification.trace_x.test_trace_x_p5_p0_qualification_gates import (
        test_txp5p0_q10_configuration_tenant_mismatch_fail_closed,
        test_txp5p0_q11_profile_binding_tenant_mismatch_fail_closed,
    )

    test_txp5p0_q10_configuration_tenant_mismatch_fail_closed()
    test_txp5p0_q11_profile_binding_tenant_mismatch_fail_closed()


def assert_hitl_resume_tenant_widen_denied() -> None:
    from tests.qualification.tenant_x._r3_adversarial import (
        assert_partial_recovery_tenant_mismatch_rejects,
    )

    assert_partial_recovery_tenant_mismatch_rejects()


def assert_parent_child_exec_cross_tenant_denied() -> None:
    from tests.qualification.tenant_x._r4_adversarial import (
        assert_background_transport_task_tenant_mismatch_rejects,
    )

    assert_background_transport_task_tenant_mismatch_rejects()


def assert_control_plane_cross_tenant_mutation_denied() -> None:
    from tests.qualification.config_x.test_config_x_adversarial_cx_gates import (
        test_cx_g_cross_tenant_provider_config_rejected_by_realization_guards,
    )

    test_cx_g_cross_tenant_provider_config_rejected_by_realization_guards()


def assert_scheduler_worker_tenant_drift_denied() -> None:
    from tests.qualification.tenant_x._r4_adversarial import (
        assert_vendor_worker_payload_job_tenant_mismatch_rejects,
    )

    assert_vendor_worker_payload_job_tenant_mismatch_rejects()


def assert_trace_replay_cross_tenant_denied() -> None:
    from tests.unit.applications.test_uca6c_marketplace_qualified_execution_composition import (
        test_tenant_mismatch_blocks_materialization_pin_and_io,
    )

    test_tenant_mismatch_blocks_materialization_pin_and_io()


def assert_continuation_aware_catalog_tool_host_cross_tenant_denied() -> None:
    from intergrax.contracts.execution_bound_catalog_tool_invocation import (
        ExecutionBoundCatalogToolInvokeRequest,
    )
    from intergrax.runtime.nexus.tools.continuation_aware_catalog_tool_host import (
        ContinuationAwareCatalogToolHost,
    )

    registry = ToolRegistry()
    tool_id = "ebh7.integrated.catalog_host"
    registry.register(
        tools_agent_make_contract(tool_id, _In, _Out),
        _EchoHandler(),
    )
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    host = ContinuationAwareCatalogToolHost(tool_invoker=invoker)
    run_id = canonical_run_id_for_tests("ebh7-ten-e2e-13")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        tool_invoker=invoker,
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=MagicMock(),
        context_builder=MagicMock(),
        rag_prompt_builder=MagicMock(),
        websearch_executor=MagicMock(),
        websearch_prompt_builder=MagicMock(),
    )
    request = RuntimeRequest(
        agent_id="agent-ebh7",
        user_id="user-ebh7",
        session_id="session-ebh7",
        tenant_id="tenant-ebh7-a",
        message="catalog",
        task_id="task_00000000000000000000000000000001",
        run_id=run_id,
    )
    state = RuntimeState(
        context=ctx,
        request=request,
        run_id=run_id,
        session=ChatSession(
            id="session-ebh7",
            tenant_id="tenant-ebh7-a",
            user_id="user-ebh7",
        ),
        tool_traces=[],
    )
    catalog_request = ExecutionBoundCatalogToolInvokeRequest(
        tool_id=tool_id,
        input=_In(),
        tenant_id="tenant-ebh7-b",
        task_id="task_00000000000000000000000000000001",
        run_id=str(run_id),
        agent_id="agent-ebh7",
        step_id="step-ebh7-13",
    )
    with (
        canonical_execution_identity_scope(run_id),
        pytest.raises(ToolScopeViolationError),
    ):
        host.invoke(state=state, request=catalog_request, declarative_grant=None)


def assert_plugin_host_compat_cross_tenant_activation_denied() -> None:
    execute_proof(
        "tests.qualification.compat_x.test_compat_x_plugin_provider_r4_gates",
        "test_r4_probe_g_cross_tenant_evidence",
    )
    assert_continuation_aware_catalog_tool_host_cross_tenant_denied()
