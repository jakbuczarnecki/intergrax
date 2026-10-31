# © Artur Czarnecki. All rights reserved.

"""EBH-7 gap-closure integrated tenant boundary assertions (inspected @ disposition)."""

from __future__ import annotations

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
from tests.qualification.compat_x.test_compat_x_plugin_provider_r4_gates import (
    test_r4_probe_g_cross_tenant_evidence,
)
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
    identity = ActiveExecutionGovernanceIdentity(
        tenant_id="tenant-ebh7-a",
        workspace_id="workspace-ebh7",
        principal_id="principal-ebh7",
    )
    with pytest.raises(GovernanceIdentityProjectionMismatchError, match="tenant"):
        validate_governance_identity_projection(identity, tenant_id="tenant-ebh7-b")


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


def assert_plugin_host_compat_cross_tenant_activation_denied() -> None:
    test_r4_probe_g_cross_tenant_evidence()
