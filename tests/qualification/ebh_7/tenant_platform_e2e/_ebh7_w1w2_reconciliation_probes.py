# © Artur Czarnecki. All rights reserved.

"""EBH-7 W1/W2 qualification reconciliation probes (PARTIAL evidence — not EXACT promotion)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from intergrax.contracts.admitted_root_governance_identity import AdmittedRootGovernanceIdentity
from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.execution_intake import (
    CanonicalExecutionIntakePort,
    CanonicalExecutionIntakeRequest,
    CanonicalExecutionIntakeResult,
)
from intergrax.contracts.root_execution_launch import (
    RootExecutionLaunchDisposition,
    RootExecutionLaunchRequest,
)
from intergrax.contracts.root_execution_operation import RootExecutionOperation
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.tools.tool_invocation_wiring_resolution import ToolInvocationContext
from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    UnknownIntegrationError,
)
from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.governance.default_root_execution_launcher import DefaultRootExecutionLauncher
from intergrax.runtime.governance.execution_admission_composition import (
    build_root_execution_authority_admission,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    DenyingRuntimeExecutionPolicyAdmission,
)
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan, ToolRuntime
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


@dataclass(frozen=True)
class _LaunchPayload:
    value: str


class _NoopIntake(CanonicalExecutionIntakePort[_LaunchPayload, str]):
    async def dispatch(
        self,
        request: CanonicalExecutionIntakeRequest[_LaunchPayload],
    ) -> CanonicalExecutionIntakeResult[str]:
        _ = request
        return CanonicalExecutionIntakeResult(result="ok")


async def probe_governed_root_launch_denied_before_intake() -> None:
    """Exercises DefaultRootExecutionLauncher → policy admission DENY (not full cross-tenant A/B)."""
    root_admission = build_root_execution_authority_admission(
        runtime_policy_admission=DenyingRuntimeExecutionPolicyAdmission(),
    )
    intake = _NoopIntake()
    launcher = DefaultRootExecutionLauncher(
        root_authority_admission=root_admission,
        execution_intake=intake,
    )
    request = RootExecutionLaunchRequest(
        admitted_governance_identity=AdmittedRootGovernanceIdentity(
            tenant_id="tenant-ebh7-a",
            workspace_id="ws-ebh7",
            principal_id="principal-ebh7",
        ),
        root_execution_operation=RootExecutionOperation.ROOT_AGENT,
        collaborative_authority_scopes=("workspace.read",),
        effective_authority_decision=EffectiveAuthorityDecision(
            decision=PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="reconciliation-probe",
                policy_rule_id="ebh7.w1w2.probe",
            ),
        ),
        payload=_LaunchPayload(value="probe"),
    )
    result = await launcher.launch(request)
    assert result.disposition is RootExecutionLaunchDisposition.DENIED


class _In(BaseModel):
    n: int = 0


class _Out(BaseModel):
    n: int = 0


class _Handler(ToolHandler[_In, _Out]):
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        _ = request
        return _Out(n=1)


async def probe_tool_runtime_invoke_chain_cross_tenant_denied() -> None:
    """ToolRuntime.invoke → configured invoker; cross-tenant via ToolInvocationContext @ invoker."""
    tool_id = "ebh7.reconciliation.tool"
    registry = ToolRegistry()
    registry.register(tools_agent_make_contract(tool_id, _In, _Out), _Handler())
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    run_id = canonical_run_id_for_tests("ebh7-w1w2-toolruntime")
    state = RuntimeState(
        context=RuntimeContext(
            config=RuntimeConfig(
                llm_adapter=FakeLLMAdapter(),
                production_mode=False,
                tool_invoker=invoker,
            ),
            session_manager=build_in_memory_session_manager(),
            prompt_registry=MagicMock(),
            context_builder=MagicMock(),
            rag_prompt_builder=MagicMock(),
            websearch_executor=MagicMock(),
            websearch_prompt_builder=MagicMock(),
        ),
        request=RuntimeRequest(
            agent_id="agent-ebh7",
            user_id="user-ebh7",
            session_id="session-ebh7",
            tenant_id="tenant-ebh7-a",
            message="tool",
            task_id="task_00000000000000000000000000000001",
            run_id=run_id,
        ),
        run_id=run_id,
        session=ChatSession(id="session-ebh7", tenant_id="tenant-ebh7-a", user_id="user-ebh7"),
        tool_traces=[],
    )

    plan = ToolInvocationPlan.from_tool_ids([tool_id])

    def _catalog_invoke_cross_tenant(
        state: RuntimeState,
        tool_ids: tuple[str, ...],
        *,
        tool_inputs: object = None,
        trace_step: str = "",
    ) -> None:
        _ = tool_ids, tool_inputs, trace_step
        exec_request = ToolExecutionRequest(
            run_id=str(run_id),
            step_id="step-w1w2",
            tool_id=tool_id,
            input=_In(),
            invocation_context=ToolInvocationContext(
                run_id=str(run_id),
                step_id="step-w1w2",
                tool_id=tool_id,
                tenant_id="tenant-ebh7-b",
            ),
        )
        with pytest.raises(ToolScopeViolationError):
            invoker.invoke(state=state, agent_id="agent-ebh7", request=exec_request)

    with (
        patch(
            "intergrax.runtime.nexus.tools.catalog_dispatch.invoke_catalog_tool_ids",
            side_effect=_catalog_invoke_cross_tenant,
        ),
        canonical_execution_identity_scope(run_id),
    ):
        await ToolRuntime.invoke(state=state, plan=plan, allowed_tools=[tool_id])


def probe_resolve_from_profile_sqlite_sanctioned(tmp_path: Path) -> None:
    clear_catalog()
    register_sqlite_integration()
    profile = IntegrationProfile(
        relational_store="sqlite",
        options={"sqlite": {"data_dir": str(tmp_path)}},
    )
    store = resolve_from_profile(profile, IntegrationCategory.RELATIONAL_STORE)
    assert isinstance(store, RelationalStoreIntegrationContract)
    store.connect()
    store.close()


def probe_resolve_from_profile_unregistered_fail_closed() -> None:
    clear_catalog()
    with pytest.raises((UnknownIntegrationError, IntegrationConfigurationError)):
        resolve_from_profile(IntegrationProfile(), IntegrationCategory.RELATIONAL_STORE)


def probe_relational_store_sqlite_duckdb_dual_slug(tmp_path: Path) -> None:
    from tests.qualification.ebh_5._package_c_v2_proof_helpers import (
        run_package_c_v2_relational_sqlite_duckdb_dual_slug,
    )

    run_package_c_v2_relational_sqlite_duckdb_dual_slug(tmp_path)


def probe_plugin_host_activation_cross_tenant_open() -> None:
    """Placeholder — canonical host activation cross-tenant test not yet implemented."""
    pytest.skip(
        "BLOCKER: continuation_aware_catalog_tool_host cross-tenant activation "
        "requires dedicated qualification harness — not COMPAT assess",
    )
