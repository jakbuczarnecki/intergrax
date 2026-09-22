# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R5-R4 — execution-bound catalog grant wiring (post-TIGAE transport removal)."""

from __future__ import annotations

import dataclasses

import pytest

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications._shared.uca6c_codecraft_qualified_execution_composition import (
    build_execution_bound_catalog_tool_invoker_for_qualified_capability,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    PolicyCompositionResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.continuation.composition import (
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
    NexusExecutionBoundCatalogToolInvoker,
)
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
from intergrax.tools.providers.sandbox.contracts import CodeExecInput
from intergrax.tools.registry import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.autonomous_work.uca6c_r5_r2_strict_fixtures import (
    uca6c_attach_catalog_hitl_grant,
    uca6c_high_risk_tool_approval_grant,
    uca6c_strict_sandbox_env_profile,
    uca6c_strict_worker_manifest,
    uca6c_strict_worker_registry,
)
from intergrax.contracts.execution_identity import mint_run_id

pytestmark = pytest.mark.unit


class _AllowMse:
    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.orchestration.tool_invocation",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        _ = request, source_agent_id, source_step_id
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="test",
            policy_rule_id="test.mse",
        )
        enforcement = CollaborativeWorkEnforcementResult(
            operation_id=request.operation_id,
            authority_scope=request.resource_scope,
            composition=PolicyCompositionResult(
                decision=decision,
                collaborative_authority=decision,
            ),
        )
        return MeaningfulSideEffectAuthorizationResult(
            permitted=True,
            decision=decision,
            enforcement_result=enforcement,
            requires_governed_continuation=False,
            governed_continuation_request=None,
        )


def _strict_catalog_invoker() -> NexusExecutionBoundCatalogToolInvoker:
    manifest = uca6c_strict_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    tool_wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled_bundles=["sandbox"]),
        wiring_context=ToolWiringContext(),
        registry=ToolRegistry(),
    )
    invoker = build_execution_bound_catalog_tool_invoker_for_qualified_capability(
        tool_wiring,
        uca6c_strict_sandbox_env_profile(),
        caller_agent_id="worker-uca6c-qualified",
        tenant_id=_TENANT,
        manifest=manifest,
        agent_registry=registry,
        meaningful_side_effect_authorization=_AllowMse(),
        continuation_dependencies=wire_execution_engine_continuation_dependencies(),
    )
    assert isinstance(invoker, NexusExecutionBoundCatalogToolInvoker)
    return invoker


def test_public_invoke_request_does_not_expose_tigae_transport_field() -> None:
    fields = {
        f.name for f in dataclasses.fields(ExecutionBoundCatalogToolInvokeRequest)
    }
    assert "governance_approval_evidence" not in fields


def test_binding_grant_maps_to_runtime_state() -> None:
    catalog = _strict_catalog_invoker()
    run_id = str(mint_run_id())
    step_id = "uca6c.bound:nexus-grant"
    grant = uca6c_high_risk_tool_approval_grant(
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=run_id,
        step_id=step_id,
    )
    uca6c_attach_catalog_hitl_grant(catalog, grant)
    state = catalog._runtime_state(
        ExecutionBoundCatalogToolInvokeRequest(
            tool_id=CODE_EXEC_TOOL_ID,
            input=CodeExecInput(code="1", language="python", timeout_s=5),
            tenant_id=_TENANT,
            task_id=str(_TASK_ID),
            run_id=run_id,
            agent_id="worker-uca6c-qualified",
            step_id=step_id,
        ),
    )
    assert state.declarative_hitl_grant is not None
    assert state.declarative_hitl_grant.grant_id == grant.grant_id
