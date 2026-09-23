# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R4.2 — single continuation dependency bundle for pause/resume/QCE."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationIdentity,
    ExecutionContinuationLookup,
    ExecutionPauseRequest,
)
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.runtime.execution.continuation.composition import (
    ExecutionEngineContinuationDependencies,
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.execution_bound_catalog_tool_composition import (
    build_execution_bound_catalog_tool_composition,
)
from tests.unit.autonomous_work.uca6c_r5_r2_strict_fixtures import (
    uca6c_strict_r6_durable_wiring,
    uca6c_strict_sandbox_env_profile,
    uca6c_strict_worker_manifest,
    uca6c_strict_worker_registry,
)

pytestmark = pytest.mark.unit


def test_qce_composition_reuses_injected_continuation_bundle(tmp_path: Path) -> None:
    from intergrax.applications._shared.agent_runtime_governance_wiring import (
        capability_grants_from_application_manifest,
    )
    from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
    from intergrax.runtime.wiring.agent_runtime_governance_factory import (
        build_agent_runtime_governance_boundary,
    )
    from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
    from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
    from intergrax.tools.registry import ToolProfile
    from intergrax.tools.registry.runtime import ToolRegistry
    from intergrax.tools.registry.wiring import ToolWiringContext

    bundle = uca6c_strict_r6_durable_wiring(tmp_path)
    deps = bundle["continuation_dependencies"]
    assert isinstance(deps, ExecutionEngineContinuationDependencies)
    manifest = uca6c_strict_worker_manifest()
    tenant_id = "tenant-uca6c"
    registry = uca6c_strict_worker_registry(manifest)
    tool_wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled_bundles=["sandbox"]),
        wiring_context=ToolWiringContext(),
        registry=ToolRegistry(),
    )
    grants = capability_grants_from_application_manifest(
        manifest,
        tenant_id=tenant_id,
        agent_registry=registry,
    )

    class _AllowMse:
        def authorize(self, *args, **kwargs):  # noqa: ANN002, ANN003
            from intergrax.contracts.collaborative_work import (
                CollaborativeWorkEnforcementResult,
                PolicyCompositionResult,
            )
            from intergrax.contracts.meaningful_side_effect_authorization import (
                MeaningfulSideEffectAuthorizationResult,
            )
            from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

            decision = PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="test",
                policy_rule_id="test.mse",
            )
            enforcement = CollaborativeWorkEnforcementResult(
                operation_id="op",
                authority_scope="scope",
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

    from intergrax.runtime.sandbox.isolation_gate import SandboxIsolationAvailability

    availability = lambda: SandboxIsolationAvailability(  # noqa: E731
        session_configured=True,
        host_configured=True,
        healthy=True,
    )
    from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle

    composition = build_execution_bound_catalog_tool_composition(
        registry=tool_wiring.registry,
        policy_bundle=RuntimePolicyBundle(),
        caller_agent_id="worker-uca6c-qualified",
        sandbox_availability=availability,
        production_mode=True,
        scope_policy=StaticToolScopePolicy(allowed_tools={CODE_EXEC_TOOL_ID}),
        agent_runtime_governance=build_agent_runtime_governance_boundary(
            capability_grants=grants,
        ),
        meaningful_side_effect_authorization=_AllowMse(),
        canonical_inner_execution_guard=None,
        document_store=bundle["document_store"],
        continuation_dependencies=deps,
        reentry_claim_owner_id="uca6c:test",
        durable_wiring_binding_resolver=bundle["durable_wiring_binding_resolver"],
        task_checkpoint_store=bundle["task_checkpoint_store"],
    )
    host_deps = composition.invoker.continuation_aware_dependencies
    assert host_deps is not None
    assert host_deps.hitl_continuation.port is deps.continuation
    assert host_deps.hitl_continuation.lifecycle_driver is deps.lifecycle_driver
    reentry = composition.suspended_work_reentry_coordinator
    assert reentry is not None
    assert reentry.continuation_port is deps.continuation


def test_pause_and_resume_use_same_continuation_port_instance(tmp_path: Path) -> None:
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )

    deps = wire_execution_engine_continuation_dependencies()
    identity = ExecutionContinuationIdentity(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    continuation_id = "gcr_uca6c_r6_single_owner"
    paused = deps.continuation.request_pause(
        ExecutionPauseRequest(
            identity=identity,
            continuation_id=continuation_id,
            reason=ContinuationReason.COMPLIANCE,
            pause_id="pause-r6",
            human_request_id="hr-r6",
            requested_at="2026-09-23T00:00:00+00:00",
        ),
    )
    assert paused.continuation_id == continuation_id
    loaded = deps.continuation.get_pending(
        ExecutionContinuationLookup(continuation_id=continuation_id),
    )
    assert loaded.continuation_id == continuation_id
