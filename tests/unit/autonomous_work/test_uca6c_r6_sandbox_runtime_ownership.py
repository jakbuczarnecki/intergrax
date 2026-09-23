# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R4.1 — single sandbox binding resolver injection (no extras lookup)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications._shared.uca6c_codecraft_qualified_execution_composition import (
    Uca6cCodecraftQualifiedExecutionCompositionError,
    _require_durable_wiring_binding_resolver,
)
from intergrax.runtime.execution.execution_bound_catalog_tool_composition import (
    build_execution_bound_catalog_tool_composition,
)
from intergrax.runtime.sandbox.durable_sandbox_wiring_binding_resolver import (
    SandboxSessionManagerDurableWiringBindingResolver,
)
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.tools.registry import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.unit.autonomous_work.uca6c_r5_r2_strict_fixtures import (
    uca6c_strict_r6_durable_wiring,
    uca6c_strict_sandbox_env_profile,
    uca6c_strict_worker_manifest,
    uca6c_strict_worker_registry,
)

pytestmark = pytest.mark.unit


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


def test_strict_composition_requires_injected_binding_resolver() -> None:
    with pytest.raises(
        Uca6cCodecraftQualifiedExecutionCompositionError,
        match="injected",
    ):
        _require_durable_wiring_binding_resolver(injected=None)


def test_initial_and_reentry_share_one_injected_resolver_instance(
    tmp_path: Path,
) -> None:
    from intergrax.applications._shared.agent_runtime_governance_wiring import (
        capability_grants_from_application_manifest,
    )
    from intergrax.runtime.wiring.agent_runtime_governance_factory import (
        build_agent_runtime_governance_boundary,
    )
    from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
    from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID

    manifest = uca6c_strict_worker_manifest()
    tenant_id = "tenant-uca6c"
    registry = uca6c_strict_worker_registry(manifest)
    bundle = uca6c_strict_r6_durable_wiring(tmp_path)
    resolver = bundle["durable_wiring_binding_resolver"]
    assert isinstance(resolver, SandboxSessionManagerDurableWiringBindingResolver)
    manager = bundle["sandbox_session_manager"]
    assert isinstance(manager, SandboxSessionManager)

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
        continuation_dependencies=bundle["continuation_dependencies"],
        reentry_claim_owner_id="uca6c:test",
        durable_wiring_binding_resolver=resolver,
        task_checkpoint_store=bundle["task_checkpoint_store"],
    )
    assert composition.suspended_work_reentry_coordinator is not None
    assert (
        composition.suspended_work_reentry_coordinator.binding_resolver
        is resolver
    )
    session = manager.open_or_create(tenant_id=tenant_id, task_id="task" + "a" * 32)
    wiring = resolver.resolve_fixed_sandbox_session_wiring(
        sandbox_session_id=session.session_id,
        tenant_id=tenant_id,
        task_id="task" + "a" * 32,
    )
    assert wiring.sandbox_session.session_id == session.session_id
