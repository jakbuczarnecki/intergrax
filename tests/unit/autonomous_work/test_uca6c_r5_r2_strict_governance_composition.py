# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R5-R2 — strict production governance composition for qualified CodeCraft."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications._shared.uca6c_codecraft_qualified_execution_composition import (
    Uca6cCodecraftQualifiedExecutionCompositionError,
    bootstrap_uca6c_code_exec_catalog_tools,
    build_production_codecraft_qualified_capability_execution_handler,
)
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionOutcome,
    CodeCraftBoundCapabilityExecutionRequest,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    PolicyCompositionResult,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    derive_qualified_capability_governance_step_id,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.codecraft.ephemeral_registry import EphemeralToolRegistryStore
from intergrax.runtime.codecraft.ownership import CodeCraftSessionOwnership
from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager
from intergrax.runtime.codecraft.wiring_bound_capability_execution import (
    WiringCodeCraftBoundCapabilityExecution,
)
from intergrax.runtime.agent_governance.errors import (
    CapabilityNotGrantedError,
    ToolGovernanceApprovalRequiredError,
    ToolGovernanceDeniedError,
)
from intergrax.runtime.agent_governance.request_builder import (
    governance_capability_for_contract,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
    NexusExecutionBoundCatalogToolInvoker,
)
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
from intergrax.tools.registry import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.codecraft_execution_environment import (
    codecraft_sandbox_execution_profile,
)
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TENANT,
    _TASK_ID,
)
from tests.unit.autonomous_work.uca6c_r5_r2_strict_fixtures import (
    build_sandbox_session,
    uca6c_attach_catalog_hitl_grant,
    uca6c_high_risk_tool_approval_evidence,
    uca6c_high_risk_tool_approval_grant,
    uca6c_strict_echo_only_worker_manifest,
    uca6c_strict_sandbox_env_profile,
    uca6c_strict_r6_durable_wiring,
    uca6c_strict_worker_manifest,
    uca6c_strict_worker_registry,
)
from tests.unit.runtime.nexus.tools.test_gr10_r8_orchestration_inner_guard import (
    _RecordingGuard,
)

pytestmark = pytest.mark.unit


def _strict_r6_kwargs() -> dict[str, object]:
    return uca6c_strict_r6_durable_wiring()


_COMPOSITION_PATH = Path(
    "intergrax/applications/_shared/uca6c_codecraft_qualified_execution_composition.py",
)


class _RecordingMsePort:
    def __init__(self, *, allow: bool) -> None:
        self.calls = 0
        self._allow = allow

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.orchestration.tool_invocation",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        self.calls += 1
        _ = request, source_agent_id, source_step_id
        action = PolicyAction.ALLOW if self._allow else PolicyAction.DENY
        decision = PolicyDecision(
            action=action,
            reason="uca6c-r5-r2-test-mse",
            policy_rule_id="test.uca6c.mse",
        )
        enforcement_result = CollaborativeWorkEnforcementResult(
            operation_id=request.operation_id,
            authority_scope=request.resource_scope,
            composition=PolicyCompositionResult(
                decision=decision,
                collaborative_authority=decision,
            ),
        )
        return MeaningfulSideEffectAuthorizationResult(
            permitted=self._allow,
            decision=decision,
            enforcement_result=enforcement_result,
            requires_governed_continuation=False,
            governed_continuation_request=None,
        )


def _codecraft_context(tmp_path: Path, craft_id: str) -> ToolWiringContext:
    sandbox = build_sandbox_session(
        tmp_path,
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
    )
    sessions = CodeCraftSessionManager()
    ownership = CodeCraftSessionOwnership(tenant_id=_TENANT, task_id=str(_TASK_ID))
    session = sessions.open(
        goal="exec",
        ownership=ownership,
        mode="autonomous",
        craft_id=craft_id,
    )
    sessions.save_owned(
        session.model_copy(update={"code": "print('uca6c-r5-r2-ok')"}),
        ownership,
    )
    registry = EphemeralToolRegistryStore()
    registry.for_craft(craft_id).register("ephemeral.r5r2.exec")
    return ToolWiringContext(
        sandbox_session=sandbox,
        extras={
            "codecraft_session_manager": sessions,
            "codecraft_ephemeral_registry": registry,
            "codecraft_profile": CodeCraftProfile(
                mode="autonomous",
                require_tests=False,
            ),
            "effective_environment_profile": codecraft_sandbox_execution_profile(),
        },
    )


def _strict_tool_wiring(ctx: ToolWiringContext) -> ApplicationToolWiring:
    return ApplicationToolWiring(
        profile=ToolProfile(enabled_bundles=["sandbox"]),
        wiring_context=ctx,
        registry=ToolRegistry(),
    )


def test_strict_production_tests_have_no_private_governance_mutation() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "_agent_runtime_governance":
            raise AssertionError(
                "strict production tests must not access _agent_runtime_governance",
            )
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "_agent_runtime_governance"
                ):
                    raise AssertionError(
                        "strict production tests must not mutate _agent_runtime_governance",
                    )
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "AllowAllPolicyProvider":
                    raise AssertionError(
                        "strict production E2E must not import AllowAllPolicyProvider",
                    )
        if isinstance(node, ast.Name) and node.id == "allow_all_agent_governance":
            raise AssertionError(
                "strict production E2E must not use allow_all_agent_governance",
            )
        if isinstance(node, ast.Attribute) and node.attr == "declarative_hitl_grant":
            raise AssertionError(
                "strict production tests must not mutate catalog.binding declarative_hitl_grant",
            )


def test_code_exec_governance_capability_matches_manifest_grant() -> None:
    from intergrax.tools.providers.sandbox.bundle import sandbox_exec_contract

    contract = sandbox_exec_contract()
    assert governance_capability_for_contract(contract) == "sandbox"
    manifest = uca6c_strict_worker_manifest()
    binding = manifest.enabled_agents()[0]
    assert "sandbox" in binding.capabilities


def test_composition_module_has_no_allow_mint() -> None:
    source = _COMPOSITION_PATH.read_text(encoding="utf-8")
    assert "PolicyAction.ALLOW" not in source
    assert "AllowAll" not in source


def test_strict_missing_mse_fails_at_composition() -> None:
    manifest = uca6c_strict_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    tool_wiring = _strict_tool_wiring(ToolWiringContext())
    with pytest.raises(
        Uca6cCodecraftQualifiedExecutionCompositionError,
        match="meaningful_side_effect_authorization",
    ):
        build_production_codecraft_qualified_capability_execution_handler(
            tool_wiring,
            uca6c_strict_sandbox_env_profile(),
            caller_agent_id="worker-uca6c-qualified",
            tenant_id=_TENANT,
            manifest=manifest,
            agent_registry=registry,
            meaningful_side_effect_authorization=None,
            **_strict_r6_kwargs(),
        )


def test_strict_missing_manifest_fails_closed() -> None:
    tool_wiring = _strict_tool_wiring(ToolWiringContext())
    mse: MeaningfulSideEffectAuthorizationPort = _RecordingMsePort(allow=True)
    with pytest.raises(
        Uca6cCodecraftQualifiedExecutionCompositionError,
        match="manifest",
    ):
        build_production_codecraft_qualified_capability_execution_handler(
            tool_wiring,
            uca6c_strict_sandbox_env_profile(),
            caller_agent_id="worker-uca6c-qualified",
            tenant_id=_TENANT,
            manifest=None,
            agent_registry=uca6c_strict_worker_registry(uca6c_strict_worker_manifest()),
            meaningful_side_effect_authorization=mse,
            **_strict_r6_kwargs(),
        )


def test_strict_missing_agent_registry_fails_closed() -> None:
    tool_wiring = _strict_tool_wiring(ToolWiringContext())
    mse: MeaningfulSideEffectAuthorizationPort = _RecordingMsePort(allow=True)
    with pytest.raises(
        Uca6cCodecraftQualifiedExecutionCompositionError,
        match="agent_registry",
    ):
        build_production_codecraft_qualified_capability_execution_handler(
            tool_wiring,
            uca6c_strict_sandbox_env_profile(),
            caller_agent_id="worker-uca6c-qualified",
            tenant_id=_TENANT,
            manifest=uca6c_strict_worker_manifest(),
            agent_registry=None,
            meaningful_side_effect_authorization=mse,
            **_strict_r6_kwargs(),
        )


def test_high_level_builder_propagates_mse_and_inner_guard() -> None:
    manifest = uca6c_strict_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    tool_wiring = _strict_tool_wiring(ToolWiringContext())
    mse: MeaningfulSideEffectAuthorizationPort = _RecordingMsePort(allow=True)
    guard = _RecordingGuard(allow=True)
    handler = build_production_codecraft_qualified_capability_execution_handler(
        tool_wiring,
        uca6c_strict_sandbox_env_profile(),
        caller_agent_id="worker-uca6c-qualified",
        tenant_id=_TENANT,
        manifest=manifest,
        agent_registry=registry,
        meaningful_side_effect_authorization=mse,
        canonical_inner_execution_guard=guard,
        **_strict_r6_kwargs(),
    )
    port = handler._execution_port
    assert isinstance(port, WiringCodeCraftBoundCapabilityExecution)
    catalog = port._catalog_tool_invoker
    assert isinstance(catalog, NexusExecutionBoundCatalogToolInvoker)
    assert catalog.production_mode is True
    runtime_invoker = catalog.tool_invoker
    assert runtime_invoker._meaningful_side_effect_authorization is mse  # noqa: SLF001
    assert runtime_invoker._inner_execution_guard is guard  # noqa: SLF001


def test_bootstrap_registers_code_exec_once_and_repeated_composition_is_stable() -> (
    None
):
    tool_wiring = _strict_tool_wiring(ToolWiringContext())
    assert not tool_wiring.registry.has(CODE_EXEC_TOOL_ID)
    bootstrap_uca6c_code_exec_catalog_tools(tool_wiring)
    assert tool_wiring.registry.has(CODE_EXEC_TOOL_ID)
    ids_after_first = frozenset(
        entry.contract.tool_id for entry in tool_wiring.registry.list()
    )
    bootstrap_uca6c_code_exec_catalog_tools(tool_wiring)
    ids_after_second = frozenset(
        entry.contract.tool_id for entry in tool_wiring.registry.list()
    )
    assert ids_after_second == ids_after_first


def test_strict_production_success_via_high_level_builder(tmp_path: Path) -> None:
    craft_id = "craft-r5r2-strict-ok"
    ctx = _codecraft_context(tmp_path, craft_id)
    manifest = uca6c_strict_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    caller_agent_id = "worker-uca6c-qualified"
    tool_wiring = _strict_tool_wiring(ctx)
    mse: MeaningfulSideEffectAuthorizationPort = _RecordingMsePort(allow=True)
    inner_guard = _RecordingGuard(allow=True)
    handler = build_production_codecraft_qualified_capability_execution_handler(
        tool_wiring,
        uca6c_strict_sandbox_env_profile(),
        caller_agent_id=caller_agent_id,
        tenant_id=_TENANT,
        manifest=manifest,
        agent_registry=registry,
        meaningful_side_effect_authorization=mse,
        canonical_inner_execution_guard=inner_guard,
        **_strict_r6_kwargs(),
    )
    port = handler._execution_port
    assert isinstance(port, WiringCodeCraftBoundCapabilityExecution)
    catalog = port._catalog_tool_invoker
    assert isinstance(catalog, NexusExecutionBoundCatalogToolInvoker)
    assert catalog.caller_agent_id == caller_agent_id
    run_id = mint_run_id()
    execution_id = mint_execution_id()
    execution_request_id = "qualified-capability-execution:uca6c-r5r2-direct:binding"
    step_id = derive_qualified_capability_governance_step_id(execution_request_id)
    uca6c_attach_catalog_hitl_grant(
        catalog,
        uca6c_high_risk_tool_approval_grant(
            tenant_id=_TENANT,
            task_id=str(_TASK_ID),
            run_id=str(run_id),
            step_id=step_id,
            agent_id=caller_agent_id,
        ),
    )
    id_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id="workspace-uca6c",
            principal_id="principal-uca6c",
        ),
    )
    try:
        result = port.execute(
            CodeCraftBoundCapabilityExecutionRequest(
                craft_id=craft_id,
                tenant_id=_TENANT,
                task_id=_TASK_ID,
                run_id=None,
                execution_id=execution_id,
                execution_request_id=execution_request_id,
            ),
        )
    finally:
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(id_token)
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.SUCCEEDED
    assert inner_guard.calls >= 1
    assert mse.calls == 1
    assert "tool_invocation_start" in catalog.last_invocation_trace_steps
    assert "tool_invocation_end" in catalog.last_invocation_trace_steps


def test_strict_mse_deny_blocks_before_success(tmp_path: Path) -> None:
    craft_id = "craft-r5r2-strict-deny"
    ctx = _codecraft_context(tmp_path, craft_id)
    manifest = uca6c_strict_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    caller_agent_id = "worker-uca6c-qualified"
    tool_wiring = _strict_tool_wiring(ctx)
    mse: MeaningfulSideEffectAuthorizationPort = _RecordingMsePort(allow=False)
    inner_guard = _RecordingGuard(allow=True)
    handler = build_production_codecraft_qualified_capability_execution_handler(
        tool_wiring,
        uca6c_strict_sandbox_env_profile(),
        caller_agent_id=caller_agent_id,
        tenant_id=_TENANT,
        manifest=manifest,
        agent_registry=registry,
        meaningful_side_effect_authorization=mse,
        canonical_inner_execution_guard=inner_guard,
        **_strict_r6_kwargs(),
    )
    port = handler._execution_port
    assert isinstance(port, WiringCodeCraftBoundCapabilityExecution)
    catalog = port._catalog_tool_invoker
    assert isinstance(catalog, NexusExecutionBoundCatalogToolInvoker)
    run_id = mint_run_id()
    execution_id = mint_execution_id()
    execution_request_id = "qualified-capability-execution:uca6c-r5r2-direct:binding"
    step_id = derive_qualified_capability_governance_step_id(execution_request_id)
    uca6c_attach_catalog_hitl_grant(
        catalog,
        uca6c_high_risk_tool_approval_grant(
            tenant_id=_TENANT,
            task_id=str(_TASK_ID),
            run_id=str(run_id),
            step_id=step_id,
            agent_id=caller_agent_id,
        ),
    )
    id_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id="workspace-uca6c",
            principal_id="principal-uca6c",
        ),
    )
    try:
        with pytest.raises(ToolGovernanceDeniedError):
            port.execute(
                CodeCraftBoundCapabilityExecutionRequest(
                    craft_id=craft_id,
                    tenant_id=_TENANT,
                    task_id=_TASK_ID,
                    run_id=None,
                    execution_id=execution_id,
                    execution_request_id=execution_request_id,
                ),
            )
    finally:
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(id_token)
    assert mse.calls == 1
    assert "tool_invocation_end" not in catalog.last_invocation_trace_steps


def test_strict_agent_governance_deny_blocks_before_mse(tmp_path: Path) -> None:
    craft_id = "craft-r5r2-gov-deny"
    ctx = _codecraft_context(tmp_path, craft_id)
    manifest = uca6c_strict_echo_only_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    caller_agent_id = "worker-uca6c-echo-only"
    tool_wiring = _strict_tool_wiring(ctx)
    mse: MeaningfulSideEffectAuthorizationPort = _RecordingMsePort(allow=True)
    inner_guard = _RecordingGuard(allow=True)
    handler = build_production_codecraft_qualified_capability_execution_handler(
        tool_wiring,
        uca6c_strict_sandbox_env_profile(),
        caller_agent_id=caller_agent_id,
        tenant_id=_TENANT,
        manifest=manifest,
        agent_registry=registry,
        meaningful_side_effect_authorization=mse,
        canonical_inner_execution_guard=inner_guard,
        **_strict_r6_kwargs(),
    )
    port = handler._execution_port
    assert isinstance(port, WiringCodeCraftBoundCapabilityExecution)
    run_id = mint_run_id()
    execution_request_id = "qualified-capability-execution:uca6c-r5r2-direct:binding"
    execution_id = mint_execution_id()
    id_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id="workspace-uca6c",
            principal_id="principal-uca6c",
        ),
    )
    try:
        with pytest.raises(CapabilityNotGrantedError):
            port.execute(
                CodeCraftBoundCapabilityExecutionRequest(
                    craft_id=craft_id,
                    tenant_id=_TENANT,
                    task_id=_TASK_ID,
                    run_id=None,
                    execution_id=execution_id,
                    execution_request_id=execution_request_id,
                ),
            )
    finally:
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(id_token)
    assert mse.calls == 0
    assert inner_guard.calls >= 1


def test_strict_high_risk_without_approval_evidence_requires_governance_approval(
    tmp_path: Path,
) -> None:
    craft_id = "craft-r5r2-approval-required"
    ctx = _codecraft_context(tmp_path, craft_id)
    manifest = uca6c_strict_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    tool_wiring = _strict_tool_wiring(ctx)
    mse: MeaningfulSideEffectAuthorizationPort = _RecordingMsePort(allow=True)
    handler = build_production_codecraft_qualified_capability_execution_handler(
        tool_wiring,
        uca6c_strict_sandbox_env_profile(),
        caller_agent_id="worker-uca6c-qualified",
        tenant_id=_TENANT,
        manifest=manifest,
        agent_registry=registry,
        meaningful_side_effect_authorization=mse,
        canonical_inner_execution_guard=_RecordingGuard(allow=True),
        **_strict_r6_kwargs(),
    )
    port = handler._execution_port
    assert isinstance(port, WiringCodeCraftBoundCapabilityExecution)
    run_id = mint_run_id()
    execution_request_id = "qualified-capability-execution:uca6c-r5r2-direct:binding"
    execution_id = mint_execution_id()
    id_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id="workspace-uca6c",
            principal_id="principal-uca6c",
        ),
    )
    try:
        with pytest.raises(ToolGovernanceApprovalRequiredError):
            port.execute(
                CodeCraftBoundCapabilityExecutionRequest(
                    craft_id=craft_id,
                    tenant_id=_TENANT,
                    task_id=_TASK_ID,
                    run_id=None,
                    execution_id=execution_id,
                    execution_request_id=execution_request_id,
                ),
            )
    finally:
        reset_active_execution_governance_identity(gov_token)
        reset_active_execution_identity(id_token)
    assert mse.calls == 0


def test_non_strict_lab_regression_still_composes_without_mse(tmp_path: Path) -> None:
    from tests.unit.autonomous_work.uca6c_r5_tool_runtime_fixtures import (
        sandbox_env_profile,
    )

    craft_id = "craft-r5r2-lab"
    ctx = _codecraft_context(tmp_path, craft_id)
    tool_wiring = _strict_tool_wiring(ctx)
    handler = build_production_codecraft_qualified_capability_execution_handler(
        tool_wiring,
        sandbox_env_profile(),
        caller_agent_id="worker-r5-e2e",
        tenant_id=_TENANT,
    )
    port = handler._execution_port
    assert isinstance(port, WiringCodeCraftBoundCapabilityExecution)
    catalog = port._catalog_tool_invoker
    assert isinstance(catalog, NexusExecutionBoundCatalogToolInvoker)
    assert catalog.production_mode is False


def test_strict_constructibility_does_not_use_runtime_state_test_builder() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "build_runtime_state_for_tests" not in (node.module or "")
        if isinstance(node, ast.Name) and node.id == "build_runtime_state_for_tests":
            raise AssertionError(
                "strict tests must not use build_runtime_state_for_tests"
            )
