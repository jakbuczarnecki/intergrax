# © Artur Czarnecki. All rights reserved.

"""EBH-4-R1-R3-B4 adversarial tenant isolation (local evidence)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.memory.memory_vector_namespace import resolve_memory_index_collection
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.contracts.agent_step import AgentStep
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.tools.providers.eval.contracts import EvalTrajectoryInput
from intergrax.runtime.codecraft.ownership import CodeCraftOwnershipError, resolve_codecraft_ownership
from intergrax.runtime.codecraft.trace import CodeCraftTraceEmitter
from intergrax.runtime.nexus.agents.acp_uaep_shim import build_step_context_from_uaep
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.execution.agent_runtime_io import RuntimeRequest
from intergrax.runtime.governance.in_memory_metrics_store import InMemoryMetricsStore
from intergrax.runtime.interactions.actor_resolution import resolve_actor_from_envelope, resolve_actor_from_task
from intergrax.runtime.nexus.agents.runtime_request_bridge import runtime_request_to_agent_run
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.plugins.bootstrap import bootstrap_runtime_plugins
from intergrax.runtime.plugins.default_plugins import default_lab_plugins
from intergrax.runtime.task.task import Task
from testing_support.builder import build_runtime_request_for_tests, canonical_task_id_for_tests
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.contracts.execution_identity import mint_run_id

pytestmark = pytest.mark.unit


class _BridgeContract:
    id = "bridge-agent"
    risk_level = AgentRiskLevel.LOW


def test_b4_a_direct_task_empty_tenant_rejected() -> None:
    with pytest.raises(ValueError, match="tenant_id must be non-empty"):
        Task(tenant_id="", user_id="u1", message="x")
    with pytest.raises(ValueError, match="tenant_id must be non-empty"):
        Task(tenant_id="   ", user_id="u1", message="x")


def test_b4_b_task_a_maps_to_actor_a() -> None:
    task = Task(tenant_id="tenant-a", user_id="u1", message="go")
    actor = resolve_actor_from_task(task)
    assert actor.tenant_id == "tenant-a"


def test_b4_b_envelope_actor_preserves_tenant() -> None:
    envelope = TaskEnvelope(tenant_id="tenant-a", user_id="u1", message="go")
    actor = resolve_actor_from_envelope(envelope)
    assert actor.tenant_id == "tenant-a"


def test_b4_c_tenantless_task_completed_skips_trace_and_metrics() -> None:
    trace_store = MagicMock()
    metrics_store = InMemoryMetricsStore()
    bus = RuntimeEventBus(record_history=False)
    bootstrap_runtime_plugins(
        default_lab_plugins(trace_store=trace_store, metrics_store=metrics_store),
        event_bus=bus,
    )
    identity = runtime_event_test_identity(
        task_id=canonical_task_id_for_tests("metrics-tenantless"),
    )
    event = RuntimeEvent(
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        tenant_id=None,
        **identity,
    )
    bus.record(event)
    trace_store.read_run.assert_not_called()
    assert len(metrics_store._data) == 0


def test_b4_c_task_completed_tenant_a_reads_trace_for_a_only() -> None:
    trace_store = MagicMock()
    trace_store.read_run.return_value = MagicMock()
    metrics_store = InMemoryMetricsStore()
    bus = RuntimeEventBus(record_history=False)
    bootstrap_runtime_plugins(
        default_lab_plugins(trace_store=trace_store, metrics_store=metrics_store),
        event_bus=bus,
    )
    identity = runtime_event_test_identity(
        task_id=canonical_task_id_for_tests("metrics-tenant-a"),
    )
    run_id = str(identity["run_id"])
    event = RuntimeEvent(
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        tenant_id="tenant-a",
        agent_id="agent-1",
        **identity,
    )
    bus.record(event)
    trace_store.read_run.assert_called_once_with(run_id, "tenant-a")


def test_b4_d_runtime_request_metadata_tenant_b_rejected() -> None:
    req = build_runtime_request_for_tests(
        seed="meta-override",
        tenant_id="tenant-a",
        metadata={"tenant_id": "tenant-b"},
    )
    with pytest.raises(ValueError, match="metadata tenant_id cannot override"):
        req.to_envelope()


def test_b4_bridge_metadata_tenant_b_rejected_without_canonical_identity() -> None:
    req = build_runtime_request_for_tests(
        seed="bridge-meta",
        tenant_id="tenant-a",
        metadata={"tenant_id": "tenant-b"},
    )
    with pytest.raises(ValueError, match="metadata tenant_id cannot override"):
        runtime_request_to_agent_run(req, contract=_BridgeContract())


def test_b4_bridge_missing_tenant_rejected() -> None:
    req = RuntimeRequest(
        agent_id="a",
        user_id="u",
        session_id="s",
        message="m",
        task_id=canonical_task_id_for_tests("bridge-missing-tenant"),
        run_id=mint_run_id(),
        tenant_id=None,
    )
    with pytest.raises(ValueError, match="tenant_id is required"):
        runtime_request_to_agent_run(req, contract=_BridgeContract())


def test_b4_memory_namespace_empty_tenant_rejected() -> None:
    with pytest.raises(ValueError, match="tenant_id or vector_index_namespace"):
        resolve_memory_index_collection(
            vector_index_namespace=None,
            tenant_id="",
            domain="ltm",
        )


def test_b4_memory_namespace_explicit_override_allowed() -> None:
    assert (
        resolve_memory_index_collection(
            vector_index_namespace="shared-ns",
            tenant_id="",
            domain="ltm",
        )
        == "shared-ns:ltm"
    )


def test_b4_delegation_actor_tenant_preserved() -> None:
    parent = ActorIdentity(
        kind=ActorKind.USER,
        actor_id="u1",
        tenant_id="tenant-a",
        permission_scopes=("read",),
    )
    assert parent.tenant_id == "tenant-a"


def test_b4_task_envelope_round_trip_tenant() -> None:
    task = Task(tenant_id="tenant-z", user_id="u1", message="hi")
    assert task.to_envelope().tenant_id == "tenant-z"
    restored = Task.from_envelope(task.to_envelope())
    assert restored.tenant_id == "tenant-z"


def test_b4_runtime_request_from_envelope_tenant_chain() -> None:
    envelope = TaskEnvelope(tenant_id="tenant-a", user_id="u1", message="go")
    task_id = canonical_task_id_for_tests("rr-chain")
    run_id = mint_run_id()
    req = RuntimeRequest.from_envelope(envelope, task_id=task_id, run_id=run_id)
    assert req.tenant_id == "tenant-a"
    assert req.to_envelope().tenant_id == "tenant-a"


def test_b4_r1_agent_step_context_blank_tenant_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentStepContext(tenant_id="", step_index=0)


def test_b4_r1_agent_step_context_missing_tenant_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentStepContext.model_validate({"step_index": 0})


def test_b4_r1_agent_step_context_tenant_a_preserved() -> None:
    ctx = AgentStepContext(tenant_id="tenant-a", step_index=0)
    assert ctx.tenant_id == "tenant-a"


def test_b4_r1_runtime_request_maps_to_agent_step_context_tenant_a() -> None:
    req = build_runtime_request_for_tests(seed="step-ctx-a", tenant_id="tenant-a")
    exec_ctx = RuntimeExecutionContext(
        task_id=req.task_id,
        run_id=req.run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        agent_id=req.agent_id,
        request=req,
        workspace_id=None,
    )
    step = AgentStep(step_index=0, step_id="s0", step_name="llm")

    class _StubAgent:
        pass

    step_ctx = build_step_context_from_uaep(_StubAgent(), step, exec_ctx)
    assert step_ctx.tenant_id == "tenant-a"


def test_b4_r1_eval_trajectory_missing_tenant_rejected() -> None:
    with pytest.raises(ValidationError):
        EvalTrajectoryInput.model_validate({"run_id": "run-1"})


def test_b4_r1_eval_trajectory_blank_tenant_rejected() -> None:
    with pytest.raises(ValidationError):
        EvalTrajectoryInput(run_id="run-1", tenant_id="  ")


def test_b4_r1_eval_trajectory_tenant_a_preserved() -> None:
    params = EvalTrajectoryInput(run_id="run-1", tenant_id="tenant-a")
    assert params.tenant_id == "tenant-a"


def test_b4_r1_user_profile_manager_missing_tenant_rejected() -> None:
    store = InMemoryUserProfileStore()
    with pytest.raises(TypeError):
        UserProfileManager(store)  # type: ignore[call-arg]


def test_b4_r1_user_profile_manager_blank_tenant_rejected() -> None:
    store = InMemoryUserProfileStore()
    with pytest.raises(ValueError, match="tenant_id must be non-empty"):
        UserProfileManager(store, tenant_id="   ")


def test_b4_r1_memory_vector_scope_same_namespace_different_tenants() -> None:
    store_a = UserProfileManager(
        InMemoryUserProfileStore(),
        tenant_id="tenant-a",
        vector_index_namespace="shared-name",
    )
    store_b = UserProfileManager(
        InMemoryUserProfileStore(),
        tenant_id="tenant-b",
        vector_index_namespace="shared-name",
    )
    scope_a = store_a._vector_scope()
    scope_b = store_b._vector_scope()
    assert scope_a.tenant_id == "tenant-a"
    assert scope_b.tenant_id == "tenant-b"
    assert scope_a.namespace == scope_b.namespace == "shared-name"
    assert scope_a != scope_b


def test_b4_r1_codecraft_ownership_omitted_caller_uses_sandbox(tmp_path) -> None:
    session = SandboxSession.create(
        tmp_path,
        tenant_id="tenant-a",
        task_id="task-a",
        allowed_operations=frozenset({"run_python"}),
    )
    ctx = ToolWiringContext(sandbox_session=session)
    ownership = resolve_codecraft_ownership(ctx)
    assert ownership.tenant_id == "tenant-a"
    assert ownership.task_id == "task-a"


def test_b4_r1_codecraft_ownership_caller_mismatch_rejected(tmp_path) -> None:
    session = SandboxSession.create(
        tmp_path,
        tenant_id="tenant-a",
        task_id="task-a",
        allowed_operations=frozenset({"run_python"}),
    )
    ctx = ToolWiringContext(sandbox_session=session)
    with pytest.raises(CodeCraftOwnershipError, match="codecraft_tenant_mismatch"):
        resolve_codecraft_ownership(ctx, caller_tenant_id="tenant-b")


def test_b4_r1_codecraft_ownership_blank_caller_rejected(tmp_path) -> None:
    session = SandboxSession.create(
        tmp_path,
        tenant_id="tenant-a",
        task_id="task-a",
        allowed_operations=frozenset({"run_python"}),
    )
    ctx = ToolWiringContext(sandbox_session=session)
    with pytest.raises(CodeCraftOwnershipError, match="codecraft_tenant_blank"):
        resolve_codecraft_ownership(ctx, caller_tenant_id="  ")


def test_b4_r1_codecraft_trace_tenant_a_on_event(tmp_path) -> None:
    emitter = CodeCraftTraceEmitter(run_id="run-a")
    evt = emitter.session_opened(
        craft_id="c1",
        mode="autonomous",
        tenant_id="tenant-a",
        task_id="task-a",
    )
    assert evt.tags["tenant_id"] == "tenant-a"


def test_b4_r1_codecraft_trace_blank_tenant_fail_closed() -> None:
    emitter = CodeCraftTraceEmitter(run_id="run-a")
    with pytest.raises(ValueError, match="tenant_id must be non-empty"):
        emitter.session_opened(
            craft_id="c1",
            mode="autonomous",
            tenant_id="",
            task_id="task-a",
        )


def test_b4_r2_step_kernel_context_missing_tenant_rejected() -> None:
    from intergrax.runtime.kernel.step_kernel import StepKernelContext

    with pytest.raises(TypeError):
        StepKernelContext(agent_id="x")  # type: ignore[call-arg]


def test_b4_r2_step_kernel_context_blank_tenant_rejected() -> None:
    from intergrax.runtime.kernel.step_kernel import StepKernelContext

    with pytest.raises(ValueError, match="tenant_id must be non-empty"):
        StepKernelContext(agent_id="x", tenant_id="   ")


def test_b4_r2_step_kernel_context_strips_tenant() -> None:
    from intergrax.runtime.kernel.step_kernel import StepKernelContext

    ctx = StepKernelContext(agent_id="x", tenant_id=" tenant-a ")
    assert ctx.tenant_id == "tenant-a"


def test_b4_r2_kernel_tenant_matches_uaep_step_context() -> None:
    from intergrax.contracts.agent_step import AgentStep
    from intergrax.runtime.kernel.step_kernel import StepKernelContext
    from intergrax.runtime.nexus.agents.uaep_step_bridge import build_uaep_step_context

    kernel_ctx = StepKernelContext(agent_id="agent-1", tenant_id="tenant-a")
    exec_ctx = RuntimeExecutionContext(
        run_id=mint_run_id(),
        task_id=canonical_task_id_for_tests("b4-r2-kernel"),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        agent_id="agent-1",
        workspace_id="ws-1",
        request=build_runtime_request_for_tests(seed="b4-r2-kernel", tenant_id="tenant-a"),
    )
    step = AgentStep(step_index=0, step_id="s0", step_name="llm")
    step_ctx = build_uaep_step_context(step, exec_ctx, kernel_ctx)
    assert step_ctx.tenant_id == kernel_ctx.tenant_id == "tenant-a"


def test_b4_r3_runtime_request_identity_metadata_substitute_rejected() -> None:
    from intergrax.runtime.nexus.agents.uaep_step_bridge import _runtime_request_identity

    req = RuntimeRequest(
        agent_id="a",
        user_id="u",
        session_id="s",
        message="m",
        task_id=canonical_task_id_for_tests("r3-meta-sub"),
        run_id=mint_run_id(),
        tenant_id=None,
        metadata={"tenant_id": "tenant-a"},
    )
    with pytest.raises(ValueError, match="tenant_id is required for RuntimeRequest"):
        _runtime_request_identity(req)


def test_b4_r3_runtime_request_identity_canonical_mismatch_rejected() -> None:
    from intergrax.runtime.nexus.agents.uaep_step_bridge import _runtime_request_identity

    req = build_runtime_request_for_tests(seed="r3-can-mis", tenant_id="tenant-a")
    req.canonical_identity = RequestIdentity(tenant_id="tenant-b", user_id="u1")
    with pytest.raises(ValueError, match="conflicts with canonical RequestIdentity"):
        _runtime_request_identity(req)


def test_b4_r3_runtime_request_identity_typed_only_derives_identity() -> None:
    from intergrax.runtime.nexus.agents.uaep_step_bridge import _runtime_request_identity

    req = build_runtime_request_for_tests(seed="r3-typed-only", tenant_id="tenant-a")
    identity = _runtime_request_identity(req)
    assert identity.tenant_id == "tenant-a"


def test_b4_r3_runtime_request_identity_canonical_agrees() -> None:
    from intergrax.runtime.nexus.agents.uaep_step_bridge import _runtime_request_identity

    req = build_runtime_request_for_tests(seed="r3-can-agree", tenant_id="tenant-a")
    req.canonical_identity = RequestIdentity(tenant_id="tenant-a", user_id="u1")
    identity = _runtime_request_identity(req)
    assert identity is req.canonical_identity
    assert identity.tenant_id == "tenant-a"


def test_b4_r3_build_kernel_session_tenant_mismatch_rejected() -> None:
    from unittest.mock import patch

    from intergrax.runtime.nexus.agents.uaep_step_bridge import build_kernel_session
    from intergrax.runtime.policy.policy_engine import PolicyEngine

    req = build_runtime_request_for_tests(seed="r3-kernel-mis", tenant_id="tenant-b")
    with (
        patch(
            "intergrax.runtime.nexus.agents.uaep_step_bridge.resolve_agentic_pre_model_scope"
        ) as scope_mock,
        pytest.raises(
            ValueError,
            match="kernel tenant_id conflicts with canonical RuntimeRequest tenant_id",
        ),
    ):
        build_kernel_session(
            agent_id="agent-1",
            run_id=req.run_id,
            task_id=req.task_id,
            tenant_id="tenant-a",
            max_steps=1,
            policy_engine=PolicyEngine(),
            request=req,
        )
    scope_mock.assert_not_called()


def test_b4_r3_build_kernel_session_tenant_match() -> None:
    from intergrax.runtime.kernel.step_kernel import StepKernelContext
    from intergrax.runtime.nexus.agents.uaep_step_bridge import build_kernel_session
    from intergrax.runtime.policy.policy_engine import PolicyEngine

    req = build_runtime_request_for_tests(seed="r3-kernel-ok", tenant_id="tenant-a")
    kernel_ctx = build_kernel_session(
        agent_id="agent-1",
        run_id=req.run_id,
        task_id=req.task_id,
        tenant_id="tenant-a",
        max_steps=1,
        policy_engine=PolicyEngine(),
        request=req,
    )
    assert isinstance(kernel_ctx, StepKernelContext)
    assert kernel_ctx.tenant_id == "tenant-a"


def test_b4_r3_acp_shim_request_step_kernel_chain() -> None:
    from intergrax.contracts.agent_contract_meta import AgentContract
    from intergrax.contracts.agent_run import AgentRunRequest
    from intergrax.runtime.kernel.step_kernel import StepKernelContext
    from intergrax.runtime.nexus.agents.acp_uaep_shim import attach_acp_catalog_exec_ctx
    from testing_support.builder import canonical_governed_execution_scope

    task_id = canonical_task_id_for_tests("r3-acp-ok")
    contract = AgentContract(
        id="bridge-agent",
        name="bridge",
        description="",
        risk_level=AgentRiskLevel.LOW,
        allowed_tools=(),
    )
    with canonical_governed_execution_scope("r3-acp-ok") as run_id:
        step_ctx = AgentStepContext(
            tenant_id="tenant-a",
            step_index=0,
            run_id=run_id,
            task_id=task_id,
        )
        kernel_ctx = StepKernelContext(agent_id="bridge-agent", tenant_id="tenant-a")
        request = AgentRunRequest(
            input="hi",
            identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        )
        attach_acp_catalog_exec_ctx(
            step_ctx,
            kernel_ctx=kernel_ctx,
            request=request,
            contract=contract,
        )
        exec_ctx = step_ctx.metadata.get("uaep_exec_ctx")
        assert isinstance(exec_ctx, RuntimeExecutionContext)
        assert exec_ctx.request is not None
        assert exec_ctx.request.tenant_id == "tenant-a"


def test_b4_r3_acp_shim_request_vs_step_mismatch_rejected() -> None:
    from intergrax.contracts.agent_contract_meta import AgentContract
    from intergrax.contracts.agent_run import AgentRunRequest
    from intergrax.runtime.kernel.step_kernel import StepKernelContext
    from intergrax.runtime.nexus.agents.acp_uaep_shim import attach_acp_catalog_exec_ctx
    from testing_support.builder import canonical_governed_execution_scope

    task_id = canonical_task_id_for_tests("r3-acp-rs")
    contract = AgentContract(
        id="bridge-agent",
        name="bridge",
        description="",
        risk_level=AgentRiskLevel.LOW,
        allowed_tools=(),
    )
    with canonical_governed_execution_scope("r3-acp-rs") as run_id:
        step_ctx = AgentStepContext(
            tenant_id="tenant-b",
            step_index=0,
            run_id=run_id,
            task_id=task_id,
        )
        kernel_ctx = StepKernelContext(agent_id="bridge-agent", tenant_id="tenant-b")
        request = AgentRunRequest(
            input="hi",
            identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        )
        with pytest.raises(
            ValueError,
            match="conflicts with AgentStepContext tenant_id",
        ):
            attach_acp_catalog_exec_ctx(
                step_ctx,
                kernel_ctx=kernel_ctx,
                request=request,
                contract=contract,
            )
        assert "uaep_exec_ctx" not in step_ctx.metadata


def test_b4_r3_acp_shim_step_vs_kernel_mismatch_rejected() -> None:
    from intergrax.contracts.agent_contract_meta import AgentContract
    from intergrax.contracts.agent_run import AgentRunRequest
    from intergrax.runtime.kernel.step_kernel import StepKernelContext
    from intergrax.runtime.nexus.agents.acp_uaep_shim import attach_acp_catalog_exec_ctx
    from testing_support.builder import canonical_governed_execution_scope

    task_id = canonical_task_id_for_tests("r3-acp-sk")
    contract = AgentContract(
        id="bridge-agent",
        name="bridge",
        description="",
        risk_level=AgentRiskLevel.LOW,
        allowed_tools=(),
    )
    with canonical_governed_execution_scope("r3-acp-sk") as run_id:
        step_ctx = AgentStepContext(
            tenant_id="tenant-a",
            step_index=0,
            run_id=run_id,
            task_id=task_id,
        )
        kernel_ctx = StepKernelContext(agent_id="bridge-agent", tenant_id="tenant-b")
        request = AgentRunRequest(
            input="hi",
            identity=RequestIdentity(tenant_id="tenant-a", user_id="u1"),
        )
        with pytest.raises(
            ValueError,
            match="conflicts with StepKernelContext tenant_id",
        ):
            attach_acp_catalog_exec_ctx(
                step_ctx,
                kernel_ctx=kernel_ctx,
                request=request,
                contract=contract,
            )
        assert "uaep_exec_ctx" not in step_ctx.metadata


def test_b4_r3_acp_shim_missing_request_tenant_rejected() -> None:
    from intergrax.contracts.agent_contract_meta import AgentContract
    from intergrax.contracts.agent_run import AgentRunRequest
    from intergrax.runtime.kernel.step_kernel import StepKernelContext
    from intergrax.runtime.nexus.agents.acp_uaep_shim import attach_acp_catalog_exec_ctx
    from testing_support.builder import canonical_governed_execution_scope

    task_id = canonical_task_id_for_tests("r3-acp-miss")
    contract = AgentContract(
        id="bridge-agent",
        name="bridge",
        description="",
        risk_level=AgentRiskLevel.LOW,
        allowed_tools=(),
    )
    with canonical_governed_execution_scope("r3-acp-miss") as run_id:
        step_ctx = AgentStepContext(
            tenant_id="tenant-a",
            step_index=0,
            run_id=run_id,
            task_id=task_id,
        )
        kernel_ctx = StepKernelContext(agent_id="bridge-agent", tenant_id="tenant-a")
        request = AgentRunRequest(
            input="hi",
            identity=RequestIdentity.model_construct(tenant_id=None, user_id="u1"),
        )
        with pytest.raises(ValueError, match="tenant_id is required for ACP UAEP shim"):
            attach_acp_catalog_exec_ctx(
                step_ctx,
                kernel_ctx=kernel_ctx,
                request=request,
                contract=contract,
            )


def test_b4_r4_resolve_request_scope_typed_a_metadata_a() -> None:
    from intergrax.agents.authoring.runtime_tool_helpers import resolve_request_scope
    from testing_support.builder import build_runtime_execution_context_for_tests, build_runtime_request_for_tests

    seed = "r4-scope-aa"
    request = build_runtime_request_for_tests(
        seed=seed,
        tenant_id="tenant-a",
        metadata={"tenant_id": "tenant-a"},
    )
    exec_ctx = build_runtime_execution_context_for_tests(seed=seed, request=request, tenant_id="tenant-a")
    scope = resolve_request_scope(exec_ctx)
    assert scope["tenant_id"] == "tenant-a"


def test_b4_r4_resolve_request_scope_typed_a_metadata_b_rejected() -> None:
    from intergrax.agents.authoring.runtime_tool_helpers import RequestScopeError, resolve_request_scope
    from testing_support.builder import build_runtime_execution_context_for_tests, build_runtime_request_for_tests

    seed = "r4-scope-ab"
    request = build_runtime_request_for_tests(
        seed=seed,
        tenant_id="tenant-a",
        metadata={"tenant_id": "tenant-b"},
    )
    exec_ctx = build_runtime_execution_context_for_tests(seed=seed, request=request, tenant_id="tenant-a")
    with pytest.raises(RequestScopeError, match="cannot override"):
        resolve_request_scope(exec_ctx)


def test_b4_r4_resolve_request_scope_metadata_only_rejected() -> None:
    from intergrax.agents.authoring.runtime_tool_helpers import RequestScopeError, resolve_request_scope
    from testing_support.builder import build_runtime_execution_context_for_tests, build_runtime_request_for_tests

    seed = "r4-scope-meta"
    base = build_runtime_request_for_tests(
        seed=seed,
        tenant_id="tenant-a",
        metadata={"tenant_id": "tenant-a"},
    )
    request = RuntimeRequest(
        agent_id=base.agent_id,
        user_id=base.user_id,
        session_id=base.session_id,
        message=base.message,
        task_id=base.task_id,
        run_id=base.run_id,
        workspace_id=base.workspace_id,
        tenant_id=None,
        metadata={"tenant_id": "tenant-a"},
        canonical_identity=base.canonical_identity,
    )
    exec_ctx = build_runtime_execution_context_for_tests(seed=seed, request=request, tenant_id="tenant-a")
    with pytest.raises(RequestScopeError):
        resolve_request_scope(exec_ctx)


@pytest.mark.asyncio
async def test_b4_r4_indexer_metadata_attack_zero_tool_calls(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    from local_indexer.steps import index_job
    from local_indexer.steps.index_job import run_index_job
    from testing_support.builder import build_runtime_execution_context_for_tests, build_runtime_request_for_tests

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    doc = allowed / "a.txt"
    doc.write_text("x", encoding="utf-8")

    seed = "r4-indexer-attack"
    request = build_runtime_request_for_tests(
        seed=seed,
        agent_id="local_indexer",
        tenant_id="tenant-a",
        metadata={
            "tenant_id": "tenant-b",
            "source_paths": [str(doc)],
            "collection_id": "c1",
        },
    )
    invoke_mock = AsyncMock()
    monkeypatch.setattr(index_job, "invoke_catalog_tool", invoke_mock)
    exec_ctx = build_runtime_execution_context_for_tests(
        seed=seed,
        agent_id="local_indexer",
        request=request,
        tenant_id="tenant-a",
    )
    step_ctx = AgentStepContext(
        tenant_id="tenant-a",
        run_id=str(exec_ctx.run_id),
        agent_id="local_indexer",
        contract_id="local_indexer",
        metadata={"uaep_exec_ctx": exec_ctx},
    )
    monkeypatch.setenv("INTERGRAX_READ_ALLOWLIST_ROOTS", str(allowed.resolve()))

    result = await run_index_job(step_ctx)
    invoke_mock.assert_not_called()
    assert result["ingest_summary"]["used"] is False


@pytest.mark.asyncio
async def test_b4_r4_search_scope_missing_metadata_tenant_zero_retrieve(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    from local_search.steps import search_job
    from local_search.steps.search_job import run_search_job
    from testing_support.builder import build_runtime_execution_context_for_tests, build_runtime_request_for_tests

    seed = "r4-search-meta-only"
    base = build_runtime_request_for_tests(
        seed=seed,
        agent_id="local_search",
        tenant_id="tenant-a",
        message="find",
        metadata={"tenant_id": "tenant-a", "query": "hello"},
    )
    request = RuntimeRequest(
        agent_id=base.agent_id,
        user_id=base.user_id,
        session_id=base.session_id,
        message=base.message,
        task_id=base.task_id,
        run_id=base.run_id,
        workspace_id=base.workspace_id,
        tenant_id=None,
        metadata={"tenant_id": "tenant-a", "query": "hello"},
        canonical_identity=base.canonical_identity,
    )
    invoke_mock = AsyncMock()
    monkeypatch.setattr(search_job, "invoke_catalog_tool", invoke_mock)
    exec_ctx = build_runtime_execution_context_for_tests(
        seed=seed,
        agent_id="local_search",
        request=request,
        tenant_id="tenant-a",
    )
    step_ctx = AgentStepContext(
        tenant_id="tenant-a",
        run_id=str(exec_ctx.run_id),
        agent_id="local_search",
        contract_id="local_search",
        message="hello",
        metadata={"uaep_exec_ctx": exec_ctx},
    )
    result = await run_search_job(step_ctx)
    invoke_mock.assert_not_called()
    assert result["search_summary"]["reason"] == "tenant_scope_invalid"


def test_b4_r4_domain_agents_do_not_import_nexus_runtime_tool_helpers() -> None:
    import ast
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    agents_root = repo / "agents"
    forbidden = "intergrax.runtime.nexus.agents.runtime_tool_helpers"
    violations: list[str] = []
    for path in agents_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if forbidden in source:
            violations.append(path.relative_to(repo).as_posix())
    assert violations == [], violations


def test_b4_r4_resolve_request_scope_single_semantic_implementation() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    authoring = repo / "intergrax" / "agents" / "authoring" / "runtime_tool_helpers.py"
    nexus = repo / "intergrax" / "runtime" / "nexus" / "agents" / "runtime_tool_helpers.py"
    assert "def resolve_request_scope" in authoring.read_text(encoding="utf-8")
    assert "def resolve_request_scope" not in nexus.read_text(encoding="utf-8")


def test_b4_r4_production_step_modules_importable() -> None:
    import importlib

    importlib.import_module("agents.local_indexer.steps.index_job")
    importlib.import_module("agents.local_search.steps.search_job")
