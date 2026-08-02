# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-5 Nexus UCL orchestration tests."""

from __future__ import annotations

import ast
import asyncio
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
)
from intergrax.context.formatter import DefaultContextFormatter
from intergrax.context.planning import (
    ContextBudgetClass,
    ContextPlan,
    ContextSourceBudgetAllocation,
    ContextSourceGroup,
)
from intergrax.context.planner import ContextPlanner
from intergrax.context.session_history import build_session_history_snapshot
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactLookupKey,
    ContextOptimizationDecision,
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    EphemeralArtifactPersistencePolicy,
    ModelCallExecutionScope,
    OptimizationArtifactType,
)
from intergrax.runtime.context_lifecycle.in_memory_repository import InMemoryOptimizationArtifactRepository
from intergrax.runtime.nexus.context.ucl_orchestration import (
    NexusUCLExecutionError,
    NexusUCLExecutionReason,
    NexusUCLRuntimeDependencies,
    resolve_ucl_context_plan,
)
from intergrax.runtime.token_optimization.message_sequence_artifact import (
    MessageSequenceArtifactExecutor,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[5]
STRATEGY_ID = "message_sequence_summarization.v1"
STRATEGY_VERSION = "1.0.0"


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _optimization_policy(
    *,
    persistence: EphemeralArtifactPersistencePolicy = (
        EphemeralArtifactPersistencePolicy.PERSIST_REUSABLE
    ),
) -> ContextOptimizationPolicy:
    return ContextOptimizationPolicy(
        policy_version="policy.v1",
        validation_contract_version="validation.v1",
        enabled=True,
        mode=ContextOptimizationMode.EPHEMERAL_ASSEMBLY,
        allow_lossy=True,
        allow_llm_summarization=True,
        allow_artifact_reuse=True,
        allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
        allowed_strategy_ids=(STRATEGY_ID,),
        ephemeral_artifact_persistence=persistence,
    )


def _request(*, run_id: str = "run-1") -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-1",
        run_id=run_id,
        task_id="task-1",
        tenant_id="tenant-1",
        assembly_scope="acp_step",
        objective="test",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=200),
        assembly_options=TaskContextAssemblyOptions(),
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
    )


def _plan_fixture(
    *,
    history_contents: Sequence[str],
    base_messages: Sequence[ChatMessage] | None = None,
    resolved_budget: int = 40,
    optimization_policy: ContextOptimizationPolicy | None = None,
) -> tuple[Any, ...]:
    base_messages = list(base_messages or [ChatMessage(role="user", content="current", entry_id="current")])
    history_messages = [
        ChatMessage(role="user", content=content, entry_id=f"m{index}")
        for index, content in enumerate(history_contents)
    ]
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev-1",
        messages=history_messages,
    )
    fragments = [
        ContextFragment(
            fragment_id=f"frag-{index}",
            source=ContextFragmentSource.SESSION_HISTORY,
            source_id="session",
            content=content,
            token_estimate=_count_tokens(content),
            relevance_score=0.5,
            freshness_score=0.5,
            confidence_score=0.5,
            mandatory=False,
            metadata={"message_id": f"m{index}"},
        )
        for index, content in enumerate(history_contents)
    ]
    formatter = DefaultContextFormatter()
    request = _request()
    fragment_messages = formatter.format(fragments, request)
    messages_for_compile = [*base_messages]
    insert_at = len(messages_for_compile) - 1
    messages_for_compile[insert_at:insert_at] = list(fragment_messages)
    planner = ContextPlanner(count_tokens=_count_tokens)
    context_plan = planner.plan(
        request,
        messages_for_compile=messages_for_compile,
        fragment_messages=fragment_messages,
        ranked_fragments=fragments,
        session_history=snapshot,
        resolved_global_budget_tokens=resolved_budget,
        optimization_policy=optimization_policy,
    )
    return (
        request,
        context_plan,
        snapshot,
        messages_for_compile,
        fragment_messages,
        fragments,
        optimization_policy,
    )


def _runtime(
    *,
    repository: InMemoryOptimizationArtifactRepository | None = None,
    invoke_model: Callable[..., LLMAdapterResponse] | None = None,
    wait_timeout_seconds: float = 0.25,
    artifact_ids: list[str] | None = None,
) -> tuple[NexusUCLRuntimeDependencies, list[str], list[int]]:
    repo = repository or InMemoryOptimizationArtifactRepository()
    model_calls: list[int] = [0]
    ids = artifact_ids if artifact_ids is not None else []
    id_iter = iter(range(1, 1000))

    def _invoke_model(call: Any) -> LLMAdapterResponse:
        model_calls[0] += 1
        if invoke_model is not None:
            return invoke_model(call)
        return LLMAdapterResponse(content="condensed session summary")

    executor = MessageSequenceArtifactExecutor(
        preflight=lambda _call: None,
        invoke_model=_invoke_model,
        count_tokens=_count_tokens,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        operation_id_factory=lambda: "internal-op",
        receipt_id_factory=lambda: "receipt-1",
    )

    def _artifact_id_factory() -> str:
        if ids:
            return ids.pop(0)
        return f"artifact-{next(id_iter)}"

    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=executor,
        strategy_versions={STRATEGY_ID: STRATEGY_VERSION},
        artifact_id_factory=_artifact_id_factory,
        wait_timeout_seconds=wait_timeout_seconds,
    )
    return runtime, model_calls, [0]


def _resolve_kwargs(
    *,
    request: ContextAssemblyRequest,
    context_plan: Any,
    snapshot: Any,
    messages_for_compile: Sequence[ChatMessage],
    fragment_messages: Sequence[ChatMessage],
    fragments: Sequence[ContextFragment],
    optimization_policy: ContextOptimizationPolicy | None,
    runtime: NexusUCLRuntimeDependencies | None,
) -> dict[str, Any]:
    return {
        "request": request,
        "context_plan": context_plan,
        "optimization_policy": optimization_policy,
        "session_history": snapshot,
        "messages_for_compile": messages_for_compile,
        "fragment_messages": fragment_messages,
        "ranked_fragments": fragments,
        "runtime": runtime,
        "count_tokens": _count_tokens,
    }


@pytest.mark.asyncio
async def test_no_op_without_runtime() -> None:
    (
        request,
        context_plan,
        snapshot,
        messages_for_compile,
        fragment_messages,
        fragments,
        _policy,
    ) = _plan_fixture(history_contents=["short"], resolved_budget=500)
    resolution = await resolve_ucl_context_plan(
        **_resolve_kwargs(
            request=request,
            context_plan=context_plan,
            snapshot=snapshot,
            messages_for_compile=messages_for_compile,
            fragment_messages=fragment_messages,
            fragments=fragments,
            optimization_policy=None,
            runtime=None,
        )
    )
    assert resolution.decision is ContextOptimizationDecision.NO_OP
    assert resolution.messages == tuple(messages_for_compile)


@pytest.mark.asyncio
async def test_select_only_without_runtime() -> None:
    drop_group = ContextSourceGroup(
        group_id="group-drop",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("m0",),
        source_content_hash="hash-drop",
        token_estimate=80,
        droppable=True,
        protected=False,
    )
    keep_group = ContextSourceGroup(
        group_id="group-keep",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("current",),
        source_content_hash="hash-keep",
        token_estimate=10,
        required=True,
        protected=True,
    )
    context_plan = ContextPlan(
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
        budget_class=ContextBudgetClass.PRIMARY_MODEL_INPUT,
        resolved_global_budget_tokens=50,
        estimated_total_tokens=90,
        source_groups=(drop_group, keep_group),
        source_allocations=(
            ContextSourceBudgetAllocation(
                source=ContextFragmentSource.SESSION_HISTORY,
                allocated_tokens=10,
                selected_group_ids=("group-keep",),
                excluded_group_ids=("group-drop",),
            ),
        ),
        selected_group_ids=("group-keep",),
        excluded_group_ids=("group-drop",),
        required_group_ids=("group-keep",),
        protected_group_ids=("group-keep",),
        compressible_group_ids=(),
        droppable_group_ids=("group-drop",),
        trim_safe_group_ids=(),
        optimization_required=False,
        artifact_requirement=None,
        final_validation_requirements=("respect_resolved_global_budget",),
    )
    fragments = (
        ContextFragment(
            fragment_id="frag-drop",
            source=ContextFragmentSource.SESSION_HISTORY,
            source_id="session",
            content="droppable history",
            token_estimate=80,
            relevance_score=0.5,
            freshness_score=0.5,
            confidence_score=0.5,
            mandatory=False,
            metadata={"message_id": "m0"},
        ),
    )
    formatter = DefaultContextFormatter()
    request = _request()
    fragment_messages = formatter.format(fragments, request)
    messages_for_compile = [
        fragment_messages[0],
        ChatMessage(role="user", content="current", entry_id="current"),
    ]
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev-1",
        messages=[ChatMessage(role="user", content="droppable history", entry_id="m0")],
    )
    resolution = await resolve_ucl_context_plan(
        **_resolve_kwargs(
            request=request,
            context_plan=context_plan,
            snapshot=snapshot,
            messages_for_compile=messages_for_compile,
            fragment_messages=fragment_messages,
            fragments=fragments,
            optimization_policy=None,
            runtime=None,
        )
    )
    assert resolution.decision is ContextOptimizationDecision.SELECT_ONLY
    assert resolution.fragments_excluded
    assert len(resolution.messages) == 1


@pytest.mark.asyncio
async def test_lookup_hit_reuses_without_executor() -> None:
    long_history = ["history block " * 30]
    (
        request,
        context_plan,
        snapshot,
        messages_for_compile,
        fragment_messages,
        fragments,
        policy,
    ) = _plan_fixture(
        history_contents=long_history,
        resolved_budget=20,
        optimization_policy=_optimization_policy(),
    )
    runtime, model_calls, _ = _runtime()
    first = await resolve_ucl_context_plan(
        **_resolve_kwargs(
            request=request,
            context_plan=context_plan,
            snapshot=snapshot,
            messages_for_compile=messages_for_compile,
            fragment_messages=fragment_messages,
            fragments=fragments,
            optimization_policy=policy,
            runtime=runtime,
        )
    )
    assert first.decision is ContextOptimizationDecision.CREATE_ARTIFACT
    second = await resolve_ucl_context_plan(
        **_resolve_kwargs(
            request=_request(run_id="run-2"),
            context_plan=context_plan,
            snapshot=snapshot,
            messages_for_compile=messages_for_compile,
            fragment_messages=fragment_messages,
            fragments=fragments,
            optimization_policy=policy,
            runtime=runtime,
        )
    )
    assert second.decision is ContextOptimizationDecision.REUSE_ARTIFACT
    assert model_calls[0] == 1
    assert sum(1 for message in second.messages if "history block" in (message.content or "")) == 0
    assert any("condensed session summary" in (message.content or "") for message in second.messages)


@pytest.mark.asyncio
async def test_create_and_store_persists_artifact() -> None:
    long_history = ["history block " * 30]
    (
        request,
        context_plan,
        snapshot,
        messages_for_compile,
        fragment_messages,
        fragments,
        policy,
    ) = _plan_fixture(
        history_contents=long_history,
        resolved_budget=20,
        optimization_policy=_optimization_policy(),
    )
    repo = InMemoryOptimizationArtifactRepository()
    runtime, model_calls, _ = _runtime(repository=repo)
    resolution = await resolve_ucl_context_plan(
        **_resolve_kwargs(
            request=request,
            context_plan=context_plan,
            snapshot=snapshot,
            messages_for_compile=messages_for_compile,
            fragment_messages=fragment_messages,
            fragments=fragments,
            optimization_policy=policy,
            runtime=runtime,
        )
    )
    assert resolution.decision is ContextOptimizationDecision.CREATE_ARTIFACT
    assert model_calls[0] == 1
    assert resolution.artifact_reference is not None
    requirement = context_plan.artifact_requirement
    assert requirement is not None
    lookup_key = ArtifactLookupKey(
        tenant_id=requirement.lookup_inputs.tenant_id,
        context_scope_id=requirement.lookup_inputs.context_scope_id,
        artifact_type=requirement.lookup_inputs.artifact_type,
        source_content_hash=requirement.lookup_inputs.source_content_hash,
        compression_target=requirement.lookup_inputs.compression_target,
        lossiness_profile=requirement.lookup_inputs.lossiness_profile,
        source_refs=requirement.lookup_inputs.source_refs,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        policy_version=policy.policy_version,
        validation_contract_version=policy.validation_contract_version,
    )
    assert repo.lookup(lookup_key) is not None


@pytest.mark.asyncio
async def test_concurrent_same_key_single_flight() -> None:
    gate = threading.Event()
    long_history = ["shared history " * 30]

    def _blocked_invoke(_call: Any) -> LLMAdapterResponse:
        gate.wait(timeout=5)
        return LLMAdapterResponse(content="shared summary text")

    (
        request,
        context_plan,
        snapshot,
        messages_for_compile,
        fragment_messages,
        fragments,
        policy,
    ) = _plan_fixture(
        history_contents=long_history,
        resolved_budget=20,
        optimization_policy=_optimization_policy(),
    )
    repo = InMemoryOptimizationArtifactRepository()
    runtime, model_calls, _ = _runtime(repository=repo, invoke_model=_blocked_invoke)

    async def _resolve(run_id: str) -> Any:
        return await resolve_ucl_context_plan(
            **_resolve_kwargs(
                request=_request(run_id=run_id),
                context_plan=context_plan,
                snapshot=snapshot,
                messages_for_compile=messages_for_compile,
                fragment_messages=fragment_messages,
                fragments=fragments,
                optimization_policy=policy,
                runtime=runtime,
            )
        )

    first_task = asyncio.create_task(_resolve("run-a"))
    await asyncio.sleep(0.05)
    second_task = asyncio.create_task(_resolve("run-b"))
    await asyncio.sleep(0.05)
    gate.set()
    first, second = await asyncio.gather(first_task, second_task)
    decisions = {first.decision, second.decision}
    assert decisions == {
        ContextOptimizationDecision.CREATE_ARTIFACT,
        ContextOptimizationDecision.REUSE_ARTIFACT,
    }
    assert model_calls[0] == 1
    assert first.messages[-2].content == second.messages[-2].content


@pytest.mark.asyncio
async def test_different_key_concurrency_allows_two_model_calls() -> None:
    (
        _request_a,
        plan_a,
        snapshot_a,
        messages_a,
        fragment_messages_a,
        fragments_a,
        policy,
    ) = _plan_fixture(history_contents=["alpha " * 30], resolved_budget=20, optimization_policy=_optimization_policy())
    (
        _request_b,
        plan_b,
        snapshot_b,
        messages_b,
        fragment_messages_b,
        fragments_b,
        _policy_b,
    ) = _plan_fixture(history_contents=["beta " * 30], resolved_budget=20, optimization_policy=_optimization_policy())
    repo = InMemoryOptimizationArtifactRepository()
    runtime, model_calls, _ = _runtime(repository=repo)
    await asyncio.gather(
        resolve_ucl_context_plan(
            **_resolve_kwargs(
                request=_request(run_id="run-a"),
                context_plan=plan_a,
                snapshot=snapshot_a,
                messages_for_compile=messages_a,
                fragment_messages=fragment_messages_a,
                fragments=fragments_a,
                optimization_policy=policy,
                runtime=runtime,
            )
        ),
        resolve_ucl_context_plan(
            **_resolve_kwargs(
                request=_request(run_id="run-b"),
                context_plan=plan_b,
                snapshot=snapshot_b,
                messages_for_compile=messages_b,
                fragment_messages=fragment_messages_b,
                fragments=fragments_b,
                optimization_policy=policy,
                runtime=runtime,
            )
        ),
    )
    assert model_calls[0] == 2


@pytest.mark.asyncio
async def test_already_in_progress_timeout_errors() -> None:
    long_history = ["blocked history " * 30]
    (
        request,
        context_plan,
        snapshot,
        messages_for_compile,
        fragment_messages,
        fragments,
        policy,
    ) = _plan_fixture(
        history_contents=long_history,
        resolved_budget=20,
        optimization_policy=_optimization_policy(),
    )
    repo = InMemoryOptimizationArtifactRepository()
    requirement = context_plan.artifact_requirement
    assert requirement is not None
    lookup_key = ArtifactLookupKey(
        **{
            **{
                "tenant_id": requirement.lookup_inputs.tenant_id,
                "context_scope_id": requirement.lookup_inputs.context_scope_id,
                "artifact_type": requirement.lookup_inputs.artifact_type,
                "source_content_hash": requirement.lookup_inputs.source_content_hash,
                "compression_target": requirement.lookup_inputs.compression_target,
                "lossiness_profile": requirement.lookup_inputs.lossiness_profile,
                "source_refs": requirement.lookup_inputs.source_refs,
                "strategy_id": STRATEGY_ID,
                "strategy_version": STRATEGY_VERSION,
                "policy_version": policy.policy_version,
                "validation_contract_version": policy.validation_contract_version,
            }
        }
    )
    repo.try_acquire_creation_reservation(
        lookup_key,
        owner_operation_id="other-owner",
        lease_seconds=60,
    )
    runtime, model_calls, _ = _runtime(repository=repo, wait_timeout_seconds=0.01)
    with pytest.raises(NexusUCLExecutionError) as exc_info:
        await resolve_ucl_context_plan(
            **_resolve_kwargs(
                request=request,
                context_plan=context_plan,
                snapshot=snapshot,
                messages_for_compile=messages_for_compile,
                fragment_messages=fragment_messages,
                fragments=fragments,
                optimization_policy=policy,
                runtime=runtime,
            )
        )
    assert exc_info.value.reason == "artifact_creation_in_progress"
    assert model_calls[0] == 0


@pytest.mark.asyncio
async def test_executor_failure_releases_reservation() -> None:
    long_history = ["fail history " * 30]
    (
        request,
        context_plan,
        snapshot,
        messages_for_compile,
        fragment_messages,
        fragments,
        policy,
    ) = _plan_fixture(
        history_contents=long_history,
        resolved_budget=20,
        optimization_policy=_optimization_policy(),
    )
    repo = InMemoryOptimizationArtifactRepository()

    call_count = [0]

    def _fail_once(_call: Any) -> LLMAdapterResponse:
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("model failed")
        return LLMAdapterResponse(content="recovered summary text")

    runtime, model_calls, _ = _runtime(repository=repo, invoke_model=_fail_once)
    with pytest.raises(NexusUCLExecutionError) as exc_info:
        await resolve_ucl_context_plan(
            **_resolve_kwargs(
                request=request,
                context_plan=context_plan,
                snapshot=snapshot,
                messages_for_compile=messages_for_compile,
                fragment_messages=fragment_messages,
                fragments=fragments,
                optimization_policy=policy,
                runtime=runtime,
            )
        )
    assert exc_info.value.reason == NexusUCLExecutionReason.ARTIFACT_CREATION_FAILED.value
    assert model_calls[0] == 1
    second = await resolve_ucl_context_plan(
        **_resolve_kwargs(
            request=_request(run_id="run-2"),
            context_plan=context_plan,
            snapshot=snapshot,
            messages_for_compile=messages_for_compile,
            fragment_messages=fragment_messages,
            fragments=fragments,
            optimization_policy=policy,
            runtime=runtime,
        )
    )
    assert second.decision is ContextOptimizationDecision.CREATE_ARTIFACT


@pytest.mark.asyncio
async def test_non_persist_policy_releases_reservation() -> None:
    long_history = ["ephemeral history " * 30]
    policy = _optimization_policy(persistence=EphemeralArtifactPersistencePolicy.DO_NOT_PERSIST)
    (
        request,
        context_plan,
        snapshot,
        messages_for_compile,
        fragment_messages,
        fragments,
        _policy,
    ) = _plan_fixture(
        history_contents=long_history,
        resolved_budget=20,
        optimization_policy=policy,
    )
    repo = InMemoryOptimizationArtifactRepository()
    runtime, _, _ = _runtime(repository=repo)
    resolution = await resolve_ucl_context_plan(
        **_resolve_kwargs(
            request=request,
            context_plan=context_plan,
            snapshot=snapshot,
            messages_for_compile=messages_for_compile,
            fragment_messages=fragment_messages,
            fragments=fragments,
            optimization_policy=policy,
            runtime=runtime,
        )
    )
    assert resolution.decision is ContextOptimizationDecision.CREATE_ARTIFACT
    assert resolution.artifact_reference is None
    requirement = context_plan.artifact_requirement
    assert requirement is not None
    lookup_key = ArtifactLookupKey(
        tenant_id=requirement.lookup_inputs.tenant_id,
        context_scope_id=requirement.lookup_inputs.context_scope_id,
        artifact_type=requirement.lookup_inputs.artifact_type,
        source_content_hash=requirement.lookup_inputs.source_content_hash,
        compression_target=requirement.lookup_inputs.compression_target,
        lossiness_profile=requirement.lookup_inputs.lossiness_profile,
        source_refs=requirement.lookup_inputs.source_refs,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        policy_version=policy.policy_version,
        validation_contract_version=policy.validation_contract_version,
    )
    assert repo.lookup(lookup_key) is None


def test_import_boundary_for_ucl_orchestration_module() -> None:
    source = (REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "ucl_orchestration.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    forbidden = {
        "InMemoryOptimizationArtifactRepository",
        "DefaultNexusContextEngine",
        "HistoryLayer",
        "ConversationalMemory",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("applications"):
                raise AssertionError(f"forbidden application import: {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden:
                    raise AssertionError(f"forbidden import: {alias.name}")
    for token in forbidden:
        assert token not in source
