# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-5 Nexus UCL orchestration tests."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
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
    ContextArtifactLookupInputs,
    ContextArtifactRequirement,
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
    ArtifactCreationCoordinationStatus,
    ArtifactLookupKey,
    ContextOptimizationDecision,
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    ContextOptimizationReasonCode,
    EphemeralArtifactPersistencePolicy,
    ModelCallExecutionScope,
    OptimizationArtifactType,
)
from intergrax.runtime.context_lifecycle.in_memory_repository import InMemoryOptimizationArtifactRepository
from intergrax.runtime.context_lifecycle.repository import (
    ArtifactCreationCoordinationResult,
    ArtifactCreationReservation,
    StoredOptimizationArtifact,
    build_optimization_artifact_reference,
    compute_artifact_content_hash,
)
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
    count_tokens: Callable[[str], int] | None = None,
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
        "count_tokens": count_tokens or _count_tokens,
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


def _dummy_reservation() -> ArtifactCreationReservation:
    acquired_at = datetime.now(UTC)
    return ArtifactCreationReservation(
        reservation_id="res-1",
        artifact_lookup_key_hash="hash",
        tenant_id="tenant",
        owner_operation_id="owner",
        acquired_at=acquired_at,
        lease_deadline=acquired_at + timedelta(seconds=60),
    )


def _policy_for_gate_case(
    base: ContextOptimizationPolicy,
    *,
    enabled: bool | None = None,
    mode: ContextOptimizationMode | None = None,
    allow_artifact_reuse: bool | None = None,
    allow_lossy: bool | None = None,
    allow_llm_summarization: bool | None = None,
    allowed_artifact_types: tuple[OptimizationArtifactType, ...] | None = None,
    require_rollback_metadata: bool = False,
    require_receipt: bool = False,
) -> ContextOptimizationPolicy:
    return ContextOptimizationPolicy(
        policy_version=base.policy_version,
        validation_contract_version=base.validation_contract_version,
        enabled=base.enabled if enabled is None else enabled,
        mode=base.mode if mode is None else mode,
        allow_lossy=base.allow_lossy if allow_lossy is None else allow_lossy,
        allow_llm_summarization=(
            base.allow_llm_summarization
            if allow_llm_summarization is None
            else allow_llm_summarization
        ),
        allow_artifact_reuse=(
            base.allow_artifact_reuse if allow_artifact_reuse is None else allow_artifact_reuse
        ),
        allowed_artifact_types=(
            base.allowed_artifact_types
            if allowed_artifact_types is None
            else allowed_artifact_types
        ),
        allowed_strategy_ids=base.allowed_strategy_ids,
        require_rollback_metadata=require_rollback_metadata,
        require_receipt=require_receipt,
        ephemeral_artifact_persistence=(
            EphemeralArtifactPersistencePolicy.DO_NOT_PERSIST
            if allow_artifact_reuse is False
            else base.ephemeral_artifact_persistence
        ),
    )


class _SpyOptimizationArtifactRepository(InMemoryOptimizationArtifactRepository):
    def __init__(self) -> None:
        super().__init__()
        self.lookup_calls = 0
        self.reservation_calls = 0
        self.wait_calls = 0
        self.store_calls = 0
        self.release_calls = 0
        self._lookup_results: list[StoredOptimizationArtifact | None] = []
        self._coordination_result: ArtifactCreationCoordinationResult | None = None
        self._release_result: bool | None = None

    def configure_lookup(self, *results: StoredOptimizationArtifact | None) -> None:
        self._lookup_results = list(results)

    def configure_coordination(self, result: ArtifactCreationCoordinationResult) -> None:
        self._coordination_result = result

    def configure_release(self, result: bool | None) -> None:
        self._release_result = result

    def lookup(self, key: ArtifactLookupKey) -> StoredOptimizationArtifact | None:
        self.lookup_calls += 1
        if self._lookup_results:
            index = min(self.lookup_calls - 1, len(self._lookup_results) - 1)
            return self._lookup_results[index]
        return super().lookup(key)

    def try_acquire_creation_reservation(
        self,
        key: ArtifactLookupKey,
        *,
        owner_operation_id: str,
        lease_seconds: int,
    ) -> ArtifactCreationCoordinationResult:
        self.reservation_calls += 1
        if self._coordination_result is not None:
            return self._coordination_result
        return super().try_acquire_creation_reservation(
            key,
            owner_operation_id=owner_operation_id,
            lease_seconds=lease_seconds,
        )

    def wait_for_artifact_or_reservation_change(
        self,
        key: ArtifactLookupKey,
        *,
        observed_state_version: int,
        timeout_seconds: float,
    ) -> None:
        self.wait_calls += 1
        return super().wait_for_artifact_or_reservation_change(
            key,
            observed_state_version=observed_state_version,
            timeout_seconds=timeout_seconds,
        )

    def store_validated_artifact(self, *, reservation: Any, artifact: StoredOptimizationArtifact) -> Any:
        self.store_calls += 1
        return super().store_validated_artifact(reservation=reservation, artifact=artifact)

    def release_creation_reservation(self, *, reservation: Any, reason_code: Any = None) -> bool:
        self.release_calls += 1
        if self._release_result is not None:
            return self._release_result
        return super().release_creation_reservation(
            reservation=reservation,
            reason_code=reason_code,
        )


def _artifact_id_call_counter() -> tuple[list[int], Callable[[], str]]:
    calls = [0]

    def _factory() -> str:
        calls[0] += 1
        return "artifact-spy-1"

    return calls, _factory


async def _create_valid_stored_artifact() -> tuple[Any, ...]:
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
    runtime, _, _ = _runtime(repository=repo)
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
    stored = repo.lookup(lookup_key)
    assert stored is not None
    return (
        request,
        context_plan,
        snapshot,
        messages_for_compile,
        fragment_messages,
        fragments,
        policy,
        lookup_key,
        stored,
    )


def _corrupt_stored_payload(
    stored: StoredOptimizationArtifact,
    *,
    payload_mutator: Callable[[dict[str, Any]], dict[str, Any]],
    media_type: str | None = None,
) -> StoredOptimizationArtifact:
    parsed = json.loads(stored.payload.decode("utf-8"))
    mutated = payload_mutator(parsed)
    payload = json.dumps(mutated, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    metadata = replace(stored.metadata, artifact_content_hash=compute_artifact_content_hash(payload))
    return StoredOptimizationArtifact(
        metadata=metadata,
        payload=payload,
        media_type=media_type if media_type is not None else stored.media_type,
        encoding=stored.encoding,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy_mutator", "plan_mutator"),
    [
        (lambda policy: _policy_for_gate_case(policy, enabled=False), None),
        (
            lambda policy: _policy_for_gate_case(
                policy,
                mode=ContextOptimizationMode.DURABLE_COMPACTION,
                require_rollback_metadata=True,
                require_receipt=True,
            ),
            None,
        ),
        (lambda policy: _policy_for_gate_case(policy, allow_artifact_reuse=False), None),
        (
            lambda policy: _policy_for_gate_case(
                policy,
                allow_lossy=False,
                allow_llm_summarization=False,
            ),
            None,
        ),
        (lambda policy: _policy_for_gate_case(policy, allow_llm_summarization=False), None),
        (lambda policy: _policy_for_gate_case(policy, allowed_artifact_types=()), None),
        (
            None,
            lambda plan: replace(
                plan,
                artifact_requirement=replace(
                    plan.artifact_requirement,
                    lookup_inputs=replace(
                        plan.artifact_requirement.lookup_inputs,
                        lossiness_profile="lossless",
                    ),
                ),
            ),
        ),
    ],
    ids=[
        "enabled_false",
        "mode_durable_compaction",
        "allow_artifact_reuse_false",
        "allow_lossy_false",
        "allow_llm_summarization_false",
        "message_sequence_not_allowed",
        "lossiness_not_lossy",
    ],
)
async def test_policy_gate_blocks_before_repository_access(
    policy_mutator: Callable[[ContextOptimizationPolicy], ContextOptimizationPolicy] | None,
    plan_mutator: Callable[[ContextPlan], ContextPlan] | None,
) -> None:
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
    if policy_mutator is not None:
        policy = policy_mutator(policy)
    if plan_mutator is not None:
        context_plan = plan_mutator(context_plan)
    repo = _SpyOptimizationArtifactRepository()
    id_calls, id_factory = _artifact_id_call_counter()
    model_calls = [0]
    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=MessageSequenceArtifactExecutor(
            preflight=lambda _call: None,
            invoke_model=lambda _call: (model_calls.__setitem__(0, model_calls[0] + 1) or LLMAdapterResponse(content="condensed session summary")),
            count_tokens=_count_tokens,
        ),
        strategy_versions={STRATEGY_ID: STRATEGY_VERSION},
        artifact_id_factory=id_factory,
    )
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
    assert exc_info.value.reason == NexusUCLExecutionReason.POLICY_BLOCKED.value
    assert repo.lookup_calls == 0
    assert repo.reservation_calls == 0
    assert repo.wait_calls == 0
    assert model_calls[0] == 0
    assert id_calls[0] == 0


@pytest.mark.asyncio
async def test_non_contiguous_target_fails_before_repository_access() -> None:
    group_a = ContextSourceGroup(
        group_id="group-a",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("m0",),
        source_content_hash="hash-a",
        token_estimate=80,
        compressible=True,
        required=False,
        protected=False,
    )
    group_mid = ContextSourceGroup(
        group_id="group-mid",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("mid",),
        source_content_hash="hash-mid",
        token_estimate=10,
        compressible=False,
        required=False,
        protected=False,
    )
    group_b = ContextSourceGroup(
        group_id="group-b",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("m1",),
        source_content_hash="hash-b",
        token_estimate=80,
        compressible=True,
        required=False,
        protected=False,
    )
    group_current = ContextSourceGroup(
        group_id="group-current",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("current",),
        source_content_hash="hash-current",
        token_estimate=10,
        required=True,
        protected=True,
    )
    lookup_inputs = ContextArtifactLookupInputs(
        tenant_id="tenant",
        context_scope_id="scope",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash=hashlib.sha256("hash-a|hash-b".encode("utf-8")).hexdigest(),
        compression_target=__import__(
            "intergrax.runtime.context_lifecycle.contracts",
            fromlist=["ArtifactCompressionTarget"],
        ).ArtifactCompressionTarget(target_tokens=18),
        lossiness_profile="lossy",
        source_refs=("m0", "m1"),
    )
    artifact_requirement = ContextArtifactRequirement(
        lookup_inputs=lookup_inputs,
        source_group_ids=("group-a", "group-b"),
        allowed_strategy_ids=(STRATEGY_ID,),
        minimum_preservation=__import__(
            "intergrax.context.planning",
            fromlist=["ContextMinimumPreservationRequirements"],
        ).ContextMinimumPreservationRequirements(
            preserve_message_order=True,
            preserve_roles=True,
            preserve_message_ids=True,
            preserve_tool_call_links=True,
            preserve_recent_tail_messages=0,
            required_group_ids=("group-current",),
            protected_group_ids=("group-current",),
        ),
    )
    context_plan = ContextPlan(
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
        budget_class=ContextBudgetClass.PRIMARY_MODEL_INPUT,
        resolved_global_budget_tokens=20,
        estimated_total_tokens=180,
        source_groups=(group_a, group_mid, group_b, group_current),
        source_allocations=(
            ContextSourceBudgetAllocation(
                source=ContextFragmentSource.SESSION_HISTORY,
                allocated_tokens=180,
                selected_group_ids=("group-a", "group-mid", "group-b", "group-current"),
                excluded_group_ids=(),
            ),
        ),
        selected_group_ids=("group-a", "group-mid", "group-b", "group-current"),
        excluded_group_ids=(),
        required_group_ids=("group-current",),
        protected_group_ids=("group-current",),
        compressible_group_ids=("group-a", "group-b"),
        droppable_group_ids=(),
        trim_safe_group_ids=(),
        optimization_required=True,
        artifact_requirement=artifact_requirement,
        final_validation_requirements=("respect_resolved_global_budget",),
    )
    fragments = (
        ContextFragment(
            fragment_id="frag-a",
            source=ContextFragmentSource.SESSION_HISTORY,
            source_id="session",
            content="history a " * 30,
            token_estimate=80,
            relevance_score=0.5,
            freshness_score=0.5,
            confidence_score=0.5,
            mandatory=False,
            metadata={"message_id": "m0"},
        ),
        ContextFragment(
            fragment_id="frag-mid",
            source=ContextFragmentSource.SESSION_HISTORY,
            source_id="session",
            content="middle",
            token_estimate=10,
            relevance_score=0.5,
            freshness_score=0.5,
            confidence_score=0.5,
            mandatory=False,
            metadata={"message_id": "mid"},
        ),
        ContextFragment(
            fragment_id="frag-b",
            source=ContextFragmentSource.SESSION_HISTORY,
            source_id="session",
            content="history b " * 30,
            token_estimate=80,
            relevance_score=0.5,
            freshness_score=0.5,
            confidence_score=0.5,
            mandatory=False,
            metadata={"message_id": "m1"},
        ),
    )
    formatter = DefaultContextFormatter()
    request = _request()
    fragment_messages = formatter.format(fragments, request)
    messages_for_compile = [
        fragment_messages[0],
        fragment_messages[1],
        fragment_messages[2],
        ChatMessage(role="user", content="current", entry_id="current"),
    ]
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev-1",
        messages=[
            ChatMessage(role="user", content="history a " * 30, entry_id="m0"),
            ChatMessage(role="user", content="middle", entry_id="mid"),
            ChatMessage(role="user", content="history b " * 30, entry_id="m1"),
        ],
    )
    repo = _SpyOptimizationArtifactRepository()
    id_calls, id_factory = _artifact_id_call_counter()
    model_calls = [0]
    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=MessageSequenceArtifactExecutor(
            preflight=lambda _call: None,
            invoke_model=lambda _call: (model_calls.__setitem__(0, model_calls[0] + 1) or LLMAdapterResponse(content="x")),
            count_tokens=_count_tokens,
        ),
        strategy_versions={STRATEGY_ID: STRATEGY_VERSION},
        artifact_id_factory=id_factory,
    )
    with pytest.raises(NexusUCLExecutionError) as exc_info:
        await resolve_ucl_context_plan(
            **_resolve_kwargs(
                request=request,
                context_plan=context_plan,
                snapshot=snapshot,
                messages_for_compile=messages_for_compile,
                fragment_messages=fragment_messages,
                fragments=fragments,
                optimization_policy=_optimization_policy(),
                runtime=runtime,
            )
        )
    assert exc_info.value.reason == NexusUCLExecutionReason.NON_CONTIGUOUS_ARTIFACT_TARGET.value
    assert repo.lookup_calls == 0
    assert model_calls[0] == 0
    assert id_calls[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("history_source_refs", "compile_history_refs"),
    [
        (("m0", "m1"), ("m0",)),
        (("m1", "m0"), ("m0", "m1")),
    ],
    ids=["missing_ref", "reversed_refs"],
)
async def test_partial_multi_message_group_fails_before_repository_access(
    history_source_refs: tuple[str, ...],
    compile_history_refs: tuple[str, ...],
) -> None:
    group_history = ContextSourceGroup(
        group_id="group-history",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=history_source_refs,
        source_content_hash="hash-history",
        token_estimate=80,
        compressible=True,
        required=False,
        protected=False,
    )
    group_current = ContextSourceGroup(
        group_id="group-current",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_refs=("current",),
        source_content_hash="hash-current",
        token_estimate=10,
        required=True,
        protected=True,
    )
    lookup_inputs = ContextArtifactLookupInputs(
        tenant_id="tenant",
        context_scope_id="scope",
        artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
        source_content_hash=hashlib.sha256("hash-history".encode("utf-8")).hexdigest(),
        compression_target=__import__(
            "intergrax.runtime.context_lifecycle.contracts",
            fromlist=["ArtifactCompressionTarget"],
        ).ArtifactCompressionTarget(target_tokens=18),
        lossiness_profile="lossy",
        source_refs=history_source_refs,
    )
    artifact_requirement = ContextArtifactRequirement(
        lookup_inputs=lookup_inputs,
        source_group_ids=("group-history",),
        allowed_strategy_ids=(STRATEGY_ID,),
        minimum_preservation=__import__(
            "intergrax.context.planning",
            fromlist=["ContextMinimumPreservationRequirements"],
        ).ContextMinimumPreservationRequirements(
            preserve_message_order=True,
            preserve_roles=True,
            preserve_message_ids=True,
            preserve_tool_call_links=True,
            preserve_recent_tail_messages=0,
            required_group_ids=("group-current",),
            protected_group_ids=("group-current",),
        ),
    )
    context_plan = ContextPlan(
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
        budget_class=ContextBudgetClass.PRIMARY_MODEL_INPUT,
        resolved_global_budget_tokens=20,
        estimated_total_tokens=90,
        source_groups=(group_history, group_current),
        source_allocations=(
            ContextSourceBudgetAllocation(
                source=ContextFragmentSource.SESSION_HISTORY,
                allocated_tokens=90,
                selected_group_ids=("group-history", "group-current"),
                excluded_group_ids=(),
            ),
        ),
        selected_group_ids=("group-history", "group-current"),
        excluded_group_ids=(),
        required_group_ids=("group-current",),
        protected_group_ids=("group-current",),
        compressible_group_ids=("group-history",),
        droppable_group_ids=(),
        trim_safe_group_ids=(),
        optimization_required=True,
        artifact_requirement=artifact_requirement,
        final_validation_requirements=("respect_resolved_global_budget",),
    )
    history_by_ref = {
        "m0": "history part zero " * 20,
        "m1": "history part one " * 20,
    }
    fragments = tuple(
        ContextFragment(
            fragment_id=f"frag-{source_ref}",
            source=ContextFragmentSource.SESSION_HISTORY,
            source_id="session",
            content=history_by_ref[source_ref],
            token_estimate=40,
            relevance_score=0.5,
            freshness_score=0.5,
            confidence_score=0.5,
            mandatory=False,
            metadata={"message_id": source_ref},
        )
        for source_ref in compile_history_refs
    )
    formatter = DefaultContextFormatter()
    request = _request()
    fragment_messages = formatter.format(fragments, request)
    messages_for_compile = [
        *fragment_messages,
        ChatMessage(role="user", content="current", entry_id="current"),
    ]
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev-1",
        messages=[
            ChatMessage(role="user", content=history_by_ref[source_ref], entry_id=source_ref)
            for source_ref in history_source_refs
        ],
    )
    repo = _SpyOptimizationArtifactRepository()
    id_calls, id_factory = _artifact_id_call_counter()
    model_calls = [0]
    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=MessageSequenceArtifactExecutor(
            preflight=lambda _call: None,
            invoke_model=lambda _call: (model_calls.__setitem__(0, model_calls[0] + 1) or LLMAdapterResponse(content="x")),
            count_tokens=_count_tokens,
        ),
        strategy_versions={STRATEGY_ID: STRATEGY_VERSION},
        artifact_id_factory=id_factory,
    )
    with pytest.raises(NexusUCLExecutionError) as exc_info:
        await resolve_ucl_context_plan(
            **_resolve_kwargs(
                request=request,
                context_plan=context_plan,
                snapshot=snapshot,
                messages_for_compile=messages_for_compile,
                fragment_messages=fragment_messages,
                fragments=fragments,
                optimization_policy=_optimization_policy(),
                runtime=runtime,
            )
        )
    assert exc_info.value.reason == NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED.value
    assert repo.lookup_calls == 0
    assert repo.reservation_calls == 0
    assert model_calls[0] == 0
    assert id_calls[0] == 0


def _ucl_replacement_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    return [message for message in messages if message.entry_id.startswith("ucl-artifact-")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "persistence",
    [
        EphemeralArtifactPersistencePolicy.DO_NOT_PERSIST,
        EphemeralArtifactPersistencePolicy.PERSIST_AFTER_HUMAN_REVIEW,
    ],
)
async def test_non_persist_flow_has_no_persistent_artifact_identity(
    persistence: EphemeralArtifactPersistencePolicy,
) -> None:
    long_history = ["ephemeral history " * 30]
    if persistence is EphemeralArtifactPersistencePolicy.PERSIST_AFTER_HUMAN_REVIEW:
        policy = ContextOptimizationPolicy(
            policy_version="policy.v1",
            validation_contract_version="validation.v1",
            enabled=True,
            mode=ContextOptimizationMode.EPHEMERAL_ASSEMBLY,
            allow_lossy=True,
            allow_llm_summarization=True,
            allow_artifact_reuse=True,
            allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
            allowed_strategy_ids=(STRATEGY_ID,),
            require_human_review=True,
            ephemeral_artifact_persistence=persistence,
        )
    else:
        policy = _optimization_policy(persistence=persistence)
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
    repo = _SpyOptimizationArtifactRepository()
    id_calls, id_factory = _artifact_id_call_counter()
    runtime, model_calls, _ = _runtime(repository=repo, artifact_ids=[])
    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=runtime.message_sequence_executor,
        strategy_versions={STRATEGY_ID: STRATEGY_VERSION},
        artifact_id_factory=id_factory,
        wait_timeout_seconds=runtime.wait_timeout_seconds,
    )
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
    assert repo.store_calls == 0
    assert repo.release_calls == 1
    assert id_calls[0] == 0
    assert model_calls[0] == 1
    replacements = _ucl_replacement_messages(resolution.messages)
    assert len(replacements) == 1
    assert "artifact_id" not in replacements[0].metadata
    synthetic_fragments = [
        fragment
        for fragment in resolution.fragments_included
        if fragment.fragment_id.startswith("ucl-artifact-")
    ]
    assert len(synthetic_fragments) == 1
    assert resolution.artifact_lookup_key_hash is not None
    assert synthetic_fragments[0].source_id == resolution.artifact_lookup_key_hash
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


@pytest.mark.asyncio
async def test_persisted_flow_exposes_persisted_artifact_identity() -> None:
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
    repo = _SpyOptimizationArtifactRepository()
    id_calls, id_factory = _artifact_id_call_counter()
    runtime, model_calls, _ = _runtime(repository=repo, artifact_ids=[])
    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=runtime.message_sequence_executor,
        strategy_versions={STRATEGY_ID: STRATEGY_VERSION},
        artifact_id_factory=id_factory,
        wait_timeout_seconds=runtime.wait_timeout_seconds,
    )
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
    assert id_calls[0] == 1
    assert repo.store_calls == 1
    assert resolution.artifact_reference is not None
    replacements = _ucl_replacement_messages(resolution.messages)
    assert len(replacements) == 1
    assert replacements[0].metadata.get("artifact_id") == resolution.artifact_reference.artifact_id
    synthetic_fragments = [
        fragment
        for fragment in resolution.fragments_included
        if fragment.fragment_id.startswith("ucl-artifact-")
    ]
    assert len(synthetic_fragments) == 1
    assert synthetic_fragments[0].source_id == resolution.artifact_reference.artifact_id


@pytest.mark.asyncio
async def test_missing_target_message_fails_before_repository_access() -> None:
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
    requirement = context_plan.artifact_requirement
    assert requirement is not None
    object.__setattr__(
        requirement,
        "source_group_ids",
        (*requirement.source_group_ids, "group-missing"),
    )
    repo = _SpyOptimizationArtifactRepository()
    id_calls, id_factory = _artifact_id_call_counter()
    model_calls = [0]
    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=MessageSequenceArtifactExecutor(
            preflight=lambda _call: None,
            invoke_model=lambda _call: (model_calls.__setitem__(0, model_calls[0] + 1) or LLMAdapterResponse(content="x")),
            count_tokens=_count_tokens,
        ),
        strategy_versions={STRATEGY_ID: STRATEGY_VERSION},
        artifact_id_factory=id_factory,
    )
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
    assert exc_info.value.reason == NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED.value
    assert repo.lookup_calls == 0
    assert model_calls[0] == 0


@pytest.mark.asyncio
async def test_summary_token_count_failure_does_not_store_artifact() -> None:
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
    repo = _SpyOptimizationArtifactRepository()
    runtime, model_calls, _ = _runtime(repository=repo)

    summary_token_calls = [0]

    def _count_tokens(text: str) -> int:
        if text == "condensed session summary":
            summary_token_calls[0] += 1
            if summary_token_calls[0] == 1:
                raise ValueError("summary tokenization failed")
        return max(1, len(text) // 4)

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
                count_tokens=_count_tokens,
            )
        )
    assert exc_info.value.reason == NexusUCLExecutionReason.ARTIFACT_CREATION_FAILED.value
    assert model_calls[0] == 1
    assert repo.store_calls == 0
    assert repo.release_calls >= 1
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
            count_tokens=_count_tokens,
        )
    )
    assert second.decision is ContextOptimizationDecision.CREATE_ARTIFACT


@pytest.mark.asyncio
async def test_non_persist_release_false_is_reservation_conflict() -> None:
    long_history = ["history block " * 30]
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
    repo = _SpyOptimizationArtifactRepository()
    repo.configure_release(False)
    runtime, model_calls, _ = _runtime(repository=repo)
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
    assert exc_info.value.reason == "artifact_creation_reservation_conflict"
    assert model_calls[0] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory",
    [
        lambda: (_ for _ in ()).throw(
            NexusUCLExecutionError(NexusUCLExecutionReason.POLICY_BLOCKED)
        ),
        lambda: "   ",
    ],
    ids=["raises_nexus_error", "whitespace_only"],
)
async def test_artifact_id_factory_error_is_normalized(factory: Callable[[], str]) -> None:
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
    repo = _SpyOptimizationArtifactRepository()
    runtime, model_calls, _ = _runtime(repository=repo, artifact_ids=[])
    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=runtime.message_sequence_executor,
        strategy_versions={STRATEGY_ID: STRATEGY_VERSION},
        artifact_id_factory=factory,
    )
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
    assert repo.release_calls >= 1


@pytest.mark.asyncio
async def test_artifact_available_retries_lookup_without_executor() -> None:
    (
        request,
        context_plan,
        snapshot,
        messages_for_compile,
        fragment_messages,
        fragments,
        policy,
        lookup_key,
        stored,
    ) = await _create_valid_stored_artifact()
    repo = _SpyOptimizationArtifactRepository()
    repo.configure_lookup(None, stored)
    repo.configure_coordination(
        ArtifactCreationCoordinationResult(
            status=ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE,
            artifact_lookup_key_hash="hash",
            state_version=1,
            reservation=None,
            artifact_reference=build_optimization_artifact_reference(stored),
            reason_code=None,
        )
    )
    model_calls = [0]
    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=MessageSequenceArtifactExecutor(
            preflight=lambda _call: None,
            invoke_model=lambda _call: (model_calls.__setitem__(0, model_calls[0] + 1) or LLMAdapterResponse(content="x")),
            count_tokens=_count_tokens,
        ),
        strategy_versions={STRATEGY_ID: STRATEGY_VERSION},
        artifact_id_factory=lambda: "artifact-spy-1",
    )
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
    assert resolution.decision is ContextOptimizationDecision.REUSE_ARTIFACT
    assert repo.lookup_calls == 2
    assert repo.reservation_calls == 1
    assert model_calls[0] == 0


@pytest.mark.asyncio
async def test_reservation_expired_skips_executor_and_reacquire() -> None:
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
    repo = _SpyOptimizationArtifactRepository()
    repo.configure_lookup(None)
    repo.configure_coordination(
        ArtifactCreationCoordinationResult(
            status=ArtifactCreationCoordinationStatus.RESERVATION_EXPIRED,
            artifact_lookup_key_hash="hash",
            state_version=1,
            reservation=_dummy_reservation(),
            artifact_reference=None,
            reason_code=ContextOptimizationReasonCode.ARTIFACT_CREATION_LEASE_EXPIRED,
        )
    )
    model_calls = [0]
    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=MessageSequenceArtifactExecutor(
            preflight=lambda _call: None,
            invoke_model=lambda _call: (model_calls.__setitem__(0, model_calls[0] + 1) or LLMAdapterResponse(content="x")),
            count_tokens=_count_tokens,
        ),
        strategy_versions={STRATEGY_ID: STRATEGY_VERSION},
        artifact_id_factory=lambda: "artifact-spy-1",
    )
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
    assert exc_info.value.reason == "artifact_creation_lease_expired"
    assert model_calls[0] == 0
    assert repo.reservation_calls == 1


@pytest.mark.asyncio
async def test_reservation_conflict_skips_executor_and_retry() -> None:
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
    repo = _SpyOptimizationArtifactRepository()
    repo.configure_lookup(None)
    repo.configure_coordination(
        ArtifactCreationCoordinationResult(
            status=ArtifactCreationCoordinationStatus.RESERVATION_CONFLICT,
            artifact_lookup_key_hash="hash",
            state_version=1,
            reservation=_dummy_reservation(),
            artifact_reference=None,
            reason_code=ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT,
        )
    )
    model_calls = [0]
    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=MessageSequenceArtifactExecutor(
            preflight=lambda _call: None,
            invoke_model=lambda _call: (model_calls.__setitem__(0, model_calls[0] + 1) or LLMAdapterResponse(content="x")),
            count_tokens=_count_tokens,
        ),
        strategy_versions={STRATEGY_ID: STRATEGY_VERSION},
        artifact_id_factory=lambda: "artifact-spy-1",
    )
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
    assert exc_info.value.reason == "artifact_creation_reservation_conflict"
    assert model_calls[0] == 0
    assert repo.lookup_calls == 1
    assert repo.reservation_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_id", "mutator"),
    [
        ("wrong_schema_version", lambda payload: {**payload, "schema_version": "bad.v9"}),
        ("wrong_source_refs", lambda payload: {**payload, "source_refs": ["other"]}),
        ("wrong_source_content_hash", lambda payload: {**payload, "source_content_hash": "bad"}),
        ("wrong_strategy_id", lambda payload: {**payload, "strategy_id": "other"}),
        ("empty_summary", lambda payload: {**payload, "summary": "   "}),
        ("wrong_media_type", None),
    ],
)
async def test_invalid_reuse_payload_cases(case_id: str, mutator: Callable[[dict[str, Any]], dict[str, Any]] | None) -> None:
    (
        request,
        context_plan,
        snapshot,
        messages_for_compile,
        fragment_messages,
        fragments,
        policy,
        _lookup_key,
        stored,
    ) = await _create_valid_stored_artifact()
    if case_id == "wrong_media_type":
        invalid = StoredOptimizationArtifact(
            metadata=stored.metadata,
            payload=stored.payload,
            media_type="application/json",
            encoding=stored.encoding,
        )
    else:
        assert mutator is not None
        invalid = _corrupt_stored_payload(stored, payload_mutator=mutator)
    repo = _SpyOptimizationArtifactRepository()
    repo.configure_lookup(invalid)
    model_calls = [0]
    runtime = NexusUCLRuntimeDependencies(
        repository=repo,
        message_sequence_executor=MessageSequenceArtifactExecutor(
            preflight=lambda _call: None,
            invoke_model=lambda _call: (model_calls.__setitem__(0, model_calls[0] + 1) or LLMAdapterResponse(content="x")),
            count_tokens=_count_tokens,
        ),
        strategy_versions={STRATEGY_ID: STRATEGY_VERSION},
        artifact_id_factory=lambda: "artifact-spy-1",
    )
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
    assert exc_info.value.reason == NexusUCLExecutionReason.ARTIFACT_PAYLOAD_INVALID.value
    assert model_calls[0] == 0


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
