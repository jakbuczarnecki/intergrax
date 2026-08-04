# © Artur Czarnecki. All rights reserved.

"""Canonical Nexus UCL orchestration (CTX-UCL-5)."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
)
from intergrax.context.planning import (
    ContextPlan,
    artifact_lookup_key_kwargs_from_context_inputs,
)
from intergrax.context.session_history import SessionHistoryMessage, SessionHistorySnapshot
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCreationCoordinationStatus,
    ArtifactLookupKey,
    ContextOptimizationDecision,
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    ContextOptimizationReasonCode,
    OptimizationArtifactType,
    EphemeralArtifactPersistencePolicy,
    ModelCallExecutionScope,
    OptimizationExecutionGuard,
    ReusableOptimizationArtifact,
)
from intergrax.runtime.context_lifecycle.repository import (
    OptimizationArtifactReference,
    OptimizationArtifactRepository,
    StoredOptimizationArtifact,
    build_optimization_artifact_reference,
)
from intergrax.runtime.context_lifecycle.serialization import compute_artifact_lookup_key_hash
from intergrax.runtime.token_optimization.message_sequence_artifact import (
    MessageSequenceArtifactExecutionRequest,
    MessageSequenceArtifactExecutor,
    MessageSequenceArtifactSourceGroupProof,
)
from intergrax.runtime.token_optimization.durable_compaction_candidate import (
    MessageSequenceArtifactValidationError,
    validate_message_sequence_payload,
    validate_stored_message_sequence_artifact,
)

logger = logging.getLogger("intergrax.nexus.ucl")

NEXUS_UCL_RUNTIME_HANDLE = "nexus_ucl_runtime"

_MEDIA_TYPE = "application/vnd.intergrax.message-sequence-summary+json"
_ENCODING = "utf-8"
_PERSIST_POLICIES = frozenset(
    {
        EphemeralArtifactPersistencePolicy.PERSIST_REUSABLE,
        EphemeralArtifactPersistencePolicy.PERSIST_AFTER_VALIDATION,
    }
)


class NexusUCLExecutionReason(StrEnum):
    PRIMARY_SCOPE_REQUIRED = "ucl_primary_scope_required"
    RUNTIME_REQUIRED = "ucl_runtime_required"
    POLICY_REQUIRED = "ucl_policy_required"
    POLICY_BLOCKED = "ucl_policy_blocked"
    STRATEGY_VERSION_UNAVAILABLE = "ucl_strategy_version_unavailable"
    PLAN_MATERIALIZATION_FAILED = "ucl_plan_materialization_failed"
    NON_CONTIGUOUS_ARTIFACT_TARGET = "ucl_non_contiguous_artifact_target"
    ARTIFACT_PAYLOAD_INVALID = "ucl_artifact_payload_invalid"
    FINAL_COMPILE_MUTATED_PLAN = "ucl_final_compile_mutated_plan"
    ARTIFACT_CREATION_FAILED = "artifact_creation_failed"


class NexusUCLExecutionError(ValueError):
    reason: str

    def __init__(self, reason: str | NexusUCLExecutionReason) -> None:
        code = reason.value if isinstance(reason, NexusUCLExecutionReason) else reason
        self.reason = code
        super().__init__(code)

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class NexusUCLRuntimeDependencies:
    repository: OptimizationArtifactRepository
    message_sequence_executor: MessageSequenceArtifactExecutor
    strategy_versions: Mapping[str, str]
    artifact_id_factory: Callable[[], str]
    wait_timeout_seconds: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.repository, OptimizationArtifactRepository):
            raise ValueError("repository must be OptimizationArtifactRepository")
        if type(self.message_sequence_executor) is not MessageSequenceArtifactExecutor:
            raise ValueError("message_sequence_executor must be MessageSequenceArtifactExecutor")
        if not isinstance(self.strategy_versions, Mapping):
            raise ValueError("strategy_versions must be a Mapping")
        if not self.strategy_versions:
            raise ValueError("strategy_versions must contain at least one entry")
        frozen_versions: dict[str, str] = {}
        for key, value in self.strategy_versions.items():
            if not isinstance(key, str) or not key:
                raise ValueError("strategy_versions keys must be non-empty strings")
            if not isinstance(value, str) or not value:
                raise ValueError("strategy_versions values must be non-empty strings")
            frozen_versions[key] = value
        object.__setattr__(self, "strategy_versions", MappingProxyType(frozen_versions))
        if not callable(self.artifact_id_factory):
            raise TypeError("artifact_id_factory must be callable")
        timeout = self.wait_timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("wait_timeout_seconds must be int or float")
        timeout_value = float(timeout)
        if not math.isfinite(timeout_value) or timeout_value < 0 or timeout_value > 5.0:
            raise ValueError("wait_timeout_seconds must be finite and in [0, 5.0]")
        object.__setattr__(self, "wait_timeout_seconds", timeout_value)


@dataclass(frozen=True, slots=True)
class _PreparedArtifactMaterialization:
    filtered_messages: tuple[ChatMessage, ...]
    filtered_group_ids: tuple[str, ...]
    target_group_ids: tuple[str, ...]
    first_target_index: int
    fragments_included: tuple[ContextFragment, ...]
    fragments_excluded: tuple[tuple[ContextFragment, str], ...]


@dataclass(frozen=True, slots=True)
class _ArtifactMaterialization:
    decision: ContextOptimizationDecision
    messages: tuple[ChatMessage, ...]
    fragments_included: tuple[ContextFragment, ...]
    fragments_excluded: tuple[tuple[ContextFragment, str], ...]


@dataclass(frozen=True, slots=True)
class NexusUCLResolution:
    decision: ContextOptimizationDecision
    messages: tuple[ChatMessage, ...] = field(repr=False)
    fragments_included: tuple[ContextFragment, ...]
    fragments_excluded: tuple[tuple[ContextFragment, str], ...]
    artifact_reference: OptimizationArtifactReference | None
    artifact_lookup_key_hash: str | None
    coordination_status: ArtifactCreationCoordinationStatus | None
    llm_transform_invoked: bool

    def __post_init__(self) -> None:
        if not isinstance(self.decision, ContextOptimizationDecision):
            raise TypeError("decision must be ContextOptimizationDecision")
        if not isinstance(self.messages, tuple):
            raise TypeError("messages must be a tuple")
        for index, message in enumerate(self.messages):
            if not isinstance(message, ChatMessage):
                raise TypeError(f"messages[{index}] must be ChatMessage")
        if not isinstance(self.fragments_included, tuple):
            raise TypeError("fragments_included must be a tuple")
        for index, fragment in enumerate(self.fragments_included):
            if not isinstance(fragment, ContextFragment):
                raise TypeError(f"fragments_included[{index}] must be ContextFragment")
        if not isinstance(self.fragments_excluded, tuple):
            raise TypeError("fragments_excluded must be a tuple")
        for index, item in enumerate(self.fragments_excluded):
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(f"fragments_excluded[{index}] must be a 2-tuple")
            fragment, reason = item
            if not isinstance(fragment, ContextFragment):
                raise TypeError(f"fragments_excluded[{index}][0] must be ContextFragment")
            if not isinstance(reason, str) or not reason:
                raise TypeError(f"fragments_excluded[{index}][1] must be a non-empty string")
        if self.artifact_reference is not None and not isinstance(
            self.artifact_reference, OptimizationArtifactReference
        ):
            raise TypeError("artifact_reference must be OptimizationArtifactReference or None")
        if self.artifact_lookup_key_hash is not None and (
            not isinstance(self.artifact_lookup_key_hash, str) or not self.artifact_lookup_key_hash
        ):
            raise TypeError("artifact_lookup_key_hash must be a non-empty string or None")
        if self.coordination_status is not None and not isinstance(
            self.coordination_status, ArtifactCreationCoordinationStatus
        ):
            raise TypeError("coordination_status must be ArtifactCreationCoordinationStatus or None")
        if not isinstance(self.llm_transform_invoked, bool):
            raise TypeError("llm_transform_invoked must be bool")

        decision = self.decision
        if decision is ContextOptimizationDecision.NO_OP:
            if (
                self.artifact_reference is not None
                or self.artifact_lookup_key_hash is not None
                or self.coordination_status is not None
                or self.llm_transform_invoked
            ):
                raise ValueError("NO_OP resolution invariants violated")
        elif decision is ContextOptimizationDecision.SELECT_ONLY:
            if (
                self.artifact_reference is not None
                or self.artifact_lookup_key_hash is not None
                or self.coordination_status is not None
                or self.llm_transform_invoked
            ):
                raise ValueError("SELECT_ONLY resolution invariants violated")
            if not self.fragments_excluded:
                raise ValueError("SELECT_ONLY requires at least one excluded fragment")
        elif decision is ContextOptimizationDecision.REUSE_ARTIFACT:
            if self.artifact_reference is None or self.artifact_lookup_key_hash is None:
                raise ValueError("REUSE_ARTIFACT requires artifact reference and lookup hash")
            if self.llm_transform_invoked:
                raise ValueError("REUSE_ARTIFACT requires llm_transform_invoked is False")
        elif decision is ContextOptimizationDecision.CREATE_ARTIFACT:
            if self.artifact_lookup_key_hash is None:
                raise ValueError("CREATE_ARTIFACT requires lookup hash")
            if self.coordination_status is not ArtifactCreationCoordinationStatus.ACQUIRED:
                raise ValueError("CREATE_ARTIFACT requires coordination_status == ACQUIRED")
            if not self.llm_transform_invoked:
                raise ValueError("CREATE_ARTIFACT requires llm_transform_invoked is True")


async def resolve_ucl_context_plan(
    *,
    request: ContextAssemblyRequest,
    context_plan: ContextPlan,
    optimization_policy: ContextOptimizationPolicy | None,
    session_history: SessionHistorySnapshot | None,
    messages_for_compile: Sequence[ChatMessage],
    fragment_messages: Sequence[ChatMessage],
    ranked_fragments: Sequence[ContextFragment],
    runtime: NexusUCLRuntimeDependencies | None,
    count_tokens: Callable[[str], int],
) -> NexusUCLResolution:
    if request.execution_scope is not ModelCallExecutionScope.PRIMARY_MODEL_CALL:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PRIMARY_SCOPE_REQUIRED)
    if not isinstance(request.run_id, str) or not request.run_id:
        raise ValueError("run_id must be a non-empty string")

    parent_guard = OptimizationExecutionGuard(
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
        operation_id=request.run_id,
        parent_operation_id=None,
        optimization_depth=0,
    )

    group_id_by_source_ref, message_group_ids = _materialize_plan_mapping(
        context_plan=context_plan,
        messages_for_compile=messages_for_compile,
        fragment_messages=fragment_messages,
        ranked_fragments=ranked_fragments,
    )
    excluded_group_ids = set(context_plan.excluded_group_ids)

    if not context_plan.optimization_required:
        return _resolve_selection_only(
            context_plan=context_plan,
            messages_for_compile=messages_for_compile,
            fragment_messages=fragment_messages,
            ranked_fragments=ranked_fragments,
            message_group_ids=message_group_ids,
            excluded_group_ids=excluded_group_ids,
        )

    if context_plan.artifact_requirement is None:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
    if optimization_policy is None:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.POLICY_REQUIRED)
    if session_history is None:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
    if runtime is None:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.RUNTIME_REQUIRED)

    requirement = context_plan.artifact_requirement
    lookup_inputs = requirement.lookup_inputs
    if lookup_inputs.tenant_id != session_history.tenant_id:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
    if lookup_inputs.context_scope_id != session_history.context_scope_id:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)

    _enforce_ucl_policy_gate(optimization_policy, requirement)

    strategy_id, strategy_version = _select_strategy(
        requirement.allowed_strategy_ids,
        optimization_policy.allowed_strategy_ids,
        runtime.strategy_versions,
    )

    prepared_materialization = _prepare_artifact_materialization(
        context_plan=context_plan,
        requirement=requirement,
        messages_for_compile=messages_for_compile,
        fragment_messages=fragment_messages,
        ranked_fragments=ranked_fragments,
        message_group_ids=message_group_ids,
        excluded_group_ids=excluded_group_ids,
    )

    lookup_key_kwargs = artifact_lookup_key_kwargs_from_context_inputs(lookup_inputs)
    lookup_key_kwargs["strategy_id"] = strategy_id
    lookup_key_kwargs["strategy_version"] = strategy_version
    lookup_key_kwargs["policy_version"] = optimization_policy.policy_version
    lookup_key_kwargs["validation_contract_version"] = optimization_policy.validation_contract_version
    lookup_key = ArtifactLookupKey(**lookup_key_kwargs)  # type: ignore[arg-type]
    lookup_hash = compute_artifact_lookup_key_hash(lookup_key)

    repository = runtime.repository
    stored_artifact = await asyncio.to_thread(repository.lookup, lookup_key)
    if stored_artifact is not None:
        summary, artifact_content_hash = _validate_stored_artifact_payload(
            stored_artifact,
            lookup_key=lookup_key,
        )
        artifact_reference = build_optimization_artifact_reference(stored_artifact)
        resolution = _materialize_artifact_messages(
            prepared=prepared_materialization,
            decision=ContextOptimizationDecision.REUSE_ARTIFACT,
            lookup_hash=lookup_hash,
            summary=summary,
            artifact_content_hash=artifact_content_hash,
            artifact_id=artifact_reference.artifact_id,
            count_tokens=count_tokens,
        )
        _log_ucl_resolution(
            decision=resolution.decision,
            lookup_hash=lookup_hash,
            coordination_status=None,
            llm_transform_invoked=False,
        )
        return _finalize_artifact_resolution(
            resolution,
            artifact_reference=artifact_reference,
            lookup_hash=lookup_hash,
            coordination_status=None,
            llm_transform_invoked=False,
        )

    coordination = await asyncio.to_thread(
        repository.try_acquire_creation_reservation,
        lookup_key,
        owner_operation_id=parent_guard.operation_id,
        lease_seconds=optimization_policy.reservation_lease_seconds,
    )

    if coordination.status is ArtifactCreationCoordinationStatus.ARTIFACT_AVAILABLE:
        stored_artifact = await asyncio.to_thread(repository.lookup, lookup_key)
        if stored_artifact is None:
            raise NexusUCLExecutionError(
                ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT.value
            )
        summary, artifact_content_hash = _validate_stored_artifact_payload(
            stored_artifact,
            lookup_key=lookup_key,
        )
        artifact_reference = build_optimization_artifact_reference(stored_artifact)
        resolution = _materialize_artifact_messages(
            prepared=prepared_materialization,
            decision=ContextOptimizationDecision.REUSE_ARTIFACT,
            lookup_hash=lookup_hash,
            summary=summary,
            artifact_content_hash=artifact_content_hash,
            artifact_id=artifact_reference.artifact_id,
            count_tokens=count_tokens,
        )
        _log_ucl_resolution(
            decision=resolution.decision,
            lookup_hash=lookup_hash,
            coordination_status=coordination.status,
            llm_transform_invoked=False,
        )
        return _finalize_artifact_resolution(
            resolution,
            artifact_reference=artifact_reference,
            lookup_hash=lookup_hash,
            coordination_status=coordination.status,
            llm_transform_invoked=False,
        )

    if coordination.status is ArtifactCreationCoordinationStatus.ALREADY_IN_PROGRESS:
        await asyncio.to_thread(
            repository.wait_for_artifact_or_reservation_change,
            lookup_key,
            observed_state_version=coordination.state_version,
            timeout_seconds=runtime.wait_timeout_seconds,
        )
        stored_artifact = await asyncio.to_thread(repository.lookup, lookup_key)
        if stored_artifact is None:
            raise NexusUCLExecutionError(
                ContextOptimizationReasonCode.ARTIFACT_CREATION_IN_PROGRESS.value
            )
        summary, artifact_content_hash = _validate_stored_artifact_payload(
            stored_artifact,
            lookup_key=lookup_key,
        )
        artifact_reference = build_optimization_artifact_reference(stored_artifact)
        resolution = _materialize_artifact_messages(
            prepared=prepared_materialization,
            decision=ContextOptimizationDecision.REUSE_ARTIFACT,
            lookup_hash=lookup_hash,
            summary=summary,
            artifact_content_hash=artifact_content_hash,
            artifact_id=artifact_reference.artifact_id,
            count_tokens=count_tokens,
        )
        _log_ucl_resolution(
            decision=resolution.decision,
            lookup_hash=lookup_hash,
            coordination_status=coordination.status,
            llm_transform_invoked=False,
        )
        return _finalize_artifact_resolution(
            resolution,
            artifact_reference=artifact_reference,
            lookup_hash=lookup_hash,
            coordination_status=coordination.status,
            llm_transform_invoked=False,
        )

    if coordination.status is ArtifactCreationCoordinationStatus.RESERVATION_EXPIRED:
        raise NexusUCLExecutionError(
            ContextOptimizationReasonCode.ARTIFACT_CREATION_LEASE_EXPIRED.value
        )

    if coordination.status is ArtifactCreationCoordinationStatus.RESERVATION_CONFLICT:
        raise NexusUCLExecutionError(
            ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT.value
        )

    if coordination.status is not ArtifactCreationCoordinationStatus.ACQUIRED:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
    if coordination.reservation is None:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)

    source_messages, source_group_proofs = _build_executor_source_inputs(
        context_plan=context_plan,
        requirement=requirement,
        session_history=session_history,
    )
    execution_request = MessageSequenceArtifactExecutionRequest(
        decision=ContextOptimizationDecision.CREATE_ARTIFACT,
        coordination=coordination,
        lookup_key=lookup_key,
        policy=optimization_policy,
        parent_guard=parent_guard,
        source_messages=source_messages,
        source_group_proofs=source_group_proofs,
    )

    artifact_reference: OptimizationArtifactReference | None = None
    try:
        execution_result = await asyncio.to_thread(
            runtime.message_sequence_executor.execute,
            execution_request,
        )
        summary, artifact_content_hash = _validate_execution_payload(
            execution_result.payload,
            execution_result.artifact_content_hash,
            lookup_key=lookup_key,
        )
        persistence = optimization_policy.ephemeral_artifact_persistence
        should_persist = persistence in _PERSIST_POLICIES

        if should_persist:
            artifact_id = _allocate_artifact_id(runtime.artifact_id_factory)
            metadata = ReusableOptimizationArtifact(
                artifact_id=artifact_id,
                lookup_key=lookup_key,
                artifact_content_hash=execution_result.artifact_content_hash,
                created_at=execution_result.receipt.created_at,
                created_by_executor="message_sequence_artifact_executor.v1",
                validation=execution_result.validation,
                receipt_ref=execution_result.receipt.receipt_id,
                safe_metadata={
                    "parent_operation_id": execution_result.receipt.parent_operation_id,
                    "internal_operation_id": execution_result.receipt.internal_operation_id,
                    "source_ref_count": execution_result.receipt.source_ref_count,
                    "input_tokens": execution_result.receipt.input_tokens,
                    "output_tokens": execution_result.receipt.output_tokens,
                    "target_tokens": execution_result.receipt.target_tokens,
                },
            )
            stored = StoredOptimizationArtifact(
                metadata=metadata,
                payload=execution_result.payload,
                media_type=execution_result.media_type,
                encoding=execution_result.encoding,
            )
            resolution = _materialize_artifact_messages(
                prepared=prepared_materialization,
                decision=ContextOptimizationDecision.CREATE_ARTIFACT,
                lookup_hash=lookup_hash,
                summary=summary,
                artifact_content_hash=artifact_content_hash,
                artifact_id=artifact_id,
                count_tokens=count_tokens,
            )
            artifact_reference = await asyncio.to_thread(
                repository.store_validated_artifact,
                reservation=coordination.reservation,
                artifact=stored,
            )
        else:
            resolution = _materialize_artifact_messages(
                prepared=prepared_materialization,
                decision=ContextOptimizationDecision.CREATE_ARTIFACT,
                lookup_hash=lookup_hash,
                summary=summary,
                artifact_content_hash=artifact_content_hash,
                artifact_id=None,
                count_tokens=count_tokens,
            )
            released = await asyncio.to_thread(
                repository.release_creation_reservation,
                reservation=coordination.reservation,
            )
            if released is not True:
                raise NexusUCLExecutionError(
                    ContextOptimizationReasonCode.ARTIFACT_CREATION_RESERVATION_CONFLICT.value
                )
            artifact_reference = None
    except NexusUCLExecutionError:
        await _release_reservation_on_failure(repository, coordination.reservation)
        raise
    except Exception:
        await _release_reservation_on_failure(repository, coordination.reservation)
        raise NexusUCLExecutionError(NexusUCLExecutionReason.ARTIFACT_CREATION_FAILED) from None

    _log_ucl_resolution(
        decision=ContextOptimizationDecision.CREATE_ARTIFACT,
        lookup_hash=lookup_hash,
        coordination_status=coordination.status,
        llm_transform_invoked=True,
    )
    return _finalize_artifact_resolution(
        resolution,
        artifact_reference=artifact_reference,
        lookup_hash=lookup_hash,
        coordination_status=coordination.status,
        llm_transform_invoked=True,
    )


def _log_ucl_resolution(
    *,
    decision: ContextOptimizationDecision,
    lookup_hash: str | None,
    coordination_status: ArtifactCreationCoordinationStatus | None,
    llm_transform_invoked: bool,
) -> None:
    logger.info(
        "ucl decision=%s lookup_hash=%s coordination=%s llm_transform_invoked=%s",
        decision.value,
        lookup_hash or "none",
        coordination_status.value if coordination_status is not None else "none",
        llm_transform_invoked,
    )


async def _release_reservation_on_failure(
    repository: OptimizationArtifactRepository,
    reservation: Any,
) -> None:
    try:
        await asyncio.to_thread(
            repository.release_creation_reservation,
            reservation=reservation,
            reason_code=ContextOptimizationReasonCode.ARTIFACT_CREATION_FAILED,
        )
    except Exception:
        logger.debug("ucl reservation release after failure failed", exc_info=True)


def _finalize_artifact_resolution(
    resolution: _ArtifactMaterialization,
    *,
    artifact_reference: OptimizationArtifactReference | None,
    lookup_hash: str,
    coordination_status: ArtifactCreationCoordinationStatus | None,
    llm_transform_invoked: bool,
) -> NexusUCLResolution:
    return NexusUCLResolution(
        decision=resolution.decision,
        messages=resolution.messages,
        fragments_included=resolution.fragments_included,
        fragments_excluded=resolution.fragments_excluded,
        artifact_reference=artifact_reference,
        artifact_lookup_key_hash=lookup_hash,
        coordination_status=coordination_status,
        llm_transform_invoked=llm_transform_invoked,
    )


def _select_strategy(
    plan_strategy_ids: tuple[str, ...],
    policy_strategy_ids: tuple[str, ...],
    strategy_versions: Mapping[str, str],
) -> tuple[str, str]:
    policy_allowed = set(policy_strategy_ids)
    for strategy_id in plan_strategy_ids:
        if strategy_id not in policy_allowed:
            continue
        version = strategy_versions.get(strategy_id)
        if version:
            return strategy_id, version
    raise NexusUCLExecutionError(NexusUCLExecutionReason.STRATEGY_VERSION_UNAVAILABLE)


def _materialize_plan_mapping(
    *,
    context_plan: ContextPlan,
    messages_for_compile: Sequence[ChatMessage],
    fragment_messages: Sequence[ChatMessage],
    ranked_fragments: Sequence[ContextFragment],
) -> tuple[dict[str, str], tuple[str, ...]]:
    if len(fragment_messages) != len(ranked_fragments):
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)

    group_id_by_source_ref: dict[str, str] = {}
    for group in context_plan.source_groups:
        for source_ref in group.source_refs:
            if source_ref in group_id_by_source_ref:
                raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
            group_id_by_source_ref[source_ref] = group.group_id

    fragment_index_by_entry_id = {
        message.entry_id: index for index, message in enumerate(fragment_messages)
    }
    plan_group_ids = {group.group_id for group in context_plan.source_groups}
    message_group_ids: list[str] = []
    present_source_refs_by_group_id: dict[str, list[str]] = {}
    seen_group_order: list[str] = []
    seen_group_set: set[str] = set()

    for message in messages_for_compile:
        if message.entry_id in fragment_index_by_entry_id:
            fragment = ranked_fragments[fragment_index_by_entry_id[message.entry_id]]
            if fragment.source is ContextFragmentSource.SESSION_HISTORY:
                source_ref = str(fragment.metadata.get("message_id") or "").strip()
                if not source_ref:
                    raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
            else:
                source_ref = fragment.fragment_id
        else:
            source_ref = message.entry_id
            if not isinstance(source_ref, str) or not source_ref:
                raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)

        group_id = group_id_by_source_ref.get(source_ref)
        if group_id is None or group_id not in plan_group_ids:
            raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
        present_source_refs_by_group_id.setdefault(group_id, []).append(source_ref)
        message_group_ids.append(group_id)
        if group_id not in seen_group_set:
            seen_group_order.append(group_id)
            seen_group_set.add(group_id)

    for group in context_plan.source_groups:
        if tuple(present_source_refs_by_group_id.get(group.group_id, ())) != group.source_refs:
            raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)

    expected_order = [group.group_id for group in context_plan.source_groups]
    if seen_group_order != expected_order:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)

    return group_id_by_source_ref, tuple(message_group_ids)


def _resolve_selection_only(
    *,
    context_plan: ContextPlan,
    messages_for_compile: Sequence[ChatMessage],
    fragment_messages: Sequence[ChatMessage],
    ranked_fragments: Sequence[ContextFragment],
    message_group_ids: tuple[str, ...],
    excluded_group_ids: set[str],
) -> NexusUCLResolution:
    selected_messages = [
        message
        for message, group_id in zip(messages_for_compile, message_group_ids, strict=True)
        if group_id not in excluded_group_ids
    ]
    fragments_included: list[ContextFragment] = []
    fragments_excluded: list[tuple[ContextFragment, str]] = []
    fragment_group_ids = _fragment_group_ids_by_plan(
        fragment_messages=fragment_messages,
        ranked_fragments=ranked_fragments,
        context_plan=context_plan,
    )
    for fragment, group_id in zip(ranked_fragments, fragment_group_ids, strict=True):
        if group_id in excluded_group_ids:
            fragments_excluded.append((fragment, "context_plan_excluded"))
        else:
            fragments_included.append(fragment)

    if excluded_group_ids:
        decision = ContextOptimizationDecision.SELECT_ONLY
    else:
        decision = ContextOptimizationDecision.NO_OP

    return NexusUCLResolution(
        decision=decision,
        messages=tuple(selected_messages),
        fragments_included=tuple(fragments_included),
        fragments_excluded=tuple(fragments_excluded),
        artifact_reference=None,
        artifact_lookup_key_hash=None,
        coordination_status=None,
        llm_transform_invoked=False,
    )


def _build_executor_source_inputs(
    *,
    context_plan: ContextPlan,
    requirement: Any,
    session_history: SessionHistorySnapshot,
) -> tuple[tuple[SessionHistoryMessage, ...], tuple[MessageSequenceArtifactSourceGroupProof, ...]]:
    groups_by_id = {group.group_id: group for group in context_plan.source_groups}
    source_refs: list[str] = []
    proofs: list[MessageSequenceArtifactSourceGroupProof] = []
    for group_id in requirement.source_group_ids:
        group = groups_by_id.get(group_id)
        if group is None:
            raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
        proofs.append(
            MessageSequenceArtifactSourceGroupProof(
                source_refs=group.source_refs,
                source_content_hash=group.source_content_hash,
            )
        )
        source_refs.extend(group.source_refs)

    messages_by_id = {message.message_id: message for message in session_history.messages}
    if len(messages_by_id) != len(session_history.messages):
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
    source_messages: list[SessionHistoryMessage] = []
    for source_ref in source_refs:
        message = messages_by_id.get(source_ref)
        if message is None:
            raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
        source_messages.append(message)
    return tuple(source_messages), tuple(proofs)


def _enforce_ucl_policy_gate(
    optimization_policy: ContextOptimizationPolicy,
    requirement: Any,
) -> None:
    if not optimization_policy.enabled:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.POLICY_BLOCKED)
    if optimization_policy.mode is not ContextOptimizationMode.EPHEMERAL_ASSEMBLY:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.POLICY_BLOCKED)
    if not optimization_policy.allow_artifact_reuse:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.POLICY_BLOCKED)
    if not optimization_policy.allow_lossy:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.POLICY_BLOCKED)
    if not optimization_policy.allow_llm_summarization:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.POLICY_BLOCKED)

    lookup_inputs = requirement.lookup_inputs
    if lookup_inputs.artifact_type is not OptimizationArtifactType.MESSAGE_SEQUENCE:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.POLICY_BLOCKED)
    if OptimizationArtifactType.MESSAGE_SEQUENCE not in optimization_policy.allowed_artifact_types:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.POLICY_BLOCKED)
    if lookup_inputs.lossiness_profile != "lossy":
        raise NexusUCLExecutionError(NexusUCLExecutionReason.POLICY_BLOCKED)


def _allocate_artifact_id(factory: Callable[[], str]) -> str:
    try:
        value = factory()
    except NexusUCLExecutionError:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.ARTIFACT_CREATION_FAILED) from None
    except Exception:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.ARTIFACT_CREATION_FAILED) from None
    if not isinstance(value, str) or not value.strip():
        raise NexusUCLExecutionError(NexusUCLExecutionReason.ARTIFACT_CREATION_FAILED)
    return value


def _prepare_artifact_materialization(
    *,
    context_plan: ContextPlan,
    requirement: Any,
    messages_for_compile: Sequence[ChatMessage],
    fragment_messages: Sequence[ChatMessage],
    ranked_fragments: Sequence[ContextFragment],
    message_group_ids: tuple[str, ...],
    excluded_group_ids: set[str],
) -> _PreparedArtifactMaterialization:
    target_group_ids_list = list(requirement.source_group_ids)
    if not target_group_ids_list:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)

    groups_by_id = {group.group_id: group for group in context_plan.source_groups}
    selected_group_ids = set(context_plan.selected_group_ids)
    target_group_ids_set = set(target_group_ids_list)

    for group_id in target_group_ids_list:
        if group_id not in groups_by_id:
            raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
        if group_id not in selected_group_ids:
            raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
        if group_id in excluded_group_ids:
            raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)

    filtered_messages: list[ChatMessage] = []
    filtered_group_ids: list[str] = []
    for message, group_id in zip(messages_for_compile, message_group_ids, strict=True):
        if group_id in excluded_group_ids:
            continue
        filtered_messages.append(message)
        filtered_group_ids.append(group_id)

    for group_id in filtered_group_ids:
        if group_id in excluded_group_ids:
            raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)

    for group_id in target_group_ids_list:
        if group_id not in filtered_group_ids:
            raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)

    target_indices = [
        index
        for index, group_id in enumerate(filtered_group_ids)
        if group_id in target_group_ids_set
    ]
    if not target_indices:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
    first_target = min(target_indices)
    last_target = max(target_indices)
    for index in range(first_target, last_target + 1):
        if filtered_group_ids[index] not in target_group_ids_set:
            raise NexusUCLExecutionError(NexusUCLExecutionReason.NON_CONTIGUOUS_ARTIFACT_TARGET)

    fragment_group_map = _fragment_group_ids_by_plan(
        fragment_messages=fragment_messages,
        ranked_fragments=ranked_fragments,
        context_plan=context_plan,
    )
    fragments_included: list[ContextFragment] = []
    fragments_excluded: list[tuple[ContextFragment, str]] = []
    for fragment, group_id in zip(ranked_fragments, fragment_group_map, strict=True):
        if group_id in excluded_group_ids:
            fragments_excluded.append((fragment, "context_plan_excluded"))
        elif group_id in target_group_ids_set:
            fragments_excluded.append((fragment, "replaced_by_optimization_artifact"))
        else:
            fragments_included.append(fragment)

    return _PreparedArtifactMaterialization(
        filtered_messages=tuple(filtered_messages),
        filtered_group_ids=tuple(filtered_group_ids),
        target_group_ids=tuple(target_group_ids_list),
        first_target_index=first_target,
        fragments_included=tuple(fragments_included),
        fragments_excluded=tuple(fragments_excluded),
    )


def _materialize_artifact_messages(
    *,
    prepared: _PreparedArtifactMaterialization,
    decision: ContextOptimizationDecision,
    lookup_hash: str,
    summary: str,
    artifact_content_hash: str,
    artifact_id: str | None,
    count_tokens: Callable[[str], int],
) -> _ArtifactMaterialization:
    target_group_ids_set = set(prepared.target_group_ids)
    replacement_metadata: dict[str, Any] = {
        "artifact_lookup_key_hash": lookup_hash,
        "artifact_content_hash": artifact_content_hash,
        "optimization_decision": decision.value,
    }
    if artifact_id is not None:
        replacement_metadata["artifact_id"] = artifact_id

    replacement_message = ChatMessage(
        role="system",
        content=summary,
        entry_id=f"ucl-artifact-{lookup_hash}",
        metadata=replacement_metadata,
    )

    final_messages: list[ChatMessage] = []
    inserted = False
    for index, message in enumerate(prepared.filtered_messages):
        group_id = prepared.filtered_group_ids[index]
        if group_id in target_group_ids_set:
            if not inserted and index == prepared.first_target_index:
                final_messages.append(replacement_message)
                inserted = True
            continue
        final_messages.append(message)
    if not inserted:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)

    synthetic_metadata: dict[str, Any] = {
        "artifact_lookup_key_hash": lookup_hash,
        "artifact_content_hash": artifact_content_hash,
        "optimization_decision": decision.value,
    }
    synthetic_fragment = ContextFragment(
        fragment_id=f"ucl-artifact-{lookup_hash}",
        source=ContextFragmentSource.SESSION_HISTORY,
        source_id=artifact_id or lookup_hash,
        content=summary,
        token_estimate=count_tokens(summary),
        relevance_score=0.0,
        freshness_score=0.0,
        confidence_score=0.0,
        mandatory=False,
        metadata=synthetic_metadata,
    )
    fragments_included = (*prepared.fragments_included, synthetic_fragment)

    return _ArtifactMaterialization(
        decision=decision,
        messages=tuple(final_messages),
        fragments_included=fragments_included,
        fragments_excluded=prepared.fragments_excluded,
    )


def _fragment_group_ids_by_plan(
    *,
    fragment_messages: Sequence[ChatMessage],
    ranked_fragments: Sequence[ContextFragment],
    context_plan: ContextPlan,
) -> tuple[str, ...]:
    group_id_by_source_ref: dict[str, str] = {}
    for group in context_plan.source_groups:
        for source_ref in group.source_refs:
            group_id_by_source_ref[source_ref] = group.group_id

    group_ids: list[str] = []
    for fragment in ranked_fragments:
        if fragment.source is ContextFragmentSource.SESSION_HISTORY:
            source_ref = str(fragment.metadata.get("message_id") or "").strip()
        else:
            source_ref = fragment.fragment_id
        group_id = group_id_by_source_ref.get(source_ref)
        if group_id is None:
            raise NexusUCLExecutionError(NexusUCLExecutionReason.PLAN_MATERIALIZATION_FAILED)
        group_ids.append(group_id)
    return tuple(group_ids)


def _validate_stored_artifact_payload(
    stored: StoredOptimizationArtifact,
    *,
    lookup_key: ArtifactLookupKey,
) -> tuple[str, str]:
    try:
        return validate_stored_message_sequence_artifact(stored, lookup_key=lookup_key)
    except MessageSequenceArtifactValidationError:
        raise NexusUCLExecutionError(NexusUCLExecutionReason.ARTIFACT_PAYLOAD_INVALID) from None


def _validate_execution_payload(
    payload: bytes,
    artifact_content_hash: str,
    *,
    lookup_key: ArtifactLookupKey,
) -> tuple[str, str]:
    try:
        return validate_message_sequence_payload(
            payload=payload,
            media_type=_MEDIA_TYPE,
            encoding=_ENCODING,
            artifact_content_hash=artifact_content_hash,
            lookup_key=lookup_key,
        )
    except MessageSequenceArtifactValidationError:
        raise NexusUCLExecutionError(
            NexusUCLExecutionReason.ARTIFACT_PAYLOAD_INVALID
        ) from None
