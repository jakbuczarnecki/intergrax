# © Artur Czarnecki. All rights reserved.

"""Deterministic context planner (CTX-UCL-3)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
)
from intergrax.context.planning import (
    MANDATORY_CONTEXT_EXCEEDS_MODEL_LIMIT,
    NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET,
    ContextArtifactLookupInputs,
    ContextArtifactRequirement,
    ContextMinimumPreservationRequirements,
    ContextPlan,
    ContextPlanningError,
    ContextSourceBudgetAllocation,
    ContextSourceGroup,
    budget_class_for_execution_scope,
)
from intergrax.context.session_history import SessionHistorySnapshot
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCompressionTarget,
    ContextOptimizationPolicy,
    ModelCallExecutionScope,
    OptimizationArtifactType,
)

_DROPPABLE_SOURCES = frozenset(
    {
        ContextFragmentSource.RAG,
        ContextFragmentSource.WEBSEARCH,
        ContextFragmentSource.LONGTERM_MEMORY,
        ContextFragmentSource.ATTACHMENT,
        ContextFragmentSource.SHARED_CONTEXT,
        ContextFragmentSource.GRAPH_PRIOR,
        ContextFragmentSource.WORKSPACE,
        ContextFragmentSource.CUSTOM,
    }
)

_REQUIRED_SOURCES = frozenset(
    {
        ContextFragmentSource.TASK_MESSAGE,
        ContextFragmentSource.SYSTEM_INSTRUCTIONS,
        ContextFragmentSource.POLICY_OVERLAY,
    }
)

_FINAL_VALIDATION_REQUIREMENTS: tuple[str, ...] = (
    "preserve_message_order",
    "preserve_roles",
    "preserve_message_ids",
    "preserve_tool_call_links",
    "respect_resolved_global_budget",
    "run_context_preflight",
)


def _deterministic_group_id(source_refs: tuple[str, ...]) -> str:
    joined = ":".join(source_refs)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"grp-{digest[:16]}"


def _group_content_hash(source_refs: tuple[str, ...], content_hashes: Sequence[str]) -> str:
  payload = "|".join(f"{ref}:{content_hash}" for ref, content_hash in zip(source_refs, content_hashes, strict=True))
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_tool_call_ids(tool_calls: tuple[object, ...]) -> tuple[str, ...]:
    ids: list[str] = []
    for call in tool_calls:
        if isinstance(call, dict):
            call_id = call.get("id")
            if call_id is not None and str(call_id).strip():
                ids.append(str(call_id).strip())
    return tuple(ids)


def group_session_history_snapshot(
    snapshot: SessionHistorySnapshot,
    *,
    count_tokens: Callable[[str], int],
) -> tuple[ContextSourceGroup, ...]:
    """Deterministic session-history grouping rules."""
    messages = snapshot.messages
    groups: list[ContextSourceGroup] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "system":
            refs = (message.message_id,)
            groups.append(
                ContextSourceGroup(
                    group_id=_deterministic_group_id(refs),
                    source=ContextFragmentSource.SESSION_HISTORY,
                    source_refs=refs,
                    source_content_hash=_group_content_hash(refs, (message.content_hash,)),
                    token_estimate=count_tokens(message.content),
                    start_sequence=message.sequence,
                    end_sequence=message.sequence,
                    protected=True,
                )
            )
            index += 1
            continue

        if message.role == "tool":
            refs = (message.message_id,)
            groups.append(
                ContextSourceGroup(
                    group_id=_deterministic_group_id(refs),
                    source=ContextFragmentSource.SESSION_HISTORY,
                    source_refs=refs,
                    source_content_hash=_group_content_hash(refs, (message.content_hash,)),
                    token_estimate=count_tokens(message.content),
                    start_sequence=message.sequence,
                    end_sequence=message.sequence,
                    protected=True,
                )
            )
            index += 1
            continue

        if message.role == "user":
            turn_refs: list[str] = [message.message_id]
            turn_hashes: list[str] = [message.content_hash]
            turn_tokens = count_tokens(message.content)
            start_sequence = message.sequence
            end_sequence = message.sequence
            index += 1
            while index < len(messages) and messages[index].role != "user":
                current = messages[index]
                if current.role == "assistant" and current.tool_calls:
                    if turn_refs != [message.message_id] or turn_tokens > count_tokens(message.content):
                        refs_tuple = tuple(turn_refs)
                        groups.append(
                            ContextSourceGroup(
                                group_id=_deterministic_group_id(refs_tuple),
                                source=ContextFragmentSource.SESSION_HISTORY,
                                source_refs=refs_tuple,
                                source_content_hash=_group_content_hash(refs_tuple, tuple(turn_hashes)),
                                token_estimate=turn_tokens,
                                start_sequence=start_sequence,
                                end_sequence=end_sequence,
                            )
                        )
                        turn_refs = []
                        turn_hashes = []
                        turn_tokens = 0

                    tool_refs = [current.message_id]
                    tool_hashes = [current.content_hash]
                    tool_tokens = count_tokens(current.content)
                    tool_start = current.sequence
                    tool_end = current.sequence
                    expected_tool_ids = set(_extract_tool_call_ids(current.tool_calls))
                    index += 1
                    received_tool_ids: set[str] = set()
                    while index < len(messages) and messages[index].role == "tool":
                        tool_message = messages[index]
                        if tool_message.tool_call_id not in expected_tool_ids:
                            break
                        tool_refs.append(tool_message.message_id)
                        tool_hashes.append(tool_message.content_hash)
                        tool_tokens += count_tokens(tool_message.content)
                        tool_end = tool_message.sequence
                        received_tool_ids.add(tool_message.tool_call_id)
                        index += 1
                    incomplete = expected_tool_ids != received_tool_ids
                    refs_tuple = tuple(tool_refs)
                    groups.append(
                        ContextSourceGroup(
                            group_id=_deterministic_group_id(refs_tuple),
                            source=ContextFragmentSource.SESSION_HISTORY,
                            source_refs=refs_tuple,
                            source_content_hash=_group_content_hash(refs_tuple, tuple(tool_hashes)),
                            token_estimate=tool_tokens,
                            start_sequence=tool_start,
                            end_sequence=tool_end,
                            protected=True,
                            required=incomplete,
                        )
                    )
                    continue

                if current.role == "assistant":
                    turn_refs.append(current.message_id)
                    turn_hashes.append(current.content_hash)
                    turn_tokens += count_tokens(current.content)
                    end_sequence = current.sequence
                    index += 1
                    continue
                break

            if turn_refs:
                refs_tuple = tuple(turn_refs)
                groups.append(
                    ContextSourceGroup(
                        group_id=_deterministic_group_id(refs_tuple),
                        source=ContextFragmentSource.SESSION_HISTORY,
                        source_refs=refs_tuple,
                        source_content_hash=_group_content_hash(refs_tuple, tuple(turn_hashes)),
                        token_estimate=turn_tokens,
                        start_sequence=start_sequence,
                        end_sequence=end_sequence,
                    )
                )
            continue

        if message.role == "assistant":
            if message.tool_calls:
                tool_refs = [message.message_id]
                tool_hashes = [message.content_hash]
                tool_tokens = count_tokens(message.content)
                tool_start = message.sequence
                tool_end = message.sequence
                expected_tool_ids = set(_extract_tool_call_ids(message.tool_calls))
                index += 1
                received_tool_ids: set[str] = set()
                while index < len(messages) and messages[index].role == "tool":
                    tool_message = messages[index]
                    if tool_message.tool_call_id not in expected_tool_ids:
                        break
                    tool_refs.append(tool_message.message_id)
                    tool_hashes.append(tool_message.content_hash)
                    tool_tokens += count_tokens(tool_message.content)
                    tool_end = tool_message.sequence
                    received_tool_ids.add(tool_message.tool_call_id)
                    index += 1
                incomplete = expected_tool_ids != received_tool_ids
                refs_tuple = tuple(tool_refs)
                groups.append(
                    ContextSourceGroup(
                        group_id=_deterministic_group_id(refs_tuple),
                        source=ContextFragmentSource.SESSION_HISTORY,
                        source_refs=refs_tuple,
                        source_content_hash=_group_content_hash(refs_tuple, tuple(tool_hashes)),
                        token_estimate=tool_tokens,
                        start_sequence=tool_start,
                        end_sequence=tool_end,
                        protected=True,
                        required=incomplete,
                    )
                )
                continue

            refs = (message.message_id,)
            groups.append(
                ContextSourceGroup(
                    group_id=_deterministic_group_id(refs),
                    source=ContextFragmentSource.SESSION_HISTORY,
                    source_refs=refs,
                    source_content_hash=_group_content_hash(refs, (message.content_hash,)),
                    token_estimate=count_tokens(message.content),
                    start_sequence=message.sequence,
                    end_sequence=message.sequence,
                    protected=True,
                )
            )
            index += 1
            continue

        index += 1

    return tuple(groups)


def _fragment_groups(
    fragments: Sequence[ContextFragment],
    *,
    count_tokens: Callable[[str], int],
) -> tuple[ContextSourceGroup, ...]:
    groups: list[ContextSourceGroup] = []
    for fragment in fragments:
        if fragment.source is ContextFragmentSource.SESSION_HISTORY:
            continue
        refs = (fragment.fragment_id,)
        groups.append(
            ContextSourceGroup(
                group_id=_deterministic_group_id(refs),
                source=fragment.source,
                source_refs=refs,
                source_content_hash=fragment.content_hash or hashlib.sha256(fragment.content.encode()).hexdigest(),
                token_estimate=fragment.token_estimate or count_tokens(fragment.content),
                required=fragment.mandatory or fragment.source in _REQUIRED_SOURCES,
                protected=fragment.mandatory or fragment.source in _REQUIRED_SOURCES,
                droppable=(
                    not fragment.mandatory
                    and fragment.source in _DROPPABLE_SOURCES
                ),
            )
        )
    return tuple(groups)


def _apply_recent_tail_protection(
    session_groups: list[ContextSourceGroup],
    *,
    recent_tail_min_messages: int,
) -> set[str]:
    if recent_tail_min_messages <= 0 or not session_groups:
        return set()
    protected_ids: set[str] = set()
    remaining = recent_tail_min_messages
    for group in reversed(session_groups):
        if remaining <= 0:
            break
        message_count = len(group.source_refs)
        protected_ids.add(group.group_id)
        remaining -= message_count
    return protected_ids


def _mark_session_groups(
    session_groups: list[ContextSourceGroup],
    *,
    recent_tail_ids: set[str],
) -> list[ContextSourceGroup]:
    marked: list[ContextSourceGroup] = []
    for group in session_groups:
        protected = group.protected or group.group_id in recent_tail_ids
        compressible = (
            group.source is ContextFragmentSource.SESSION_HISTORY
            and not protected
            and not group.required
        )
        marked.append(
            ContextSourceGroup(
                group_id=group.group_id,
                source=group.source,
                source_refs=group.source_refs,
                source_content_hash=group.source_content_hash,
                token_estimate=group.token_estimate,
                start_sequence=group.start_sequence,
                end_sequence=group.end_sequence,
                required=group.required,
                protected=protected,
                compressible=compressible,
                droppable=group.droppable,
                trim_safe=False,
            )
        )
    return marked


def _total_tokens(group_ids: Sequence[str], groups_by_id: dict[str, ContextSourceGroup]) -> int:
    return sum(groups_by_id[group_id].token_estimate for group_id in group_ids)


def _source_allocations(
    selected_ids: tuple[str, ...],
    excluded_ids: tuple[str, ...],
    groups_by_id: dict[str, ContextSourceGroup],
) -> tuple[ContextSourceBudgetAllocation, ...]:
    by_source: dict[ContextFragmentSource, list[str]] = {}
    for group_id in selected_ids:
        source = groups_by_id[group_id].source
        by_source.setdefault(source, []).append(group_id)

    allocations: list[ContextSourceBudgetAllocation] = []
    for source in sorted(by_source, key=lambda item: item.value):
        selected = tuple(by_source[source])
        excluded_for_source = tuple(
            group_id
            for group_id in excluded_ids
            if groups_by_id[group_id].source is source
        )
        allocated = sum(groups_by_id[group_id].token_estimate for group_id in selected)
        allocations.append(
            ContextSourceBudgetAllocation(
                source=source,
                allocated_tokens=allocated,
                selected_group_ids=selected,
                excluded_group_ids=excluded_for_source,
            )
        )
    return tuple(allocations)


class ContextPlanner:
    """Injectable CE planner — no repository, executor, or LLM access."""

    def __init__(
        self,
        *,
        count_tokens: Callable[[str], int],
    ) -> None:
        self._count_tokens = count_tokens

    def plan(
        self,
        request: ContextAssemblyRequest,
        *,
        messages_for_compile: Sequence[ChatMessage],
        ranked_fragments: Sequence[ContextFragment],
        session_history: SessionHistorySnapshot | None,
        resolved_global_budget_tokens: int,
        optimization_policy: ContextOptimizationPolicy | None = None,
        model_family: str | None = None,
        locale: str | None = None,
    ) -> ContextPlan:
        _ = messages_for_compile
        execution_scope = request.execution_scope
        if not isinstance(execution_scope, ModelCallExecutionScope):
            raise ValueError("execution_scope must be ModelCallExecutionScope")

        session_groups: list[ContextSourceGroup] = []
        if session_history is not None:
            session_groups = list(
                group_session_history_snapshot(session_history, count_tokens=self._count_tokens)
            )

        fragment_groups = list(
            _fragment_groups(ranked_fragments, count_tokens=self._count_tokens)
        )
        all_groups = session_groups + fragment_groups
        groups_by_id = {group.group_id: group for group in all_groups}

        recent_tail = optimization_policy.recent_tail_min_messages if optimization_policy else 0
        recent_tail_ids = _apply_recent_tail_protection(session_groups, recent_tail_min_messages=recent_tail)
        session_groups = _mark_session_groups(session_groups, recent_tail_ids=recent_tail_ids)
        all_groups = session_groups + fragment_groups
        groups_by_id = {group.group_id: group for group in all_groups}

        selected_ids = tuple(group.group_id for group in all_groups)
        excluded_ids: tuple[str, ...] = ()
        required_ids = tuple(group.group_id for group in all_groups if group.required)
        protected_ids = tuple(group.group_id for group in all_groups if group.protected)
        compressible_ids = tuple(group.group_id for group in all_groups if group.compressible)
        droppable_ids = tuple(group.group_id for group in all_groups if group.droppable)
        trim_safe_ids = tuple(group.group_id for group in all_groups if group.trim_safe)

        mandatory_tokens = _total_tokens(
            tuple(group.group_id for group in all_groups if group.required or group.protected),
            groups_by_id,
        )
        if mandatory_tokens > resolved_global_budget_tokens:
            raise ContextPlanningError(MANDATORY_CONTEXT_EXCEEDS_MODEL_LIMIT)

        estimated_total = _total_tokens(selected_ids, groups_by_id)
        if estimated_total <= resolved_global_budget_tokens:
            return self._build_plan(
                request=request,
                execution_scope=execution_scope,
                resolved_global_budget_tokens=resolved_global_budget_tokens,
                estimated_total_tokens=estimated_total,
                all_groups=all_groups,
                selected_ids=selected_ids,
                excluded_ids=excluded_ids,
                required_ids=required_ids,
                protected_ids=protected_ids,
                compressible_ids=compressible_ids,
                droppable_ids=droppable_ids,
                trim_safe_ids=trim_safe_ids,
                optimization_required=False,
                artifact_requirement=None,
                groups_by_id=groups_by_id,
            )

        droppable_ordered = tuple(
            group.group_id for group in all_groups if group.droppable
        )
        selected_set = set(selected_ids)
        for group_id in droppable_ordered:
            selected_set.remove(group_id)
            excluded_ids = tuple(sorted(set(excluded_ids) | {group_id}))
            estimated_total = _total_tokens(tuple(selected_set), groups_by_id)
            if estimated_total <= resolved_global_budget_tokens:
                selected_ids = tuple(group.group_id for group in all_groups if group.group_id in selected_set)
                return self._build_plan(
                    request=request,
                    execution_scope=execution_scope,
                    resolved_global_budget_tokens=resolved_global_budget_tokens,
                    estimated_total_tokens=estimated_total,
                    all_groups=all_groups,
                    selected_ids=selected_ids,
                    excluded_ids=excluded_ids,
                    required_ids=required_ids,
                    protected_ids=protected_ids,
                    compressible_ids=compressible_ids,
                    droppable_ids=droppable_ids,
                    trim_safe_ids=trim_safe_ids,
                    optimization_required=False,
                    artifact_requirement=None,
                    groups_by_id=groups_by_id,
                )

        target_groups: list[ContextSourceGroup] = []
        started = False
        for group in session_groups:
            if group.compressible and group.group_id in selected_set:
                started = True
                target_groups.append(group)
            elif started:
                break
        if not target_groups:
            raise ContextPlanningError(NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET)

        target_group_ids = tuple(group.group_id for group in target_groups)
        target_source_refs: list[str] = []
        for group in target_groups:
            target_source_refs.extend(group.source_refs)

        source_content_hash = hashlib.sha256(
            "|".join(group.source_content_hash for group in target_groups).encode("utf-8")
        ).hexdigest()

        non_target_tokens = sum(
            groups_by_id[group_id].token_estimate
            for group_id in selected_set
            if group_id not in set(target_group_ids)
        )
        target_token_budget = max(
            1,
            resolved_global_budget_tokens - non_target_tokens,
        )

        allowed_strategy_ids: tuple[str, ...] = ()
        if optimization_policy is not None:
            allowed_strategy_ids = optimization_policy.allowed_strategy_ids

        lossiness_profile = "lossless"
        if optimization_policy is not None and optimization_policy.allow_lossy:
            lossiness_profile = "lossy"

        protected_policy_version = None
        if optimization_policy is not None:
            protected_policy_version = optimization_policy.protected_region_policy_version

        lookup_inputs = ContextArtifactLookupInputs(
            tenant_id=session_history.tenant_id if session_history else request.tenant_id,
            context_scope_id=(
                session_history.context_scope_id if session_history else request.task_id
            ),
            artifact_type=OptimizationArtifactType.MESSAGE_SEQUENCE,
            source_content_hash=source_content_hash,
            compression_target=ArtifactCompressionTarget(target_tokens=target_token_budget),
            lossiness_profile=lossiness_profile,
            source_refs=tuple(target_source_refs),
            protected_region_policy_version=protected_policy_version,
            model_family=model_family,
            locale=locale,
        )

        preservation = ContextMinimumPreservationRequirements(
            preserve_message_order=True,
            preserve_roles=True,
            preserve_message_ids=True,
            preserve_tool_call_links=True,
            preserve_recent_tail_messages=recent_tail,
            required_group_ids=required_ids,
            protected_group_ids=protected_ids,
        )

        artifact_requirement = ContextArtifactRequirement(
            lookup_inputs=lookup_inputs,
            source_group_ids=target_group_ids,
            allowed_strategy_ids=allowed_strategy_ids,
            minimum_preservation=preservation,
        )

        return self._build_plan(
            request=request,
            execution_scope=execution_scope,
            resolved_global_budget_tokens=resolved_global_budget_tokens,
            estimated_total_tokens=estimated_total,
            all_groups=all_groups,
            selected_ids=selected_ids,
            excluded_ids=excluded_ids,
            required_ids=required_ids,
            protected_ids=protected_ids,
            compressible_ids=compressible_ids,
            droppable_ids=droppable_ids,
            trim_safe_ids=trim_safe_ids,
            optimization_required=True,
            artifact_requirement=artifact_requirement,
            groups_by_id=groups_by_id,
        )

    def _build_plan(
        self,
        *,
        request: ContextAssemblyRequest,
        execution_scope: ModelCallExecutionScope,
        resolved_global_budget_tokens: int,
        estimated_total_tokens: int,
        all_groups: list[ContextSourceGroup],
        selected_ids: tuple[str, ...],
        excluded_ids: tuple[str, ...],
        required_ids: tuple[str, ...],
        protected_ids: tuple[str, ...],
        compressible_ids: tuple[str, ...],
        droppable_ids: tuple[str, ...],
        trim_safe_ids: tuple[str, ...],
        optimization_required: bool,
        artifact_requirement: ContextArtifactRequirement | None,
        groups_by_id: dict[str, ContextSourceGroup],
    ) -> ContextPlan:
        _ = request
        return ContextPlan(
            execution_scope=execution_scope,
            budget_class=budget_class_for_execution_scope(execution_scope),
            resolved_global_budget_tokens=resolved_global_budget_tokens,
            estimated_total_tokens=estimated_total_tokens,
            source_groups=tuple(all_groups),
            source_allocations=_source_allocations(selected_ids, excluded_ids, groups_by_id),
            selected_group_ids=selected_ids,
            excluded_group_ids=excluded_ids,
            required_group_ids=required_ids,
            protected_group_ids=protected_ids,
            compressible_group_ids=compressible_ids,
            droppable_group_ids=droppable_ids,
            trim_safe_group_ids=trim_safe_ids,
            optimization_required=optimization_required,
            artifact_requirement=artifact_requirement,
            final_validation_requirements=_FINAL_VALIDATION_REQUIREMENTS,
        )
