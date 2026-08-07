# © Artur Czarnecki. All rights reserved.

"""Deterministic context planner (CTX-UCL-3)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    content_hash_for_text,
)
from intergrax.context.planning import (
    MANDATORY_CONTEXT_EXCEEDS_MODEL_LIMIT,
    NO_ALLOWED_CONTEXT_OPTIMIZATION_STRATEGY,
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
from intergrax.context.session_history import SessionHistorySnapshot, session_history_message_to_chat_message
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


def _strict_tool_call_id(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _extract_tool_call_ids(tool_calls: tuple[object, ...] | list[object] | None) -> tuple[str, ...]:
    ids: list[str] = []
    if not tool_calls:
        return ()
    for call in tool_calls:
        if not isinstance(call, Mapping):
            return ()
        call_id = _strict_tool_call_id(call.get("id"))
        if call_id is None:
            return ()
        ids.append(call_id)
    return tuple(ids)


def _tool_call_group_complete(
    assistant: ChatMessage,
    tool_messages: Sequence[ChatMessage],
) -> bool:
    tool_calls = assistant.tool_calls
    if not tool_calls:
        return False
    call_ids: list[str] = []
    for call in tool_calls:
        if not isinstance(call, Mapping):
            return False
        call_id = _strict_tool_call_id(call.get("id"))
        if call_id is None:
            return False
        call_ids.append(call_id)
    if not call_ids:
        return False
    if len(call_ids) != len(set(call_ids)):
        return False
    received: dict[str, int] = {}
    for tool_message in tool_messages:
        if tool_message.role != "tool":
            return False
        call_id = _strict_tool_call_id(tool_message.tool_call_id)
        if call_id is None:
            return False
        if call_id not in set(call_ids):
            return False
        if call_id in received:
            return False
        received[call_id] = 1
    return set(call_ids) == set(received.keys())


def _base_message_group_id(message: ChatMessage, position: int) -> str:
    content_hash = content_hash_for_text(message.content or "")
    identity = f"{message.entry_id}:{position}:{message.role}:{content_hash}"
    return _deterministic_group_id((identity,))


def _base_message_content_hashes(messages: Sequence[ChatMessage]) -> tuple[str, ...]:
    return tuple(content_hash_for_text(message.content or "") for message in messages)


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
                    required=True,
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
                        turn_refs = []
                        turn_hashes = []
                        turn_tokens = 0

                    tool_refs = [current.message_id]
                    tool_hashes = [current.content_hash]
                    tool_tokens = count_tokens(current.content)
                    tool_start = current.sequence
                    tool_end = current.sequence
                    index += 1
                    tool_messages: list = []
                    while index < len(messages) and messages[index].role == "tool":
                        tool_message = messages[index]
                        tool_messages.append(tool_message)
                        tool_refs.append(tool_message.message_id)
                        tool_hashes.append(tool_message.content_hash)
                        tool_tokens += count_tokens(tool_message.content)
                        tool_end = tool_message.sequence
                        index += 1
                    assistant_msg = session_history_message_to_chat_message(current)
                    complete = _tool_call_group_complete(assistant_msg, tool_messages)
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
                            protected=not complete,
                            required=not complete,
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
                index += 1
                tool_messages = []
                while index < len(messages) and messages[index].role == "tool":
                    tool_message = messages[index]
                    tool_messages.append(tool_message)
                    tool_refs.append(tool_message.message_id)
                    tool_hashes.append(tool_message.content_hash)
                    tool_tokens += count_tokens(tool_message.content)
                    tool_end = tool_message.sequence
                    index += 1
                assistant_msg = session_history_message_to_chat_message(message)
                complete = _tool_call_group_complete(assistant_msg, tool_messages)
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
                        protected=not complete,
                        required=not complete,
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


def _group_base_messages(
    messages_for_compile: Sequence[ChatMessage],
    fragment_entry_ids: set[str],
    *,
    count_tokens: Callable[[str], int],
) -> tuple[list[ContextSourceGroup], set[int], dict[int, str]]:
    groups: list[ContextSourceGroup] = []
    assigned_indices: set[int] = set()
    base_group_id_by_message_index: dict[int, str] = {}
    last_base_user_index = -1
    for index, message in enumerate(messages_for_compile):
        if message.entry_id in fragment_entry_ids:
            continue
        if message.role == "user":
            last_base_user_index = index

    index = 0
    while index < len(messages_for_compile):
        message = messages_for_compile[index]
        if message.entry_id in fragment_entry_ids:
            index += 1
            continue

        if message.role == "assistant" and message.tool_calls:
            refs = [message.entry_id]
            hashes = [content_hash_for_text(message.content or "")]
            token_total = count_tokens(message.content or "")
            start_position = index
            index += 1
            tool_messages: list[ChatMessage] = []
            while index < len(messages_for_compile):
                current = messages_for_compile[index]
                if current.entry_id in fragment_entry_ids:
                    break
                if current.role != "tool":
                    break
                tool_messages.append(current)
                refs.append(current.entry_id)
                hashes.append(content_hash_for_text(current.content or ""))
                token_total += count_tokens(current.content or "")
                index += 1
            incomplete = not _tool_call_group_complete(message, tool_messages)
            refs_tuple = tuple(refs)
            group_id = _base_message_group_id(message, start_position)
            groups.append(
                ContextSourceGroup(
                    group_id=group_id,
                    source=ContextFragmentSource.SESSION_HISTORY,
                    source_refs=refs_tuple,
                    source_content_hash=_group_content_hash(refs_tuple, tuple(hashes)),
                    token_estimate=token_total,
                    required=incomplete,
                    protected=incomplete,
                    compressible=False,
                    droppable=False,
                    trim_safe=False,
                )
            )
            for assigned_index in range(start_position, index):
                assigned_indices.add(assigned_index)
                base_group_id_by_message_index[assigned_index] = group_id
            continue

        if message.role == "tool":
            refs = (message.entry_id,)
            group_id = _base_message_group_id(message, index)
            groups.append(
                ContextSourceGroup(
                    group_id=group_id,
                    source=ContextFragmentSource.SESSION_HISTORY,
                    source_refs=refs,
                    source_content_hash=_group_content_hash(refs, _base_message_content_hashes([message])),
                    token_estimate=count_tokens(message.content or ""),
                    required=True,
                    protected=True,
                    compressible=False,
                    droppable=False,
                    trim_safe=False,
                )
            )
            assigned_indices.add(index)
            base_group_id_by_message_index[index] = group_id
            index += 1
            continue

        source = ContextFragmentSource.SESSION_HISTORY
        required = False
        protected = False
        compressible = False
        droppable = False
        if message.role == "system":
            source = ContextFragmentSource.SYSTEM_INSTRUCTIONS
            required = True
            protected = True
        elif message.role == "user" and index == last_base_user_index:
            source = ContextFragmentSource.TASK_MESSAGE
            required = True
            protected = True

        refs = (message.entry_id,)
        group_id = _base_message_group_id(message, index)
        groups.append(
            ContextSourceGroup(
                group_id=group_id,
                source=source,
                source_refs=refs,
                source_content_hash=_group_content_hash(refs, _base_message_content_hashes([message])),
                token_estimate=count_tokens(message.content or ""),
                required=required,
                protected=protected,
                compressible=compressible,
                droppable=droppable,
                trim_safe=False,
            )
        )
        assigned_indices.add(index)
        base_group_id_by_message_index[index] = group_id
        index += 1

    return groups, assigned_indices, base_group_id_by_message_index


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
    canonical_snapshot_group_ids: set[str],
) -> list[ContextSourceGroup]:
    marked: list[ContextSourceGroup] = []
    for group in session_groups:
        protected = group.protected or group.group_id in recent_tail_ids
        compressible = (
            group.group_id in canonical_snapshot_group_ids
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
    by_source: dict[ContextFragmentSource, dict[str, list[str]]] = {}
    for group_id in selected_ids:
        source = groups_by_id[group_id].source
        bucket = by_source.setdefault(source, {"selected": [], "excluded": []})
        bucket["selected"].append(group_id)
    for group_id in excluded_ids:
        source = groups_by_id[group_id].source
        bucket = by_source.setdefault(source, {"selected": [], "excluded": []})
        bucket["excluded"].append(group_id)

    allocations: list[ContextSourceBudgetAllocation] = []
    for source in sorted(by_source, key=lambda item: item.value):
        bucket = by_source[source]
        selected = tuple(bucket["selected"])
        excluded_for_source = tuple(bucket["excluded"])
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
        fragment_messages: Sequence[ChatMessage],
        ranked_fragments: Sequence[ContextFragment],
        session_history: SessionHistorySnapshot | None,
        resolved_global_budget_tokens: int,
        optimization_policy: ContextOptimizationPolicy | None = None,
        model_family: str | None = None,
        locale: str | None = None,
    ) -> ContextPlan:
        if len(fragment_messages) != len(ranked_fragments):
            raise ContextPlanningError("fragment_message_mapping_mismatch")

        execution_scope = request.execution_scope
        if not isinstance(execution_scope, ModelCallExecutionScope):
            raise ValueError("execution_scope must be ModelCallExecutionScope")

        estimated_total_tokens = sum(
            self._count_tokens(message.content or "") for message in messages_for_compile
        )

        fragment_entry_ids = {message.entry_id for message in fragment_messages}
        index_by_entry_id = {message.entry_id: index for index, message in enumerate(messages_for_compile)}

        group_id_by_message_index: dict[int, str] = {}
        groups_by_id: dict[str, ContextSourceGroup] = {}

        session_groups_raw: list[ContextSourceGroup] = []
        canonical_snapshot_group_ids: set[str] = set()
        message_id_to_group_id: dict[str, str] = {}
        if session_history is not None:
            session_groups_raw = list(
                group_session_history_snapshot(session_history, count_tokens=self._count_tokens)
            )
            canonical_snapshot_group_ids = {group.group_id for group in session_groups_raw}
            for group in session_groups_raw:
                for ref in group.source_refs:
                    message_id_to_group_id[ref] = group.group_id

        session_token_accum = {group.group_id: 0 for group in session_groups_raw}
        present_snapshot_refs_by_group_id: dict[str, list[str]] = {
            group.group_id: [] for group in session_groups_raw
        }

        for fragment_message, fragment in zip(fragment_messages, ranked_fragments, strict=True):
            message_index = index_by_entry_id.get(fragment_message.entry_id)
            if message_index is None:
                raise ContextPlanningError("incomplete_model_input_plan")
            if message_index in group_id_by_message_index:
                if fragment.source is ContextFragmentSource.SESSION_HISTORY:
                    raise ContextPlanningError("incomplete_canonical_snapshot_group")
                raise ContextPlanningError("incomplete_model_input_plan")
            token_estimate = self._count_tokens(fragment_message.content or "")

            if fragment.source is ContextFragmentSource.SESSION_HISTORY:
                message_id = str(fragment.metadata.get("message_id") or "").strip()
                if not message_id or message_id not in message_id_to_group_id:
                    refs = (fragment.fragment_id,)
                    group_id = _deterministic_group_id(refs)
                    groups_by_id[group_id] = ContextSourceGroup(
                        group_id=group_id,
                        source=fragment.source,
                        source_refs=refs,
                        source_content_hash=fragment.content_hash
                        or content_hash_for_text(fragment.content),
                        token_estimate=token_estimate,
                        required=fragment.mandatory,
                        protected=fragment.mandatory,
                        droppable=False,
                        compressible=False,
                        trim_safe=False,
                    )
                    group_id_by_message_index[message_index] = group_id
                    continue
                group_id = message_id_to_group_id[message_id]
                present_refs = present_snapshot_refs_by_group_id[group_id]
                if message_id in present_refs:
                    raise ContextPlanningError("incomplete_canonical_snapshot_group")
                present_refs.append(message_id)
                session_token_accum[group_id] += token_estimate
                group_id_by_message_index[message_index] = group_id
                continue

            refs = (fragment.fragment_id,)
            group_id = _deterministic_group_id(refs)
            groups_by_id[group_id] = ContextSourceGroup(
                group_id=group_id,
                source=fragment.source,
                source_refs=refs,
                source_content_hash=fragment.content_hash or content_hash_for_text(fragment.content),
                token_estimate=token_estimate,
                required=fragment.mandatory or fragment.source in _REQUIRED_SOURCES,
                protected=fragment.mandatory or fragment.source in _REQUIRED_SOURCES,
                droppable=(
                    not fragment.mandatory
                    and fragment.source in _DROPPABLE_SOURCES
                ),
                compressible=False,
                trim_safe=False,
            )
            group_id_by_message_index[message_index] = group_id

        for group in session_groups_raw:
            present_refs = present_snapshot_refs_by_group_id[group.group_id]
            if not present_refs:
                continue
            if tuple(present_refs) != group.source_refs:
                raise ContextPlanningError("incomplete_canonical_snapshot_group")
            groups_by_id[group.group_id] = ContextSourceGroup(
                group_id=group.group_id,
                source=group.source,
                source_refs=group.source_refs,
                source_content_hash=group.source_content_hash,
                token_estimate=session_token_accum[group.group_id],
                start_sequence=group.start_sequence,
                end_sequence=group.end_sequence,
                required=group.required,
                protected=group.protected,
                compressible=group.compressible,
                droppable=group.droppable,
                trim_safe=group.trim_safe,
            )

        base_groups, base_assigned, base_group_id_by_message_index = _group_base_messages(
            messages_for_compile,
            fragment_entry_ids,
            count_tokens=self._count_tokens,
        )
        if set(base_group_id_by_message_index) & set(group_id_by_message_index):
            raise ContextPlanningError("incomplete_model_input_plan")
        for message_index, group_id in base_group_id_by_message_index.items():
            group_id_by_message_index[message_index] = group_id
        for group in base_groups:
            groups_by_id[group.group_id] = group

        if len(group_id_by_message_index) != len(messages_for_compile):
            raise ContextPlanningError("incomplete_model_input_plan")

        ordered_group_ids: list[str] = []
        seen_group_ids: set[str] = set()
        for message_index in range(len(messages_for_compile)):
            group_id = group_id_by_message_index[message_index]
            if group_id not in groups_by_id:
                raise ContextPlanningError("incomplete_model_input_plan")
            if group_id not in seen_group_ids:
                ordered_group_ids.append(group_id)
                seen_group_ids.add(group_id)

        if set(ordered_group_ids) != set(groups_by_id):
            raise ContextPlanningError("incomplete_model_input_plan")
        if len(ordered_group_ids) != len(seen_group_ids):
            raise ContextPlanningError("incomplete_model_input_plan")

        all_groups = [groups_by_id[group_id] for group_id in ordered_group_ids]

        group_token_total = sum(group.token_estimate for group in all_groups)
        if group_token_total != estimated_total_tokens:
            raise ContextPlanningError("incomplete_model_input_plan")

        session_groups_in_order = [
            group for group in all_groups if group.source is ContextFragmentSource.SESSION_HISTORY
        ]
        recent_tail = optimization_policy.recent_tail_min_messages if optimization_policy else 0
        recent_tail_ids = _apply_recent_tail_protection(
            session_groups_in_order,
            recent_tail_min_messages=recent_tail,
        )
        marked_session_by_id = {
            group.group_id: group
            for group in _mark_session_groups(
                session_groups_in_order,
                recent_tail_ids=recent_tail_ids,
                canonical_snapshot_group_ids=canonical_snapshot_group_ids,
            )
        }
        all_groups = [
            marked_session_by_id.get(group.group_id, group) for group in all_groups
        ]
        groups_by_id = {group.group_id: group for group in all_groups}

        selected_set = {group.group_id for group in all_groups}
        excluded_set: set[str] = set()
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

        if estimated_total_tokens <= resolved_global_budget_tokens:
            selected_ids = tuple(group.group_id for group in all_groups if group.group_id in selected_set)
            excluded_ids = tuple(group.group_id for group in all_groups if group.group_id in excluded_set)
            if not set(selected_ids).isdisjoint(excluded_ids):
                raise ContextPlanningError("incomplete_model_input_plan")
            if set(selected_ids) | set(excluded_ids) != set(groups_by_id):
                raise ContextPlanningError("incomplete_model_input_plan")
            return self._build_plan(
                request=request,
                execution_scope=execution_scope,
                resolved_global_budget_tokens=resolved_global_budget_tokens,
                estimated_total_tokens=estimated_total_tokens,
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

        droppable_ordered = tuple(group.group_id for group in all_groups if group.droppable)
        for group_id in droppable_ordered:
            selected_set.remove(group_id)
            excluded_set.add(group_id)
            post_drop_total = _total_tokens(tuple(selected_set), groups_by_id)
            if post_drop_total <= resolved_global_budget_tokens:
                selected_ids = tuple(
                    group.group_id for group in all_groups if group.group_id in selected_set
                )
                excluded_ids = tuple(
                    group.group_id for group in all_groups if group.group_id in excluded_set
                )
                if not set(selected_ids).isdisjoint(excluded_ids):
                    raise ContextPlanningError("incomplete_model_input_plan")
                if set(selected_ids) | set(excluded_ids) != set(groups_by_id):
                    raise ContextPlanningError("incomplete_model_input_plan")
                return self._build_plan(
                    request=request,
                    execution_scope=execution_scope,
                    resolved_global_budget_tokens=resolved_global_budget_tokens,
                    estimated_total_tokens=estimated_total_tokens,
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
        for group in all_groups:
            if group.source is not ContextFragmentSource.SESSION_HISTORY:
                continue
            if group.compressible and group.group_id in selected_set:
                started = True
                target_groups.append(group)
            elif started:
                break
        if not target_groups:
            raise ContextPlanningError(NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET)

        if session_history is None:
            raise ContextPlanningError(NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET)

        target_group_ids = tuple(group.group_id for group in target_groups)
        if not all(group_id in canonical_snapshot_group_ids for group_id in target_group_ids):
            raise ContextPlanningError(NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET)

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
        available_target_tokens = resolved_global_budget_tokens - non_target_tokens
        if available_target_tokens <= 0:
            raise ContextPlanningError(NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET)
        minimum_target_tokens = max(1, len(target_source_refs) * 3)
        if available_target_tokens < minimum_target_tokens:
            raise ContextPlanningError(NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET)
        target_token_budget = available_target_tokens

        if optimization_policy is None or not optimization_policy.allowed_strategy_ids:
            raise ContextPlanningError(NO_ALLOWED_CONTEXT_OPTIMIZATION_STRATEGY)

        allowed_strategy_ids = optimization_policy.allowed_strategy_ids

        lossiness_profile = "lossless"
        if optimization_policy.allow_lossy:
            lossiness_profile = "lossy"

        protected_policy_version = optimization_policy.protected_region_policy_version

        lookup_inputs = ContextArtifactLookupInputs(
            tenant_id=session_history.tenant_id,
            context_scope_id=session_history.context_scope_id,
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

        selected_ids = tuple(group.group_id for group in all_groups if group.group_id in selected_set)
        excluded_ids = tuple(group.group_id for group in all_groups if group.group_id in excluded_set)
        if not set(selected_ids).isdisjoint(excluded_ids):
            raise ContextPlanningError("incomplete_model_input_plan")
        if set(selected_ids) | set(excluded_ids) != set(groups_by_id):
            raise ContextPlanningError("incomplete_model_input_plan")

        return self._build_plan(
            request=request,
            execution_scope=execution_scope,
            resolved_global_budget_tokens=resolved_global_budget_tokens,
            estimated_total_tokens=estimated_total_tokens,
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
