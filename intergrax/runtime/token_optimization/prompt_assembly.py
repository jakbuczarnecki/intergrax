# © Artur Czarnecki. All rights reserved.

"""Cache-stable prompt assembly runtime (TOKEN-10B).

Provider-neutral and application-neutral: assembles exact model-facing messages,
deterministic tool envelopes, and redaction-safe prefix state.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from intergrax.llm.messages import ChatMessage, compute_model_facing_messages_hash
from intergrax.tools.exporters.openai import compute_openai_tools_schema_hash
from intergrax.runtime.token_optimization.contracts import PromptCacheInvalidationReason
from intergrax.runtime.token_optimization.prompt_cache import (
    PREFIX_STABILITY_INITIAL,
    PREFIX_STABILITY_INVALIDATED,
    PREFIX_STABILITY_STABLE,
    PromptCacheBlock,
    PromptCachePrefixSnapshot,
    build_prefix_snapshot,
    compute_prefix_hash,
)

_VOLATILE_PREFIX_MARKERS: tuple[str, ...] = (
    "run_id:",
    "trace_id:",
    "request_id:",
    "timestamp:",
    "step_id:",
)


@dataclass(frozen=True, slots=True)
class PromptAssemblyMessageBlock:
    """Explicit stable message block for cache-stable assembly."""

    block_id: str
    message: ChatMessage

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise ValueError("block_id must be non-empty after stripping")
        if self.message.content is None:
            raise ValueError("message content must not be None")


@dataclass(frozen=True, slots=True)
class PromptCacheBlockFingerprint:
    """Append-only comparison fingerprint without raw prompt content."""

    block_id: str
    content_hash: str
    content_chars: int


@dataclass(frozen=True, slots=True)
class CacheStablePromptState:
    """Redaction-safe state passed explicitly between assembly calls."""

    prefix_hash: str
    stable_block_fingerprints: tuple[PromptCacheBlockFingerprint, ...]
    tool_envelope_hash: str | None
    tool_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CacheStableToolEnvelope:
    """Deterministic native tool schema envelope."""

    tools_schema: tuple[Mapping[str, Any], ...]
    tool_ids: tuple[str, ...]
    envelope_hash: str


@dataclass(frozen=True, slots=True)
class CacheStablePromptAssemblyReport:
    """Safe scalar report of a cache-stable assembly result."""

    prefix_hash: str
    prefix_stability_status: str
    invalidation_reason: PromptCacheInvalidationReason
    stable_block_count: int
    append_only_thread_message_count: int
    dynamic_tail_message_count: int
    cacheable_prefix_chars: int
    dynamic_tail_chars: int
    append_only_valid: bool
    append_only_extended: bool
    reusable_prefix_block_count: int
    tool_envelope_hash: str | None
    tool_envelope_stable: bool | None
    tool_count: int
    raw_content_included: bool = False


@dataclass(frozen=True, slots=True)
class CacheStablePromptAssembly:
    """Prepared model request plus safe runtime state."""

    messages: tuple[ChatMessage, ...]
    messages_hash: str
    tool_envelope: CacheStableToolEnvelope | None
    state: CacheStablePromptState
    report: CacheStablePromptAssemblyReport


@dataclass(frozen=True, slots=True)
class CacheStablePromptSendPayload:
    """Validated defensive copy of the exact model-facing send payload."""

    messages: tuple[ChatMessage, ...]
    tools_schema: tuple[Mapping[str, Any], ...]
    messages_hash: str
    tool_envelope_hash: str | None


class CacheStablePromptIntegrityError(ValueError):
    """Raised when assembled payload no longer matches its recorded fingerprints."""


def _canonical_message_payload(message: ChatMessage) -> dict[str, Any]:
    return message.to_dict()


def _canonical_message_json(message: ChatMessage) -> str:
    payload = _canonical_message_payload(message)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def message_content_hash(message: ChatMessage) -> str:
    """SHA-256 over canonical model-facing message fields."""
    digest = hashlib.sha256()
    digest.update(_canonical_message_json(message).encode("utf-8"))
    return digest.hexdigest()


def _copy_model_facing_message(message: ChatMessage) -> ChatMessage:
    """Independent snapshot of fields that reach providers through ``to_dict()``."""
    tool_calls = copy.deepcopy(message.tool_calls) if message.tool_calls else None
    return ChatMessage(
        role=message.role,
        content=message.content,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=tool_calls,
    )


def _thread_block_id(index: int, message: ChatMessage) -> str:
    return f"thread.{index:06d}.{message.role}"


def _tail_block_id(index: int, message: ChatMessage) -> str:
    return f"tail.{index:06d}.{message.role}"


def _validate_unique_block_ids(block_ids: Sequence[str]) -> None:
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("duplicate block IDs in assembly request")


def _stable_prefix_has_dynamic_data(content: str) -> bool:
    for marker in _VOLATILE_PREFIX_MARKERS:
        if marker in content:
            return True
    return False


def _fingerprint_for_block(block_id: str, message: ChatMessage) -> PromptCacheBlockFingerprint:
    canonical = _canonical_message_json(message)
    return PromptCacheBlockFingerprint(
        block_id=block_id,
        content_hash=message_content_hash(message),
        content_chars=len(canonical),
    )


def _message_to_cache_block(block_id: str, message: ChatMessage, *, cacheable: bool) -> PromptCacheBlock:
    return PromptCacheBlock(
        block_id=block_id,
        content=_canonical_message_json(message),
        cacheable=cacheable,
        dynamic=not cacheable,
    )


def _build_current_prefix_snapshot(
    prefix_blocks: Sequence[PromptCacheBlock],
) -> PromptCachePrefixSnapshot:
    return PromptCachePrefixSnapshot(
        stable_blocks=tuple(prefix_blocks),
        dynamic_tail_blocks=(),
    )


def _fingerprint_prefix_stability(
    *,
    previous_state: CacheStablePromptState | None,
    current_fingerprints: tuple[PromptCacheBlockFingerprint, ...],
    has_dynamic_data_in_prefix: bool,
) -> tuple[str, PromptCacheInvalidationReason, bool, bool, int]:
    if has_dynamic_data_in_prefix:
        return (
            PREFIX_STABILITY_INVALIDATED,
            PromptCacheInvalidationReason.DYNAMIC_DATA_IN_PREFIX,
            False,
            False,
            0,
        )

    if previous_state is None:
        return (
            PREFIX_STABILITY_INITIAL,
            PromptCacheInvalidationReason.NONE,
            True,
            False,
            0,
        )

    previous_fps = previous_state.stable_block_fingerprints
    if previous_fps == current_fingerprints:
        return (
            PREFIX_STABILITY_STABLE,
            PromptCacheInvalidationReason.NONE,
            True,
            False,
            len(previous_fps),
        )

    if len(current_fingerprints) >= len(previous_fps):
        preserved = True
        for previous_fp, current_fp in zip(previous_fps, current_fingerprints, strict=False):
            if (
                previous_fp.block_id != current_fp.block_id
                or previous_fp.content_hash != current_fp.content_hash
            ):
                preserved = False
                break
        if preserved:
            return (
                PREFIX_STABILITY_STABLE,
                PromptCacheInvalidationReason.NONE,
                True,
                len(current_fingerprints) > len(previous_fps),
                len(previous_fps),
            )

    return (
        PREFIX_STABILITY_INVALIDATED,
        PromptCacheInvalidationReason.APPEND_ONLY_VIOLATION,
        False,
        False,
        0,
    )


def _resolve_invalidation_reason(
    *,
    previous_state: CacheStablePromptState | None,
    prefix_invalidation_reason: PromptCacheInvalidationReason,
    tool_envelope_hash: str | None,
    append_only_valid: bool,
) -> PromptCacheInvalidationReason:
    if prefix_invalidation_reason is not PromptCacheInvalidationReason.NONE:
        return prefix_invalidation_reason

    if previous_state is None:
        return PromptCacheInvalidationReason.NONE

    # Prompt-safety reasons (e.g. APPEND_ONLY_VIOLATION) retain precedence above;
    # tool-envelope transitions are reported only when the prefix is otherwise reusable.
    if (
        previous_state.tool_envelope_hash != tool_envelope_hash
        and append_only_valid
    ):
        return PromptCacheInvalidationReason.TOOL_ENVELOPE_CHANGED

    return PromptCacheInvalidationReason.NONE


def _compute_tool_envelope_stable(
    *,
    previous_state: CacheStablePromptState | None,
    tool_envelope_hash: str | None,
) -> bool | None:
    if previous_state is None:
        return None
    return previous_state.tool_envelope_hash == tool_envelope_hash


def build_cache_stable_tool_envelope(
    tools_schema: Sequence[Mapping[str, Any]],
) -> CacheStableToolEnvelope:
    """Canonicalize and fingerprint an exported OpenAI tool schema list."""
    if not tools_schema:
        return CacheStableToolEnvelope(
            tools_schema=(),
            tool_ids=(),
            envelope_hash=hashlib.sha256(b"").hexdigest(),
        )

    canonical_entries: list[dict[str, Any]] = []
    tool_names: list[str] = []
    for entry in tools_schema:
        if not isinstance(entry, Mapping):
            raise ValueError("tools_schema entries must be mappings")
        if entry.get("type") != "function":
            raise ValueError("tools_schema entries must be OpenAI function tools")
        function = entry.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("function tool must include function object")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("function.name must be non-empty")
        if name in tool_names:
            raise ValueError(f"duplicate tool name: {name}")
        tool_names.append(name)
        canonical_entries.append(copy.deepcopy(dict(entry)))

    envelope_hash = compute_openai_tools_schema_hash(canonical_entries)
    return CacheStableToolEnvelope(
        tools_schema=tuple(canonical_entries),
        tool_ids=tuple(tool_names),
        envelope_hash=envelope_hash,
    )


def _observed_tool_ids_from_schema(
    tools_schema: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    observed: list[str] = []
    for entry in tools_schema:
        if not isinstance(entry, Mapping):
            raise ValueError("tools_schema entries must be mappings")
        function = entry.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("function tool must include function object")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("function.name must be non-empty")
        observed.append(name)
    return tuple(observed)


def assemble_cache_stable_prompt(
    *,
    stable_prefix_blocks: Sequence[PromptAssemblyMessageBlock],
    append_only_thread: Sequence[ChatMessage] = (),
    dynamic_tail: Sequence[ChatMessage] = (),
    tools_schema: Sequence[Mapping[str, Any]] = (),
    previous_state: CacheStablePromptState | None = None,
) -> CacheStablePromptAssembly:
    """Assemble cache-stable messages, tool envelope, state, and safe report."""
    stable_block_ids = [block.block_id for block in stable_prefix_blocks]
    thread_ids = [_thread_block_id(index, message) for index, message in enumerate(append_only_thread)]
    tail_ids = [_tail_block_id(index, message) for index, message in enumerate(dynamic_tail)]
    _validate_unique_block_ids((*stable_block_ids, *thread_ids, *tail_ids))

    prefix_blocks: list[PromptCacheBlock] = []
    fingerprint_blocks: list[PromptCacheBlockFingerprint] = []

    copied_stable_messages = tuple(
        _copy_model_facing_message(block.message) for block in stable_prefix_blocks
    )
    copied_thread_messages = tuple(
        _copy_model_facing_message(message) for message in append_only_thread
    )
    copied_tail_messages = tuple(
        _copy_model_facing_message(message) for message in dynamic_tail
    )

    for block, message in zip(stable_prefix_blocks, copied_stable_messages, strict=True):
        prefix_blocks.append(_message_to_cache_block(block.block_id, message, cacheable=True))
        fingerprint_blocks.append(_fingerprint_for_block(block.block_id, message))

    for index, message in enumerate(copied_thread_messages):
        block_id = thread_ids[index]
        prefix_blocks.append(_message_to_cache_block(block_id, message, cacheable=True))
        fingerprint_blocks.append(_fingerprint_for_block(block_id, message))

    dynamic_blocks: list[PromptCacheBlock] = []
    for index, message in enumerate(copied_tail_messages):
        block_id = tail_ids[index]
        dynamic_blocks.append(_message_to_cache_block(block_id, message, cacheable=False))

    all_blocks = prefix_blocks + dynamic_blocks
    _ = build_prefix_snapshot(all_blocks)
    prefix_hash = compute_prefix_hash(_build_current_prefix_snapshot(prefix_blocks))

    has_dynamic_data_in_prefix = any(
        _stable_prefix_has_dynamic_data(message.content)
        for message in copied_stable_messages
    ) or any(
        _stable_prefix_has_dynamic_data(message.content)
        for message in copied_thread_messages
    )

    (
        prefix_stability_status,
        prefix_invalidation_reason,
        append_only_valid,
        append_only_extended,
        reusable_count,
    ) = _fingerprint_prefix_stability(
        previous_state=previous_state,
        current_fingerprints=tuple(fingerprint_blocks),
        has_dynamic_data_in_prefix=has_dynamic_data_in_prefix,
    )

    tool_envelope = (
        build_cache_stable_tool_envelope(tools_schema) if tools_schema else None
    )
    tool_envelope_hash = tool_envelope.envelope_hash if tool_envelope is not None else None

    invalidation_reason = _resolve_invalidation_reason(
        previous_state=previous_state,
        prefix_invalidation_reason=prefix_invalidation_reason,
        tool_envelope_hash=tool_envelope_hash,
        append_only_valid=append_only_valid,
    )

    tool_envelope_stable = _compute_tool_envelope_stable(
        previous_state=previous_state,
        tool_envelope_hash=tool_envelope_hash,
    )

    cacheable_prefix_chars = sum(fp.content_chars for fp in fingerprint_blocks)
    dynamic_tail_chars = sum(len(block.content) for block in dynamic_blocks)

    messages = copied_stable_messages + copied_thread_messages + copied_tail_messages
    messages_hash = compute_model_facing_messages_hash(messages)

    state = CacheStablePromptState(
        prefix_hash=prefix_hash,
        stable_block_fingerprints=tuple(fingerprint_blocks),
        tool_envelope_hash=tool_envelope_hash,
        tool_ids=tool_envelope.tool_ids if tool_envelope is not None else (),
    )

    report = CacheStablePromptAssemblyReport(
        prefix_hash=prefix_hash,
        prefix_stability_status=prefix_stability_status,
        invalidation_reason=invalidation_reason,
        stable_block_count=len(fingerprint_blocks),
        append_only_thread_message_count=len(append_only_thread),
        dynamic_tail_message_count=len(dynamic_tail),
        cacheable_prefix_chars=cacheable_prefix_chars,
        dynamic_tail_chars=dynamic_tail_chars,
        append_only_valid=append_only_valid,
        append_only_extended=append_only_extended,
        reusable_prefix_block_count=reusable_count,
        tool_envelope_hash=tool_envelope_hash,
        tool_envelope_stable=tool_envelope_stable,
        tool_count=len(tool_envelope.tool_ids) if tool_envelope is not None else 0,
        raw_content_included=False,
    )

    return CacheStablePromptAssembly(
        messages=messages,
        messages_hash=messages_hash,
        tool_envelope=tool_envelope,
        state=state,
        report=report,
    )


def materialize_cache_stable_send_payload(
    assembly: CacheStablePromptAssembly,
) -> CacheStablePromptSendPayload:
    """Validate assembly integrity and return defensive copies for adapter invocation."""
    current_messages_hash = compute_model_facing_messages_hash(assembly.messages)
    if current_messages_hash != assembly.messages_hash:
        raise CacheStablePromptIntegrityError("messages hash mismatch")

    prefix_count = len(assembly.state.stable_block_fingerprints)
    if prefix_count > len(assembly.messages):
        raise CacheStablePromptIntegrityError("stable prefix fingerprint count mismatch")
    recomputed_fingerprints = tuple(
        _fingerprint_for_block(fp.block_id, message)
        for fp, message in zip(
            assembly.state.stable_block_fingerprints,
            assembly.messages[:prefix_count],
            strict=True,
        )
    )
    if recomputed_fingerprints != assembly.state.stable_block_fingerprints:
        raise CacheStablePromptIntegrityError("stable prefix fingerprint mismatch")

    tool_envelope_hash: str | None = None
    tool_ids: tuple[str, ...] = ()
    tools_schema: tuple[Mapping[str, Any], ...] = ()
    if assembly.tool_envelope is not None:
        observed_tool_ids = _observed_tool_ids_from_schema(
            assembly.tool_envelope.tools_schema
        )
        if observed_tool_ids != assembly.tool_envelope.tool_ids:
            raise CacheStablePromptIntegrityError("tool envelope ordering mismatch")
        current_exact_hash = compute_openai_tools_schema_hash(
            assembly.tool_envelope.tools_schema
        )
        if current_exact_hash != assembly.tool_envelope.envelope_hash:
            raise CacheStablePromptIntegrityError("tool envelope hash mismatch")
        tool_envelope_hash = assembly.tool_envelope.envelope_hash
        tool_ids = assembly.tool_envelope.tool_ids
        tools_schema = tuple(
            copy.deepcopy(dict(entry)) for entry in assembly.tool_envelope.tools_schema
        )
    elif assembly.state.tool_envelope_hash is not None or assembly.state.tool_ids:
        raise CacheStablePromptIntegrityError("tool envelope state mismatch")

    if tool_ids != assembly.state.tool_ids:
        raise CacheStablePromptIntegrityError("tool ids mismatch")

    defensive_messages = tuple(_copy_model_facing_message(message) for message in assembly.messages)
    return CacheStablePromptSendPayload(
        messages=defensive_messages,
        tools_schema=tools_schema,
        messages_hash=assembly.messages_hash,
        tool_envelope_hash=tool_envelope_hash,
    )


def cache_stable_prompt_assembly_to_safe_dict(
    report: CacheStablePromptAssemblyReport,
) -> dict[str, object]:
    """Serialize assembly report fields without raw content."""
    return {
        "prefix_hash": report.prefix_hash,
        "prefix_stability_status": report.prefix_stability_status,
        "prefix_invalidation_reason": report.invalidation_reason.value,
        "append_only_valid": report.append_only_valid,
        "append_only_extended": report.append_only_extended,
        "reusable_prefix_block_count": report.reusable_prefix_block_count,
        "stable_block_count": report.stable_block_count,
        "append_only_thread_message_count": report.append_only_thread_message_count,
        "dynamic_tail_message_count": report.dynamic_tail_message_count,
        "cacheable_prefix_chars": report.cacheable_prefix_chars,
        "dynamic_tail_chars": report.dynamic_tail_chars,
        "tool_envelope_hash": report.tool_envelope_hash,
        "tool_envelope_stable": report.tool_envelope_stable,
        "tool_count": report.tool_count,
        "raw_content_included": report.raw_content_included,
    }
