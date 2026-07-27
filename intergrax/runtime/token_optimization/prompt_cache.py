# © Artur Czarnecki. All rights reserved.

"""Provider-neutral prompt-cache prefix helpers (TOKEN-OPT-5B).

Helper-level only: no provider API calls, no production prompt assembly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.runtime.token_optimization.contracts import PromptCacheInvalidationReason

PREFIX_STABILITY_INITIAL = "initial"
PREFIX_STABILITY_STABLE = "stable"
PREFIX_STABILITY_INVALIDATED = "invalidated"

_VOLATILE_PREFIX_MARKERS: tuple[str, ...] = (
    "run_id:",
    "trace_id:",
    "request_id:",
    "timestamp:",
    "step_id:",
)

_TOOL_ENVELOPE_ID_MARKERS: tuple[str, ...] = (
    "tool_envelope",
    "tool_catalog",
    "tool",
)


@dataclass(frozen=True, slots=True)
class PromptCacheBlock:
    """Single prompt block for helper-level cache-prefix modeling."""

    block_id: str
    content: str
    cacheable: bool
    dynamic: bool = False

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise ValueError("block_id cannot be empty")
        if self.content is None:
            raise ValueError("content must not be None")
        if self.cacheable and self.dynamic:
            raise ValueError("a block cannot be both cacheable=True and dynamic=True")


@dataclass(frozen=True, slots=True)
class PromptCachePrefixSnapshot:
    """Stable-prefix vs dynamic-tail partition of prompt blocks."""

    stable_blocks: tuple[PromptCacheBlock, ...]
    dynamic_tail_blocks: tuple[PromptCacheBlock, ...] = ()

    def __post_init__(self) -> None:
        for block in self.stable_blocks:
            if not block.cacheable or block.dynamic:
                raise ValueError(
                    "stable_blocks must contain only cacheable non-dynamic blocks"
                )
        # dynamic_tail_blocks may include later cacheable blocks that appear after
        # the first dynamic/non-cacheable cut (they are not part of the stable prefix).
        all_ids = [block.block_id for block in self.stable_blocks]
        all_ids.extend(block.block_id for block in self.dynamic_tail_blocks)
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("block IDs must be unique across the full snapshot")


@dataclass(frozen=True, slots=True)
class PromptCachePrefixStabilityResult:
    """Prefix-stability evaluation outcome without raw prompt content."""

    prefix_hash: str
    prefix_stability_status: str
    invalidation_reason: PromptCacheInvalidationReason
    stable_block_count: int
    dynamic_tail_block_count: int
    cacheable_prefix_chars: int
    dynamic_tail_chars: int


def build_prefix_snapshot(
    blocks: Sequence[PromptCacheBlock],
) -> PromptCachePrefixSnapshot:
    """Split leading cacheable blocks from the dynamic/non-cacheable tail."""
    stable: list[PromptCacheBlock] = []
    dynamic_tail: list[PromptCacheBlock] = []
    tail_started = False
    for block in blocks:
        if not tail_started and block.cacheable and not block.dynamic:
            stable.append(block)
            continue
        tail_started = True
        dynamic_tail.append(block)
    return PromptCachePrefixSnapshot(
        stable_blocks=tuple(stable),
        dynamic_tail_blocks=tuple(dynamic_tail),
    )


def compute_prefix_hash(snapshot: PromptCachePrefixSnapshot) -> str:
    """Deterministic sha256 over stable block IDs and contents only."""
    hasher = hashlib.sha256()
    for block in snapshot.stable_blocks:
        hasher.update(block.block_id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(block.content.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _stable_prefix_has_dynamic_data(snapshot: PromptCachePrefixSnapshot) -> bool:
    for block in snapshot.stable_blocks:
        for marker in _VOLATILE_PREFIX_MARKERS:
            if marker in block.content:
                return True
    return False


def _looks_like_tool_envelope_block_id(block_id: str) -> bool:
    lowered = block_id.lower()
    return any(marker in lowered for marker in _TOOL_ENVELOPE_ID_MARKERS)


def _tool_envelope_ids_changed(
    previous: PromptCachePrefixSnapshot,
    current: PromptCachePrefixSnapshot,
) -> bool:
    previous_ids = tuple(block.block_id for block in previous.stable_blocks)
    current_ids = tuple(block.block_id for block in current.stable_blocks)
    if previous_ids == current_ids:
        return False
    involved = set(previous_ids) ^ set(current_ids)
    # Order-only change: same set, different sequence.
    if not involved:
        involved = set(previous_ids) | set(current_ids)
    return any(_looks_like_tool_envelope_block_id(block_id) for block_id in involved)


def evaluate_prefix_stability(
    previous: PromptCachePrefixSnapshot | None,
    current: PromptCachePrefixSnapshot,
) -> PromptCachePrefixStabilityResult:
    """Compare prefix snapshots and report stability without raw content."""
    prefix_hash = compute_prefix_hash(current)
    stable_block_count = len(current.stable_blocks)
    dynamic_tail_block_count = len(current.dynamic_tail_blocks)
    cacheable_prefix_chars = sum(len(block.content) for block in current.stable_blocks)
    dynamic_tail_chars = sum(len(block.content) for block in current.dynamic_tail_blocks)

    if _stable_prefix_has_dynamic_data(current):
        return PromptCachePrefixStabilityResult(
            prefix_hash=prefix_hash,
            prefix_stability_status=PREFIX_STABILITY_INVALIDATED,
            invalidation_reason=PromptCacheInvalidationReason.DYNAMIC_DATA_IN_PREFIX,
            stable_block_count=stable_block_count,
            dynamic_tail_block_count=dynamic_tail_block_count,
            cacheable_prefix_chars=cacheable_prefix_chars,
            dynamic_tail_chars=dynamic_tail_chars,
        )

    if previous is None:
        return PromptCachePrefixStabilityResult(
            prefix_hash=prefix_hash,
            prefix_stability_status=PREFIX_STABILITY_INITIAL,
            invalidation_reason=PromptCacheInvalidationReason.NONE,
            stable_block_count=stable_block_count,
            dynamic_tail_block_count=dynamic_tail_block_count,
            cacheable_prefix_chars=cacheable_prefix_chars,
            dynamic_tail_chars=dynamic_tail_chars,
        )

    previous_hash = compute_prefix_hash(previous)
    if previous_hash == prefix_hash:
        return PromptCachePrefixStabilityResult(
            prefix_hash=prefix_hash,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            invalidation_reason=PromptCacheInvalidationReason.NONE,
            stable_block_count=stable_block_count,
            dynamic_tail_block_count=dynamic_tail_block_count,
            cacheable_prefix_chars=cacheable_prefix_chars,
            dynamic_tail_chars=dynamic_tail_chars,
        )

    reason = PromptCacheInvalidationReason.PREFIX_CHANGED
    if _tool_envelope_ids_changed(previous, current):
        reason = PromptCacheInvalidationReason.TOOL_ENVELOPE_CHANGED

    return PromptCachePrefixStabilityResult(
        prefix_hash=prefix_hash,
        prefix_stability_status=PREFIX_STABILITY_INVALIDATED,
        invalidation_reason=reason,
        stable_block_count=stable_block_count,
        dynamic_tail_block_count=dynamic_tail_block_count,
        cacheable_prefix_chars=cacheable_prefix_chars,
        dynamic_tail_chars=dynamic_tail_chars,
    )


def preserves_append_only_prefix(
    previous: PromptCachePrefixSnapshot,
    current: PromptCachePrefixSnapshot,
) -> bool:
    """Return True when current stable prefix preserves previous blocks in order."""
    if len(current.stable_blocks) < len(previous.stable_blocks):
        return False
    for previous_block, current_block in zip(
        previous.stable_blocks, current.stable_blocks, strict=False
    ):
        if (
            previous_block.block_id != current_block.block_id
            or previous_block.content != current_block.content
            or previous_block.cacheable != current_block.cacheable
            or previous_block.dynamic != current_block.dynamic
        ):
            return False
    return True
