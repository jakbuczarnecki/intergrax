# © Artur Czarnecki. All rights reserved.

"""Synthetic corpus for prompt-cache prefix stability (TOKEN-OPT-5B)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.token_optimization.contracts import PromptCacheInvalidationReason
from intergrax.runtime.token_optimization.prompt_cache import (
    PREFIX_STABILITY_INITIAL,
    PREFIX_STABILITY_INVALIDATED,
    PREFIX_STABILITY_STABLE,
    PromptCacheBlock,
    PromptCachePrefixSnapshot,
    build_prefix_snapshot,
)

PROMPT_CACHE_PREFIX_SYNTHETIC_CORPUS_MARKER = "SYNTHETIC_PROMPT_CACHE_PREFIX_CORPUS_V1"

_SYSTEM = PromptCacheBlock(
    block_id="system_policy",
    content="You are a synthetic assistant for prefix-stability evaluation.",
    cacheable=True,
    dynamic=False,
)
_TOOLS_V1 = PromptCacheBlock(
    block_id="tool_catalog_v1",
    content="tools: echo, summarize",
    cacheable=True,
    dynamic=False,
)
_TOOLS_V2 = PromptCacheBlock(
    block_id="tool_catalog_v2",
    content="tools: echo, summarize, search",
    cacheable=True,
    dynamic=False,
)
_USER_TURN_1 = PromptCacheBlock(
    block_id="user_turn_1",
    content="Question: what is the status of synthetic case A?",
    cacheable=False,
    dynamic=True,
)
_USER_TURN_2 = PromptCacheBlock(
    block_id="user_turn_2",
    content="Follow-up: confirm the same synthetic status.",
    cacheable=False,
    dynamic=True,
)
_ASSISTANT_TURN_1 = PromptCacheBlock(
    block_id="assistant_turn_1",
    content="Answer: synthetic status is stable.",
    cacheable=False,
    dynamic=True,
)
_SYSTEM_CHANGED = PromptCacheBlock(
    block_id="system_policy",
    content="You are a revised synthetic assistant for prefix-stability evaluation.",
    cacheable=True,
    dynamic=False,
)
_SYSTEM_WITH_RUN_ID = PromptCacheBlock(
    block_id="system_policy",
    content="Policy context run_id: synth-run-001 for evaluation only.",
    cacheable=True,
    dynamic=False,
)
_LATE_CACHEABLE = PromptCacheBlock(
    block_id="late_cacheable_note",
    content="This cacheable block appears after dynamic tail start.",
    cacheable=True,
    dynamic=False,
)
_POLICY_A = PromptCacheBlock(
    block_id="policy_a",
    content="Policy block A.",
    cacheable=True,
    dynamic=False,
)
_POLICY_B = PromptCacheBlock(
    block_id="policy_b",
    content="Policy block B.",
    cacheable=True,
    dynamic=False,
)


@dataclass(frozen=True, slots=True)
class PromptCachePrefixCorpusCase:
    """One synthetic prefix-stability evaluation case."""

    case_id: str
    title: str
    previous_blocks: tuple[PromptCacheBlock, ...] | None
    current_blocks: tuple[PromptCacheBlock, ...]
    expected_prefix_stability_status: str
    expected_invalidation_reason: PromptCacheInvalidationReason
    synthetic_marker: str = PROMPT_CACHE_PREFIX_SYNTHETIC_CORPUS_MARKER

    def previous_snapshot(self) -> PromptCachePrefixSnapshot | None:
        if self.previous_blocks is None:
            return None
        return build_prefix_snapshot(self.previous_blocks)

    def current_snapshot(self) -> PromptCachePrefixSnapshot:
        return build_prefix_snapshot(self.current_blocks)


PROMPT_CACHE_PREFIX_CORPUS: tuple[PromptCachePrefixCorpusCase, ...] = (
    PromptCachePrefixCorpusCase(
        case_id="prompt_cache_prefix.initial_stable_prefix",
        title="Initial stable prefix without previous baseline",
        previous_blocks=None,
        current_blocks=(_SYSTEM, _TOOLS_V1, _USER_TURN_1),
        expected_prefix_stability_status=PREFIX_STABILITY_INITIAL,
        expected_invalidation_reason=PromptCacheInvalidationReason.NONE,
    ),
    PromptCachePrefixCorpusCase(
        case_id="prompt_cache_prefix.append_dynamic_tail_keeps_prefix_stable",
        title="Appending dynamic tail keeps stable prefix hash",
        previous_blocks=(_SYSTEM, _TOOLS_V1, _USER_TURN_1),
        current_blocks=(_SYSTEM, _TOOLS_V1, _USER_TURN_1, _ASSISTANT_TURN_1, _USER_TURN_2),
        expected_prefix_stability_status=PREFIX_STABILITY_STABLE,
        expected_invalidation_reason=PromptCacheInvalidationReason.NONE,
    ),
    PromptCachePrefixCorpusCase(
        case_id="prompt_cache_prefix.stable_prefix_content_changed",
        title="Stable prefix content change invalidates cache prefix",
        previous_blocks=(_SYSTEM, _TOOLS_V1, _USER_TURN_1),
        current_blocks=(_SYSTEM_CHANGED, _TOOLS_V1, _USER_TURN_1),
        expected_prefix_stability_status=PREFIX_STABILITY_INVALIDATED,
        expected_invalidation_reason=PromptCacheInvalidationReason.PREFIX_CHANGED,
    ),
    PromptCachePrefixCorpusCase(
        case_id="prompt_cache_prefix.tool_envelope_changed",
        title="Tool catalog identity change uses tool envelope invalidation",
        previous_blocks=(_SYSTEM, _TOOLS_V1, _USER_TURN_1),
        current_blocks=(_SYSTEM, _TOOLS_V2, _USER_TURN_1),
        expected_prefix_stability_status=PREFIX_STABILITY_INVALIDATED,
        expected_invalidation_reason=PromptCacheInvalidationReason.TOOL_ENVELOPE_CHANGED,
    ),
    PromptCachePrefixCorpusCase(
        case_id="prompt_cache_prefix.dynamic_run_id_in_prefix",
        title="Volatile run_id marker in stable prefix invalidates",
        previous_blocks=(_SYSTEM, _TOOLS_V1, _USER_TURN_1),
        current_blocks=(_SYSTEM_WITH_RUN_ID, _TOOLS_V1, _USER_TURN_1),
        expected_prefix_stability_status=PREFIX_STABILITY_INVALIDATED,
        expected_invalidation_reason=PromptCacheInvalidationReason.DYNAMIC_DATA_IN_PREFIX,
    ),
    PromptCachePrefixCorpusCase(
        case_id="prompt_cache_prefix.dynamic_tail_change_does_not_change_prefix_hash",
        title="Dynamic tail rewrite does not change prefix hash",
        previous_blocks=(_SYSTEM, _TOOLS_V1, _USER_TURN_1),
        current_blocks=(_SYSTEM, _TOOLS_V1, _USER_TURN_2),
        expected_prefix_stability_status=PREFIX_STABILITY_STABLE,
        expected_invalidation_reason=PromptCacheInvalidationReason.NONE,
    ),
    PromptCachePrefixCorpusCase(
        case_id="prompt_cache_prefix.cacheable_block_after_dynamic_tail_is_not_prefix",
        title="Cacheable block after dynamic tail stays in dynamic tail",
        previous_blocks=(_SYSTEM, _USER_TURN_1),
        current_blocks=(_SYSTEM, _USER_TURN_1, _LATE_CACHEABLE),
        expected_prefix_stability_status=PREFIX_STABILITY_STABLE,
        expected_invalidation_reason=PromptCacheInvalidationReason.NONE,
    ),
    PromptCachePrefixCorpusCase(
        case_id="prompt_cache_prefix.reordered_stable_blocks_invalidates_prefix",
        title="Reordered stable blocks invalidate prefix",
        previous_blocks=(_POLICY_A, _POLICY_B, _USER_TURN_1),
        current_blocks=(_POLICY_B, _POLICY_A, _USER_TURN_1),
        expected_prefix_stability_status=PREFIX_STABILITY_INVALIDATED,
        expected_invalidation_reason=PromptCacheInvalidationReason.PREFIX_CHANGED,
    ),
)

REQUIRED_PROMPT_CACHE_PREFIX_CASE_IDS: frozenset[str] = frozenset(
    case.case_id for case in PROMPT_CACHE_PREFIX_CORPUS
)
