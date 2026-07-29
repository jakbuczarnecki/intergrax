# © Artur Czarnecki. All rights reserved.

"""TOKEN-OPT-5B: helper-level cache-prefix stability tests."""

from __future__ import annotations

import dataclasses

import pytest

from intergrax.runtime.token_optimization.contracts import PromptCacheInvalidationReason
from intergrax.runtime.token_optimization.prompt_cache import (
    PREFIX_STABILITY_INVALIDATED,
    PREFIX_STABILITY_STABLE,
    PromptCacheBlock,
    PromptCachePrefixStabilityResult,
    build_prefix_snapshot,
    compute_prefix_hash,
    evaluate_prefix_stability,
    preserves_append_only_prefix,
)
from tests.fixtures.token_optimization.prompt_cache_prefix_corpus import (
    PROMPT_CACHE_PREFIX_CORPUS,
    PROMPT_CACHE_PREFIX_SYNTHETIC_CORPUS_MARKER,
    REQUIRED_PROMPT_CACHE_PREFIX_CASE_IDS,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _case_by_id(case_id: str):
    return next(case for case in PROMPT_CACHE_PREFIX_CORPUS if case.case_id == case_id)


def test_prefix_snapshot_splits_stable_prefix_and_dynamic_tail() -> None:
    blocks = (
        PromptCacheBlock("system", "stable-a", cacheable=True),
        PromptCacheBlock("tools", "stable-b", cacheable=True),
        PromptCacheBlock("user", "dynamic-1", cacheable=False, dynamic=True),
        PromptCacheBlock("late", "should-be-tail", cacheable=True),
    )
    snapshot = build_prefix_snapshot(blocks)
    assert [block.block_id for block in snapshot.stable_blocks] == ["system", "tools"]
    assert [block.block_id for block in snapshot.dynamic_tail_blocks] == ["user", "late"]


def test_prefix_hash_is_deterministic() -> None:
    snapshot = build_prefix_snapshot(
        (
            PromptCacheBlock("system", "alpha", cacheable=True),
            PromptCacheBlock("user", "beta", cacheable=False, dynamic=True),
        )
    )
    assert compute_prefix_hash(snapshot) == compute_prefix_hash(snapshot)


def test_prefix_hash_ignores_dynamic_tail() -> None:
    base = (
        PromptCacheBlock("system", "stable", cacheable=True),
        PromptCacheBlock("user_a", "tail-a", cacheable=False, dynamic=True),
    )
    changed_tail = (
        PromptCacheBlock("system", "stable", cacheable=True),
        PromptCacheBlock("user_b", "tail-b", cacheable=False, dynamic=True),
    )
    assert compute_prefix_hash(build_prefix_snapshot(base)) == compute_prefix_hash(
        build_prefix_snapshot(changed_tail)
    )


def test_append_dynamic_tail_keeps_prefix_stable() -> None:
    case = _case_by_id("prompt_cache_prefix.append_dynamic_tail_keeps_prefix_stable")
    result = evaluate_prefix_stability(case.previous_snapshot(), case.current_snapshot())
    assert result.prefix_stability_status == PREFIX_STABILITY_STABLE
    assert result.invalidation_reason is PromptCacheInvalidationReason.NONE
    assert preserves_append_only_prefix(
        case.previous_snapshot(),  # type: ignore[arg-type]
        case.current_snapshot(),
    )


def test_stable_prefix_content_change_invalidates_prefix() -> None:
    case = _case_by_id("prompt_cache_prefix.stable_prefix_content_changed")
    result = evaluate_prefix_stability(case.previous_snapshot(), case.current_snapshot())
    assert result.prefix_stability_status == PREFIX_STABILITY_INVALIDATED
    assert result.invalidation_reason is PromptCacheInvalidationReason.PREFIX_CHANGED


def test_tool_envelope_change_uses_tool_envelope_invalidation_reason() -> None:
    case = _case_by_id("prompt_cache_prefix.tool_envelope_changed")
    result = evaluate_prefix_stability(case.previous_snapshot(), case.current_snapshot())
    assert result.invalidation_reason is PromptCacheInvalidationReason.TOOL_ENVELOPE_CHANGED


def test_dynamic_run_id_in_prefix_reports_dynamic_data_in_prefix() -> None:
    case = _case_by_id("prompt_cache_prefix.dynamic_run_id_in_prefix")
    result = evaluate_prefix_stability(case.previous_snapshot(), case.current_snapshot())
    assert result.invalidation_reason is PromptCacheInvalidationReason.DYNAMIC_DATA_IN_PREFIX


def test_cacheable_block_after_dynamic_tail_is_not_prefix() -> None:
    case = _case_by_id(
        "prompt_cache_prefix.cacheable_block_after_dynamic_tail_is_not_prefix"
    )
    snapshot = case.current_snapshot()
    assert [block.block_id for block in snapshot.stable_blocks] == ["system_policy"]
    assert "late_cacheable_note" in {
        block.block_id for block in snapshot.dynamic_tail_blocks
    }
    result = evaluate_prefix_stability(case.previous_snapshot(), snapshot)
    assert result.prefix_stability_status == PREFIX_STABILITY_STABLE


def test_reordered_stable_blocks_break_append_only_invariant() -> None:
    case = _case_by_id("prompt_cache_prefix.reordered_stable_blocks_invalidates_prefix")
    previous = case.previous_snapshot()
    current = case.current_snapshot()
    assert previous is not None
    assert preserves_append_only_prefix(previous, current) is False
    result = evaluate_prefix_stability(previous, current)
    assert result.prefix_stability_status == PREFIX_STABILITY_INVALIDATED
    assert result.invalidation_reason is PromptCacheInvalidationReason.PREFIX_CHANGED


def test_append_only_extension_keeps_prefix_stable() -> None:
    previous = build_prefix_snapshot(
        (
            PromptCacheBlock("system", "SYNTH-STABLE-POLICY", cacheable=True),
        )
    )
    current = build_prefix_snapshot(
        (
            PromptCacheBlock("system", "SYNTH-STABLE-POLICY", cacheable=True),
            PromptCacheBlock("thread.000000.user", "SYNTH-HISTORY", cacheable=True),
            PromptCacheBlock("thread.000001.assistant", "SYNTH-REPLY", cacheable=True),
        )
    )
    result = evaluate_prefix_stability(previous, current)
    assert result.prefix_stability_status == PREFIX_STABILITY_STABLE
    assert result.invalidation_reason is PromptCacheInvalidationReason.NONE
    assert preserves_append_only_prefix(previous, current) is True


def test_removed_stable_block_reports_append_only_violation() -> None:
    previous = build_prefix_snapshot(
        (
            PromptCacheBlock("system", "SYNTH-STABLE-POLICY", cacheable=True),
            PromptCacheBlock("thread.000000.user", "SYNTH-HISTORY", cacheable=True),
        )
    )
    current = build_prefix_snapshot(
        (
            PromptCacheBlock("system", "SYNTH-STABLE-POLICY", cacheable=True),
        )
    )
    result = evaluate_prefix_stability(previous, current)
    assert result.prefix_stability_status == PREFIX_STABILITY_INVALIDATED
    assert result.invalidation_reason is PromptCacheInvalidationReason.APPEND_ONLY_VIOLATION


def test_prefix_stability_reports_are_raw_content_safe() -> None:
    for case in PROMPT_CACHE_PREFIX_CORPUS:
        result = evaluate_prefix_stability(case.previous_snapshot(), case.current_snapshot())
        assert isinstance(result, PromptCachePrefixStabilityResult)
        field_names = {field.name for field in dataclasses.fields(result)}
        assert "content" not in field_names
        assert "stable_blocks" not in field_names
        assert "dynamic_tail_blocks" not in field_names
        payload = dataclasses.asdict(result)
        for block in case.current_blocks:
            if block.content:
                assert block.content not in str(payload)
        for block in case.previous_blocks or ():
            if block.content:
                assert block.content not in str(payload)


def test_synthetic_corpus_case_ids_are_unique() -> None:
    case_ids = [case.case_id for case in PROMPT_CACHE_PREFIX_CORPUS]
    assert len(case_ids) == len(set(case_ids))


def test_synthetic_corpus_has_required_cases() -> None:
    assert REQUIRED_PROMPT_CACHE_PREFIX_CASE_IDS.issubset(
        {case.case_id for case in PROMPT_CACHE_PREFIX_CORPUS}
    )
    required = {
        "prompt_cache_prefix.initial_stable_prefix",
        "prompt_cache_prefix.append_dynamic_tail_keeps_prefix_stable",
        "prompt_cache_prefix.stable_prefix_content_changed",
        "prompt_cache_prefix.tool_envelope_changed",
        "prompt_cache_prefix.dynamic_run_id_in_prefix",
        "prompt_cache_prefix.dynamic_tail_change_does_not_change_prefix_hash",
        "prompt_cache_prefix.cacheable_block_after_dynamic_tail_is_not_prefix",
        "prompt_cache_prefix.reordered_stable_blocks_invalidates_prefix",
    }
    assert required.issubset({case.case_id for case in PROMPT_CACHE_PREFIX_CORPUS})
    for case in PROMPT_CACHE_PREFIX_CORPUS:
        result = evaluate_prefix_stability(case.previous_snapshot(), case.current_snapshot())
        assert result.prefix_stability_status == case.expected_prefix_stability_status
        assert result.invalidation_reason is case.expected_invalidation_reason


def test_synthetic_corpus_has_marker() -> None:
    assert PROMPT_CACHE_PREFIX_SYNTHETIC_CORPUS_MARKER == (
        "SYNTHETIC_PROMPT_CACHE_PREFIX_CORPUS_V1"
    )
    for case in PROMPT_CACHE_PREFIX_CORPUS:
        assert case.synthetic_marker == PROMPT_CACHE_PREFIX_SYNTHETIC_CORPUS_MARKER
