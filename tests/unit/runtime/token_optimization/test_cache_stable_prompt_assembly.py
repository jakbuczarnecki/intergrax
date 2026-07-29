# © Artur Czarnecki. All rights reserved.

"""TOKEN-10B: cache-stable prompt assembly tests."""

from __future__ import annotations

import dataclasses
import json

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.token_optimization.contracts import PromptCacheInvalidationReason
from intergrax.runtime.token_optimization.prompt_assembly import (
    CacheStablePromptState,
    PromptAssemblyMessageBlock,
    assemble_cache_stable_prompt,
    cache_stable_prompt_assembly_to_safe_dict,
    message_content_hash,
)
from intergrax.runtime.token_optimization.prompt_cache import PREFIX_STABILITY_INITIAL, PREFIX_STABILITY_STABLE

pytestmark = pytest.mark.unit


def _stable_block(block_id: str, content: str) -> PromptAssemblyMessageBlock:
    return PromptAssemblyMessageBlock(
        block_id=block_id,
        message=ChatMessage(role="system", content=content),
    )


def test_message_order_stable_prefix_thread_tail() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-POLICY"),),
        append_only_thread=(
            ChatMessage(role="user", content="SYNTH-USER-HISTORY"),
            ChatMessage(role="assistant", content="SYNTH-ASSISTANT-HISTORY"),
        ),
        dynamic_tail=(ChatMessage(role="user", content="SYNTH-DYNAMIC-TAIL"),),
    )
    roles = [message.role for message in assembly.messages]
    contents = [message.content for message in assembly.messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert contents == [
        "SYNTH-POLICY",
        "SYNTH-USER-HISTORY",
        "SYNTH-ASSISTANT-HISTORY",
        "SYNTH-DYNAMIC-TAIL",
    ]


def test_dynamic_tail_change_does_not_change_prefix_hash() -> None:
    base = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        dynamic_tail=(ChatMessage(role="user", content="SYNTH-TAIL-A"),),
    )
    changed = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        dynamic_tail=(ChatMessage(role="user", content="SYNTH-TAIL-B"),),
        previous_state=base.state,
    )
    assert changed.report.prefix_hash == base.report.prefix_hash
    assert changed.report.prefix_stability_status == PREFIX_STABILITY_STABLE


def test_entry_id_does_not_affect_message_fingerprint() -> None:
    first = ChatMessage(role="user", content="SYNTH-SAME", entry_id="entry-a")
    second = ChatMessage(role="user", content="SYNTH-SAME", entry_id="entry-b")
    assert message_content_hash(first) == message_content_hash(second)


def test_created_at_does_not_affect_message_fingerprint() -> None:
    first = ChatMessage(role="user", content="SYNTH-SAME", created_at="2020-01-01T00:00:00")
    second = ChatMessage(role="user", content="SYNTH-SAME", created_at="2026-01-01T00:00:00")
    assert message_content_hash(first) == message_content_hash(second)


def test_identical_stable_prefix_is_stable() -> None:
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
    )
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        previous_state=first.state,
    )
    assert second.report.prefix_stability_status == PREFIX_STABILITY_STABLE
    assert second.report.invalidation_reason is PromptCacheInvalidationReason.NONE


def test_append_only_thread_extension_is_stable_and_reusable() -> None:
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
    )
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        append_only_thread=(ChatMessage(role="user", content="SYNTH-HISTORY"),),
        previous_state=first.state,
    )
    assert second.report.prefix_stability_status == PREFIX_STABILITY_STABLE
    assert second.report.append_only_valid is True
    assert second.report.append_only_extended is True
    assert second.report.reusable_prefix_block_count == 1


def test_rewritten_historical_message_causes_append_only_violation() -> None:
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        append_only_thread=(ChatMessage(role="user", content="SYNTH-ORIGINAL"),),
    )
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        append_only_thread=(ChatMessage(role="user", content="SYNTH-REWRITTEN"),),
        previous_state=first.state,
    )
    assert second.report.append_only_valid is False
    assert second.report.invalidation_reason is PromptCacheInvalidationReason.APPEND_ONLY_VIOLATION


def test_removed_historical_message_causes_append_only_violation() -> None:
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        append_only_thread=(
            ChatMessage(role="user", content="SYNTH-HISTORY"),
            ChatMessage(role="assistant", content="SYNTH-REPLY"),
        ),
    )
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        append_only_thread=(ChatMessage(role="user", content="SYNTH-HISTORY"),),
        previous_state=first.state,
    )
    assert second.report.append_only_valid is False
    assert second.report.invalidation_reason is PromptCacheInvalidationReason.APPEND_ONLY_VIOLATION


def test_reordered_historical_messages_cause_append_only_violation() -> None:
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        append_only_thread=(
            ChatMessage(role="user", content="SYNTH-FIRST"),
            ChatMessage(role="assistant", content="SYNTH-SECOND"),
        ),
    )
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        append_only_thread=(
            ChatMessage(role="assistant", content="SYNTH-SECOND"),
            ChatMessage(role="user", content="SYNTH-FIRST"),
        ),
        previous_state=first.state,
    )
    assert second.report.append_only_valid is False
    assert second.report.invalidation_reason is PromptCacheInvalidationReason.APPEND_ONLY_VIOLATION


def test_inserted_message_before_history_causes_append_only_violation() -> None:
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        append_only_thread=(ChatMessage(role="assistant", content="SYNTH-REPLY"),),
    )
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        append_only_thread=(
            ChatMessage(role="user", content="SYNTH-INSERTED"),
            ChatMessage(role="assistant", content="SYNTH-REPLY"),
        ),
        previous_state=first.state,
    )
    assert second.report.append_only_valid is False
    assert second.report.invalidation_reason is PromptCacheInvalidationReason.APPEND_ONLY_VIOLATION


def test_volatile_data_in_stable_prefix_invalidates() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "run_id: SYNTH-VOLATILE"),),
    )
    assert assembly.report.invalidation_reason is PromptCacheInvalidationReason.DYNAMIC_DATA_IN_PREFIX


def test_volatile_data_in_dynamic_tail_does_not_invalidate_prefix() -> None:
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
    )
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        dynamic_tail=(ChatMessage(role="user", content="run_id: SYNTH-OK-IN-TAIL"),),
        previous_state=first.state,
    )
    assert second.report.invalidation_reason is PromptCacheInvalidationReason.NONE


def test_duplicate_stable_block_ids_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate block IDs"):
        assemble_cache_stable_prompt(
            stable_prefix_blocks=(
                _stable_block("dup", "SYNTH-A"),
                _stable_block("dup", "SYNTH-B"),
            ),
        )


def test_safe_state_contains_no_raw_message_content() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-SECRET-PREFIX"),),
        dynamic_tail=(ChatMessage(role="user", content="SYNTH-SECRET-TAIL"),),
    )
    payload = dataclasses.asdict(assembly.state)
    dumped = json.dumps(payload, default=str)
    assert "SYNTH-SECRET-PREFIX" not in dumped
    assert "SYNTH-SECRET-TAIL" not in dumped
    assert "message" not in payload


def test_safe_report_contains_no_raw_message_content() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-SECRET-PREFIX"),),
        dynamic_tail=(ChatMessage(role="user", content="SYNTH-SECRET-TAIL"),),
    )
    safe = cache_stable_prompt_assembly_to_safe_dict(assembly.report)
    dumped = json.dumps(safe)
    assert "SYNTH-SECRET-PREFIX" not in dumped
    assert "SYNTH-SECRET-TAIL" not in dumped
    assert safe["raw_content_included"] is False


def test_initial_request_reports_initial_status() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
    )
    assert assembly.report.prefix_stability_status == PREFIX_STABILITY_INITIAL
    assert assembly.report.append_only_valid is True
    assert assembly.report.append_only_extended is False


def test_previous_state_none_on_first_assembly() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
    )
    assert isinstance(assembly.state, CacheStablePromptState)
