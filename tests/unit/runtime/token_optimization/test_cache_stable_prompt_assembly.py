# © Artur Czarnecki. All rights reserved.

"""TOKEN-10B: cache-stable prompt assembly tests."""

from __future__ import annotations

import dataclasses
import json

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.token_optimization.contracts import PromptCacheInvalidationReason
from intergrax.runtime.token_optimization.prompt_assembly import (
    CacheStablePromptIntegrityError,
    CacheStablePromptState,
    CacheStableToolEnvelope,
    PromptAssemblyMessageBlock,
    assemble_cache_stable_prompt,
    build_cache_stable_tool_envelope,
    cache_stable_prompt_assembly_to_safe_dict,
    materialize_cache_stable_send_payload,
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


def _router_tool_schema() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "alpha.tool",
                "description": "SYNTH-ALPHA",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "beta.tool",
                "description": "SYNTH-BETA",
                "parameters": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                },
            },
        },
    ]


def test_mutating_original_stable_message_does_not_change_assembly() -> None:
    stable = ChatMessage(role="system", content="SYNTH-STABLE")
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(
            PromptAssemblyMessageBlock(block_id="policy", message=stable),
        ),
        dynamic_tail=(ChatMessage(role="user", content="SYNTH-TAIL"),),
    )
    snapshot = (
        assembly.messages,
        assembly.messages_hash,
        assembly.state,
        assembly.report,
    )
    stable.content = "SYNTH-MUTATED"
    assert (
        assembly.messages,
        assembly.messages_hash,
        assembly.state,
        assembly.report,
    ) == snapshot
    payload = materialize_cache_stable_send_payload(assembly)
    assert payload.messages[0].content == "SYNTH-STABLE"


def test_mutating_original_dynamic_tail_does_not_change_assembly() -> None:
    tail = ChatMessage(role="user", content="SYNTH-TAIL")
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        dynamic_tail=(tail,),
    )
    snapshot = (assembly.messages[-1].content, assembly.messages_hash, assembly.report)
    tail.content = "SYNTH-MUTATED-TAIL"
    assert assembly.messages[-1].content == snapshot[0]
    assert assembly.messages_hash == snapshot[1]
    assert assembly.report == snapshot[2]


def test_mutating_original_tool_schema_does_not_change_envelope() -> None:
    schema = _router_tool_schema()
    envelope = build_cache_stable_tool_envelope(schema)
    snapshot = (envelope.envelope_hash, envelope.tool_ids, envelope.tools_schema)
    schema[0]["function"]["description"] = "SYNTH-MUTATED"
    schema[1]["function"]["parameters"]["properties"]["count"]["type"] = "string"
    assert (
        envelope.envelope_hash,
        envelope.tool_ids,
        envelope.tools_schema,
    ) == snapshot


def test_mutating_assembly_stable_message_causes_materialization_failure() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
    )
    assembly.messages[0].content = "SYNTH-MUTATED"
    with pytest.raises(CacheStablePromptIntegrityError):
        materialize_cache_stable_send_payload(assembly)


def test_mutating_dynamic_tail_inside_assembly_causes_materialization_failure() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        dynamic_tail=(ChatMessage(role="user", content="SYNTH-TAIL"),),
    )
    assembly.messages[-1].content = "SYNTH-MUTATED-TAIL"
    with pytest.raises(CacheStablePromptIntegrityError):
        materialize_cache_stable_send_payload(assembly)


def test_mutating_nested_tool_calls_causes_materialization_failure() -> None:
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "alpha.tool", "arguments": "{}"},
        }
    ]
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        append_only_thread=(
            ChatMessage(role="assistant", content="", tool_calls=tool_calls),
        ),
    )
    assert assembly.messages[1].tool_calls is not None
    assembly.messages[1].tool_calls[0]["function"]["arguments"] = '{"value":"x"}'
    with pytest.raises(CacheStablePromptIntegrityError):
        materialize_cache_stable_send_payload(assembly)


def test_mutating_nested_tool_parameters_causes_materialization_failure() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=_router_tool_schema(),
    )
    assert assembly.tool_envelope is not None
    first = assembly.tool_envelope.tools_schema[0]
    assert isinstance(first, dict)
    function = first["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    value = properties["value"]
    assert isinstance(value, dict)
    value["type"] = "integer"
    with pytest.raises(CacheStablePromptIntegrityError):
        materialize_cache_stable_send_payload(assembly)


def test_changing_tool_order_causes_materialization_failure() -> None:
    schema = _router_tool_schema()
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=schema,
    )
    assert assembly.tool_envelope is not None
    tampered_envelope = CacheStableToolEnvelope(
        tools_schema=assembly.tool_envelope.tools_schema,
        tool_ids=("beta.tool", "alpha.tool"),
        envelope_hash=assembly.tool_envelope.envelope_hash,
    )
    object.__setattr__(assembly, "tool_envelope", tampered_envelope)
    with pytest.raises(CacheStablePromptIntegrityError, match="tool envelope ordering mismatch"):
        materialize_cache_stable_send_payload(assembly)


def test_reordered_assembly_schema_causes_materialization_failure() -> None:
    schema = _router_tool_schema()
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=schema,
    )
    assert assembly.tool_envelope is not None
    reversed_schema = tuple(reversed(assembly.tool_envelope.tools_schema))
    tampered_envelope = CacheStableToolEnvelope(
        tools_schema=reversed_schema,
        tool_ids=assembly.tool_envelope.tool_ids,
        envelope_hash=assembly.tool_envelope.envelope_hash,
    )
    object.__setattr__(assembly, "tool_envelope", tampered_envelope)
    with pytest.raises(CacheStablePromptIntegrityError, match="tool envelope ordering mismatch"):
        materialize_cache_stable_send_payload(assembly)


def test_reordered_schema_with_reordered_ids_but_old_hash_fails() -> None:
    schema = _router_tool_schema()
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=schema,
    )
    assert assembly.tool_envelope is not None
    reversed_schema = tuple(reversed(assembly.tool_envelope.tools_schema))
    tampered_envelope = CacheStableToolEnvelope(
        tools_schema=reversed_schema,
        tool_ids=("beta.tool", "alpha.tool"),
        envelope_hash=assembly.tool_envelope.envelope_hash,
    )
    object.__setattr__(assembly, "tool_envelope", tampered_envelope)
    with pytest.raises(CacheStablePromptIntegrityError):
        materialize_cache_stable_send_payload(assembly)


def test_canonical_initial_envelope_sorts_tools_and_hashes_ordered_sequence() -> None:
    schema = list(reversed(_router_tool_schema()))
    envelope = build_cache_stable_tool_envelope(schema)
    assert envelope.tool_ids == ("alpha.tool", "beta.tool")
    names = [entry["function"]["name"] for entry in envelope.tools_schema]
    assert names == ["alpha.tool", "beta.tool"]
    from intergrax.tools.exporters.openai import compute_openai_tools_schema_hash

    assert envelope.envelope_hash == compute_openai_tools_schema_hash(envelope.tools_schema)


def test_unchanged_envelope_preserves_order_on_materialization() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=_router_tool_schema(),
    )
    payload = materialize_cache_stable_send_payload(assembly)
    assert assembly.tool_envelope is not None
    payload_names = [entry["function"]["name"] for entry in payload.tools_schema]
    envelope_names = [entry["function"]["name"] for entry in assembly.tool_envelope.tools_schema]
    assert payload_names == envelope_names == ["alpha.tool", "beta.tool"]


def test_unchanged_assembly_materializes_successfully() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        dynamic_tail=(ChatMessage(role="user", content="SYNTH-TAIL"),),
        tools_schema=_router_tool_schema(),
    )
    payload = materialize_cache_stable_send_payload(assembly)
    assert payload.messages_hash == assembly.messages_hash
    assert payload.tool_envelope_hash == assembly.tool_envelope.envelope_hash


def test_materialized_payload_is_defensive_copy() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=_router_tool_schema(),
    )
    payload = materialize_cache_stable_send_payload(assembly)
    assert payload.messages is not assembly.messages
    assert payload.messages[0] is not assembly.messages[0]
    payload.messages[0].content = "SYNTH-MUTATED"
    assert assembly.messages[0].content == "SYNTH-STABLE"
    assert payload.tools_schema is not assembly.tool_envelope.tools_schema
    first_payload = payload.tools_schema[0]
    assert isinstance(first_payload, dict)
    first_payload["function"]["description"] = "SYNTH-MUTATED"
    first_envelope = assembly.tool_envelope.tools_schema[0]
    assert isinstance(first_envelope, dict)
    assert first_envelope["function"]["description"] == "SYNTH-ALPHA"


def test_materialized_full_message_hash_matches_sent_sequence() -> None:
    from intergrax.llm.messages import compute_model_facing_messages_hash

    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        dynamic_tail=(ChatMessage(role="user", content="SYNTH-TAIL"),),
    )
    payload = materialize_cache_stable_send_payload(assembly)
    assert compute_model_facing_messages_hash(payload.messages) == payload.messages_hash


def test_none_to_none_envelope_transition_is_stable() -> None:
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
    )
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        previous_state=first.state,
    )
    assert second.report.tool_envelope_stable is True
    assert second.report.invalidation_reason is PromptCacheInvalidationReason.NONE


def test_none_to_hash_reports_tool_envelope_changed() -> None:
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
    )
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=_router_tool_schema(),
        previous_state=first.state,
    )
    assert second.report.tool_envelope_stable is False
    assert second.report.invalidation_reason is PromptCacheInvalidationReason.TOOL_ENVELOPE_CHANGED


def test_hash_to_none_reports_tool_envelope_changed() -> None:
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=_router_tool_schema(),
    )
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        previous_state=first.state,
    )
    assert second.report.tool_envelope_stable is False
    assert second.report.invalidation_reason is PromptCacheInvalidationReason.TOOL_ENVELOPE_CHANGED


def test_hash_a_to_hash_b_reports_tool_envelope_changed() -> None:
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=_router_tool_schema(),
    )
    mutated_schema = _router_tool_schema()
    mutated_schema[0]["function"]["description"] = "SYNTH-CHANGED"
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=mutated_schema,
        previous_state=first.state,
    )
    assert second.report.tool_envelope_stable is False
    assert second.report.invalidation_reason is PromptCacheInvalidationReason.TOOL_ENVELOPE_CHANGED


def test_hash_a_to_hash_a_remains_stable() -> None:
    schema = _router_tool_schema()
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=schema,
    )
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=schema,
        previous_state=first.state,
    )
    assert second.report.tool_envelope_stable is True
    assert second.report.invalidation_reason is PromptCacheInvalidationReason.NONE


def test_prompt_safety_invalidation_retains_precedence_over_envelope_change() -> None:
    schema = _router_tool_schema()
    first = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-STABLE"),),
        tools_schema=schema,
    )
    mutated_schema = _router_tool_schema()
    mutated_schema[0]["function"]["description"] = "SYNTH-CHANGED"
    second = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-REWRITTEN"),),
        tools_schema=mutated_schema,
        previous_state=first.state,
    )
    assert second.report.tool_envelope_stable is False
    assert second.report.invalidation_reason is PromptCacheInvalidationReason.APPEND_ONLY_VIOLATION


def test_safe_reports_still_contain_no_raw_content_or_schema() -> None:
    assembly = assemble_cache_stable_prompt(
        stable_prefix_blocks=(_stable_block("policy", "SYNTH-SECRET-PREFIX"),),
        dynamic_tail=(ChatMessage(role="user", content="SYNTH-SECRET-TAIL"),),
        tools_schema=_router_tool_schema(),
    )
    safe = cache_stable_prompt_assembly_to_safe_dict(assembly.report)
    dumped = json.dumps(safe)
    assert "SYNTH-SECRET-PREFIX" not in dumped
    assert "SYNTH-SECRET-TAIL" not in dumped
    assert "function" not in dumped
    assert safe["raw_content_included"] is False
