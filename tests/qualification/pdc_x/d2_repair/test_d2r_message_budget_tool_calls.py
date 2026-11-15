# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-B pinned tests: F-D2-01 (tool-call payload in the hard budget) and F-D2-17 (drop order).

Every positive case is red on the unchanged code because the budget chain sizes a message as
``count_tokens(message.content)`` only; the negative cases keep the fail-closed and linkage promises.
"""

from __future__ import annotations

import json

import pytest
from intergrax.context.budget.contracts import ContextBudgetUnsatisfiableError
from intergrax.context.budget.mandatory_base_messages import estimate_mandatory_protected_message_tokens
from intergrax.context.budget.token_counter import CharEstimateContextTokenCounter
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler, classify_candidates
from intergrax.runtime.nexus.context.context_preflight import count_message_tokens, verify_context_preflight
from tests.qualification.pdc_x.d2._support import FakeAdapter, tool_linkage_violations

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_BIG = 200_000


def _count(text: str) -> int:
    return len(text)


def _tool_call(call_id: str, arguments: str = "{}", name: str = "lookup") -> dict[str, object]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _big_group(entry_prefix: str = "big", chars: int = _BIG) -> list[ChatMessage]:
    """Assistant message with empty content and a huge arguments payload, plus its result."""
    return [
        ChatMessage(
            role="assistant",
            content="",
            entry_id=f"{entry_prefix}-asst",
            tool_calls=[_tool_call(f"{entry_prefix}-call", arguments="x" * chars)],
        ),
        ChatMessage(role="tool", content="ok", entry_id=f"{entry_prefix}-tool", tool_call_id=f"{entry_prefix}-call"),
    ]


def _config() -> RuntimeConfig:
    return RuntimeConfig(llm_adapter=FakeAdapter(window=4096), production_mode=False)


def _expected_cost(message: ChatMessage) -> int:
    cost = _count(message.content or "")
    if message.tool_calls:
        cost += _count(json.dumps(message.tool_calls, sort_keys=True, default=str))
    if message.name:
        cost += _count(message.name)
    if message.tool_call_id:
        cost += _count(message.tool_call_id)
    return cost


# ----------------------------------------------------------------- F-D2-01 positive


def test_f_d2_01_compile_trims_a_message_whose_budget_is_only_its_tool_call_payload() -> None:
    messages = [
        ChatMessage(role="system", content="SYS", entry_id="sys-1"),
        *_big_group(),
        ChatMessage(role="user", content="final", entry_id="user-final"),
    ]
    result = ContextCompiler(count_tokens=_count).compile(
        list(messages), _config(), max_output_tokens=64, input_budget_tokens=200
    )

    assert result.trimmed is True
    assert result.total_tokens <= 200
    assert all(not m.tool_calls for m in result.messages)
    assert tool_linkage_violations(list(result.messages)) == []
    assert [m.entry_id for m in result.messages][0] == "sys-1"
    assert [m.entry_id for m in result.messages][-1] == "user-final"


def test_f_d2_01_candidate_cost_counts_tool_calls_name_and_tool_call_id() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content="calling",
            entry_id="a",
            tool_calls=[_tool_call("call-1", arguments="y" * 500)],
        ),
        ChatMessage(role="tool", content="result", entry_id="t", tool_call_id="call-1", name="lookup"),
        ChatMessage(role="user", content="final", entry_id="u"),
    ]
    candidates = classify_candidates(messages, count_tokens=_count)

    assert [c.token_estimate for c in candidates] == [_expected_cost(m) for m in messages]
    assert candidates[0].token_estimate > 500


def test_f_d2_01_cost_is_deterministic() -> None:
    messages = _big_group(chars=1000)
    first = [c.token_estimate for c in classify_candidates(messages, count_tokens=_count)]
    second = [c.token_estimate for c in classify_candidates(list(messages), count_tokens=_count)]
    assert first == second


def test_f_d2_01_messages_without_payload_fields_cost_exactly_their_content() -> None:
    """Frozen table taken from the unchanged code: plain messages are sized as before."""
    contents = ["", "a", "abc" * 10, "x" * 333]
    messages = [ChatMessage(role="assistant", content=text, entry_id=f"m{i}") for i, text in enumerate(contents)]
    messages.append(ChatMessage(role="user", content="final", entry_id="u"))
    candidates = classify_candidates(messages, count_tokens=_count)
    assert [c.token_estimate for c in candidates] == [0, 1, 30, 333, 5]


def test_f_d2_01_final_window_check_counts_the_payload() -> None:
    messages = [*_big_group(), ChatMessage(role="user", content="final", entry_id="u")]

    assert count_message_tokens(messages, count_tokens=_count) > _BIG
    with pytest.raises(ValueError, match="Context overflow"):
        verify_context_preflight(list(messages), FakeAdapter(window=4096), count_tokens=_count)


def test_f_d2_01_char_counter_count_messages_counts_the_payload() -> None:
    counter = CharEstimateContextTokenCounter()
    messages = [*_big_group(), ChatMessage(role="user", content="final", entry_id="u")]
    assert counter.count_messages(messages) > _BIG // 4


def test_f_d2_01_mandatory_reserve_counts_the_payload_of_a_protected_message() -> None:
    messages = [
        ChatMessage(role="system", content="RULES", entry_id="sys-1", name="n" * 50),
        ChatMessage(role="user", content="final", entry_id="u"),
    ]
    reserve = estimate_mandatory_protected_message_tokens(messages, count_text=_count)
    assert reserve >= len("RULES") + 50 + len("final")


def test_f_d2_01_estimate_is_of_the_same_order_as_the_adapter_estimator() -> None:
    """Parity with the D3 estimator (T-IGX-041-C): same recipe, same order of magnitude."""
    messages = [
        ChatMessage(
            role="assistant",
            content="",
            entry_id="a",
            tool_calls=[_tool_call("call-1", arguments="lorem ipsum " * 16_667)],
        )
    ]

    def char_tokens(text: str) -> int:
        return max(1, len(text) // 4) if text else 0

    compiler_cost = classify_candidates(messages, count_tokens=char_tokens)[0].token_estimate
    adapter_estimate = FakeAdapter().estimate_tokens_for_messages(messages)

    assert adapter_estimate > 10_000
    assert 0.5 <= compiler_cost / adapter_estimate <= 2.0


# ----------------------------------------------------------------- F-D2-01 negatives


def test_f_d2_01_oversized_group_is_dropped_atomically_by_the_hard_budget() -> None:
    messages = [
        ChatMessage(role="system", content="SYS", entry_id="sys-1"),
        *_big_group(),
        ChatMessage(role="user", content="final", entry_id="user-final"),
    ]
    result = ContextCompiler(count_tokens=_count)._enforce_hard_budget(list(messages), budget_tokens=200)

    ids = [m.entry_id for m in result]
    assert "big-asst" not in ids and "big-tool" not in ids
    assert tool_linkage_violations(result) == []
    assert ids == ["sys-1", "user-final"]


def test_f_d2_01_a_mandatory_message_that_cannot_fit_fails_closed() -> None:
    messages = [
        ChatMessage(role="system", content="RULES", entry_id="sys-1", name="n" * 500),
        ChatMessage(role="assistant", content="h" * 40, entry_id="hist"),
        ChatMessage(role="user", content="final", entry_id="user-final"),
    ]
    with pytest.raises(ContextBudgetUnsatisfiableError):
        ContextCompiler(count_tokens=_count)._enforce_hard_budget(list(messages), budget_tokens=100)


def test_f_d2_01_an_oversized_last_user_turn_fails_closed() -> None:
    messages = [
        ChatMessage(role="system", content="SYS", entry_id="sys-1"),
        ChatMessage(role="user", content="final", entry_id="user-final", name="n" * 500),
    ]
    with pytest.raises(ContextBudgetUnsatisfiableError):
        ContextCompiler(count_tokens=_count)._enforce_hard_budget(list(messages), budget_tokens=100)


def test_f_d2_01_a_non_serializable_tool_calls_value_does_not_raise() -> None:
    odd = [{"id": "call-1", ("tuple", "key"): object(), "function": {"name": "x", "arguments": {1, 2}}}]
    messages = [ChatMessage(role="assistant", content="", entry_id="a", tool_calls=odd)]
    candidates = classify_candidates(messages, count_tokens=_count)
    assert candidates[0].token_estimate > 0


def test_f_d2_01_a_group_that_fits_is_kept_whole() -> None:
    messages = [
        ChatMessage(role="system", content="SYS", entry_id="sys-1"),
        *_big_group(chars=20),
        ChatMessage(role="user", content="final", entry_id="user-final"),
    ]
    result = ContextCompiler(count_tokens=_count).compile(
        list(messages), _config(), max_output_tokens=64, input_budget_tokens=4000
    )
    assert result.trimmed is False
    assert [m.entry_id for m in result.messages] == [m.entry_id for m in messages]


# ----------------------------------------------------------------- F-D2-17


def _history(n: int = 6) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content="SYS", entry_id="sys-1"),
        *[ChatMessage(role="assistant", content="h" * 20, entry_id=f"h{i}") for i in range(n)],
        ChatMessage(role="user", content="final", entry_id="user-final"),
    ]


def test_f_d2_17_last_resort_drop_removes_the_oldest_history_first() -> None:
    # 3 (SYS) + 5 (final) + 3 * 20 = 68: exactly three history messages fit.
    result = ContextCompiler(count_tokens=_count)._enforce_hard_budget(_history(), budget_tokens=68)
    assert [m.entry_id for m in result] == ["sys-1", "h3", "h4", "h5", "user-final"]


def test_f_d2_17_mandatory_and_final_user_are_never_dropped() -> None:
    result = ContextCompiler(count_tokens=_count)._enforce_hard_budget(_history(), budget_tokens=8)
    assert [m.entry_id for m in result] == ["sys-1", "user-final"]


def test_f_d2_17_tool_call_groups_still_leave_atomically_oldest_first() -> None:
    messages = [
        ChatMessage(role="system", content="SYS", entry_id="sys-1"),
        ChatMessage(role="assistant", content="c" * 10, entry_id="old-asst", tool_calls=[_tool_call("old")]),
        ChatMessage(role="tool", content="r" * 10, entry_id="old-tool", tool_call_id="old"),
        ChatMessage(role="assistant", content="n" * 10, entry_id="newer"),
        ChatMessage(role="user", content="final", entry_id="user-final"),
    ]
    compiler = ContextCompiler(count_tokens=_count)
    full = sum(_expected_cost(m) for m in messages)
    old_group = _expected_cost(messages[1]) + _expected_cost(messages[2])

    result = compiler._enforce_hard_budget(list(messages), budget_tokens=full - old_group)

    assert [m.entry_id for m in result] == ["sys-1", "newer", "user-final"]
    assert tool_linkage_violations(result) == []
