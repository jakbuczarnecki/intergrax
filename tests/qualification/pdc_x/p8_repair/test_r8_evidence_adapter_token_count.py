# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-M / R8 (G-05): the evidence adapter forwards ``count_messages_tokens`` to the inner adapter."""

from __future__ import annotations

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.llm.model_call_runtime_evidence_adapter import ModelCallRuntimeEvidenceAdapter
from intergrax.runtime.nexus.context.context_preflight import verify_context_preflight

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _Inner:
    context_window_tokens = 8192
    model = "fake-model"
    provider = "fake"

    def __init__(self) -> None:
        self.counted: list[int] = []

    def count_messages_tokens(self, messages) -> int:
        self.counted.append(len(messages))
        return 7 * len(messages)


def _messages() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="hello"), ChatMessage(role="assistant", content="world")]


def test_count_messages_tokens_returns_the_inner_value() -> None:
    inner = _Inner()
    wrapped = ModelCallRuntimeEvidenceAdapter(inner)  # type: ignore[arg-type]

    assert wrapped.count_messages_tokens(_messages()) == 14
    assert inner.counted == [2]


def test_the_context_preflight_accepts_the_wrapped_adapter() -> None:
    inner = _Inner()
    wrapped = ModelCallRuntimeEvidenceAdapter(inner)  # type: ignore[arg-type]

    result = verify_context_preflight(_messages(), wrapped)  # type: ignore[arg-type]

    assert result.ok is True
    assert inner.counted == [2]
