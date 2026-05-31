# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.tracking.metrics import get_llm_metrics_collector, set_metrics_enabled

pytestmark = pytest.mark.unit


def test_llm_metrics_recorded_on_end_call() -> None:
    from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter
    from unittest.mock import MagicMock

    set_metrics_enabled(True)
    collector = get_llm_metrics_collector()
    collector.reset()

    client = MagicMock()
    usage = MagicMock(input_tokens=4, output_tokens=2)
    response = MagicMock(usage=usage, output_text="hi", output=[], status="completed")
    client.responses.create.return_value = response

    adapter = OpenAIChatResponsesAdapter(client=client, model="gpt-4o-mini")
    adapter.generate_messages([ChatMessage(role="user", content="x")], run_id="m1")

    snap = collector.snapshot()
    match = [k for k in snap if "openai" in k and "gpt-4o-mini" in k]
    assert match, f"expected openai:gpt-4o-mini in snapshot keys: {list(snap)}"
    assert snap[match[0]]["calls"] == 1
    lines = collector.prometheus_lines()
    assert any("intergrax_llm_calls_total" in line for line in lines)

    set_metrics_enabled(False)
