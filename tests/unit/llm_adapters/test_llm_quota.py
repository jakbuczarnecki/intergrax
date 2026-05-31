# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.governance.quota import LLMQuotaExceeded, check_llm_tenant_quota
from intergrax.llm_adapters.tracking.context import set_llm_tenant_id
from intergrax.llm_adapters.tracking.context import clear_llm_tenant_id
from intergrax.llm_adapters.tracking.metrics import get_llm_metrics_collector, record_llm_call, set_metrics_enabled
from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


def test_check_llm_tenant_quota_raises_when_over_limit() -> None:
    set_metrics_enabled(True)
    get_llm_metrics_collector().reset()
    clear_llm_tenant_id()
    record_llm_call(
        provider="openai",
        model="m",
        run_id="r",
        input_tokens=500,
        output_tokens=100,
        duration_ms=1,
        success=True,
    )
    with patch.dict("os.environ", {"INTERGRAX_LLM_TENANT_MAX_TOKENS": "1000"}, clear=False):
        check_llm_tenant_quota("_platform", additional_tokens=0)
        with pytest.raises(LLMQuotaExceeded):
            check_llm_tenant_quota("_platform", additional_tokens=500)
    set_metrics_enabled(False)


def test_adapter_execute_enforces_quota() -> None:
    set_metrics_enabled(True)
    get_llm_metrics_collector().reset()
    set_llm_tenant_id("t1")
    record_llm_call(
        provider="openai",
        model="gpt-4o-mini",
        run_id="r",
        input_tokens=1000,
        output_tokens=0,
        duration_ms=1,
        success=True,
    )
    client = MagicMock()
    with patch.dict("os.environ", {"INTERGRAX_LLM_TENANT_MAX_TOKENS": "1000"}, clear=False):
        adapter = OpenAIChatResponsesAdapter(client=client, model="gpt-4o-mini")
        with pytest.raises(LLMQuotaExceeded):
            adapter.generate_messages([ChatMessage(role="user", content="x")], run_id="q1")
    set_metrics_enabled(False)
