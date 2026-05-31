# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters._shared.retry import call_with_retry

pytestmark = pytest.mark.unit


def test_call_with_retry_succeeds_after_transient_error() -> None:
    config = LLMCallConfig(max_retries=2, retry_backoff_sec=0.0)
    calls = {"n": 0}

    class TransientError(RuntimeError):
        status_code = 503

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise TransientError("unavailable")
        return "ok"

    assert call_with_retry(fn, config=config) == "ok"
    assert calls["n"] == 2
