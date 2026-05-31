# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.distributed.contracts.rate_limiter import RateLimitResult
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters._shared.resilience import (
    LLMRateLimitError,
    execute_with_resilience,
    reset_provider_resilience,
    set_llm_distributed_rate_limiter,
)
from intergrax.llm_adapters.tracking.context import set_llm_tenant_id

pytestmark = pytest.mark.unit


def test_distributed_rate_limit_blocks_when_not_allowed() -> None:
    limiter = MagicMock()
    limiter.acquire.return_value = RateLimitResult(
        allowed=False, remaining_tokens=0.0, retry_after_seconds=1.0
    )
    set_llm_distributed_rate_limiter(limiter)
    set_llm_tenant_id("t1")
    cfg = LLMCallConfig(use_distributed_rate_limit=True, calls_per_minute=10)
    reset_provider_resilience("openai")

    with pytest.raises(LLMRateLimitError):
        execute_with_resilience(
            lambda: "ok",
            provider="openai",
            config=cfg,
            retry_fn=lambda f: f(),
            tenant_id="t1",
        )
    set_llm_distributed_rate_limiter(None)
    reset_provider_resilience("openai")
