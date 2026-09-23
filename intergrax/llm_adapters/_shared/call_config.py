# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.llm_adapters.contracts.call_config import LLMCallConfig

__all__ = ["LLMCallConfig", "parse_call_config"]


def parse_call_config(defaults: dict) -> LLMCallConfig:
    """Extract ``LLMCallConfig`` fields from adapter constructor kwargs."""
    known = {
        "temperature",
        "max_tokens",
        "timeout_sec",
        "max_retries",
        "retry_backoff_sec",
        "max_retry_after_sec",
        "retry_on_status",
        "rate_limit_wait_timeout_sec",
        "calls_per_minute",
        "circuit_breaker_threshold",
        "circuit_breaker_cooldown_sec",
        "use_distributed_rate_limit",
    }
    cfg_kwargs = {k: defaults[k] for k in known if k in defaults}
    return LLMCallConfig(**cfg_kwargs)
