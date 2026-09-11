# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Injectable provider rate limit port with process-local default (W2-C)."""

from __future__ import annotations

from intergrax.contracts.provider_rate_limit import ProviderRateLimitPort
from intergrax.runtime.resilience.local_provider_rate_limit import LocalProviderRateLimit

_default_local = LocalProviderRateLimit()
_port: ProviderRateLimitPort | None = None


def set_llm_provider_rate_limit_port(port: ProviderRateLimitPort | None) -> None:
    global _port
    if port is not None and not isinstance(port, ProviderRateLimitPort):
        raise TypeError("port must implement ProviderRateLimitPort or be None")
    _port = port


def get_llm_provider_rate_limit_port() -> ProviderRateLimitPort:
    if _port is not None:
        return _port
    return _default_local


def get_default_local_provider_rate_limit() -> LocalProviderRateLimit:
    return _default_local
