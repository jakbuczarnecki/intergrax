# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""
Per-tenant LLM token budgets (platform governance hook).

Uses in-process metrics from :mod:`intergrax.llm_adapters.tracking.metrics`.
Configure via ``INTERGRAX_LLM_TENANT_MAX_TOKENS`` (unset or 0 = disabled; any other value
that is not a non-negative integer denies every call with ``LLMQuotaConfigurationError``).
"""

from __future__ import annotations

import os

from intergrax.llm_adapters.tracking.metrics import get_llm_metrics_collector

LLM_TENANT_MAX_TOKENS_ENV = "INTERGRAX_LLM_TENANT_MAX_TOKENS"


class LLMQuotaExceeded(RuntimeError):
    """Raised when a tenant exceeds configured LLM token budget."""


class LLMQuotaConfigurationError(LLMQuotaExceeded):
    """The token budget variable is set but is not a non-negative integer (F-D3-12, fail closed).

    Subclasses :class:`LLMQuotaExceeded` so every existing quota handler maps it to a denial.
    The message names the variable only, never the configured value.
    """


def _max_tokens_per_tenant() -> int:
    """Configured per-tenant limit; unset or ``0`` means disabled, anything else malformed denies."""
    raw = os.getenv(LLM_TENANT_MAX_TOKENS_ENV)
    if raw is None:
        return 0
    try:
        value = int(raw.strip())
    except ValueError:
        raise LLMQuotaConfigurationError(
            f"{LLM_TENANT_MAX_TOKENS_ENV} must be a non-negative integer; LLM calls are denied"
        ) from None
    if value < 0:
        raise LLMQuotaConfigurationError(
            f"{LLM_TENANT_MAX_TOKENS_ENV} must be a non-negative integer; LLM calls are denied"
        )
    return value


def check_llm_tenant_quota(
    tenant_id: str | None,
    *,
    additional_tokens: int = 0,
) -> None:
    """
    Fail fast when tenant cumulative tokens exceed ``INTERGRAX_LLM_TENANT_MAX_TOKENS``.

    Call before LLM SDK invocations (wired in ``LLMAdapter._execute``).
    """
    limit = _max_tokens_per_tenant()
    if limit <= 0:
        return

    tenant = (tenant_id or "").strip() or "_platform"
    used = get_llm_metrics_collector().tenant_total_tokens(tenant)
    if used >= limit:
        raise LLMQuotaExceeded(f"LLM token quota exceeded for tenant='{tenant}': used={used} limit={limit}")
    projected = used + max(0, int(additional_tokens))
    if projected > limit:
        raise LLMQuotaExceeded(f"LLM token quota exceeded for tenant='{tenant}': projected={projected} limit={limit}")
