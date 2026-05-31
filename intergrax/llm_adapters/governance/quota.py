# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""
Per-tenant LLM token budgets (platform governance hook).

Uses in-process metrics from :mod:`intergrax.llm_adapters.tracking.metrics`.
Configure via ``INTERGRAX_LLM_TENANT_MAX_TOKENS`` (0 = disabled).
"""

from __future__ import annotations

import os
from typing import Optional

from intergrax.llm_adapters.tracking.metrics import get_llm_metrics_collector


class LLMQuotaExceeded(RuntimeError):
    """Raised when a tenant exceeds configured LLM token budget."""


def _max_tokens_per_tenant() -> int:
    raw = os.getenv("INTERGRAX_LLM_TENANT_MAX_TOKENS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def check_llm_tenant_quota(
    tenant_id: Optional[str],
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
        raise LLMQuotaExceeded(
            f"LLM token quota exceeded for tenant='{tenant}': "
            f"used={used} limit={limit}"
        )
    projected = used + max(0, int(additional_tokens))
    if projected > limit:
        raise LLMQuotaExceeded(
            f"LLM token quota exceeded for tenant='{tenant}': "
            f"projected={projected} limit={limit}"
        )
