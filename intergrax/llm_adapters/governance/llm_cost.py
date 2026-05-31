# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Post-run LLM cost signals for governance (platform, not agent-specific)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from intergrax.llm_adapters.tracking.metrics import get_llm_metrics_collector


@dataclass(frozen=True)
class LLMCostEvaluation:
    tenant_id: str
    run_id: str
    total_tokens: int
    total_calls: int
    total_errors: int
    warn_threshold_exceeded: bool
    reasons: List[str]
    per_provider: Dict[str, Dict[str, int]]


def _warn_token_threshold() -> int:
    raw = os.getenv("INTERGRAX_LLM_GOVERNANCE_WARN_TOKENS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def evaluate_llm_run_cost(
    *,
    tenant_id: str,
    run_id: str,
) -> LLMCostEvaluation:
    """
    Summarize tenant LLM usage for governance logging after a Nexus task completes.

    Does not block execution — use ``check_llm_tenant_quota`` for hard caps.
    """
    snapshot = get_llm_metrics_collector().snapshot_for_tenant(tenant_id)
    total_tokens = 0
    total_calls = 0
    total_errors = 0
    for stats in snapshot.values():
        total_tokens += stats["input_tokens"] + stats["output_tokens"]
        total_calls += stats["calls"]
        total_errors += stats["errors"]

    reasons: List[str] = []
    threshold = _warn_token_threshold()
    warn = threshold > 0 and total_tokens >= threshold
    if warn:
        reasons.append(f"llm_tokens>={threshold}")

    return LLMCostEvaluation(
        tenant_id=tenant_id,
        run_id=run_id,
        total_tokens=total_tokens,
        total_calls=total_calls,
        total_errors=total_errors,
        warn_threshold_exceeded=warn,
        reasons=reasons,
        per_provider=snapshot,
    )
