# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.llm_adapters.governance.llm_cost import evaluate_llm_run_cost
from intergrax.llm_adapters.tracking.metrics import get_llm_metrics_collector, record_llm_call, set_metrics_enabled

pytestmark = pytest.mark.unit


def test_evaluate_llm_run_cost_warn_threshold() -> None:
    set_metrics_enabled(True)
    get_llm_metrics_collector().reset()
    from intergrax.llm_adapters.tracking.context import clear_llm_tenant_id

    clear_llm_tenant_id()
    record_llm_call(
        provider="groq",
        model="m",
        run_id="r",
        input_tokens=600,
        output_tokens=500,
        duration_ms=1,
        success=True,
    )
    with patch.dict("os.environ", {"INTERGRAX_LLM_GOVERNANCE_WARN_TOKENS": "1000"}, clear=False):
        ev = evaluate_llm_run_cost(tenant_id="_platform", run_id="run-9")
    assert ev.total_tokens == 1100
    assert ev.warn_threshold_exceeded is True
    assert ev.run_id == "run-9"
    set_metrics_enabled(False)
