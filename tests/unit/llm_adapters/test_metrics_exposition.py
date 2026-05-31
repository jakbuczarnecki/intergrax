# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.tracking.context import set_llm_tenant_id
from intergrax.llm_adapters.tracking.exposition import render_otlp_json, render_prometheus_text
from intergrax.llm_adapters.tracking.metrics import get_llm_metrics_collector, set_metrics_enabled

pytestmark = pytest.mark.unit


def test_prometheus_and_otlp_include_tenant() -> None:
    set_metrics_enabled(True)
    collector = get_llm_metrics_collector()
    collector.reset()
    set_llm_tenant_id("tenant-a")

    from intergrax.llm_adapters.tracking.metrics import record_llm_call

    record_llm_call(
        provider="openai",
        model="gpt-4o-mini",
        run_id="r1",
        input_tokens=10,
        output_tokens=5,
        duration_ms=100,
        success=True,
    )

    prom = render_prometheus_text()
    assert 'tenant_id="tenant-a"' in prom
    otlp = render_otlp_json()
    assert otlp["resourceMetrics"]
    attrs = otlp["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]["attributes"]
    assert attrs["tenant_id"] == "tenant-a"

    set_metrics_enabled(False)
