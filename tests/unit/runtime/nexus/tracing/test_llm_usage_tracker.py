# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.llm_adapters.base.usage_log import LLMRunStats

from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker
from testing_support.builder import FakeLLMAdapter


pytestmark = pytest.mark.unit


def test_llm_usage_tracker_export_shape_and_total_dedup() -> None:
    run_id = "run_test_001"
    tracker = LLMUsageTracker(run_id=run_id)

    adapter = FakeLLMAdapter(fixed_text="OK")

    call = adapter.usage.begin_call(run_id=run_id)
    adapter.usage.end_call(call, input_tokens=10, output_tokens=20, success=True)

    # Register the same adapter instance twice under two labels to verify total dedup by instance id.
    tracker.register_adapter(adapter, label="primary")
    tracker.register_adapter(adapter, label="alias")

    payload = tracker.export()

    # Contract: export must follow LLMUsageReport.to_dict() shape (asdict).
    assert payload["run_id"] == run_id
    assert "total" in payload
    assert "entries" in payload
    assert "by_provider_model" in payload
    assert "adapter_instance_ids" in payload

    # Guardrail: legacy export used "adapters" key (dict per label). This must not appear.
    assert "adapters" not in payload

    total = payload["total"]
    assert total["calls"] == 1
    assert total["input_tokens"] == 10
    assert total["output_tokens"] == 20
    assert total["total_tokens"] == 30
    assert isinstance(total["duration_ms"], int)
    assert total["duration_ms"] >= 0
    assert total["errors"] == 0

    # Even though we registered two labels, entries list can contain two entries,
    # but total must be deduplicated by adapter instance.
    assert len(payload["entries"]) == 2


def test_llm_usage_tracker_same_label_two_instances_aggregates_usage() -> None:
    run_id = "run_multi_instance"
    tracker = LLMUsageTracker(run_id=run_id)
    label = "core_inner:vllm:model-x"

    adapter_a = FakeLLMAdapter(fixed_text="a")
    tracker.register_adapter(adapter_a, label=label)
    call_a = adapter_a.usage.begin_call(run_id=run_id)
    adapter_a.usage.end_call(call_a, input_tokens=1, output_tokens=1, success=True)

    adapter_b = FakeLLMAdapter(fixed_text="b")
    tracker.register_adapter(adapter_b, label=label)
    call_b = adapter_b.usage.begin_call(run_id=run_id)
    adapter_b.usage.end_call(call_b, input_tokens=2, output_tokens=2, success=True)

    report = tracker.build_report()
    by_label = {e.label: e for e in report.entries}
    assert len(by_label) == 1
    assert by_label[label].stats.calls == 2
    assert report.total.calls == 2
