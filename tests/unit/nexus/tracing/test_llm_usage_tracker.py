# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.llm_adapters.llm_adapter import LLMRunStats

from intergrax.llm_adapters.llm_usage_track import LLMUsageTracker
from tests._support.builder import FakeLLMAdapter


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
