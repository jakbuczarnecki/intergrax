# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from intergrax.llm_adapters.base.usage_log import LLMRunStats, LLMUsageTrackable
from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker
from testing_support.builder import FakeLLMAdapter

pytestmark = pytest.mark.unit


@dataclass
class _NullStatsReader:
    def get_run_stats(self, run_id: Optional[str] = None) -> LLMRunStats | None:
        return None


class _TrackableWithNullStats:
    provider = "openai"
    model = "null-stats"

    def __init__(self) -> None:
        self.usage = _NullStatsReader()


def test_usage_tracker_multiple_labels_dedup_total_and_by_provider_model() -> None:
    run_id = "run_multi"
    tracker = LLMUsageTracker(run_id=run_id)
    a1 = FakeLLMAdapter(fixed_text="a")
    a2 = FakeLLMAdapter(fixed_text="b")
    a2.model = "m2"

    for adapter in (a1, a2):
        call = adapter.usage.begin_call(run_id=run_id)
        adapter.usage.end_call(call, input_tokens=1, output_tokens=1, success=True)

    for adapter, label in ((a1, "one"), (a1, "one_alias"), (a2, "two")):
        tracker.register_adapter(adapter, label=label)

    total = tracker.total()
    assert total.calls == 2
    assert total.input_tokens == 2

    report = tracker.build_report()
    assert len(report.entries) == 3
    assert len(report.by_provider_model) == 2


def test_usage_tracker_null_stats_yields_empty_entry_stats() -> None:
    tracker = LLMUsageTracker(run_id="run-null")
    trackable = _TrackableWithNullStats()
    assert isinstance(trackable, LLMUsageTrackable)
    tracker.register_adapter(trackable, label="null")
    entry = tracker.build_report().entries[0]
    assert entry.stats.calls == 0
    assert tracker.total().calls == 0
