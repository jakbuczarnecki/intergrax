# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from intergrax.llm_adapters.base.usage_log import LLMRunStats, LLMUsageTrackable
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
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


def test_usage_tracker_default_label_is_semantic_without_object_id() -> None:
    tracker = LLMUsageTracker(run_id="run-label")
    a1 = _TrackableWithNullStats()
    a2 = _TrackableWithNullStats()

    tracker.register_adapter(a1)
    tracker.register_adapter(a2)

    labels = tracker.registered_labels()
    assert labels == ["openai:null-stats"]
    assert "@" not in labels[0]
    assert str(id(a1)) not in labels[0]
    assert str(id(a2)) not in labels[0]


def test_usage_tracker_default_label_idempotent_for_same_semantics() -> None:
    tracker = LLMUsageTracker(run_id="run-idem")
    first = _TrackableWithNullStats()
    second = _TrackableWithNullStats()

    tracker.register_adapter(first)
    tracker.register_adapter(second)

    assert tracker.registered_labels() == ["openai:null-stats"]
    report = tracker.build_report()
    assert len(report.entries) == 1
    assert report.adapter_instance_ids["openai:null-stats"] == id(first)


def test_usage_tracker_adapter_instance_id_distinct_from_logical_label() -> None:
    tracker = LLMUsageTracker(run_id="run-inst")
    trackable = _TrackableWithNullStats()
    tracker.register_adapter(trackable, label="custom-label")

    entry = tracker.build_report().entries[0]
    assert entry.label == "custom-label"
    assert entry.adapter_instance_id == id(trackable)
    assert entry.label != str(entry.adapter_instance_id)


def test_usage_tracker_canonical_provider_from_enum() -> None:
    tracker = LLMUsageTracker(run_id="run-enum")

    class _EnumProviderTrackable:
        provider = LLMProvider.OPENAI
        model = "gpt-4o-mini"

        def __init__(self) -> None:
            self.usage = _NullStatsReader()

    trackable = _EnumProviderTrackable()
    assert isinstance(trackable, LLMUsageTrackable)
    tracker.register_adapter(trackable)

    assert tracker.build_report().entries[0].meta.provider == "openai"


def test_usage_tracker_null_stats_yields_empty_entry_stats() -> None:
    tracker = LLMUsageTracker(run_id="run-null")
    trackable = _TrackableWithNullStats()
    assert isinstance(trackable, LLMUsageTrackable)
    tracker.register_adapter(trackable, label="null")
    entry = tracker.build_report().entries[0]
    assert entry.stats.calls == 0
    assert tracker.total().calls == 0
