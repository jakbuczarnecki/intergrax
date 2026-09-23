# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker
from testing_support.builder import FakeLLMAdapter

pytestmark = pytest.mark.unit

_PROVIDER_MODEL_KEY = "fake:fake"


def _one_call(adapter: FakeLLMAdapter, run_id: str) -> None:
    call = adapter.usage.begin_call(run_id=run_id)
    adapter.usage.end_call(call, input_tokens=1, output_tokens=1, success=True)


def _register_blocker_case(
    tracker: LLMUsageTracker,
    *,
    label1_first: bool,
) -> None:
    run_id = tracker.run_id
    source_a = FakeLLMAdapter(fixed_text="a")
    source_b = FakeLLMAdapter(fixed_text="b")
    _one_call(source_a, run_id)
    _one_call(source_b, run_id)

    if label1_first:
        tracker.register_adapter(source_a, label="label1")
        tracker.register_adapter(source_a, label="label2")
        tracker.register_adapter(source_b, label="label2")
    else:
        tracker.register_adapter(source_a, label="label2")
        tracker.register_adapter(source_b, label="label2")
        tracker.register_adapter(source_a, label="label1")


def test_by_provider_model_physical_sources_blocker_reproduction() -> None:
    """Test A: label1→A, label2→A+B; same provider/model; total and by_provider_model == 2."""
    run_id = "run-blocker-a"
    tracker = LLMUsageTracker(run_id=run_id)
    _register_blocker_case(tracker, label1_first=True)

    total = tracker.total()
    report = tracker.build_report()
    pm = report.by_provider_model[_PROVIDER_MODEL_KEY]

    assert total.calls == 2
    assert pm.calls == 2
    assert pm.input_tokens == 2


def test_by_provider_model_order_independent_registration() -> None:
    """Test B: reversed registration order yields identical accounting."""
    run_id = "run-order-b"
    tracker_first = LLMUsageTracker(run_id=run_id)
    _register_blocker_case(tracker_first, label1_first=True)

    tracker_second = LLMUsageTracker(run_id=run_id)
    _register_blocker_case(tracker_second, label1_first=False)

    r1 = tracker_first.build_report()
    r2 = tracker_second.build_report()

    assert tracker_first.total().calls == tracker_second.total().calls == 2
    assert r1.by_provider_model == r2.by_provider_model
    assert r1.by_provider_model[_PROVIDER_MODEL_KEY].calls == 2


def test_by_provider_model_separate_keys_for_different_provider_model() -> None:
    """Test C: distinct provider/model buckets."""
    run_id = "run-diff-pm-c"
    tracker = LLMUsageTracker(run_id=run_id)
    a = FakeLLMAdapter(fixed_text="a")
    b = FakeLLMAdapter(fixed_text="b")
    b.model = "other-model"
    _one_call(a, run_id)
    _one_call(b, run_id)
    tracker.register_adapter(a, label="la")
    tracker.register_adapter(b, label="lb")

    report = tracker.build_report()
    assert len(report.by_provider_model) == 2
    assert report.by_provider_model["fake:fake"].calls == 1
    assert report.by_provider_model["fake:other-model"].calls == 1
    assert report.total.calls == 2


def test_by_provider_model_alias_dedupes_physical_instance() -> None:
    """Test D: one physical instance under two labels counts once."""
    run_id = "run-alias-d"
    tracker = LLMUsageTracker(run_id=run_id)
    adapter = FakeLLMAdapter(fixed_text="x")
    _one_call(adapter, run_id)
    tracker.register_adapter(adapter, label="primary")
    tracker.register_adapter(adapter, label="alias")

    report = tracker.build_report()
    assert report.total.calls == 1
    assert report.by_provider_model[_PROVIDER_MODEL_KEY].calls == 1
    assert len(report.by_provider_model) == 1


def test_by_provider_model_multi_instance_same_logical_label() -> None:
    """Test E: one logical label with A+B aggregates entry, total, and by_provider_model."""
    run_id = "run-multi-e"
    tracker = LLMUsageTracker(run_id=run_id)
    label = "shared-label"
    a = FakeLLMAdapter(fixed_text="a")
    b = FakeLLMAdapter(fixed_text="b")
    _one_call(a, run_id)
    _one_call(b, run_id)
    tracker.register_adapter(a, label=label)
    tracker.register_adapter(b, label=label)

    report = tracker.build_report()
    assert len(report.entries) == 1
    assert report.entries[0].stats.calls == 2
    assert report.total.calls == 2
    assert report.by_provider_model[_PROVIDER_MODEL_KEY].calls == 2
