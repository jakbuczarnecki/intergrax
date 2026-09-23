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


def _one_call(adapter: FakeLLMAdapter, run_id: str, *, tokens: int = 1) -> None:
    call = adapter.usage.begin_call(run_id=run_id)
    adapter.usage.end_call(
        call, input_tokens=tokens, output_tokens=tokens, success=True
    )


@dataclass
class _NullStatsReader:
    def get_run_stats(self, run_id: Optional[str] = None) -> LLMRunStats | None:
        return None


def test_semantic_same_provider_model_multi_instance_accepted() -> None:
    """S1: two instances under one label aggregate stats."""
    run_id = "run-s1"
    tracker = LLMUsageTracker(run_id=run_id)
    label = "shared"
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


def test_semantic_different_provider_rejected() -> None:
    """S2: mixed provider under one logical label fails closed."""

    class _OpenAITrackable:
        provider = "openai"
        model = "gpt-x"

        def __init__(self) -> None:
            self.usage = _NullStatsReader()

    class _VllmTrackable:
        provider = "vllm"
        model = "gpt-x"

        def __init__(self) -> None:
            self.usage = _NullStatsReader()

    tracker = LLMUsageTracker(run_id="run-s2")
    label = "mixed-provider"
    a = _OpenAITrackable()
    b = _VllmTrackable()
    assert isinstance(a, LLMUsageTrackable)
    assert isinstance(b, LLMUsageTrackable)

    tracker.register_adapter(a, label=label)
    with pytest.raises(ValueError, match="already represents provider/model openai:gpt-x"):
        tracker.register_adapter(b, label=label)


def test_semantic_different_model_rejected() -> None:
    """S3: same provider, different model under one label fails closed."""
    tracker = LLMUsageTracker(run_id="run-s3")
    label = "mixed-model"
    a = FakeLLMAdapter(fixed_text="a")
    b = FakeLLMAdapter(fixed_text="b")
    b.model = "other-model"

    tracker.register_adapter(a, label=label)
    with pytest.raises(ValueError, match="already represents provider/model fake:fake"):
        tracker.register_adapter(b, label=label)


def test_semantic_canonical_provider_equality() -> None:
    """S4: string and enum providers canonicalize to the same slug."""

    class _SpacedOpenAITrackable:
        provider = " OPENAI "
        model = "gpt-x"

        def __init__(self) -> None:
            self.usage = _NullStatsReader()

    class _EnumOpenAITrackable:
        provider = LLMProvider.OPENAI
        model = "gpt-x"

        def __init__(self) -> None:
            self.usage = _NullStatsReader()

    tracker = LLMUsageTracker(run_id="run-s4")
    label = "canonical-openai"
    first = _SpacedOpenAITrackable()
    second = _EnumOpenAITrackable()
    assert isinstance(first, LLMUsageTrackable)
    assert isinstance(second, LLMUsageTrackable)

    tracker.register_adapter(first, label=label)
    tracker.register_adapter(second, label=label)

    assert len(tracker.registered_labels()) == 1
    assert len(tracker.build_report().entries) == 1


def test_unregister_removes_source_from_all_alias_labels() -> None:
    """U1: unregister drops every logical label that only held this instance."""
    run_id = "run-u1"
    tracker = LLMUsageTracker(run_id=run_id)
    adapter = FakeLLMAdapter(fixed_text="x")
    tracker.register_adapter(adapter, label="L1")
    tracker.register_adapter(adapter, label="L2")

    tracker.unregister_adapter(adapter)

    assert tracker.registered_labels() == []


def test_unregister_leaves_other_physical_sources() -> None:
    """U2: unregister one of two sources under the same label."""
    run_id = "run-u2"
    tracker = LLMUsageTracker(run_id=run_id)
    label = "L"
    a = FakeLLMAdapter(fixed_text="a")
    b = FakeLLMAdapter(fixed_text="b")
    _one_call(a, run_id, tokens=3)
    _one_call(b, run_id, tokens=7)

    tracker.register_adapter(a, label=label)
    tracker.register_adapter(b, label=label)
    tracker.unregister_adapter(a)

    assert tracker.registered_labels() == [label]
    entry = tracker.build_report().entries[0]
    assert entry.stats.calls == 1
    assert entry.stats.input_tokens == 7


def test_unregister_idempotent() -> None:
    """U3: repeated unregister is safe."""
    tracker = LLMUsageTracker(run_id="run-u3")
    adapter = FakeLLMAdapter(fixed_text="x")
    tracker.register_adapter(adapter, label="L")

    tracker.unregister_adapter(adapter)
    tracker.unregister_adapter(adapter)

    assert tracker.registered_labels() == []


def test_unregister_alias_and_other_source() -> None:
    """U4: L1→A only; L2→A+B; unregister A removes L1, L2 keeps B."""
    run_id = "run-u4"
    tracker = LLMUsageTracker(run_id=run_id)
    a = FakeLLMAdapter(fixed_text="a")
    b = FakeLLMAdapter(fixed_text="b")
    _one_call(a, run_id)
    _one_call(b, run_id)

    tracker.register_adapter(a, label="L1")
    tracker.register_adapter(a, label="L2")
    tracker.register_adapter(b, label="L2")

    tracker.unregister_adapter(a)

    labels = sorted(tracker.registered_labels())
    assert labels == ["L2"]
    entry = tracker.build_report().entries[0]
    assert entry.label == "L2"
    assert entry.stats.calls == 1
