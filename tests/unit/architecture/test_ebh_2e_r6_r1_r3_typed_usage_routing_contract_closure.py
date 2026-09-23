# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6-R1-R3 — usage observability & routing profile typed contract closure."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from intergrax.llm_adapters.base.usage_log import LLMRunStats, LLMUsageTrackable
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
from intergrax.llm_adapters.contracts.routing_profile import LLMRoutingProfile
from intergrax.llm_adapters.routing.profile_source import RoutingProfileSource
from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker
from tests.unit.architecture.ebh_2e_external_structural_llm_adapter import (
    ExternalStructuralAdapter,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_USAGE_TRACKER_SOURCE = _REPO_ROOT / "intergrax/llm_adapters/tracking/llm_usage_track.py"
_PROFILE_SOURCE_SOURCE = _REPO_ROOT / "intergrax/llm_adapters/routing/profile_source.py"


def test_ebh_2e_r6_r1_r3_usage_tracker_register_declares_trackable_not_plain_adapter() -> None:
    tree = ast.parse(_USAGE_TRACKER_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "register_adapter":
            continue
        for arg in node.args.args:
            if arg.arg == "trackable" and arg.annotation is not None:
                ann = ast.unparse(arg.annotation)
                assert "LLMUsageTrackable" in ann
                return
    pytest.fail("register_adapter must declare LLMUsageTrackable parameter")


def test_ebh_2e_r6_r1_r3_usage_tracker_source_no_adapter_id_or_plain_usage() -> None:
    text = _USAGE_TRACKER_SOURCE.read_text(encoding="utf-8")
    assert "adapter.id" not in text
    assert "ad.usage" not in text


def test_ebh_2e_r6_r1_r3_routing_profile_source_no_any() -> None:
    text = _PROFILE_SOURCE_SOURCE.read_text(encoding="utf-8")
    assert "Any" not in text
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RoutingProfileSource":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and item.annotation is not None:
                    pytest.fail("RoutingProfileSource must use @property, not bare annotation")
            return


@dataclass
class _InMemoryRunStatsReader:
    _by_run: dict[str, LLMRunStats]

    def get_run_stats(self, run_id: Optional[str] = None) -> LLMRunStats | None:
        rid = run_id or "general"
        return self._by_run.get(rid)


class ExternalUsageTrackableAdapter:
    """Structural usage-trackable adapter — no BaseLLMAdapter inheritance."""

    provider = "external-provider"
    model = "ext-usage-model"

    def __init__(self) -> None:
        self.usage = _InMemoryRunStatsReader(
            _by_run={"run-struct": LLMRunStats(calls=2, input_tokens=3, output_tokens=5, total_tokens=8)}
        )


def test_ebh_2e_r6_r1_r3_structural_usage_trackable_registration_and_report() -> None:
    adapter = ExternalUsageTrackableAdapter()
    assert isinstance(adapter, LLMUsageTrackable)

    tracker = LLMUsageTracker(run_id="run-struct")
    tracker.register_adapter(adapter, label="external_usage")

    report = tracker.build_report()
    assert report.total.calls == 2
    assert report.total.total_tokens == 8
    assert report.entries[0].meta.provider == "external-provider"


def test_ebh_2e_r6_r1_r3_plain_llm_adapter_not_trackable_explicit_type_error() -> None:
    plain = ExternalStructuralAdapter()
    assert not isinstance(plain, LLMUsageTrackable)
    tracker = LLMUsageTracker(run_id="run-plain")
    with pytest.raises(TypeError, match="LLMUsageTrackable"):
        tracker.register_adapter(plain)  # type: ignore[arg-type]


class ExternalRoutingProfileSource:
    def __init__(self, profile: LLMRoutingProfile | None) -> None:
        self._profile = profile

    @property
    def llm_routing_profile(self) -> LLMRoutingProfile | None:
        return self._profile


def test_ebh_2e_r6_r1_r3_structural_routing_profile_source() -> None:
    profile = LLMRoutingProfile(default_profile=LLMProfile(provider="openai", model="gpt-4o-mini"))
    source: RoutingProfileSource = ExternalRoutingProfileSource(profile)
    assert source.llm_routing_profile is profile
