# © Artur Czarnecki. All rights reserved.

"""DecisionPluginProfile configuration contract (P0-A-R3 migration)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.applications.contracts.environment_profile import DecisionPluginProfile
from intergrax.core.plugins.discovery import EP_DECISION_STRATEGIES
from intergrax.core.plugins.selection_ref import PlatformPluginSelectionRef

pytestmark = pytest.mark.unit


def _strategy_ref(
    plugin_id: str,
    *,
    entry_point_name: str = "external_council",
    distribution: str = "my-decision-strategy-pkg",
) -> PlatformPluginSelectionRef:
    return PlatformPluginSelectionRef(
        plugin_id=plugin_id,
        entry_point_group=EP_DECISION_STRATEGIES,
        entry_point_name=entry_point_name,
        distribution=distribution,
    )


def test_legacy_kind_fields_rejected_by_extra_forbid() -> None:
    with pytest.raises(ValidationError, match="strategy_kinds"):
        DecisionPluginProfile.model_validate(
            {"strategy_kinds": ["council"], "discover_entry_points": True},
        )


def test_strategy_plugins_accepted() -> None:
    profile = DecisionPluginProfile(
        discover_entry_points=True,
        strategy_plugins=[_strategy_ref("external_council_variant")],
    )
    assert len(profile.strategy_plugins) == 1
    assert profile.strategy_plugins[0].plugin_id == "external_council_variant"


def test_duplicate_plugin_id_rejected() -> None:
    ref = _strategy_ref("dup.strategy")
    with pytest.raises(ValidationError, match="duplicate plugin_id"):
        DecisionPluginProfile(
            strategy_plugins=[ref, ref],
        )


def test_duplicate_locator_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate plugin locator"):
        DecisionPluginProfile(
            strategy_plugins=[
                _strategy_ref("strategy.a", entry_point_name="ep_one"),
                _strategy_ref("strategy.b", entry_point_name="ep_one"),
            ],
        )


def test_empty_plugin_id_rejected() -> None:
    with pytest.raises(ValidationError):
        PlatformPluginSelectionRef(
            plugin_id="   ",
            entry_point_group=EP_DECISION_STRATEGIES,
            entry_point_name="ep",
            distribution="pkg",
        )


def test_selection_ref_normalizes_distribution_identity() -> None:
    base = {
        "plugin_id": "plugin.example",
        "entry_point_group": EP_DECISION_STRATEGIES,
        "entry_point_name": "example",
    }
    ref_underscore = PlatformPluginSelectionRef(distribution="My_Plugin", **base)
    ref_hyphen = PlatformPluginSelectionRef(distribution="my-plugin", **base)
    ref_dot = PlatformPluginSelectionRef(distribution="my.plugin", **base)
    assert ref_underscore.distribution == "my-plugin"
    assert ref_hyphen.distribution == "my-plugin"
    assert ref_dot.distribution == "my-plugin"
