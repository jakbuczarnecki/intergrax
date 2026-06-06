# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.adaptive_feature_flag_gate import (
    effective_adaptive_mode,
    resolve_effective_adaptive_profile,
)
from intergrax.applications.contracts.environment_profile import AdaptiveProfile
from intergrax.integrations.contracts.feature_flag import FeatureFlagEvaluation
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _StubFeatureFlagBackend:
    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled

    def is_enabled(self, flag_key: str, *, tenant_id: str, user_id: str = "") -> bool:
        del flag_key, tenant_id, user_id
        return self._enabled

    def evaluate(
        self,
        flag_key: str,
        *,
        tenant_id: str,
        user_id: str = "",
    ) -> FeatureFlagEvaluation:
        return FeatureFlagEvaluation(key=flag_key, enabled=self._enabled)


def test_observe_mode_bypasses_feature_flag_gate() -> None:
    profile = AdaptiveProfile(enabled=True, mode="observe", rollout_flag_key="harness.adaptive.recommend")
    integration = IntegrationProfile(feature_flag=_StubFeatureFlagBackend(enabled=False))
    effective = resolve_effective_adaptive_profile(
        profile,
        integration_profile=integration,
        tenant_id="tenant-a",
    )
    assert effective.mode == "observe"


def test_recommend_downgrades_when_flag_disabled() -> None:
    profile = AdaptiveProfile(
        enabled=True,
        mode="recommend",
        feature_flag_slug="unleash",
        rollout_flag_key="harness.adaptive.recommend",
    )
    integration = IntegrationProfile(feature_flag=_StubFeatureFlagBackend(enabled=False))
    effective = resolve_effective_adaptive_profile(
        profile,
        integration_profile=integration,
        tenant_id="tenant-a",
    )
    assert effective.mode == "observe"


def test_recommend_kept_when_flag_enabled() -> None:
    profile = AdaptiveProfile(
        enabled=True,
        mode="recommend",
        feature_flag_slug="unleash",
        rollout_flag_key="harness.adaptive.recommend",
    )
    integration = IntegrationProfile(feature_flag=_StubFeatureFlagBackend(enabled=True))
    effective = resolve_effective_adaptive_profile(
        profile,
        integration_profile=integration,
        tenant_id="tenant-a",
    )
    assert effective.mode == "recommend"
    assert effective_adaptive_mode(
        profile,
        integration_profile=integration,
        tenant_id="tenant-a",
    ) == "recommend"
