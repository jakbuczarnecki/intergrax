# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from intergrax.applications._shared.adaptive_runtime_bridge import apply_adaptive_profiles_from_environment
from intergrax.applications._shared.adaptive_wiring import wire_adaptive_profile
from intergrax.applications.contracts.environment_profile import AdaptiveProfile, ApplicationEnvironmentProfile
from intergrax.debug.app import create_debug_app
from intergrax.integrations.contracts.feature_flag import FeatureFlagEvaluation
from intergrax.integrations.registry.harness_lab_stack import (
    HARNESS_LAB_STABLE_SLUGS,
    HARNESS_M6_P4_PROBE_SLUGS,
    HARNESS_M6_P5_PROBE_SLUGS,
    HARNESS_M6_P6_PROBE_SLUGS,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.nexus.config import RuntimeConfig
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class _DisabledFlagBackend:
    def is_enabled(self, flag_key: str, *, tenant_id: str, user_id: str = "") -> bool:
        del flag_key, tenant_id, user_id
        return False

    def evaluate(
        self,
        flag_key: str,
        *,
        tenant_id: str,
        user_id: str = "",
    ) -> FeatureFlagEvaluation:
        return FeatureFlagEvaluation(key=flag_key, enabled=False)


def test_apply_adaptive_profiles_uses_gated_wiring_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "adaptive_profile": AdaptiveProfile(
                enabled=True,
                mode="recommend",
                feature_flag_slug="unleash",
                rollout_flag_key="harness.adaptive.recommend",
            ),
            "integration_profile": IntegrationProfile(feature_flag=_DisabledFlagBackend()),
        }
    )
    wiring = wire_adaptive_profile(env, tenant_id="tenant-a")
    assert wiring.profile.mode == "observe"

    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), tenant_id="tenant-a")
    updated = apply_adaptive_profiles_from_environment(
        config,
        env,
        wiring=wiring,
        tenant_id="tenant-a",
    )
    assert updated.adaptive_profile is not None
    assert updated.adaptive_profile.mode == "observe"


def test_integration_health_debug_route_lab_stack() -> None:
    app = create_debug_app(include_integration_health_routes=True)
    client = TestClient(app)
    response = client.get("/debug/integrations/health?stack=lab")
    assert response.status_code == 200
    payload = response.json()
    assert payload["stack"] == "lab"
    assert payload["count"] == len(HARNESS_LAB_STABLE_SLUGS)
    slugs = {item["slug"] for item in payload["probes"]}
    assert slugs == set(HARNESS_LAB_STABLE_SLUGS)


def test_integration_health_debug_route_m6_p4_stack() -> None:
    app = create_debug_app(include_integration_health_routes=True)
    client = TestClient(app)
    response = client.get("/debug/integrations/health?stack=m6_p4")
    assert response.status_code == 200
    payload = response.json()
    assert payload["stack"] == "m6_p4"
    assert payload["count"] == len(HARNESS_M6_P4_PROBE_SLUGS)
    slugs = {item["slug"] for item in payload["probes"]}
    assert slugs == set(HARNESS_M6_P4_PROBE_SLUGS)


def test_integration_health_debug_route_m6_p5_stack() -> None:
    app = create_debug_app(include_integration_health_routes=True)
    client = TestClient(app)
    response = client.get("/debug/integrations/health?stack=m6_p5")
    assert response.status_code == 200
    payload = response.json()
    assert payload["stack"] == "m6_p5"
    assert payload["count"] == len(HARNESS_M6_P5_PROBE_SLUGS)
    slugs = {item["slug"] for item in payload["probes"]}
    assert slugs == set(HARNESS_M6_P5_PROBE_SLUGS)


def test_integration_health_debug_route_m6_p6_stack() -> None:
    app = create_debug_app(include_integration_health_routes=True)
    client = TestClient(app)
    response = client.get("/debug/integrations/health?stack=m6_p6")
    assert response.status_code == 200
    payload = response.json()
    assert payload["stack"] == "m6_p6"
    assert payload["count"] == len(HARNESS_M6_P6_PROBE_SLUGS)
    slugs = {item["slug"] for item in payload["probes"]}
    assert slugs == set(HARNESS_M6_P6_PROBE_SLUGS)


def test_integration_health_debug_route_disabled_by_default() -> None:
    app = create_debug_app(include_integration_health_routes=False)
    client = TestClient(app)
    response = client.get("/debug/integrations/health")
    assert response.status_code == 404
