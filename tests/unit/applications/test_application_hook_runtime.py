# © Artur Czarnecki. All rights reserved.

"""APP-CON-5 — hook runtime guard wired on harness host runtime."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_product_defaults_use_250ms_hook_timeout() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    assert env.reliability_profile.middleware_hook_timeout_seconds == 0.25


def test_build_harness_host_runtime_configures_hook_runtime_guard() -> None:
    manifest = ApplicationManifest.lab(
        app_id="hook_guard_wiring_test",
        name="Hook Guard Wiring Test",
        route_prefix="/v1/hook_guard_wiring_test",
        env_prefix="HOOK_GUARD_WIRING_TEST_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="hook_guard.lab").model_copy(
        update={
            "reliability_profile": ApplicationEnvironmentProfile.lab_defaults()
            .reliability_profile.model_copy(update={"middleware_hook_timeout_seconds": 0.5})
        }
    )
    runtime = build_harness_host_runtime(
        manifest,
        environment,
        use_in_memory_trace=True,
    )
    pipeline = resolve_harness_host_nexus_loop_legacy(runtime).middleware
    assert isinstance(pipeline, MiddlewarePipeline)
    assert pipeline._hook_timeout_seconds == 0.5  # noqa: SLF001
    assert pipeline._event_bus is resolve_harness_host_nexus_loop_legacy(runtime).event_bus  # noqa: SLF001
