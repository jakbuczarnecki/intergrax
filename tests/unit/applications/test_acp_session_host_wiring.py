# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.acp_session_host_wiring import (
    build_acp_session_host_from_harness,
)
from intergrax.applications._shared.harness_host_composition import (
    HarnessHostInternalComposition,
    HarnessHostPluginRegistrationSurface,
)
from intergrax.applications._shared.runtime_boundary_adapters import (
    application_profile_to_runtime_profile,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_build_acp_session_host_from_harness_attaches_decision_gate() -> None:
    decision_gate = MagicMock()
    runtime = MagicMock()
    runtime.environment = ApplicationEnvironmentProfile.lab_defaults()
    nexus_loop = MagicMock()
    nexus_loop.peek_decision_flow_gate.return_value = decision_gate
    runtime._internal_composition = HarnessHostInternalComposition(
        execution_terminal=MagicMock(),
        event_bus=MagicMock(),
        decision_flow_gate=decision_gate,
        middleware_pipeline=MagicMock(),
        lifecycle_hook_coordinator=MagicMock(),
        plugin_surface=HarnessHostPluginRegistrationSurface(
            event_bus=MagicMock(),
            hook_registry=MagicMock(),
            policy_engine=MagicMock(),
        ),
        runtime_event_persistence=None,
        _orchestration_backend=nexus_loop,
    )
    runtime.env_wiring.tool_wiring = MagicMock()

    host_ctx = build_acp_session_host_from_harness(runtime)
    assert host_ctx.decision_flow_gate is decision_gate
    assert host_ctx.runtime_profile == application_profile_to_runtime_profile(runtime.environment)
