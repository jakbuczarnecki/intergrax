# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.acp_session_host_wiring import (
    build_acp_session_host_from_harness,
)
from intergrax.applications._shared.runtime_boundary_adapters import (
    application_profile_to_runtime_profile,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_build_acp_session_host_from_harness_attaches_critic_hooks() -> None:
    critic_hooks = MagicMock()
    runtime = MagicMock()
    runtime.environment = ApplicationEnvironmentProfile.lab_defaults()
    runtime.critic.graph_hooks = critic_hooks
    runtime.env_wiring.tool_wiring = MagicMock()

    host_ctx = build_acp_session_host_from_harness(runtime)
    assert host_ctx.critic_graph_hooks is critic_hooks
    assert host_ctx.runtime_profile == application_profile_to_runtime_profile(runtime.environment)
