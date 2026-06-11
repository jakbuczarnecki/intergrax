# © Artur Czarnecki. All rights reserved.

"""Shared fixtures for cognitive pattern unit tests (ACP-10)."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity


@pytest.fixture
def pattern_run_request() -> AgentRunRequest:
    return AgentRunRequest(
        input="pattern-probe",
        identity=RequestIdentity(tenant_id="t-pattern", user_id="u-pattern"),
    )
