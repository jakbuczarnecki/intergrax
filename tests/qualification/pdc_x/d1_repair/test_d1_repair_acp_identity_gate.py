# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-A / F-D1-11 (ACP group): the PRE_MODEL governance identity gate stays fail-closed.

The 12 stale ACP tests now bind an active governance identity explicitly (fixture
``bound_governance_identity`` in ``tests/unit/agents/authoring/conftest.py``). These tests pin
that the product gate itself is unchanged: no identity, or an identity of another tenant,
still stops the run.
"""

from __future__ import annotations

import pytest
from intergrax.agents.authoring.patterns.reference import PatternReflexProbe
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    peek_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.policy.pre_model_policy_errors import PreModelPolicyConfigurationError

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request() -> AgentRunRequest:
    return AgentRunRequest(
        input="pattern-probe",
        identity=RequestIdentity(tenant_id="t-pattern", user_id="u-pattern"),
    )


async def test_acp_run_without_an_active_governance_identity_is_refused() -> None:
    assert peek_active_execution_governance_identity() is None

    with pytest.raises(PreModelPolicyConfigurationError, match="governance identity unavailable"):
        await PatternReflexProbe().run(_request())


async def test_acp_run_with_a_governance_identity_of_another_tenant_is_refused() -> None:
    token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(tenant_id="t-other", workspace_id="ws", principal_id="u-pattern")
    )
    try:
        with pytest.raises(PreModelPolicyConfigurationError):
            await PatternReflexProbe().run(_request())
    finally:
        reset_active_execution_governance_identity(token)


async def test_acp_run_with_a_matching_governance_identity_succeeds() -> None:
    token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(tenant_id="t-pattern", workspace_id="ws", principal_id="u-pattern")
    )
    try:
        result = await PatternReflexProbe().run(_request())
    finally:
        reset_active_execution_governance_identity(token)

    assert result.status == AgentRunStatus.SUCCEEDED
