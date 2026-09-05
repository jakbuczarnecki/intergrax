# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from vendor_discovery.vendor_discovery_agent import VendorDiscoveryAgent
from vendor_discovery.contract import build_agent_contract
from vendor_discovery.schemas.output import VendorDiscoveryOutput
from vendor_discovery.steps.domain import build_stub_vendor_discovery_output
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus
from testing_support.builder import canonical_execution_identity_scope


@pytest.mark.unit
def test_stub_output_is_valid_vendor_discovery_schema() -> None:
    report = build_stub_vendor_discovery_output("CRM for SMB")
    assert report.candidates
    assert report.candidates[0].vendor_id == "stub-1"
    assert 0.0 <= report.confidence <= 1.0


@pytest.mark.unit
def test_contract_declares_canon_capabilities() -> None:
    contract = VendorDiscoveryAgent().get_contract()
    assert "vendor_discovery.search" in contract.capabilities
    assert any(s.skill_id == "research.literature_scan" for s in contract.skills)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_vendor_discovery_agent_typed_run_smoke() -> None:
    agent = VendorDiscoveryAgent()
    contract = build_agent_contract()
    with canonical_execution_identity_scope("agent-smoke"):
        result = await agent.run(
            AgentRunRequest(
                input="need CRM vendor",
                identity=RequestIdentity(tenant_id="t1", user_id="u1"),
                agent_id=contract.id,
            )
        )
    assert result.status == AgentRunStatus.SUCCEEDED
    assert "stub-1" in str(result.output)
    assert "CRM vendor" in str(result.output)
