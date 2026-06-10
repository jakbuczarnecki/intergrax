# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from vendor_discovery.vendor_discovery_agent import VendorDiscoveryAgent
from vendor_discovery.schemas.output import VendorDiscoveryOutput
from vendor_discovery.steps.domain import build_stub_vendor_discovery_output
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState


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
@pytest.mark.integration
@pytest.mark.gate
async def test_vendor_discovery_runs_through_nexus() -> None:
    registry = AgentRegistry()
    registry.register(VendorDiscoveryAgent(), requires_uaep=True)
    loop = NexusLoop(registry)
    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="need CRM vendor",
            context=TaskContext(capability="vendor_discovery.search"),
        )
    )
    assert result.state == TaskState.COMPLETED
    assert result.agent_id == "vendor_discovery"
    payload = json.loads(result.answer)
    parsed = VendorDiscoveryOutput.model_validate(payload)
    assert parsed.candidates
    assert "CRM vendor" in parsed.candidates[0].name
