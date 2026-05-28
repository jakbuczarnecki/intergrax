# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.registry.bootstrap import build_organization_worker_registry
from intergrax.runtime.task.task import Task, TaskContext
from organization_worker.organization_worker_agent import (
    ORG_VENDOR_REPORT_CAPABILITY,
    OrganizationWorkerAgent,
)


@pytest.mark.unit
@pytest.mark.gate
def test_organization_worker_agent_requests_hitl_before_complete():
    agent = OrganizationWorkerAgent()
    contract = agent.get_contract()
    assert ORG_VENDOR_REPORT_CAPABILITY in contract.capabilities

    match = agent.can_handle(TaskContext(capability=ORG_VENDOR_REPORT_CAPABILITY))
    assert match.matched is True


@pytest.mark.unit
@pytest.mark.gate
def test_build_organization_worker_registry():
    registry = build_organization_worker_registry()
    assert registry.get("organization_worker") is not None
