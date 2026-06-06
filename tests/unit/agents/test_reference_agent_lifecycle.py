# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.registry.agent_assembly_resolver import validate_agent_assembly
from legal.legal_agent import LegalAgent
from organization_worker.organization_worker_agent import OrganizationWorkerAgent
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent
from signoff_probe.signoff_probe_agent import SignoffProbeAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REFERENCE_AGENT_FACTORIES: tuple[type, ...] = (
    EchoAgent,
    LegalAgent,
    ResearchAgent,
    SummaryAgent,
    SignoffProbeAgent,
    OrganizationWorkerAgent,
)


@pytest.mark.parametrize("factory", _REFERENCE_AGENT_FACTORIES)
def test_reference_agents_declare_lifecycle_metadata(factory: type) -> None:
    agent = factory()
    contract: AgentContract = agent.get_contract()
    assert (contract.owner_team or "").strip(), f"{contract.id} requires owner_team"
    result = validate_agent_assembly(contract)
    assert result.valid, f"{contract.id}: {result.errors}"


def test_echo_agent_is_production_eligible_with_complete_metadata() -> None:
    contract = EchoAgent().get_contract()
    assert contract.production_eligible is True
    assert (contract.owner_contact or "").strip()
    assert (contract.runbook_ref or "").strip()
