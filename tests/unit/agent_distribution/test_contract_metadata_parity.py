# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.agent_distribution.agent_capability_metadata import AgentCapabilityDescriptor
from intergrax.agent_distribution.agent_project_metadata import (
    parse_agent_project_pyproject,
    project_agent_capability_descriptors,
)
from intergrax.agent_distribution.builtin_capability_metadata import (
    PackageAgentCapabilityMetadataProvider,
)
from intergrax.agent_distribution.contract_metadata_parity import (
    AgentContractMetadataParityError,
    validate_agent_contract_metadata_parity,
)
from legal.legal_agent import LegalAgent
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _descriptor_for_contract_id(contract_id: str) -> AgentCapabilityDescriptor:
    provider = PackageAgentCapabilityMetadataProvider(
        package_roots=(
            REPO_ROOT / "agents" / "echo",
            REPO_ROOT / "agents" / "legal",
            REPO_ROOT / "agents" / "research",
        )
    )
    descriptors = {
        item.contract_id: item for item in provider.list_agent_capability_descriptors()
    }
    return descriptors[contract_id]


@pytest.mark.parametrize(
    ("contract_id", "agent_factory"),
    [
        ("echo", EchoAgent),
        ("legal", LegalAgent),
        ("research", ResearchAgent),
        ("research-summary", SummaryAgent),
    ],
)
def test_representative_package_metadata_matches_runtime_contract(
    contract_id: str,
    agent_factory: type,
) -> None:
    descriptor = _descriptor_for_contract_id(contract_id)
    contract = agent_factory().get_contract()
    validate_agent_contract_metadata_parity(descriptor=descriptor, contract=contract)


def test_parity_fails_when_package_version_differs_from_runtime_contract_version() -> None:
    descriptor = AgentCapabilityDescriptor(
        contract_id="echo",
        agent_version="0.1.0",
        capabilities=("echo.basic",),
        skill_ids=("harness.tool_smoke",),
    )
    contract = EchoAgent().get_contract()
    with pytest.raises(AgentContractMetadataParityError, match="agent_version"):
        validate_agent_contract_metadata_parity(descriptor=descriptor, contract=contract)


def test_parity_fails_on_capability_drift() -> None:
    descriptor = AgentCapabilityDescriptor(
        contract_id="echo",
        agent_version="1.0.0",
        capabilities=("echo.basic", "echo.extra"),
        skill_ids=("harness.tool_smoke",),
    )
    contract = EchoAgent().get_contract()
    with pytest.raises(AgentContractMetadataParityError, match="capabilities"):
        validate_agent_contract_metadata_parity(descriptor=descriptor, contract=contract)


def test_parity_fails_on_skill_id_drift() -> None:
    descriptor = AgentCapabilityDescriptor(
        contract_id="legal",
        agent_version="0.1.0",
        capabilities=("legal.review",),
        skill_ids=("legal.contract_review", "legal.case_research"),
    )
    contract = LegalAgent().get_contract()
    with pytest.raises(AgentContractMetadataParityError, match="skill_ids"):
        validate_agent_contract_metadata_parity(descriptor=descriptor, contract=contract)


def test_parity_fails_on_tool_id_drift() -> None:
    descriptor = AgentCapabilityDescriptor(
        contract_id="echo",
        agent_version="1.0.0",
        capabilities=("echo.basic",),
        skill_ids=("harness.tool_smoke",),
        tool_ids=("rag.retrieve",),
    )
    contract = EchoAgent().get_contract()
    with pytest.raises(AgentContractMetadataParityError, match="tool_ids"):
        validate_agent_contract_metadata_parity(descriptor=descriptor, contract=contract)


def test_research_package_projects_both_contract_versions_independently() -> None:
    metadata = parse_agent_project_pyproject(
        (REPO_ROOT / "agents" / "research" / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert metadata.package_version == "0.1.0"
    descriptors = {
        item.contract_id: item
        for item in project_agent_capability_descriptors(metadata)
    }
    assert descriptors["research"].agent_version == "0.1.0"
    assert descriptors["research-summary"].agent_version == "0.1.0"
    validate_agent_contract_metadata_parity(
        descriptor=descriptors["research"],
        contract=ResearchAgent().get_contract(),
    )
    validate_agent_contract_metadata_parity(
        descriptor=descriptors["research-summary"],
        contract=SummaryAgent().get_contract(),
    )
