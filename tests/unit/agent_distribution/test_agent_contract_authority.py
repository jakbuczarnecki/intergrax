# © Artur Czarnecki. All rights reserved.

"""Package AgentContract authority persistence and integrity (EBH-2E-AR1-A-R1)."""

from __future__ import annotations

import sys

import pytest

from intergrax.agent_distribution.agent_contract_authority import (
    AgentPackageContractAuthorityService,
    PackageAgentContractAuthorityRecord,
    contract_metadata_content_digest,
)
from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentArtifactMetadataStore,
)
from intergrax.contracts.agent_contract_meta import AgentContract

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _echo_contract() -> AgentContract:
    sys.path.insert(0, "agents/echo")
    try:
        from contract import build_agent_contract

        return build_agent_contract()
    finally:
        sys.path.pop(0)


def _echo_metadata() -> AgentProjectMetadata:
    return AgentProjectMetadata(
        distribution_package_id="intergrax-echo-agent",
        package_version="0.1.0",
        declared_contracts=(),
    )


def test_package_contract_authority_immutable_and_digest_bound() -> None:
    contract = _echo_contract()
    digest = "sha256:" + ("b" * 64)
    record = PackageAgentContractAuthorityRecord.from_validated_contract(
        package_digest=digest,
        distribution_package_id="intergrax-echo-agent",
        artifact_store_ref="artifact://echo",
        agent_project_metadata_ref="meta://echo",
        contract=contract,
    )
    assert record.metadata_digest == contract_metadata_content_digest(contract)
    store = InMemoryAgentArtifactMetadataStore(AgentDistributionStoreState())
    service = AgentPackageContractAuthorityService(store)
    persisted = service.persist_authority_record(record)
    loaded = service.resolve_contract(package_digest=digest, contract_id="echo")
    assert loaded.id == "echo"
    assert persisted.metadata_digest == record.metadata_digest


def test_mismatched_package_digest_on_read_fails() -> None:
    contract = _echo_contract()
    digest = "sha256:" + ("c" * 64)
    record = PackageAgentContractAuthorityRecord.from_validated_contract(
        package_digest=digest,
        distribution_package_id="intergrax-echo-agent",
        artifact_store_ref="artifact://echo",
        agent_project_metadata_ref="meta://echo",
        contract=contract,
    )
    store = InMemoryAgentArtifactMetadataStore(AgentDistributionStoreState())
    service = AgentPackageContractAuthorityService(store)
    service.persist_authority_record(record)
    other_digest = "sha256:" + ("d" * 64)
    with pytest.raises(Exception, match="missing package contract authority"):
        service.resolve_contract(package_digest=other_digest, contract_id="echo")
