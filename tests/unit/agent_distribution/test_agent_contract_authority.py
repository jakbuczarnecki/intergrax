# © Artur Czarnecki. All rights reserved.

"""Package AgentContract authority persistence and integrity (EBH-2E-AR1-A-R1)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from intergrax.agent_distribution.admin_models import InstallAgentRequest, InstallationMutationResult
from intergrax.agent_distribution.agent_contract_authority import (
    AgentPackageContractAuthorityService,
    PackageAgentContractAuthorityRecord,
    contract_metadata_content_digest,
)
from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentArtifactMetadataStore,
)
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
    QualificationStatus,
)
from intergrax.contracts.agent_contract_meta import AgentContract
from testing_support.acquisition_install_contract_authority_port import (
    ProofInstallContractAuthorityLifecyclePort,
)
from testing_support.agent_platform_admin_harness import admin_test_principal
from testing_support.canonical_agent_lifecycle_composition import (
    CANONICAL_PING_CONTRACT_ID,
    default_stage15_proof_config,
)
from testing_support.canonical_lifecycle_ping_agent import CanonicalPingAgent

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


def test_proof_install_authority_port_enriches_digest_bound_authority() -> None:
    digest = "sha256:" + ("e" * 64)
    package_identity = AgentPackageIdentity(
        distribution_package_id="intergrax-canonical-ping",
        package_version="0.0.1",
        package_digest=digest,
    )
    trust = AgentInstallationTrustRecord(
        qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=digest,
        publisher_identity_ref="publisher:proof",
        source_provider_id="builtin",
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id="evidence:proof",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )
    request = InstallAgentRequest(
        mutation_id="mut-proof-authority",
        installation_id="inst-proof",
        installation_slot_id="slot-proof",
        package_identity=package_identity,
        artifact_store_ref="artifact://proof",
        trust_record=trust,
        agent_project_metadata_ref="meta://proof",
    )
    inner = MagicMock()
    inner.install_agent.return_value = MagicMock(spec=InstallationMutationResult)
    proof_config = default_stage15_proof_config()
    port = ProofInstallContractAuthorityLifecyclePort(inner=inner, proof_config=proof_config)
    principal = admin_test_principal()
    port.install_agent(
        application_id="app-proof",
        application_environment_id="env-proof",
        request=request,
        principal=principal,
    )
    enriched = inner.install_agent.call_args.kwargs["request"]
    assert enriched.package_contract_authority
    record = enriched.package_contract_authority[0]
    assert record.package_digest == digest
    assert record.metadata_digest == contract_metadata_content_digest(CanonicalPingAgent().get_contract())
    assert record.contract_id == CANONICAL_PING_CONTRACT_ID


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
