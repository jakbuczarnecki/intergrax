# © Artur Czarnecki. All rights reserved.

"""EBH-2E-AR1-A-R1-R1 — install authority atomicity and revision serving closure gates."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from intergrax.agent_distribution.admin_models import InstallAgentRequest
from intergrax.agent_distribution.agent_contract_authority import (
    AgentPackageContractAuthorityService,
    PackageAgentContractAuthorityError,
    PackageAgentContractAuthorityRecord,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentPackageContractDeclaration,
    AgentProjectMetadata,
)
from intergrax.agent_distribution.contract_metadata_parity import (
    AgentContractMetadataParityError,
)
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.installation import InstallationState
from intergrax.agent_distribution.stores import AgentArtifactMetadataStore
from intergrax.applications._shared import agent_certification_wiring
from intergrax.applications._shared import capability_graph_deploy_gate
from intergrax.applications._shared import health_score_wiring
from intergrax.applications._shared import package_wiring
from intergrax.applications._shared.roster_agent_contract_authority import (
    RosterAgentContractAuthority,
)
from testing_support.agent_distribution.agent_platform_admin_qualification_harness import (
    QUALIFICATION_ENVIRONMENT_ID,
    QUALIFICATION_PACKAGE_DIGEST,
    QUALIFICATION_PACKAGE_ID,
    build_agent_platform_admin_qualification_stack,
    qualification_install_request,
    qualification_trust_record,
)
from testing_support.agent_platform_admin_harness import admin_test_principal

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_STRICT_SHARED_MODULES = (
    agent_certification_wiring,
    capability_graph_deploy_gate,
)


def _echo_contract():
    sys.path.insert(0, "agents/echo")
    try:
        from contract import build_agent_contract

        return build_agent_contract()
    finally:
        sys.path.pop(0)


def _declared_contract(contract) -> AgentPackageContractDeclaration:
    return AgentPackageContractDeclaration(
        contract_id=contract.id,
        contract_version=contract.version,
        capabilities=tuple(
            item.id if hasattr(item, "id") else str(item)
            for item in contract.capabilities
        ),
        skill_ids=tuple(
            skill.skill_id for skill in contract.skills if skill.skill_id
        ),
        tool_ids=tuple(
            tool.tool_id
            for tool in getattr(contract, "allowed_tools", ())
            if getattr(tool, "tool_id", None)
        ),
    )


def test_agent_contract_authority_module_has_single_persistence_port_type() -> None:
    path = Path("intergrax/agent_distribution/agent_contract_authority.py")
    source = path.read_text(encoding="utf-8")
    assert "class AgentPackageContractAuthorityStore" not in source
    assert "AgentArtifactMetadataStore" in source


def test_strict_shared_modules_do_not_reference_lab_compat_helper() -> None:
    forbidden = "materialize_manifest_contract_authority_lab_compat"
    for module in _STRICT_SHARED_MODULES:
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert forbidden not in source, module.__name__


def test_install_invalid_authority_leaves_slot_unchanged() -> None:
    stack = build_agent_platform_admin_qualification_stack()
    principal = admin_test_principal()
    contract = _echo_contract()
    bad_record = PackageAgentContractAuthorityRecord.from_validated_contract(
        package_digest=QUALIFICATION_PACKAGE_DIGEST,
        distribution_package_id=QUALIFICATION_PACKAGE_ID,
        artifact_store_ref="store://artifacts/inst-bad",
        agent_project_metadata_ref="meta://search",
        contract=contract.model_copy(update={"version": "9.9.9"}),
    )
    request = InstallAgentRequest(
        mutation_id="mut-bad-authority",
        installation_id="inst-bad",
        installation_slot_id="slot-search",
        package_identity=AgentPackageIdentity(
            distribution_package_id=QUALIFICATION_PACKAGE_ID,
            package_version="1.0.0",
            package_digest=QUALIFICATION_PACKAGE_DIGEST,
        ),
        artifact_store_ref="store://artifacts/inst-bad",
        trust_record=qualification_trust_record(),
        agent_project_metadata_ref="meta://search",
        package_contract_authority=(bad_record,),
    )
    metadata = stack.service._metadata_provider.get_metadata("meta://search")
    assert metadata is not None
    stack.service._metadata_provider._records["meta://search"] = AgentProjectMetadata(
        distribution_package_id=QUALIFICATION_PACKAGE_ID,
        dependencies=(),
        declared_contracts=(_declared_contract(contract),),
    )
    with pytest.raises(
        (PackageAgentContractAuthorityError, AgentContractMetadataParityError)
    ):
        stack.service.install_agent(
            application_id="app-a",
            application_environment_id=QUALIFICATION_ENVIRONMENT_ID,
            request=request,
            principal=principal,
        )
    active = stack.service._installation_service.resolve_active_for_slot(
        QUALIFICATION_ENVIRONMENT_ID,
        "slot-search",
    )
    assert active is None


def test_install_authority_persist_failure_does_not_activate() -> None:
    stack = build_agent_platform_admin_qualification_stack()
    principal = admin_test_principal()
    contract = _echo_contract()
    record = PackageAgentContractAuthorityRecord.from_validated_contract(
        package_digest=QUALIFICATION_PACKAGE_DIGEST,
        distribution_package_id=QUALIFICATION_PACKAGE_ID,
        artifact_store_ref="store://artifacts/inst-1",
        agent_project_metadata_ref="meta://search",
        contract=contract,
    )
    stack.service._metadata_provider._records["meta://search"] = AgentProjectMetadata(
        distribution_package_id=QUALIFICATION_PACKAGE_ID,
        dependencies=(),
        declared_contracts=(_declared_contract(contract),),
    )

    class _FailingArtifactStore:
        def __init__(self, inner: AgentArtifactMetadataStore) -> None:
            self._inner = inner

        def get_by_digest(self, package_digest: str):
            return self._inner.get_by_digest(package_digest)

        def persist_metadata(self, metadata):
            return self._inner.persist_metadata(metadata)

        def get_package_contract_authority(self, package_digest: str, contract_id: str):
            return self._inner.get_package_contract_authority(package_digest, contract_id)

        def persist_package_contract_authority(self, record):
            raise PackageAgentContractAuthorityError("simulated persistence failure")

    stack.service._artifact_metadata_store = _FailingArtifactStore(
        stack.service._artifact_metadata_store
    )
    stack.service._package_contract_authority_service = AgentPackageContractAuthorityService(
        stack.service._artifact_metadata_store
    )
    request = qualification_install_request().model_copy(
        update={"package_contract_authority": (record,)}
    )
    with pytest.raises(PackageAgentContractAuthorityError, match="simulated"):
        stack.service.install_agent(
            application_id="app-a",
            application_environment_id=QUALIFICATION_ENVIRONMENT_ID,
            request=request,
            principal=principal,
        )
    active = stack.service._installation_service.resolve_active_for_slot(
        QUALIFICATION_ENVIRONMENT_ID,
        "slot-search",
    )
    assert active is None


def test_valid_install_persists_before_activation() -> None:
    stack = build_agent_platform_admin_qualification_stack()
    principal = admin_test_principal()
    contract = _echo_contract()
    record = PackageAgentContractAuthorityRecord.from_validated_contract(
        package_digest=QUALIFICATION_PACKAGE_DIGEST,
        distribution_package_id=QUALIFICATION_PACKAGE_ID,
        artifact_store_ref="store://artifacts/inst-1",
        agent_project_metadata_ref="meta://search",
        contract=contract,
    )
    stack.service._metadata_provider._records["meta://search"] = AgentProjectMetadata(
        distribution_package_id=QUALIFICATION_PACKAGE_ID,
        dependencies=(),
        declared_contracts=(_declared_contract(contract),),
    )
    request = qualification_install_request().model_copy(
        update={"package_contract_authority": (record,)}
    )
    result = stack.service.install_agent(
        application_id="app-a",
        application_environment_id=QUALIFICATION_ENVIRONMENT_ID,
        request=request,
        principal=principal,
    )
    assert result.installation.installation_state is InstallationState.INSTALLED_ACTIVE
    loaded = stack.service._artifact_metadata_store.get_package_contract_authority(
        QUALIFICATION_PACKAGE_DIGEST,
        contract.id,
    )
    assert loaded is not None
    assert loaded.metadata_digest == record.metadata_digest


def test_roster_agent_contract_authority_resolves_without_live_contract_module() -> None:
    from intergrax.agent_distribution.in_memory_stores import (
        AgentDistributionStoreState,
        InMemoryAgentArtifactMetadataStore,
    )
    from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
    from intergrax.agent_distribution.runtime_revision import RuntimeRevision, RuntimeRevisionState
    from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

    contract = _echo_contract()
    digest_a = "sha256:" + ("f1" * 32)
    store = InMemoryAgentArtifactMetadataStore(AgentDistributionStoreState())
    service = AgentPackageContractAuthorityService(store)
    record = PackageAgentContractAuthorityRecord.from_validated_contract(
        package_digest=digest_a,
        distribution_package_id="intergrax-echo-agent",
        artifact_store_ref="artifact://a",
        agent_project_metadata_ref="meta://echo",
        contract=contract,
    )
    service.persist_authority_record(record)
    manifest = ApplicationManifest.lab(
        app_id="app",
        name="App",
        agents=[
            AgentBinding(
                contract_id="echo",
                import_path="echo.echo_agent.EchoAgent",
                enabled=True,
            )
        ],
    )
    roster = EffectiveRoster(
        application_id="app",
        application_environment_id="env",
        manifest_release_id="rel",
        effective_roster_revision_id="roster-a",
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="echo",
                installation_slot_id="slot-echo",
                package_digest=digest_a,
                distribution_package_id="intergrax-echo-agent",
                effective_enablement=True,
                manifest_origin_ref="manifest:agents/echo",
            ),
        ),
    )
    revision = RuntimeRevision(
        runtime_revision_id="rev-a",
        application_id="app",
        application_environment_id="env",
        application_release_id="rel",
        platform_version="0.1.0",
        revision_state=RuntimeRevisionState.CANDIDATE,
        effective_roster_revision_id="roster-a",
        installed_agent_package_digests=(digest_a,),
    )
    authority = RosterAgentContractAuthority.from_revision_roster(
        artifact_metadata_store=store,
        runtime_revision=revision,
        effective_roster=roster,
        manifest=manifest,
    )
    sys.path[:] = [p for p in sys.path if not p.endswith("agents/echo")]
    binding = manifest.enabled_agents()[0]
    resolved = authority.contract_for_binding(binding)
    assert resolved.id == "echo"
    assert resolved.version == contract.version
