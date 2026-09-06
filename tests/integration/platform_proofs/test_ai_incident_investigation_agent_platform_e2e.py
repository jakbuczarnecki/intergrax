# © Artur Czarnecki. All rights reserved.

"""AIPV-1 — AI Incident Investigation canonical Agent Platform lifecycle integration."""

from __future__ import annotations

import pytest

from pathlib import Path

from intergrax.agent_distribution.admin_models import (
    BindAgentRequest,
    InstallAgentRequest,
)
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    RuntimeAgentFactoryResolutionError,
)
from intergrax.core.qualification import QualificationStatus
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_contract import (
    INVESTIGATOR_AGENT_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    build_runtime_bundle,
)
from platform_proofs.scenarios.ai_incident_investigation.integration.package_identity import (
    INCIDENT_INVESTIGATOR_APPLICATION_BINDING_ID,
    INCIDENT_INVESTIGATOR_APPLICATION_ID,
    INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID,
    INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
    INCIDENT_INVESTIGATOR_INSTALLATION_ID,
    INCIDENT_INVESTIGATOR_INSTALLATION_SLOT_ID,
    INCIDENT_INVESTIGATOR_METADATA_REF,
    INCIDENT_INVESTIGATOR_PACKAGE_DIGEST,
    INCIDENT_INVESTIGATOR_PACKAGE_VERSION,
)
from platform_proofs.scenarios.ai_incident_investigation.integration.production_validation import (
    IncidentInvestigatorAgentPlatformProofStack,
)
from testing_support.agent_platform_admin_harness import admin_test_principal

pytestmark = [pytest.mark.integration, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _stub_scenario_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.platform_proofs.scenarios.ai_incident_investigation.planner_doubles import (
        ScriptedIncidentInvestigationLLM,
    )

    adapter = ScriptedIncidentInvestigationLLM()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition.resolve_llm_adapter",
        _resolve,
    )


def test_production_validation_module_has_no_lab_shortcuts() -> None:
    module_path = (
        Path(__file__).resolve().parents[3]
        / "platform_proofs"
        / "scenarios"
        / "ai_incident_investigation"
        / "integration"
        / "production_validation.py"
    )
    source = module_path.read_text(encoding="utf-8")
    for forbidden in (
        "build_scenario_lab_agent_registry",
        "ApplicationManifest.lab",
    ):
        assert forbidden not in source
    assert "ScenarioRuntimeMode.LAB" not in source.replace(
        "ScenarioRuntimeMode.PRODUCTION_ATTACHED",
        "",
    )


def test_ai_incident_investigator_executes_from_canonical_active_revision(
    tmp_path: Path,
) -> None:
    stack = IncidentInvestigatorAgentPlatformProofStack.build(tmp_path)
    proof = stack.run_happy_path()

    assert proof.package_digest == INCIDENT_INVESTIGATOR_PACKAGE_DIGEST
    assert proof.runtime_revision_id is not None
    assert proof.traffic_serving_revision_id == proof.runtime_revision_id
    assert proof.execution_outcome == OUTCOME_RESOLVED

    serving = stack.admin.inspect_serving(
        application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
        application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
    )
    assert serving.traffic_serving_revision_id == proof.runtime_revision_id

    projection = stack.resolve_serving_projection()
    assert projection.evidence.runtime_revision_id == proof.runtime_revision_id
    assert projection.agent_registry.has(INVESTIGATOR_AGENT_ID)


@pytest.mark.asyncio
async def test_production_execution_fails_closed_without_active_revision(
    tmp_path: Path,
) -> None:
    stack = IncidentInvestigatorAgentPlatformProofStack.build(tmp_path)
    stack.install_from_catalog()
    stack.bind_enabled_agent()
    stack.build_revision()

    serving = stack.admin.inspect_serving(
        application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
        application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
    )
    assert serving.traffic_serving_revision_id is None

    with pytest.raises(AssertionError):
        stack.resolve_serving_projection()

    with pytest.raises(AssertionError):
        await stack.execute_incident_scenario()


def test_factory_reference_mismatch_blocks_projection_activation(
    tmp_path: Path,
) -> None:
    stack = IncidentInvestigatorAgentPlatformProofStack.build(tmp_path)
    stack.install_from_catalog()
    principal = admin_test_principal()
    stack.admin.bind_agent(
        application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
        application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
        request=BindAgentRequest(
            mutation_id="mut-aipv1-bind-bad-factory",
            application_binding_id=INCIDENT_INVESTIGATOR_APPLICATION_BINDING_ID,
            logical_agent_id=INVESTIGATOR_AGENT_ID,
            installation_slot_id=INCIDENT_INVESTIGATOR_INSTALLATION_SLOT_ID,
            factory_reference=AgentBindingFactoryReference(
                factory_path="incident_investigator_agent.factory.missing_factory",
            ),
            enablement=True,
        ),
        principal=principal,
    )
    built = stack.build_revision()
    with pytest.raises(
        (RuntimeAgentFactoryResolutionError, Exception),
        match="missing_factory|not found|RegistryProjectionError|factory",
    ):
        stack.register_projection_and_activate(built)


def test_trust_digest_mismatch_blocks_install(tmp_path: Path) -> None:
    stack = IncidentInvestigatorAgentPlatformProofStack.build(tmp_path)
    principal = admin_test_principal()
    wrong_digest = "sha256:" + ("f" * 64)
    with pytest.raises(Exception, match="digest|trust|mismatch"):
        stack.admin.install_agent(
            application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
            application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
            request=InstallAgentRequest(
                mutation_id="mut-aipv1-install-bad-trust",
                installation_id=INCIDENT_INVESTIGATOR_INSTALLATION_ID,
                installation_slot_id=INCIDENT_INVESTIGATOR_INSTALLATION_SLOT_ID,
                package_identity=AgentPackageIdentity(
                    distribution_package_id=INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID,
                    package_version=INCIDENT_INVESTIGATOR_PACKAGE_VERSION,
                    package_digest=INCIDENT_INVESTIGATOR_PACKAGE_DIGEST,
                ),
                artifact_store_ref=f"store://artifacts/{INCIDENT_INVESTIGATOR_INSTALLATION_ID}",
                trust_record=AgentInstallationTrustRecord(
                    qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
                    package_digest=wrong_digest,
                    publisher_identity_ref="publisher:ai-incident-investigator",
                    source_provider_id="builtin-ai-incident-investigator",
                    trust_evidence_refs=(
                        AgentTrustEvidenceRef(
                            evidence_id="evidence:aipv-1-bad",
                            kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                        ),
                    ),
                ),
                agent_project_metadata_ref=INCIDENT_INVESTIGATOR_METADATA_REF,
            ),
            principal=principal,
        )


@pytest.mark.asyncio
async def test_lab_authoring_runtime_remains_independent() -> None:
    bundle = build_runtime_bundle()
    assert bundle.runtime_composition.is_platform_attached
    assert bundle.investigator.get_contract().id == INVESTIGATOR_AGENT_ID
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED
