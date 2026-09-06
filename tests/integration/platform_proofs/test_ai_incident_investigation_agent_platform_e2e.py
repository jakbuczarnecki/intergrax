# © Artur Czarnecki. All rights reserved.

"""AIPV-1 — AI Incident Investigation canonical Agent Platform lifecycle integration."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.agent_distribution.admin_models import BindAgentRequest
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.in_memory_stores import InMemoryAgentInstallationStore
from intergrax.agent_distribution.trust import (
    AgentPackageTrustOutcome,
    AgentPackageTrustReasonCode,
    AgentPackageTrustRevocationState,
    AgentQualificationEvidenceKind,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    RuntimeAgentFactoryResolutionError,
)
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
    INCIDENT_INVESTIGATOR_CATALOG_SOURCE_ID,
    INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID,
    INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
    INCIDENT_INVESTIGATOR_INSTALLATION_ID,
    INCIDENT_INVESTIGATOR_INSTALLATION_SLOT_ID,
    INCIDENT_INVESTIGATOR_PACKAGE_DIGEST,
    INCIDENT_INVESTIGATOR_PUBLISHER_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.integration.production_validation import (
    IncidentInvestigatorAgentPlatformProofStack,
)
from platform_proofs.scenarios.ai_incident_investigation.integration.trust_fixture import (
    AIPV_EVALUATED_AT,
    AIPV_QUALIFIED_AT,
    IncidentInvestigatorCanonicalTrustRecordFactory,
    IncidentInvestigatorTrustFixture,
    evaluate_incident_investigator_trust,
)
from testing_support.agent_platform_admin_harness import admin_test_principal

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_INTEGRATION_DIR = (
    Path(__file__).resolve().parents[3]
    / "platform_proofs"
    / "scenarios"
    / "ai_incident_investigation"
    / "integration"
)


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


def _assert_no_direct_trust_record_construction(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "AgentInstallationTrustRecord":
            pytest.fail(
                f"{path}: direct AgentInstallationTrustRecord construction is forbidden"
            )
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "AgentInstallationTrustRecord"
        ):
            pytest.fail(
                f"{path}: direct AgentInstallationTrustRecord construction is forbidden"
            )


def test_production_validation_module_has_no_lab_shortcuts() -> None:
    module_path = _INTEGRATION_DIR / "production_validation.py"
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


def test_production_validation_integration_has_no_manual_trust_records() -> None:
    for path in sorted(_INTEGRATION_DIR.glob("*.py")):
        _assert_no_direct_trust_record_construction(path)


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


def test_canonical_trust_coordinator_produces_installable_record(
    tmp_path: Path,
) -> None:
    fixture = IncidentInvestigatorTrustFixture.build()
    decision = evaluate_incident_investigator_trust()
    assert decision.outcome is AgentPackageTrustOutcome.ALLOW
    record = decision.trust_record
    assert record is not None
    assert record.package_digest == INCIDENT_INVESTIGATOR_PACKAGE_DIGEST
    assert record.publisher_identity_ref == INCIDENT_INVESTIGATOR_PUBLISHER_ID
    assert record.source_provider_id == INCIDENT_INVESTIGATOR_CATALOG_SOURCE_ID
    assert record.policy_fingerprint == fixture.policy.policy_fingerprint
    assert record.revocation_checked_at == AIPV_EVALUATED_AT
    assert record.qualification_qualified_at == AIPV_QUALIFIED_AT
    assert any(
        ref.kind is AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION
        for ref in record.trust_evidence_refs
    )

    stack = IncidentInvestigatorAgentPlatformProofStack.build(tmp_path)
    stack.install_from_catalog()
    state = stack.composition.agent_platform_runtime.distribution_state
    installation = InMemoryAgentInstallationStore(state).get_installation(
        INCIDENT_INVESTIGATOR_INSTALLATION_ID,
    )
    assert installation is not None
    assert installation.trust_record == record


def test_canonical_trust_record_factory_delegates_to_coordinator() -> None:
    factory = IncidentInvestigatorCanonicalTrustRecordFactory()
    record = factory.build_trust_record(
        package_digest=INCIDENT_INVESTIGATOR_PACKAGE_DIGEST,
        package_id=INCIDENT_INVESTIGATOR_DISTRIBUTION_PACKAGE_ID,
    )
    assert record.policy_fingerprint is not None
    assert record.revocation_checked_at == AIPV_EVALUATED_AT
    assert record.qualification_qualified_at == AIPV_QUALIFIED_AT


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


def test_revoked_package_digest_denies_trust_before_install(tmp_path: Path) -> None:
    decision = evaluate_incident_investigator_trust(
        revocation_state=AgentPackageTrustRevocationState(
            revoked_package_digests=frozenset({INCIDENT_INVESTIGATOR_PACKAGE_DIGEST}),
        ),
    )
    assert decision.outcome is AgentPackageTrustOutcome.DENY
    assert decision.reason_code is AgentPackageTrustReasonCode.PACKAGE_DIGEST_REVOKED
    assert decision.trust_record is None

    stack = IncidentInvestigatorAgentPlatformProofStack.build(tmp_path)
    installed = stack.admin.list_installed(
        application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
        application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
    )
    assert installed.installations == ()
    bindings = stack.admin.list_bindings(
        application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
        application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
    )
    assert bindings.bindings == ()
    serving = stack.admin.inspect_serving(
        application_id=INCIDENT_INVESTIGATOR_APPLICATION_ID,
        application_environment_id=INCIDENT_INVESTIGATOR_ENVIRONMENT_ID,
    )
    assert serving.traffic_serving_revision_id is None


@pytest.mark.asyncio
async def test_lab_authoring_runtime_remains_independent() -> None:
    bundle = build_runtime_bundle()
    assert bundle.runtime_composition.is_platform_attached
    assert bundle.investigator.get_contract().id == INVESTIGATOR_AGENT_ID
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED
