# © Artur Czarnecki. All rights reserved.

"""Focused flagship proof composition tests (COMM-5F3-F)."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.evidence.obligation_derivation import (
    DeterministicEvidenceObligationDerivation,
)
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    EvidenceObligationDerivationContextV1,
    MaxAgeTemporalConstraintV1,
)
from intergrax.runtime.vendor_knowledge.live.governance_approval import (
    GOVERNANCE_APPROVAL_READ_CAPABILITY_ID,
)
from intergrax.runtime.vendor_knowledge.live.project_status import (
    PROJECT_STATUS_READ_CAPABILITY_ID,
)
from local_workspace_application.workspaces.hybrid_ask_policy_derivation import (
    map_derived_obligation,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    LiveResultRetentionV1,
    QueryPolicyModeV2,
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceKnowledgeConfigurationV1,
    WorkspaceLiveAccessBinding,
    WorkspaceQueryPolicyV2,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_docker_scenario import (
    FlagshipControllableOrchestratorV1,
    FlagshipMutableConfigurationServiceV1,
    _all_bindings,
    _configuration as build_flagship_configuration,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_formatter import (
    build_history_comparison,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_models import (
    FlagshipRequirementProofV1,
    FlagshipScenarioIdV1,
    FlagshipScenarioProofV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_policy import (
    FLAGSHIP_BINDING_GOVERNANCE,
    FLAGSHIP_CONN_GOVERNANCE,
    FLAGSHIP_CONN_READINESS,
    FLAGSHIP_POLICY_REV_17,
    FLAGSHIP_POLICY_REV_18,
    FLAGSHIP_TENANT_ID,
    FLAGSHIP_WORKSPACE_ID,
    MutableFlagshipPolicyRulesPort,
    UnsupportedFlagshipPolicyRevisionError,
    build_flagship_deployment_policy_rules,
    validate_flagship_policy_revision,
)

_NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
_SCENARIO_SOURCE = Path(
    "proof_infrastructure/governed_hybrid_knowledge_proof/flagship_docker_scenario.py"
)


def _binding(
    *,
    binding_id: str,
    connection_ref: str,
    capability_id: str,
) -> WorkspaceLiveAccessBinding:
    return WorkspaceLiveAccessBinding(
        live_access_binding_id=binding_id,
        tenant_id=FLAGSHIP_TENANT_ID,
        workspace_id=FLAGSHIP_WORKSPACE_ID,
        connection_ref=connection_ref,
        allowed_capability_ids=(capability_id,),
        derived_provider_id="provider-proof",
        derived_integration_kind=IntegrationCategory.ISSUE_TRACKER,
        derived_safe_display_label=f"Binding {binding_id}",
        status=LiveAccessBindingStatusV1.ACTIVE,
        mutation_id=f"mutation-{binding_id}",
        effective_revision=1,
        semantic_identity_hash=sha256(binding_id.encode()).hexdigest(),
        created_at=_NOW,
        updated_at=_NOW,
    )


def _configuration(
    bindings: tuple[WorkspaceLiveAccessBinding, ...],
) -> WorkspaceKnowledgeConfigurationV1:
    connection_refs = tuple(binding.connection_ref for binding in bindings)
    capability_ids = tuple(
        capability
        for binding in bindings
        for capability in binding.allowed_capability_ids
    )
    return WorkspaceKnowledgeConfigurationV1(
        tenant_id=FLAGSHIP_TENANT_ID,
        workspace_id=FLAGSHIP_WORKSPACE_ID,
        configuration_revision=1,
        connection_attachments=tuple(
            WorkspaceConnectionAttachment(
                attachment_id=f"attachment-{connection_ref}",
                tenant_id=FLAGSHIP_TENANT_ID,
                workspace_id=FLAGSHIP_WORKSPACE_ID,
                connection_ref=connection_ref,
                safe_display_label=f"Attachment {connection_ref}",
                status=WorkspaceConnectionAttachmentStatusV1.ATTACHED,
                mutation_id=f"mutation-{connection_ref}",
                effective_revision=1,
                created_at=_NOW,
                updated_at=_NOW,
            )
            for connection_ref in connection_refs
        ),
        indexed_sources=(),
        live_access_bindings=bindings,
        query_policy=WorkspaceQueryPolicyV2(
            tenant_id=FLAGSHIP_TENANT_ID,
            workspace_id=FLAGSHIP_WORKSPACE_ID,
            mode=QueryPolicyModeV2.LIVE_ONLY,
            allowed_connection_refs=connection_refs,
            allowed_capability_ids=capability_ids,
            max_live_calls=len(bindings),
            max_total_duration_ms=30_000,
            max_result_items=10,
            max_result_bytes=1_048_576,
            live_result_retention=LiveResultRetentionV1.EPHEMERAL,
            mutation_id="mutation-flagship-policy",
            effective_revision=1,
            updated_at=_NOW,
        ),
        updated_at=_NOW,
    )


def test_flagship_policy_revision_changes_snapshot_without_requirement_id_change() -> None:
    engine = DeterministicEvidenceObligationDerivation()
    rev17 = engine.derive(
        EvidenceObligationDerivationContextV1(
            tenant_id=FLAGSHIP_TENANT_ID,
            workspace_id=FLAGSHIP_WORKSPACE_ID,
            configuration_revision=17,
            resolved_policy_rules=build_flagship_deployment_policy_rules(
                policy_revision=FLAGSHIP_POLICY_REV_17,
            ),
        )
    )
    rev18 = engine.derive(
        EvidenceObligationDerivationContextV1(
            tenant_id=FLAGSHIP_TENANT_ID,
            workspace_id=FLAGSHIP_WORKSPACE_ID,
            configuration_revision=18,
            resolved_policy_rules=build_flagship_deployment_policy_rules(
                policy_revision=FLAGSHIP_POLICY_REV_18,
            ),
        )
    )
    rev17_security = next(
        item for item in rev17.derived_obligations if item.requirement_id.endswith(":security")
    )
    rev18_security = next(
        item for item in rev18.derived_obligations if item.requirement_id.endswith(":security")
    )
    rev17_obligation = map_derived_obligation(rev17_security)
    rev18_obligation = map_derived_obligation(rev18_security)
    assert rev17_obligation.requirement_id == rev18_obligation.requirement_id
    assert rev17.derivation_snapshot_id != rev18.derivation_snapshot_id
    rev17_constraint = rev17_security.temporal_constraint
    rev18_constraint = rev18_security.temporal_constraint
    assert isinstance(rev17_constraint, MaxAgeTemporalConstraintV1)
    assert isinstance(rev18_constraint, MaxAgeTemporalConstraintV1)
    assert rev17_constraint.max_age_seconds == 86_400
    assert rev18_constraint.max_age_seconds == 3_600


def test_flagship_history_comparison_highlights_revision_delta() -> None:
    rev17 = FlagshipScenarioProofV1(
        scenario_id=FlagshipScenarioIdV1.REV17_ALL_SATISFIED,
        derivation_snapshot_id="snapshot-rev17",
        requirements=(
            FlagshipRequirementProofV1(
                requirement_id="policy:security-policy:RULE-SECURITY:security",
                policy_revision_id="17",
                temporal_constraint=MaxAgeTemporalConstraintV1(max_age_seconds=86_400),
            ),
        ),
        passed=True,
        detail="ok",
        llm_calls=1,
        run_id="run-rev17",
    )
    rev18 = FlagshipScenarioProofV1(
        scenario_id=FlagshipScenarioIdV1.REV18_STALE_SECURITY,
        derivation_snapshot_id="snapshot-rev18",
        requirements=(
            FlagshipRequirementProofV1(
                requirement_id="policy:security-policy:RULE-SECURITY:security",
                policy_revision_id="18",
                temporal_constraint=MaxAgeTemporalConstraintV1(max_age_seconds=3_600),
            ),
        ),
        passed=True,
        detail="ok",
        llm_calls=0,
        run_id="run-rev18",
    )
    comparison = build_history_comparison(rev17=rev17, rev18=rev18)
    assert "snapshot-rev17" in comparison
    assert "snapshot-rev18" in comparison
    assert "max_age=86400s" in comparison
    assert "max_age=3600s" in comparison


def test_flagship_policy_rules_emit_four_mandatory_obligations() -> None:
    rules = build_flagship_deployment_policy_rules(policy_revision=FLAGSHIP_POLICY_REV_17)
    engine = DeterministicEvidenceObligationDerivation()
    contract = engine.derive(
        EvidenceObligationDerivationContextV1(
            tenant_id=FLAGSHIP_TENANT_ID,
            workspace_id=FLAGSHIP_WORKSPACE_ID,
            configuration_revision=17,
            resolved_policy_rules=rules,
        )
    )
    keys = {
        item.requirement_id.rsplit(":", maxsplit=1)[-1]
        for item in contract.derived_obligations
    }
    assert keys == {"readiness", "security", "change", "architecture"}


@pytest.mark.parametrize(
    "policy_revision",
    ("19", "999", "banana", ""),
)
def test_unknown_flagship_policy_revision_fails_closed(policy_revision: str) -> None:
    with pytest.raises(UnsupportedFlagshipPolicyRevisionError) as exc_info:
        validate_flagship_policy_revision(policy_revision)
    assert exc_info.value.args[0] == f"unsupported_flagship_policy_revision:{policy_revision}"


def test_mutable_flagship_policy_port_rejects_unknown_revision_on_set() -> None:
    port = MutableFlagshipPolicyRulesPort(policy_revision=FLAGSHIP_POLICY_REV_17)
    with pytest.raises(UnsupportedFlagshipPolicyRevisionError):
        port.set_revision("unknown")


def test_flagship_policy_switch_preserves_port_identity_and_changes_basis() -> None:
    port = MutableFlagshipPolicyRulesPort(policy_revision=FLAGSHIP_POLICY_REV_17)
    port_id = id(port)
    engine = DeterministicEvidenceObligationDerivation()
    rev17 = engine.derive(
        EvidenceObligationDerivationContextV1(
            tenant_id=FLAGSHIP_TENANT_ID,
            workspace_id=FLAGSHIP_WORKSPACE_ID,
            configuration_revision=17,
            resolved_policy_rules=port.resolve_policy_rules(),
        )
    )
    port.set_revision(FLAGSHIP_POLICY_REV_18)
    rev18 = engine.derive(
        EvidenceObligationDerivationContextV1(
            tenant_id=FLAGSHIP_TENANT_ID,
            workspace_id=FLAGSHIP_WORKSPACE_ID,
            configuration_revision=18,
            resolved_policy_rules=port.resolve_policy_rules(),
        )
    )
    assert id(port) == port_id
    assert port.policy_revision == FLAGSHIP_POLICY_REV_18
    assert rev17.derivation_snapshot_id != rev18.derivation_snapshot_id
    rev17_ids = {item.requirement_id for item in rev17.derived_obligations}
    rev18_ids = {item.requirement_id for item in rev18.derived_obligations}
    assert rev17_ids == rev18_ids
    rev17_security = next(
        item for item in rev17.derived_obligations if item.requirement_id.endswith(":security")
    )
    rev18_security = next(
        item for item in rev18.derived_obligations if item.requirement_id.endswith(":security")
    )
    rev17_constraint = rev17_security.temporal_constraint
    rev18_constraint = rev18_security.temporal_constraint
    assert isinstance(rev17_constraint, MaxAgeTemporalConstraintV1)
    assert isinstance(rev18_constraint, MaxAgeTemporalConstraintV1)
    assert rev17_constraint.max_age_seconds == 86_400
    assert rev18_constraint.max_age_seconds == 3_600


@pytest.mark.asyncio
async def test_flagship_controllable_orchestrator_revoke_once() -> None:
    bindings = (
        _binding(
            binding_id=FLAGSHIP_BINDING_GOVERNANCE,
            connection_ref=FLAGSHIP_CONN_GOVERNANCE,
            capability_id=GOVERNANCE_APPROVAL_READ_CAPABILITY_ID,
        ),
        _binding(
            binding_id="binding-flagship-readiness",
            connection_ref=FLAGSHIP_CONN_READINESS,
            capability_id=PROJECT_STATUS_READ_CAPABILITY_ID,
        ),
    )
    configuration_service = FlagshipMutableConfigurationServiceV1(_configuration(bindings))
    inner_calls: list[str] = []

    class _FakeInner:
        async def execute(self, **_: object) -> str:
            inner_calls.append("executed")
            return "ok"

    orchestrator = FlagshipControllableOrchestratorV1(
        inner=_FakeInner(),  # type: ignore[arg-type]
        configuration_service=configuration_service,
        governance_binding_id=FLAGSHIP_BINDING_GOVERNANCE,
    )
    orchestrator_id = id(orchestrator)

    await orchestrator.execute(run_id="normal")
    assert inner_calls == ["executed"]
    governance = next(
        binding
        for binding in configuration_service.get_configuration(
            tenant_id=FLAGSHIP_TENANT_ID,
            workspace_id=FLAGSHIP_WORKSPACE_ID,
        ).live_access_bindings
        if binding.live_access_binding_id == FLAGSHIP_BINDING_GOVERNANCE
    )
    assert governance.status is LiveAccessBindingStatusV1.ACTIVE

    orchestrator.arm_governance_revocation_once()
    await orchestrator.execute(run_id="revoked")
    governance = next(
        binding
        for binding in configuration_service.get_configuration(
            tenant_id=FLAGSHIP_TENANT_ID,
            workspace_id=FLAGSHIP_WORKSPACE_ID,
        ).live_access_bindings
        if binding.live_access_binding_id == FLAGSHIP_BINDING_GOVERNANCE
    )
    assert governance.status is LiveAccessBindingStatusV1.DISABLED
    assert inner_calls == ["executed", "executed"]

    await orchestrator.execute(run_id="normal-again")
    assert inner_calls == ["executed", "executed", "executed"]
    assert id(orchestrator) == orchestrator_id


def test_flagship_configuration_service_uses_explicit_binding_methods() -> None:
    bindings = (
        _binding(
            binding_id=FLAGSHIP_BINDING_GOVERNANCE,
            connection_ref=FLAGSHIP_CONN_GOVERNANCE,
            capability_id=GOVERNANCE_APPROVAL_READ_CAPABILITY_ID,
        ),
    )
    configuration_service = FlagshipMutableConfigurationServiceV1(_configuration(bindings))
    configuration_service.disable_binding(FLAGSHIP_BINDING_GOVERNANCE)
    disabled = configuration_service.get_configuration(
        tenant_id=FLAGSHIP_TENANT_ID,
        workspace_id=FLAGSHIP_WORKSPACE_ID,
    )
    assert disabled is not None
    assert disabled.live_access_bindings[0].status is LiveAccessBindingStatusV1.DISABLED
    configuration_service.enable_all_bindings()
    restored = configuration_service.get_configuration(
        tenant_id=FLAGSHIP_TENANT_ID,
        workspace_id=FLAGSHIP_WORKSPACE_ID,
    )
    assert restored is not None
    assert restored.live_access_bindings[0].status is LiveAccessBindingStatusV1.ACTIVE


def test_flagship_scenario_source_has_no_private_service_mutation() -> None:
    source = _SCENARIO_SOURCE.read_text(encoding="utf-8")
    assert "service._resolved_policy_rules_port" not in source
    assert "service._orchestrator =" not in source
    assert "configuration_service._configuration =" not in source


def test_flagship_configuration_uses_deterministic_connection_order() -> None:
    bindings = _all_bindings()
    first = build_flagship_configuration(bindings)
    second = build_flagship_configuration(bindings)
    assert first.query_policy.allowed_connection_refs == second.query_policy.allowed_connection_refs
    assert first.connection_attachments == second.connection_attachments
    assert first.live_access_bindings == second.live_access_bindings
    assert first.query_policy.allowed_connection_refs == tuple(
        sorted(binding.connection_ref for binding in bindings)
    )


def test_flagship_configuration_rejects_duplicate_connection_refs() -> None:
    binding = _all_bindings()[0]
    duplicate_bindings = (
        binding,
        binding.model_copy(update={"live_access_binding_id": "duplicate-binding-id"}),
    )
    with pytest.raises(ValueError, match="duplicate_connection_ref_in_flagship_topology"):
        build_flagship_configuration(duplicate_bindings)
