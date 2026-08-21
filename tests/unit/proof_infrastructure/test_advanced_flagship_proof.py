# © Artur Czarnecki. All rights reserved.

"""Focused flagship proof composition tests (COMM-5F3-F)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.runtime.evidence.obligation_derivation import (
    DeterministicEvidenceObligationDerivation,
)
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    EvidenceObligationDerivationContextV1,
    MaxAgeTemporalConstraintV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy_derivation import (
    map_derived_obligation,
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
    FLAGSHIP_POLICY_REV_17,
    FLAGSHIP_POLICY_REV_18,
    FLAGSHIP_TENANT_ID,
    FLAGSHIP_WORKSPACE_ID,
    build_flagship_deployment_policy_rules,
)

_NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


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
