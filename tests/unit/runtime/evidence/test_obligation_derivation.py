# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from local_workspace_application.workspaces.hybrid_ask_policy import (
    HybridAskPolicyError,
    IndexedEvidenceRequirementV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy_derivation import (
    compose_authoritative_evidence_obligations,
)
from intergrax.runtime.evidence.obligation_derivation import (
    DeterministicEvidenceObligationDerivation,
    _validate_derived_contract,
    derive_derivation_snapshot_id,
)
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    DerivedIndexedEvidenceObligationV1,
    DerivedLiveCallProposalV1,
    DerivedLiveEvidenceObligationV1,
    EvidenceObligationDerivationContextV1,
    EvidenceObligationDerivationError,
    RequireIndexedEvidencePolicyRuleV1,
    RequireIndexedEvidenceRuleParametersV1,
    RequireLiveEvidencePolicyRuleV1,
    RequireLiveEvidenceRuleParametersV1,
    ResolvedPolicyRuleV1,
)


_TENANT = "tenant-1"
_WORKSPACE = "workspace-1"
_CONFIG_REVISION = 17
_POLICY_DOC = "org-policy-generic"
_POLICY_DOC_A = "deployment-policy"
_POLICY_DOC_B = "security-policy"


def _indexed_rule(
    *,
    policy_document_id: str = _POLICY_DOC,
    revision_id: str,
    rule_id: str,
    requirement_key: str,
    semantic_role: str,
) -> RequireIndexedEvidencePolicyRuleV1:
    return RequireIndexedEvidencePolicyRuleV1(
        policy_document_id=policy_document_id,
        revision_id=revision_id,
        rule_id=rule_id,
        parameters=RequireIndexedEvidenceRuleParametersV1(
            semantic_role=semantic_role,
            requirement_key=requirement_key,
        ),
    )


def _revision_rules(revision_id: str) -> tuple[ResolvedPolicyRuleV1, ...]:
    base = (
        _indexed_rule(
            revision_id=revision_id,
            rule_id="RULE-A",
            requirement_key="indexed-a",
            semantic_role="Indexed evidence A",
        ),
        _indexed_rule(
            revision_id=revision_id,
            rule_id="RULE-B",
            requirement_key="indexed-b",
            semantic_role="Indexed evidence B",
        ),
        _indexed_rule(
            revision_id=revision_id,
            rule_id="RULE-C",
            requirement_key="indexed-c",
            semantic_role="Indexed evidence C",
        ),
    )
    if revision_id == "18":
        return (
            *base,
            _indexed_rule(
                revision_id=revision_id,
                rule_id="RULE-D",
                requirement_key="indexed-d",
                semantic_role="Indexed evidence D",
            ),
        )
    return base


def _context(
    *,
    revision_id: str,
    configuration_revision: int = _CONFIG_REVISION,
) -> EvidenceObligationDerivationContextV1:
    return EvidenceObligationDerivationContextV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=configuration_revision,
        resolved_policy_rules=_revision_rules(revision_id),
    )


def test_revision_17_derives_three_authoritative_obligations() -> None:
    engine = DeterministicEvidenceObligationDerivation()
    contract = engine.derive(_context(revision_id="17"))

    assert len(contract.derived_obligations) == 3
    assert contract.source_rule_ids == ("RULE-A", "RULE-B", "RULE-C")
    assert contract.derived_live_call_proposals == ()
    requirement_ids = tuple(
        obligation.requirement_id for obligation in contract.derived_obligations
    )
    assert requirement_ids == (
        f"policy:{_POLICY_DOC}:RULE-A:indexed-a",
        f"policy:{_POLICY_DOC}:RULE-B:indexed-b",
        f"policy:{_POLICY_DOC}:RULE-C:indexed-c",
    )


def test_revision_18_adds_exactly_one_new_obligation() -> None:
    engine = DeterministicEvidenceObligationDerivation()
    rev17 = engine.derive(_context(revision_id="17"))
    rev18 = engine.derive(_context(revision_id="18"))

    assert len(rev18.derived_obligations) == 4
    assert rev18.source_rule_ids == ("RULE-A", "RULE-B", "RULE-C", "RULE-D")
    assert rev17.derivation_snapshot_id != rev18.derivation_snapshot_id

    rev17_ids = {
        obligation.requirement_id for obligation in rev17.derived_obligations
    }
    rev18_ids = {
        obligation.requirement_id for obligation in rev18.derived_obligations
    }
    assert rev18_ids - rev17_ids == {
        f"policy:{_POLICY_DOC}:RULE-D:indexed-d"
    }
    for obligation in rev17.derived_obligations:
        assert obligation.requirement_id in rev18_ids


def test_derivation_is_repeatable() -> None:
    engine = DeterministicEvidenceObligationDerivation()
    context = _context(revision_id="17")
    first = engine.derive(context)
    second = engine.derive(context)

    assert first == second
    assert first.derivation_snapshot_id == second.derivation_snapshot_id


def test_empty_policy_returns_empty_contract() -> None:
    engine = DeterministicEvidenceObligationDerivation()
    context = EvidenceObligationDerivationContextV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=_CONFIG_REVISION,
        resolved_policy_rules=(),
    )
    contract = engine.derive(context)

    assert contract.derived_obligations == ()
    assert contract.derived_live_call_proposals == ()
    assert contract.derivation_snapshot_id == derive_derivation_snapshot_id(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=_CONFIG_REVISION,
        resolved_policy_rules=(),
    )


def test_duplicate_rule_id_fails_closed() -> None:
    engine = DeterministicEvidenceObligationDerivation()
    duplicate = _indexed_rule(
        revision_id="17",
        rule_id="RULE-A",
        requirement_key="indexed-a-dup",
        semantic_role="Duplicate",
    )
    context = EvidenceObligationDerivationContextV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=_CONFIG_REVISION,
        resolved_policy_rules=(
            _indexed_rule(
                revision_id="17",
                rule_id="RULE-A",
                requirement_key="indexed-a",
                semantic_role="Indexed evidence A",
            ),
            duplicate,
        ),
    )
    with pytest.raises(EvidenceObligationDerivationError) as exc:
        engine.derive(context)
    assert exc.value.error_code == "duplicate_rule_id"


def test_same_rule_id_across_different_policies_succeeds() -> None:
    engine = DeterministicEvidenceObligationDerivation()
    context = EvidenceObligationDerivationContextV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=_CONFIG_REVISION,
        resolved_policy_rules=(
            _indexed_rule(
                policy_document_id=_POLICY_DOC_A,
                revision_id="rev1",
                rule_id="RULE-1",
                requirement_key="deployment-index",
                semantic_role="Deployment evidence",
            ),
            _indexed_rule(
                policy_document_id=_POLICY_DOC_B,
                revision_id="rev1",
                rule_id="RULE-1",
                requirement_key="security-index",
                semantic_role="Security evidence",
            ),
        ),
    )
    contract = engine.derive(context)

    assert len(contract.derived_obligations) == 2
    requirement_ids = tuple(
        obligation.requirement_id for obligation in contract.derived_obligations
    )
    assert requirement_ids == (
        f"policy:{_POLICY_DOC_A}:RULE-1:deployment-index",
        f"policy:{_POLICY_DOC_B}:RULE-1:security-index",
    )
    assert contract.source_policy_document_ids == (_POLICY_DOC_A, _POLICY_DOC_B)
    assert contract.source_rule_ids == ("RULE-1", "RULE-1")


def test_conflicting_policy_rule_revision_fails_closed() -> None:
    engine = DeterministicEvidenceObligationDerivation()
    context = EvidenceObligationDerivationContextV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=_CONFIG_REVISION,
        resolved_policy_rules=(
            _indexed_rule(
                policy_document_id=_POLICY_DOC_A,
                revision_id="rev1",
                rule_id="RULE-1",
                requirement_key="deployment-index",
                semantic_role="Deployment evidence",
            ),
            _indexed_rule(
                policy_document_id=_POLICY_DOC_A,
                revision_id="rev2",
                rule_id="RULE-1",
                requirement_key="deployment-index-v2",
                semantic_role="Deployment evidence v2",
            ),
        ),
    )
    with pytest.raises(EvidenceObligationDerivationError) as exc:
        engine.derive(context)
    assert exc.value.error_code == "conflicting_policy_rule_revision"


def test_multi_policy_derivation_is_order_independent() -> None:
    engine = DeterministicEvidenceObligationDerivation()
    rule_a = _indexed_rule(
        policy_document_id=_POLICY_DOC_A,
        revision_id="rev1",
        rule_id="RULE-1",
        requirement_key="deployment-index",
        semantic_role="Deployment evidence",
    )
    rule_b = _indexed_rule(
        policy_document_id=_POLICY_DOC_B,
        revision_id="rev1",
        rule_id="RULE-1",
        requirement_key="security-index",
        semantic_role="Security evidence",
    )
    forward = EvidenceObligationDerivationContextV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=_CONFIG_REVISION,
        resolved_policy_rules=(rule_a, rule_b),
    )
    reverse = EvidenceObligationDerivationContextV1(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        configuration_revision=_CONFIG_REVISION,
        resolved_policy_rules=(rule_b, rule_a),
    )

    forward_contract = engine.derive(forward)
    reverse_contract = engine.derive(reverse)

    assert forward_contract == reverse_contract
    assert (
        forward_contract.derivation_snapshot_id
        == reverse_contract.derivation_snapshot_id
    )


def test_duplicate_requirement_id_fails_closed() -> None:
    duplicate_id = f"policy:{_POLICY_DOC}:RULE-A:indexed-a"
    obligations = (
        DerivedIndexedEvidenceObligationV1(
            requirement_id=duplicate_id,
            semantic_role="First",
            source_policy_document_id=_POLICY_DOC,
            source_revision_id="17",
            source_rule_id="RULE-A",
        ),
        DerivedIndexedEvidenceObligationV1(
            requirement_id=duplicate_id,
            semantic_role="Second",
            source_policy_document_id=_POLICY_DOC,
            source_revision_id="17",
            source_rule_id="RULE-B",
        ),
    )
    with pytest.raises(EvidenceObligationDerivationError) as exc:
        _validate_derived_contract(obligations=obligations, proposals=())
    assert exc.value.error_code == "duplicate_requirement_id"


def test_unknown_live_call_reference_fails_closed() -> None:
    obligation = DerivedLiveEvidenceObligationV1(
        requirement_id=f"policy:{_POLICY_DOC}:RULE-LIVE:live-1",
        semantic_role="Live evidence",
        call_id="policy-call:missing",
        source_policy_document_id=_POLICY_DOC,
        source_revision_id="17",
        source_rule_id="RULE-LIVE",
    )
    proposal = DerivedLiveCallProposalV1(
        call_id="policy-call:present",
        live_access_binding_id="binding-1",
        capability_id="vendor.generic.status.read",
        source_policy_document_id=_POLICY_DOC,
        source_revision_id="17",
        source_rule_id="RULE-LIVE",
    )
    with pytest.raises(EvidenceObligationDerivationError) as exc:
        _validate_derived_contract(
            obligations=(obligation,),
            proposals=(proposal,),
        )
    assert exc.value.error_code == "unknown_live_call_reference"


def test_duplicate_call_id_fails_closed() -> None:
    proposal = DerivedLiveCallProposalV1(
        call_id="policy-call:dup",
        live_access_binding_id="binding-1",
        capability_id="vendor.generic.status.read",
        source_policy_document_id=_POLICY_DOC,
        source_revision_id="17",
        source_rule_id="RULE-LIVE-A",
    )
    with pytest.raises(EvidenceObligationDerivationError) as exc:
        _validate_derived_contract(
            obligations=(),
            proposals=(proposal, proposal),
        )
    assert exc.value.error_code == "duplicate_call_id"


def test_live_rule_emits_matching_call_and_obligation() -> None:
    engine = DeterministicEvidenceObligationDerivation()
    rule = RequireLiveEvidencePolicyRuleV1(
        policy_document_id=_POLICY_DOC,
        revision_id="17",
        rule_id="RULE-LIVE",
        parameters=RequireLiveEvidenceRuleParametersV1(
            semantic_role="Live evidence",
            requirement_key="live-1",
            capability_id="vendor.generic.status.read",
            live_access_binding_id="binding-1",
            live_call_descriptor_ref="status-read",
        ),
    )
    contract = engine.derive(
        EvidenceObligationDerivationContextV1(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            configuration_revision=_CONFIG_REVISION,
            resolved_policy_rules=(rule,),
        )
    )

    assert len(contract.derived_live_call_proposals) == 1
    assert len(contract.derived_obligations) == 1
    proposal = contract.derived_live_call_proposals[0]
    obligation = contract.derived_obligations[0]
    assert obligation.call_id == proposal.call_id
    assert proposal.call_id == (
        f"policy-call:{_POLICY_DOC}:RULE-LIVE:status-read"
    )


def test_caller_cannot_remove_policy_derived_authority() -> None:
    policy_derived = (
        IndexedEvidenceRequirementV1(
            requirement_id="policy:derived:required",
            semantic_role="Policy derived",
        ),
    )
    with pytest.raises(HybridAskPolicyError) as exc:
        compose_authoritative_evidence_obligations(
            product=(),
            policy_derived=policy_derived,
            provider=(),
            caller_additive=(
                IndexedEvidenceRequirementV1(
                    requirement_id="policy:derived:required",
                    semantic_role="Caller override attempt",
                ),
            ),
        )
    assert exc.value.error_code == "duplicate_requirement_id"

    composed = compose_authoritative_evidence_obligations(
        product=(),
        policy_derived=policy_derived,
        provider=(),
        caller_additive=(
            IndexedEvidenceRequirementV1(
                requirement_id="caller:additive",
                semantic_role="Caller additive",
            ),
        ),
    )
    assert composed == (
        IndexedEvidenceRequirementV1(
            requirement_id="policy:derived:required",
            semantic_role="Policy derived",
        ),
        IndexedEvidenceRequirementV1(
            requirement_id="caller:additive",
            semantic_role="Caller additive",
        ),
    )
