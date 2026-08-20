# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Deterministic evidence obligation derivation engine (COMM-5F3-A)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from intergrax.runtime.attestation.canonical_json import stable_payload_hash
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    DerivedEvidenceContractV1,
    DerivedEvidenceObligationV1,
    DerivedIndexedEvidenceObligationV1,
    DerivedLiveCallProposalV1,
    DerivedLiveEvidenceObligationV1,
    EvidenceObligationDerivationContextV1,
    EvidenceObligationDerivationError,
    EvidenceObligationDerivationPort,
    RequireIndexedEvidencePolicyRuleV1,
    RequireLiveEvidencePolicyRuleV1,
    ResolvedPolicyRuleKindV1,
    ResolvedPolicyRuleV1,
    TypedCapabilityRequestEntryV1,
)


class _CanonicalTypedCapabilityRequestEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    value: str


class _CanonicalIndexedRuleParametersV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_role: str
    requirement_key: str
    indexed_source_binding_id: str | None


class _CanonicalLiveRuleParametersV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_role: str
    requirement_key: str
    capability_id: str
    live_access_binding_id: str
    live_call_descriptor_ref: str
    typed_capability_request: tuple[_CanonicalTypedCapabilityRequestEntryV1, ...]


class _CanonicalSerializedRuleV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_document_id: str
    revision_id: str
    rule_id: str
    rule_kind: str
    parameters: _CanonicalIndexedRuleParametersV1 | _CanonicalLiveRuleParametersV1


class _DerivationSnapshotPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configuration_revision: int
    resolved_policy_rules: tuple[_CanonicalSerializedRuleV1, ...]
    tenant_id: str
    workspace_id: str


def _canonical_rule_identity_key(
    rule: ResolvedPolicyRuleV1,
) -> tuple[str, str]:
    return (rule.policy_document_id, rule.rule_id)


def _canonical_rule_sort_key(rule: ResolvedPolicyRuleV1) -> tuple[str, str, str]:
    return (rule.policy_document_id, rule.revision_id, rule.rule_id)


def _derive_requirement_id(
    *,
    policy_document_id: str,
    rule_id: str,
    requirement_key: str,
) -> str:
    return f"policy:{policy_document_id}:{rule_id}:{requirement_key}"


def _derive_call_id(
    *,
    policy_document_id: str,
    rule_id: str,
    live_call_descriptor_ref: str,
) -> str:
    return (
        "policy-call:"
        f"{policy_document_id}:{rule_id}:{live_call_descriptor_ref}"
    )


def _serialize_typed_request(
    entries: tuple[TypedCapabilityRequestEntryV1, ...],
) -> tuple[_CanonicalTypedCapabilityRequestEntryV1, ...]:
    return tuple(
        _CanonicalTypedCapabilityRequestEntryV1(key=entry.key, value=entry.value)
        for entry in sorted(entries, key=lambda item: item.key)
    )


def _serialize_rule(rule: ResolvedPolicyRuleV1) -> _CanonicalSerializedRuleV1:
    if isinstance(rule, RequireIndexedEvidencePolicyRuleV1):
        parameters = _CanonicalIndexedRuleParametersV1(
            semantic_role=rule.parameters.semantic_role,
            requirement_key=rule.parameters.requirement_key,
            indexed_source_binding_id=rule.parameters.indexed_source_binding_id,
        )
        rule_kind = ResolvedPolicyRuleKindV1.REQUIRE_INDEXED_EVIDENCE.value
    else:
        parameters = _CanonicalLiveRuleParametersV1(
            semantic_role=rule.parameters.semantic_role,
            requirement_key=rule.parameters.requirement_key,
            capability_id=rule.parameters.capability_id,
            live_access_binding_id=rule.parameters.live_access_binding_id,
            live_call_descriptor_ref=rule.parameters.live_call_descriptor_ref,
            typed_capability_request=_serialize_typed_request(
                rule.parameters.typed_capability_request
            ),
        )
        rule_kind = ResolvedPolicyRuleKindV1.REQUIRE_LIVE_EVIDENCE.value
    return _CanonicalSerializedRuleV1(
        policy_document_id=rule.policy_document_id,
        revision_id=rule.revision_id,
        rule_id=rule.rule_id,
        rule_kind=rule_kind,
        parameters=parameters,
    )


def derive_derivation_snapshot_id(
    *,
    tenant_id: str,
    workspace_id: str,
    configuration_revision: int,
    resolved_policy_rules: tuple[ResolvedPolicyRuleV1, ...],
) -> str:
    canonical_rules = tuple(
        _serialize_rule(rule)
        for rule in sorted(resolved_policy_rules, key=_canonical_rule_sort_key)
    )
    payload = _DerivationSnapshotPayloadV1(
        configuration_revision=configuration_revision,
        resolved_policy_rules=canonical_rules,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
    )
    return stable_payload_hash(payload.model_dump())


def _derive_indexed_obligation(
    rule: RequireIndexedEvidencePolicyRuleV1,
) -> DerivedIndexedEvidenceObligationV1:
    return DerivedIndexedEvidenceObligationV1(
        requirement_id=_derive_requirement_id(
            policy_document_id=rule.policy_document_id,
            rule_id=rule.rule_id,
            requirement_key=rule.parameters.requirement_key,
        ),
        semantic_role=rule.parameters.semantic_role,
        indexed_source_binding_id=rule.parameters.indexed_source_binding_id,
        source_policy_document_id=rule.policy_document_id,
        source_revision_id=rule.revision_id,
        source_rule_id=rule.rule_id,
    )


def _derive_live_artifacts(
    rule: RequireLiveEvidencePolicyRuleV1,
) -> tuple[DerivedLiveCallProposalV1, DerivedLiveEvidenceObligationV1]:
    call_id = _derive_call_id(
        policy_document_id=rule.policy_document_id,
        rule_id=rule.rule_id,
        live_call_descriptor_ref=rule.parameters.live_call_descriptor_ref,
    )
    requirement_id = _derive_requirement_id(
        policy_document_id=rule.policy_document_id,
        rule_id=rule.rule_id,
        requirement_key=rule.parameters.requirement_key,
    )
    proposal = DerivedLiveCallProposalV1(
        call_id=call_id,
        live_access_binding_id=rule.parameters.live_access_binding_id,
        capability_id=rule.parameters.capability_id,
        typed_capability_request=rule.parameters.typed_capability_request,
        source_policy_document_id=rule.policy_document_id,
        source_revision_id=rule.revision_id,
        source_rule_id=rule.rule_id,
    )
    obligation = DerivedLiveEvidenceObligationV1(
        requirement_id=requirement_id,
        semantic_role=rule.parameters.semantic_role,
        call_id=call_id,
        source_policy_document_id=rule.policy_document_id,
        source_revision_id=rule.revision_id,
        source_rule_id=rule.rule_id,
    )
    return proposal, obligation


def _validate_derived_contract(
  *,
  obligations: tuple[DerivedEvidenceObligationV1, ...],
  proposals: tuple[DerivedLiveCallProposalV1, ...],
) -> None:
    planned_call_ids = {proposal.call_id for proposal in proposals}
    seen_requirement_ids: set[str] = set()
    seen_call_ids: set[str] = set()
    for proposal in proposals:
        if proposal.call_id in seen_call_ids:
            raise EvidenceObligationDerivationError("duplicate_call_id")
        seen_call_ids.add(proposal.call_id)
    for obligation in obligations:
        if obligation.requirement_id in seen_requirement_ids:
            raise EvidenceObligationDerivationError("duplicate_requirement_id")
        seen_requirement_ids.add(obligation.requirement_id)
        if isinstance(obligation, DerivedLiveEvidenceObligationV1):
            if obligation.call_id not in planned_call_ids:
                raise EvidenceObligationDerivationError(
                    "unknown_live_call_reference"
                )


class DeterministicEvidenceObligationDerivation(
    EvidenceObligationDerivationPort,
):
    """Canonical server-side derivation with stable ordering and identity."""

    def derive(
        self,
        context: EvidenceObligationDerivationContextV1,
    ) -> DerivedEvidenceContractV1:
        seen_rule_identities: dict[tuple[str, str], str] = {}
        obligations: list[DerivedEvidenceObligationV1] = []
        proposals: list[DerivedLiveCallProposalV1] = []
        policy_document_ids: list[str] = []
        revision_ids: list[str] = []
        rule_ids: list[str] = []

        for rule in sorted(
            context.resolved_policy_rules, key=_canonical_rule_sort_key
        ):
            rule_identity = _canonical_rule_identity_key(rule)
            previous_revision_id = seen_rule_identities.get(rule_identity)
            if previous_revision_id is not None:
                if previous_revision_id == rule.revision_id:
                    raise EvidenceObligationDerivationError("duplicate_rule_id")
                raise EvidenceObligationDerivationError(
                    "conflicting_policy_rule_revision"
                )
            seen_rule_identities[rule_identity] = rule.revision_id
            policy_document_ids.append(rule.policy_document_id)
            revision_ids.append(rule.revision_id)
            rule_ids.append(rule.rule_id)

            if isinstance(rule, RequireIndexedEvidencePolicyRuleV1):
                obligations.append(_derive_indexed_obligation(rule))
                continue

            proposal, obligation = _derive_live_artifacts(rule)
            proposals.append(proposal)
            obligations.append(obligation)

        ordered_obligations = tuple(obligations)
        ordered_proposals = tuple(proposals)
        _validate_derived_contract(
            obligations=ordered_obligations,
            proposals=ordered_proposals,
        )

        snapshot_id = derive_derivation_snapshot_id(
            tenant_id=context.tenant_id,
            workspace_id=context.workspace_id,
            configuration_revision=context.configuration_revision,
            resolved_policy_rules=context.resolved_policy_rules,
        )
        return DerivedEvidenceContractV1(
            derivation_snapshot_id=snapshot_id,
            source_policy_document_ids=tuple(policy_document_ids),
            source_revision_ids=tuple(revision_ids),
            source_rule_ids=tuple(rule_ids),
            derived_obligations=ordered_obligations,
            derived_live_call_proposals=ordered_proposals,
        )
