# © Artur Czarnecki. All rights reserved.

"""Qualification evidence helpers."""

from __future__ import annotations

from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage

from testing_support.decision_e2e.contracts import QualificationEvidenceRef
from testing_support.decision_e2e.environment import ProviderBindingEvidence


def provider_evidence_ref(binding: ProviderBindingEvidence) -> QualificationEvidenceRef:
    model = binding.model or "default"
    detail = f"provider={binding.provider};model={model}"
    if binding.host:
        detail = f"{detail};host={binding.host}"
    return QualificationEvidenceRef(
        kind="provider_binding",
        ref=binding.profile_id,
        detail=detail,
    )


def decision_identity_evidence(identity: DecisionIdentity) -> QualificationEvidenceRef:
    return QualificationEvidenceRef(
        kind="decision_identity",
        ref=str(identity.decision_id),
        detail=(
            f"version={identity.version.value};tenant={identity.tenant_id};"
            f"scope={identity.scope.namespace}/{identity.scope.subject}"
        ),
    )


def lifecycle_stage_evidence(stage: DecisionLifecycleStage) -> QualificationEvidenceRef:
    return QualificationEvidenceRef(
        kind="lifecycle_stage",
        ref=stage.value,
    )


def invocation_count_evidence(count: int) -> QualificationEvidenceRef:
    return QualificationEvidenceRef(
        kind="provider_invocation_count",
        ref=str(count),
    )


def runtime_event_count_evidence(count: int) -> QualificationEvidenceRef:
    return QualificationEvidenceRef(
        kind="runtime_event_count",
        ref=str(count),
    )
