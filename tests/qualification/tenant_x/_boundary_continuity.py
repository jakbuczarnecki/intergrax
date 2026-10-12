# © Artur Czarnecki. All rights reserved.

"""TENANT-X-P0-R3 credential/recovery boundary qualification (no production changes)."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.tenant_x._types import BoundaryContinuityClass

_CREDENTIAL_BROKER_SURFACE = (
    "intergrax/integrations/credentials/broker.py::ScopedCredentialBroker::canonical.credential_broker"
)
_CONFIG_VALIDATOR_SURFACE = (
    "intergrax/integrations/_shared/p3/configs.py::require_tenant_id::validator.marker"
)


@dataclass(frozen=True, slots=True)
class CredentialContinuityQualification:
    status: BoundaryContinuityClass
    tenant_in_credential_reference: str
    lookup_scoped_by_tenant: str
    cross_tenant_consumption_risk: str
    mismatch_rejection: str
    note: str


@dataclass(frozen=True, slots=True)
class RecoveryContinuityQualification:
    status: BoundaryContinuityClass
    original_tenant_in_checkpoint: str
    restored_from_durable_artifact: str
    caller_substitution_blocked: str
    mismatch_fail_closed: str
    note: str


def qualify_credential_boundary() -> CredentialContinuityQualification:
    return CredentialContinuityQualification(
        status=BoundaryContinuityClass.PARTIAL_EVIDENCE,
        tenant_in_credential_reference="qualification: grant carries credential_ref; assert_tenant_consistency at broker",
        lookup_scoped_by_tenant="qualification: scope.tenant_id passed to broker resolve paths",
        cross_tenant_consumption_risk="GAP_QUALIFICATION_ONLY — provider consumption edge still MISSING (gap.credential_to_provider_consumption)",
        mismatch_rejection="qualification: assert_tenant_consistency(ref, grant, scope) before use",
        note=f"anchors {_CONFIG_VALIDATOR_SURFACE} → {_CREDENTIAL_BROKER_SURFACE}; consumption not proven at current HEAD",
    )


def qualify_recovery_boundary() -> RecoveryContinuityQualification:
    return RecoveryContinuityQualification(
        status=BoundaryContinuityClass.GAP_QUALIFICATION_ONLY,
        original_tenant_in_checkpoint="not proven — checkpoint tenant carriers not continuity-proven",
        restored_from_durable_artifact="not proven — resume/recovery typed edge missing",
        caller_substitution_blocked="not proven at qualification layer",
        mismatch_fail_closed="not proven at qualification layer",
        note="gap.checkpoint_to_recovery remains MISSING — no substitute edge",
    )
