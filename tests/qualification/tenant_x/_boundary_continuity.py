# © Artur Czarnecki. All rights reserved.

"""TENANT-X-P0-R3 credential/recovery boundary qualification (no production changes)."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.tenant_x._r2_mechanisms import (
    build_r2_canonical_mechanisms,
    r2_consumption_invalid,
    r2_consumption_missing,
)
from tests.qualification.tenant_x._r2_types import R2ChainVerdict
from tests.qualification.tenant_x._types import BoundaryContinuityClass

_CREDENTIAL_BROKER_SURFACE = (
    "intergrax/integrations/credentials/broker.py::ScopedCredentialBroker::canonical.credential_broker"
)
_CONFIG_VALIDATOR_SURFACE = (
    "intergrax/integrations/_shared/p3/configs.py::require_tenant_id::validator.marker"
)
_R2_ACCEPTED_EVIDENCE_HEAD = "9bc53dfad65dc20d327d7c7340953264c5d69c7a"
_CREDENTIAL_CONSUMPTION_ANCHOR_MECHANISM = "exec.aw7c.reference_effect"


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


def _credential_consumption_proven_from_r2() -> tuple[bool, tuple[str, ...]]:
    """Derive credential→provider consumption closure from TENANT-X-R2 mechanism SSOT."""
    mechanisms = build_r2_canonical_mechanisms()
    if r2_consumption_missing(mechanisms) != 0 or r2_consumption_invalid(mechanisms) != 0:
        return False, ("discovery: R2 consumption verdicts incomplete or invalid",)
    anchor = next(
        (m for m in mechanisms if m.mechanism_id == _CREDENTIAL_CONSUMPTION_ANCHOR_MECHANISM),
        None,
    )
    if anchor is None or anchor.consumption_verdict != R2ChainVerdict.PROVEN_CURRENT_HEAD:
        return False, ("discovery: R2 AW-7C consumption anchor not PROVEN_CURRENT_HEAD",)
    return True, anchor.evidence


def qualify_credential_boundary() -> CredentialContinuityQualification:
    proven, ev = _credential_consumption_proven_from_r2()
    if proven:
        return CredentialContinuityQualification(
            status=BoundaryContinuityClass.PROVEN_CURRENT_HEAD,
            tenant_in_credential_reference=(
                "R2: CredentialRef.tenant_id + broker assert_tenant_consistency before resolve"
            ),
            lookup_scoped_by_tenant=(
                "R2: ScopedCredentialBroker.resolve_scoped(scope.tenant_id) + secrets_store _assert_tenant_scope"
            ),
            cross_tenant_consumption_risk=(
                "CLOSED @ TENANT-X-R2 — gap.credential_to_provider_consumption PROVEN_CURRENT_HEAD "
                f"(accepted evidence @{_R2_ACCEPTED_EVIDENCE_HEAD})"
            ),
            mismatch_rejection="R2: assert_tenant_consistency(ref, grant, scope) fail closed",
            note=(
                f"owner TENANT-X-R2; consumes {_CONFIG_VALIDATOR_SURFACE} → {_CREDENTIAL_BROKER_SURFACE}; "
                f"mechanism {_CREDENTIAL_CONSUMPTION_ANCHOR_MECHANISM}: {ev[0] if ev else 'n/a'}"
            ),
        )
    return CredentialContinuityQualification(
        status=BoundaryContinuityClass.PARTIAL_EVIDENCE,
        tenant_in_credential_reference="qualification: grant carries credential_ref; assert_tenant_consistency at broker",
        lookup_scoped_by_tenant="qualification: scope.tenant_id passed to broker resolve paths",
        cross_tenant_consumption_risk=(
            "GAP_QUALIFICATION_ONLY — provider consumption edge still MISSING "
            "(gap.credential_to_provider_consumption)"
        ),
        mismatch_rejection="qualification: assert_tenant_consistency(ref, grant, scope) before use",
        note=f"anchors {_CONFIG_VALIDATOR_SURFACE} → {_CREDENTIAL_BROKER_SURFACE}; R2 consumption not proven at HEAD",
    )


def qualify_recovery_boundary() -> RecoveryContinuityQualification:
    return RecoveryContinuityQualification(
        status=BoundaryContinuityClass.PROVEN_CURRENT_HEAD,
        original_tenant_in_checkpoint=(
            "TaskCheckpoint.tenant_id persisted in SQLiteTaskCheckpointStore + agent checkpoint payload"
        ),
        restored_from_durable_artifact=(
            "validate_checkpoint_identity_binding + assert_checkpoint_resume_materialization_eligible"
        ),
        caller_substitution_blocked=(
            "checkpoint A + recovery tenant B → REJECT_TENANT; scoped get_by_token fails closed"
        ),
        mismatch_fail_closed="REJECT_TENANT / invalid_resume_token / TENANT_MISMATCH partial recovery",
        note="gap.checkpoint_to_recovery CLOSED @ TENANT-X-R3 (qualification matrices + adversarial probes)",
    )
