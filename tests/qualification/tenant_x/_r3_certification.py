# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R3 qualification report and closure gates."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.tenant_x._r3_chain import (
    build_r3_e2e_chain,
    r3_chain_invalid_count,
    r3_chain_missing_count,
)
from tests.qualification.tenant_x._r3_matrices import (
    CHECKPOINT_LOAD_MATRIX,
    CHECKPOINT_WRITE_MATRIX,
    CROSS_TENANT_RECOVERY_MATRIX,
    DURABLE_KEY_ISOLATION_MATRIX,
    MISSING_DEFAULT_GLOBAL_MATRIX,
    RECOVERY_RESUME_MATRIX,
    RECOVERY_TENANT_AUTHORITY_MATRIX,
    RETRY_REPLAY_MATRIX,
    STATE_TENANT_CONTINUITY_MATRIX,
)
from tests.qualification.tenant_x._r3_mechanisms import (
    build_r3_recovery_families,
    r3_critical_unknown_families,
)
from tests.qualification.tenant_x._r3_types import (
    R3ChainVerdict,
    R3CrossTenantClass,
    R3FamilyBinding,
    R3RecoveryAuthorityRole,
    R3RecoveryPathClass,
    R3SubstitutionClass,
)


@dataclass(frozen=True, slots=True)
class TenantXR3QualificationReport:
    recovery_families_total: int
    tenant_bound_families: int
    non_tenant_families: int
    critical_unknown_families: int
    checkpoint_write_proven: int
    checkpoint_write_missing: int
    checkpoint_write_invalid: int
    checkpoint_load_proven: int
    checkpoint_load_missing: int
    checkpoint_load_invalid: int
    recovery_resume_proven: int
    recovery_resume_missing: int
    recovery_resume_invalid: int
    retry_replay_proven: int
    retry_replay_missing: int
    retry_replay_invalid: int
    durable_key_proven: int
    durable_key_missing: int
    durable_key_invalid: int
    cross_tenant_same_tenant_proven: int
    cross_tenant_rejected: int
    cross_tenant_allowed: int
    cross_tenant_unknown: int
    authority_carrier: int
    authority_validator: int
    authority_restorer: int
    authority_mint: int
    authority_unknown: int
    substitution_illegal_present: int
    substitution_illegal_input_rejected: int
    substitution_legal_non_tenant: int
    substitution_unknown: int
    checkpoint_to_recovery_missing: int
    checkpoint_to_recovery_invalid: int
    b2b_nordea_verdict: str
    r3_ready_for_audit: bool
    remediation_children: tuple[str, ...]


def _checkpoint_write_counts() -> tuple[int, int, int]:
    proven = sum(1 for r in CHECKPOINT_WRITE_MATRIX if r.verdict == R3ChainVerdict.PROVEN_CURRENT_HEAD)
    missing = sum(1 for r in CHECKPOINT_WRITE_MATRIX if r.verdict == R3ChainVerdict.MISSING)
    invalid = sum(1 for r in CHECKPOINT_WRITE_MATRIX if r.verdict == R3ChainVerdict.INVALID)
    return proven, missing, invalid


def _checkpoint_load_counts() -> tuple[int, int, int]:
    ok = {
        R3RecoveryPathClass.DURABLE_TENANT_AUTHORITATIVE,
        R3RecoveryPathClass.CALLER_TENANT_VALIDATED_AGAINST_DURABLE,
    }
    proven = sum(1 for r in CHECKPOINT_LOAD_MATRIX if r.verdict in ok)
    missing = sum(1 for r in CHECKPOINT_LOAD_MATRIX if r.verdict == R3RecoveryPathClass.UNKNOWN)
    invalid = sum(
        1
        for r in CHECKPOINT_LOAD_MATRIX
        if r.verdict
        in {R3RecoveryPathClass.CALLER_TENANT_AUTHORITATIVE, R3RecoveryPathClass.NO_TENANT_CHECK}
    )
    return proven, missing, invalid


def _recovery_resume_counts() -> tuple[int, int, int]:
    ok = {
        R3RecoveryPathClass.DURABLE_TENANT_AUTHORITATIVE,
        R3RecoveryPathClass.CALLER_TENANT_VALIDATED_AGAINST_DURABLE,
    }
    proven = sum(1 for r in RECOVERY_RESUME_MATRIX if r.verdict in ok)
    missing = sum(1 for r in RECOVERY_RESUME_MATRIX if r.verdict == R3RecoveryPathClass.UNKNOWN)
    invalid = sum(
        1
        for r in RECOVERY_RESUME_MATRIX
        if r.verdict
        in {R3RecoveryPathClass.CALLER_TENANT_AUTHORITATIVE, R3RecoveryPathClass.NO_TENANT_CHECK}
    )
    return proven, missing, invalid


def _retry_replay_counts() -> tuple[int, int, int]:
    proven = sum(1 for r in RETRY_REPLAY_MATRIX if r.verdict == R3ChainVerdict.PROVEN_CURRENT_HEAD)
    missing = sum(1 for r in RETRY_REPLAY_MATRIX if r.verdict == R3ChainVerdict.MISSING)
    invalid = sum(1 for r in RETRY_REPLAY_MATRIX if r.verdict == R3ChainVerdict.INVALID)
    return proven, missing, invalid


def _durable_key_counts() -> tuple[int, int, int]:
    proven = sum(1 for r in DURABLE_KEY_ISOLATION_MATRIX if r.verdict == R3ChainVerdict.PROVEN_CURRENT_HEAD)
    missing = sum(1 for r in DURABLE_KEY_ISOLATION_MATRIX if r.verdict == R3ChainVerdict.MISSING)
    invalid = sum(1 for r in DURABLE_KEY_ISOLATION_MATRIX if r.verdict == R3ChainVerdict.INVALID)
    return proven, missing, invalid


def build_r3_qualification_report() -> TenantXR3QualificationReport:
    families = build_r3_recovery_families()
    tenant_bound = sum(1 for f in families if f.binding == R3FamilyBinding.TENANT_BOUND)
    non_tenant = sum(1 for f in families if f.binding == R3FamilyBinding.NON_TENANT)
    unknown_families = len(r3_critical_unknown_families())

    cw_p, cw_m, cw_i = _checkpoint_write_counts()
    cl_p, cl_m, cl_i = _checkpoint_load_counts()
    rr_p, rr_m, rr_i = _recovery_resume_counts()
    tr_p, tr_m, tr_i = _retry_replay_counts()
    dk_p, dk_m, dk_i = _durable_key_counts()

    ct_rows = CROSS_TENANT_RECOVERY_MATRIX
    ct_same = sum(1 for r in ct_rows if r.verdict == R3CrossTenantClass.SAME_TENANT_PROVEN)
    ct_rej = sum(1 for r in ct_rows if r.verdict == R3CrossTenantClass.CROSS_TENANT_REJECTED)
    ct_allow = sum(1 for r in ct_rows if r.verdict == R3CrossTenantClass.CROSS_TENANT_ALLOWED)
    ct_unk = sum(1 for r in ct_rows if r.verdict == R3CrossTenantClass.UNKNOWN)

    auth_rows = RECOVERY_TENANT_AUTHORITY_MATRIX
    auth_carrier = sum(1 for r in auth_rows if r.verdict == R3RecoveryAuthorityRole.TENANT_CARRIER)
    auth_validator = sum(1 for r in auth_rows if r.verdict == R3RecoveryAuthorityRole.TENANT_VALIDATOR)
    auth_restorer = sum(
        1 for r in auth_rows if r.verdict == R3RecoveryAuthorityRole.TENANT_RESTORER_FROM_DURABLE_TRUTH
    )
    auth_mint = sum(1 for r in auth_rows if r.verdict == R3RecoveryAuthorityRole.TENANT_AUTHORITY)
    auth_unk = sum(1 for r in auth_rows if r.verdict == R3RecoveryAuthorityRole.UNKNOWN)

    sub_rows = MISSING_DEFAULT_GLOBAL_MATRIX
    sub_illegal = sum(1 for r in sub_rows if r.verdict == R3SubstitutionClass.ILLEGAL_SUBSTITUTION_PRESENT)
    sub_rej = sum(1 for r in sub_rows if r.verdict == R3SubstitutionClass.ILLEGAL_INPUT_REJECTED)
    sub_legal = sum(1 for r in sub_rows if r.verdict == R3SubstitutionClass.LEGAL_NON_TENANT)
    sub_unk = sum(1 for r in sub_rows if r.verdict == R3SubstitutionClass.UNKNOWN)

    cp_to_rec_missing = r3_chain_missing_count()
    cp_to_rec_invalid = r3_chain_invalid_count()

    remediation: list[str] = []
    if unknown_families:
        remediation.append("TENANT-X-R3-R1 - critical recovery family UNKNOWN reconciliation")
    if cp_to_rec_missing or cp_to_rec_invalid:
        remediation.append("TENANT-X-R3-R2 - checkpoint→recovery E2E structural proof gaps")
    if cl_m or cl_i or rr_m or rr_i:
        remediation.append("TENANT-X-R3-R3 - recovery/resume path classification gaps")
    if ct_allow or ct_unk:
        remediation.append("TENANT-X-R3-R4 - cross-tenant recovery adversarial closure")
    if auth_mint or auth_unk:
        remediation.append("TENANT-X-R3-R5 - recovery tenant authority reconciliation")
    if sub_illegal or sub_unk:
        remediation.append("TENANT-X-R3-R6 - illegal default/global substitution closure")
    if dk_m or dk_i:
        remediation.append("TENANT-X-R3-R7 - durable tenant-key isolation gaps")

    ready = (
        unknown_families == 0
        and cp_to_rec_missing == 0
        and cp_to_rec_invalid == 0
        and cl_m == 0
        and cl_i == 0
        and rr_m == 0
        and rr_i == 0
        and ct_allow == 0
        and ct_unk == 0
        and auth_mint == 0
        and auth_unk == 0
        and dk_m == 0
        and dk_i == 0
        and sub_illegal == 0
        and sub_unk == 0
    )

    return TenantXR3QualificationReport(
        recovery_families_total=len(families),
        tenant_bound_families=tenant_bound,
        non_tenant_families=non_tenant,
        critical_unknown_families=unknown_families,
        checkpoint_write_proven=cw_p,
        checkpoint_write_missing=cw_m,
        checkpoint_write_invalid=cw_i,
        checkpoint_load_proven=cl_p,
        checkpoint_load_missing=cl_m,
        checkpoint_load_invalid=cl_i,
        recovery_resume_proven=rr_p,
        recovery_resume_missing=rr_m,
        recovery_resume_invalid=rr_i,
        retry_replay_proven=tr_p,
        retry_replay_missing=tr_m,
        retry_replay_invalid=tr_i,
        durable_key_proven=dk_p,
        durable_key_missing=dk_m,
        durable_key_invalid=dk_i,
        cross_tenant_same_tenant_proven=ct_same,
        cross_tenant_rejected=ct_rej,
        cross_tenant_allowed=ct_allow,
        cross_tenant_unknown=ct_unk,
        authority_carrier=auth_carrier,
        authority_validator=auth_validator,
        authority_restorer=auth_restorer,
        authority_mint=auth_mint,
        authority_unknown=auth_unk,
        substitution_illegal_present=sub_illegal,
        substitution_illegal_input_rejected=sub_rej,
        substitution_legal_non_tenant=sub_legal,
        substitution_unknown=sub_unk,
        checkpoint_to_recovery_missing=cp_to_rec_missing,
        checkpoint_to_recovery_invalid=cp_to_rec_invalid,
        b2b_nordea_verdict="READY WITH PLATFORM GAPS",
        r3_ready_for_audit=ready,
        remediation_children=tuple(remediation),
    )


def summarize_r3_metrics(report: TenantXR3QualificationReport) -> dict[str, int | str | tuple[str, ...]]:
    return {
        "recovery_families_total": report.recovery_families_total,
        "tenant_bound_families": report.tenant_bound_families,
        "non_tenant_families": report.non_tenant_families,
        "critical_unknown_families": report.critical_unknown_families,
        "checkpoint_write_proven": report.checkpoint_write_proven,
        "checkpoint_write_missing": report.checkpoint_write_missing,
        "checkpoint_write_invalid": report.checkpoint_write_invalid,
        "checkpoint_load_proven": report.checkpoint_load_proven,
        "checkpoint_load_missing": report.checkpoint_load_missing,
        "checkpoint_load_invalid": report.checkpoint_load_invalid,
        "recovery_resume_proven": report.recovery_resume_proven,
        "recovery_resume_missing": report.recovery_resume_missing,
        "recovery_resume_invalid": report.recovery_resume_invalid,
        "retry_replay_proven": report.retry_replay_proven,
        "retry_replay_missing": report.retry_replay_missing,
        "retry_replay_invalid": report.retry_replay_invalid,
        "durable_key_proven": report.durable_key_proven,
        "durable_key_missing": report.durable_key_missing,
        "durable_key_invalid": report.durable_key_invalid,
        "cross_tenant_same_tenant_proven": report.cross_tenant_same_tenant_proven,
        "cross_tenant_rejected": report.cross_tenant_rejected,
        "cross_tenant_allowed": report.cross_tenant_allowed,
        "cross_tenant_unknown": report.cross_tenant_unknown,
        "authority_carrier": report.authority_carrier,
        "authority_validator": report.authority_validator,
        "authority_restorer": report.authority_restorer,
        "authority_mint": report.authority_mint,
        "authority_unknown": report.authority_unknown,
        "substitution_illegal_present": report.substitution_illegal_present,
        "substitution_illegal_input_rejected": report.substitution_illegal_input_rejected,
        "substitution_legal_non_tenant": report.substitution_legal_non_tenant,
        "substitution_unknown": report.substitution_unknown,
        "checkpoint_to_recovery_missing": report.checkpoint_to_recovery_missing,
        "checkpoint_to_recovery_invalid": report.checkpoint_to_recovery_invalid,
        "e2e_chain_transitions": len(build_r3_e2e_chain()),
        "state_continuity_rows": len(STATE_TENANT_CONTINUITY_MATRIX),
        "b2b_nordea": report.b2b_nordea_verdict,
        "remediation_children": report.remediation_children,
    }
