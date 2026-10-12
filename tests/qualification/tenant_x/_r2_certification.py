# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R2 qualification report and closure gates."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.tenant_x._r2_chain import (
    build_r2_e2e_chain,
    r2_chain_invalid_count,
    r2_chain_missing_count,
)
from tests.qualification.tenant_x._r2_matrices import (
    CREDENTIAL_BROKER_MATRIX,
    CREDENTIAL_REFERENCE_MATRIX,
    CROSS_TENANT_CREDENTIAL_MATRIX,
    MISSING_DEFAULT_GLOBAL_MATRIX,
    PROVIDER_CONFIG_CONTINUITY_MATRIX,
    PROVIDER_CONSUMPTION_MATRIX,
    PROVIDER_PLUGIN_REWRITE_MATRIX,
)
from tests.qualification.tenant_x._r2_mechanisms import (
    build_r2_canonical_mechanisms,
    r2_consumption_invalid,
    r2_consumption_missing,
    r2_critical_unknown_mechanisms,
)
from tests.qualification.tenant_x._r2_types import (
    R2ChainVerdict,
    R2CrossTenantClass,
    R2ProviderRewriteClass,
    R2SubstitutionClass,
)


@dataclass(frozen=True, slots=True)
class TenantXR2QualificationReport:
    canonical_mechanisms_total: int
    continuity_proven: int
    continuity_missing: int
    continuity_invalid: int
    credential_resolution_proven: int
    credential_resolution_missing: int
    credential_resolution_invalid: int
    consumption_proven: int
    consumption_missing: int
    consumption_invalid: int
    consumption_not_applicable: int
    cross_tenant_same_tenant_proven: int
    cross_tenant_rejected: int
    cross_tenant_allowed: int
    cross_tenant_unknown: int
    provider_scope_preserved: int
    provider_local_scope_pin: int
    provider_rewrite_rejected: int
    provider_rewrite_allowed: int
    provider_rewrite_unknown: int
    substitution_illegal_present: int
    substitution_illegal_input_rejected: int
    substitution_legal_non_tenant: int
    substitution_unknown: int
    b2b_nordea_verdict: str
    r2_ready_for_audit: bool
    remediation_children: tuple[str, ...]


def _continuity_counts() -> tuple[int, int, int]:
    mechanisms = build_r2_canonical_mechanisms()
    proven = sum(1 for m in mechanisms if m.continuity_verdict == R2ChainVerdict.PROVEN_CURRENT_HEAD)
    missing = sum(1 for m in mechanisms if m.continuity_verdict == R2ChainVerdict.MISSING)
    invalid = sum(1 for m in mechanisms if m.continuity_verdict == R2ChainVerdict.INVALID)
    return proven, missing, invalid


def _consumption_counts() -> tuple[int, int, int, int]:
    mechanisms = build_r2_canonical_mechanisms()
    proven = sum(1 for m in mechanisms if m.consumption_verdict == R2ChainVerdict.PROVEN_CURRENT_HEAD)
    missing = sum(1 for m in mechanisms if m.consumption_verdict == R2ChainVerdict.MISSING)
    invalid = sum(1 for m in mechanisms if m.consumption_verdict == R2ChainVerdict.INVALID)
    n_a = sum(1 for m in mechanisms if m.consumption_verdict == R2ChainVerdict.NOT_APPLICABLE)
    return proven, missing, invalid, n_a


def _credential_resolution_counts() -> tuple[int, int, int]:
    rows = CREDENTIAL_REFERENCE_MATRIX + CREDENTIAL_BROKER_MATRIX
    proven = sum(1 for r in rows if r.verdict == R2ChainVerdict.PROVEN_CURRENT_HEAD)
    missing = sum(1 for r in rows if r.verdict == R2ChainVerdict.MISSING)
    invalid = sum(1 for r in rows if r.verdict == R2ChainVerdict.INVALID)
    return proven, missing, invalid


def _cross_tenant_counts() -> tuple[int, int, int, int]:
    rows = CROSS_TENANT_CREDENTIAL_MATRIX
    same = sum(1 for r in rows if r.verdict == R2CrossTenantClass.SAME_TENANT_PROVEN)
    rejected = sum(1 for r in rows if r.verdict == R2CrossTenantClass.CROSS_TENANT_REJECTED)
    allowed = sum(1 for r in rows if r.verdict == R2CrossTenantClass.CROSS_TENANT_ALLOWED)
    unknown = sum(1 for r in rows if r.verdict == R2CrossTenantClass.UNKNOWN)
    return same, rejected, allowed, unknown


def _provider_rewrite_counts() -> tuple[int, int, int, int, int]:
    rows = PROVIDER_PLUGIN_REWRITE_MATRIX
    preserved = sum(1 for r in rows if r.verdict == R2ProviderRewriteClass.SCOPE_PRESERVED)
    local_pin = sum(1 for r in rows if r.verdict == R2ProviderRewriteClass.LOCAL_SCOPE_PIN)
    rejected = sum(1 for r in rows if r.verdict == R2ProviderRewriteClass.REWRITE_REJECTED)
    allowed = sum(1 for r in rows if r.verdict == R2ProviderRewriteClass.REWRITE_ALLOWED)
    unknown = sum(1 for r in rows if r.verdict == R2ProviderRewriteClass.UNKNOWN)
    return preserved, local_pin, rejected, allowed, unknown


def _substitution_counts() -> tuple[int, int, int, int]:
    rows = MISSING_DEFAULT_GLOBAL_MATRIX
    illegal_present = sum(
        1 for r in rows if r.verdict == R2SubstitutionClass.ILLEGAL_SUBSTITUTION_PRESENT
    )
    illegal_rejected = sum(1 for r in rows if r.verdict == R2SubstitutionClass.ILLEGAL_INPUT_REJECTED)
    legal = sum(1 for r in rows if r.verdict == R2SubstitutionClass.LEGAL_NON_TENANT)
    unknown = sum(1 for r in rows if r.verdict == R2SubstitutionClass.UNKNOWN)
    return illegal_present, illegal_rejected, legal, unknown


def build_r2_qualification_report() -> TenantXR2QualificationReport:
    mechanisms = build_r2_canonical_mechanisms()
    cont_proven, cont_missing, cont_invalid = _continuity_counts()
    cred_res_proven, cred_res_missing, cred_res_invalid = _credential_resolution_counts()
    cons_proven, cons_missing, cons_invalid, cons_na = _consumption_counts()

    ct_same, ct_rejected, ct_allowed, ct_unknown = _cross_tenant_counts()
    pr_preserved, pr_pin, pr_rejected, pr_allowed, pr_unknown = _provider_rewrite_counts()
    sub_illegal, sub_rejected, sub_legal, sub_unknown = _substitution_counts()

    remediation: list[str] = []
    if r2_consumption_missing() or r2_consumption_invalid():
        remediation.append("TENANT-X-R2-R1 - credential→provider consumption gaps")
    if ct_allowed > 0 or ct_unknown > 0:
        remediation.append("TENANT-X-R2-R1 - cross-tenant credential resolution closure")
    if pr_allowed > 0 or pr_unknown > 0:
        remediation.append("TENANT-X-R2-R1 - provider/plugin tenant rewrite closure")
    if sub_illegal > 0 or sub_unknown > 0:
        remediation.append("TENANT-X-R2-R1 - default/global substitution closure")
    if r2_critical_unknown_mechanisms():
        remediation.append("TENANT-X-R2-R2 - critical provider/config mechanism reconciliation")
    if r2_chain_missing_count() or r2_chain_invalid_count():
        remediation.append("TENANT-X-R2-R3 - E2E chain structural proof gaps")
    if cred_res_missing > 0 or cred_res_invalid > 0:
        remediation.append("TENANT-X-R2-R4 - credential resolution matrix gaps")
    if cont_missing > 0 or cont_invalid > 0:
        remediation.append("TENANT-X-R2-R5 - provider/config continuity gaps")

    ready = (
        cons_missing == 0
        and cons_invalid == 0
        and ct_allowed == 0
        and ct_unknown == 0
        and pr_allowed == 0
        and pr_unknown == 0
        and sub_illegal == 0
        and sub_unknown == 0
        and cred_res_missing == 0
        and cred_res_invalid == 0
        and cont_missing == 0
        and cont_invalid == 0
        and r2_chain_missing_count() == 0
        and r2_chain_invalid_count() == 0
        and r2_critical_unknown_mechanisms() == 0
    )

    return TenantXR2QualificationReport(
        canonical_mechanisms_total=len(mechanisms),
        continuity_proven=cont_proven,
        continuity_missing=cont_missing,
        continuity_invalid=cont_invalid,
        credential_resolution_proven=cred_res_proven,
        credential_resolution_missing=cred_res_missing,
        credential_resolution_invalid=cred_res_invalid,
        consumption_proven=cons_proven,
        consumption_missing=cons_missing,
        consumption_invalid=cons_invalid,
        consumption_not_applicable=cons_na,
        cross_tenant_same_tenant_proven=ct_same,
        cross_tenant_rejected=ct_rejected,
        cross_tenant_allowed=ct_allowed,
        cross_tenant_unknown=ct_unknown,
        provider_scope_preserved=pr_preserved,
        provider_local_scope_pin=pr_pin,
        provider_rewrite_rejected=pr_rejected,
        provider_rewrite_allowed=pr_allowed,
        provider_rewrite_unknown=pr_unknown,
        substitution_illegal_present=sub_illegal,
        substitution_illegal_input_rejected=sub_rejected,
        substitution_legal_non_tenant=sub_legal,
        substitution_unknown=sub_unknown,
        b2b_nordea_verdict="READY WITH PLATFORM GAPS",
        r2_ready_for_audit=ready,
        remediation_children=tuple(remediation),
    )


def summarize_r2_metrics(report: TenantXR2QualificationReport) -> dict[str, int | str | tuple[str, ...]]:
    return {
        "canonical_mechanisms_total": report.canonical_mechanisms_total,
        "continuity_proven": report.continuity_proven,
        "continuity_missing": report.continuity_missing,
        "continuity_invalid": report.continuity_invalid,
        "credential_resolution_proven": report.credential_resolution_proven,
        "credential_resolution_missing": report.credential_resolution_missing,
        "credential_resolution_invalid": report.credential_resolution_invalid,
        "consumption_proven": report.consumption_proven,
        "consumption_missing": report.consumption_missing,
        "consumption_invalid": report.consumption_invalid,
        "consumption_not_applicable": report.consumption_not_applicable,
        "cross_tenant_same_tenant_proven": report.cross_tenant_same_tenant_proven,
        "cross_tenant_rejected": report.cross_tenant_rejected,
        "cross_tenant_allowed": report.cross_tenant_allowed,
        "cross_tenant_unknown": report.cross_tenant_unknown,
        "provider_scope_preserved": report.provider_scope_preserved,
        "provider_local_scope_pin": report.provider_local_scope_pin,
        "provider_rewrite_rejected": report.provider_rewrite_rejected,
        "provider_rewrite_allowed": report.provider_rewrite_allowed,
        "provider_rewrite_unknown": report.provider_rewrite_unknown,
        "substitution_illegal_present": report.substitution_illegal_present,
        "substitution_illegal_input_rejected": report.substitution_illegal_input_rejected,
        "substitution_legal_non_tenant": report.substitution_legal_non_tenant,
        "substitution_unknown": report.substitution_unknown,
        "e2e_chain_transitions": len(build_r2_e2e_chain()),
        "provider_config_continuity_rows": len(PROVIDER_CONFIG_CONTINUITY_MATRIX),
        "credential_reference_rows": len(CREDENTIAL_REFERENCE_MATRIX),
        "credential_broker_rows": len(CREDENTIAL_BROKER_MATRIX),
        "provider_consumption_rows": len(PROVIDER_CONSUMPTION_MATRIX),
        "cross_tenant_rows": len(CROSS_TENANT_CREDENTIAL_MATRIX),
        "plugin_rewrite_rows": len(PROVIDER_PLUGIN_REWRITE_MATRIX),
        "substitution_rows": len(MISSING_DEFAULT_GLOBAL_MATRIX),
        "b2b_nordea": report.b2b_nordea_verdict,
        "remediation_children": report.remediation_children,
    }
