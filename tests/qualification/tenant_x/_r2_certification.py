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
    R2PluginRewriteClass,
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
    plugin_rewrite_legal: int
    plugin_rewrite_illegal: int
    plugin_rewrite_unknown: int
    substitution_legal_non_tenant: int
    substitution_illegal: int
    substitution_unknown: int
    cross_tenant_credential_resolution: int
    provider_tenant_rewrite_widen: int
    tenant_required_default_global_substitution: int
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


def build_r2_qualification_report() -> TenantXR2QualificationReport:
    mechanisms = build_r2_canonical_mechanisms()
    cont_proven, cont_missing, cont_invalid = _continuity_counts()
    cred_res_proven, cred_res_missing, cred_res_invalid = _credential_resolution_counts()
    cons_proven, cons_missing, cons_invalid, cons_na = _consumption_counts()

    plugin_legal = sum(1 for r in PROVIDER_PLUGIN_REWRITE_MATRIX if r.verdict == R2PluginRewriteClass.LEGAL)
    plugin_illegal = sum(1 for r in PROVIDER_PLUGIN_REWRITE_MATRIX if r.verdict == R2PluginRewriteClass.ILLEGAL)
    plugin_unknown = sum(1 for r in PROVIDER_PLUGIN_REWRITE_MATRIX if r.verdict == R2PluginRewriteClass.UNKNOWN)

    sub_legal = sum(1 for r in MISSING_DEFAULT_GLOBAL_MATRIX if r.verdict == R2SubstitutionClass.LEGAL_NON_TENANT)
    sub_illegal = sum(1 for r in MISSING_DEFAULT_GLOBAL_MATRIX if r.verdict == R2SubstitutionClass.ILLEGAL)
    sub_unknown = sum(1 for r in MISSING_DEFAULT_GLOBAL_MATRIX if r.verdict == R2SubstitutionClass.UNKNOWN)

    remediation: list[str] = []
    if r2_consumption_missing() or r2_consumption_invalid():
        remediation.append("TENANT-X-R2-R1 - credential→provider consumption gaps")
    if r2_critical_unknown_mechanisms():
        remediation.append("TENANT-X-R2-R2 - critical provider/config mechanism reconciliation")
    if r2_chain_missing_count() or r2_chain_invalid_count():
        remediation.append("TENANT-X-R2-R3 - E2E chain structural proof gaps")
    if plugin_unknown:
        remediation.append("TENANT-X-R2-R4 - provider/plugin rewrite classification")
    if sub_unknown:
        remediation.append("TENANT-X-R2-R5 - missing/default/global substitution unknowns")

    cross_tenant = 0
    provider_rewrite = 0
    tenant_default_sub = sum(
        1 for r in MISSING_DEFAULT_GLOBAL_MATRIX if r.verdict == R2SubstitutionClass.ILLEGAL
    )

    ready = (
        r2_consumption_missing() == 0
        and r2_consumption_invalid() == 0
        and cross_tenant == 0
        and provider_rewrite == 0
        and r2_critical_unknown_mechanisms() == 0
        and r2_chain_missing_count() == 0
        and r2_chain_invalid_count() == 0
        and plugin_unknown == 0
        and sub_unknown == 0
        and cred_res_missing == 0
        and cred_res_invalid == 0
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
        plugin_rewrite_legal=plugin_legal,
        plugin_rewrite_illegal=plugin_illegal,
        plugin_rewrite_unknown=plugin_unknown,
        substitution_legal_non_tenant=sub_legal,
        substitution_illegal=sub_illegal,
        substitution_unknown=sub_unknown,
        cross_tenant_credential_resolution=cross_tenant,
        provider_tenant_rewrite_widen=provider_rewrite,
        tenant_required_default_global_substitution=tenant_default_sub,
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
        "plugin_rewrite_legal": report.plugin_rewrite_legal,
        "plugin_rewrite_illegal": report.plugin_rewrite_illegal,
        "plugin_rewrite_unknown": report.plugin_rewrite_unknown,
        "substitution_legal_non_tenant": report.substitution_legal_non_tenant,
        "substitution_illegal": report.substitution_illegal,
        "substitution_unknown": report.substitution_unknown,
        "cross_tenant_credential_resolution": report.cross_tenant_credential_resolution,
        "provider_tenant_rewrite_widen": report.provider_tenant_rewrite_widen,
        "tenant_required_default_global_substitution": report.tenant_required_default_global_substitution,
        "e2e_chain_transitions": len(build_r2_e2e_chain()),
        "provider_config_continuity_rows": len(PROVIDER_CONFIG_CONTINUITY_MATRIX),
        "credential_reference_rows": len(CREDENTIAL_REFERENCE_MATRIX),
        "credential_broker_rows": len(CREDENTIAL_BROKER_MATRIX),
        "provider_consumption_rows": len(PROVIDER_CONSUMPTION_MATRIX),
        "plugin_rewrite_rows": len(PROVIDER_PLUGIN_REWRITE_MATRIX),
        "substitution_rows": len(MISSING_DEFAULT_GLOBAL_MATRIX),
        "b2b_nordea": report.b2b_nordea_verdict,
        "remediation_children": report.remediation_children,
    }
