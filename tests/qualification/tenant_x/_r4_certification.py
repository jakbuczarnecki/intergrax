# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R4 qualification report and closure gates."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.tenant_x._propagation import TENANT_PROPAGATION_GAPS, PropagationEdgeStatus
from tests.qualification.tenant_x._r4_call_sites import (
    build_record_context_assembly_inventory,
    build_record_llm_call_inventory,
)
from tests.qualification.tenant_x._r4_chain import (
    build_r4_e2e_chain,
    context_model_chain_counts,
    context_model_chain_proven,
    evidence_event_chain_counts,
    queue_worker_chain_counts,
    r4_chain_invalid_count,
    r4_chain_missing_count,
    r4_chain_proven_count,
)
from tests.qualification.tenant_x._r4_matrices import (
    ASYNC_DOMAIN_WORKER_MATRIX,
    BACKGROUND_EXECUTION_IDENTITY_MATRIX,
    CONTEXT_MODEL_CONTINUITY_MATRIX,
    CONTEXT_VIEW_ISOLATION_MATRIX,
    CROSS_TENANT_R4_MATRIX,
    EVIDENCE_TENANT_CONTINUITY_MATRIX,
    EVENT_TRACE_CONTINUITY_MATRIX,
    EXTENSION_SCOPE_MATRIX,
    MISSING_DEFAULT_GLOBAL_MATRIX,
    MODEL_ATTRIBUTION_OPTIONAL_TENANT_MATRIX,
    QUEUE_WORKER_CONTINUITY_MATRIX,
)
from tests.qualification.tenant_x._r4_mechanisms import build_r4_families, r4_critical_unknown_families
from tests.qualification.tenant_x._r4_types import (
    R4AttributionTenantOptionalClass,
    R4ChainVerdict,
    R4CrossTenantClass,
    R4ExtensionTenantClass,
    R4FamilyBinding,
    R4ObservabilityRole,
    R4SubstitutionClass,
)


@dataclass(frozen=True, slots=True)
class TenantXR4QualificationReport:
    r4_families_total: int
    tenant_bound_families: int
    non_tenant_families: int
    critical_unknown_families: int
    context_model_proven: int
    context_model_missing: int
    context_model_invalid: int
    context_view_proven: int
    context_view_missing: int
    context_view_invalid: int
    evidence_proven: int
    evidence_missing: int
    evidence_invalid: int
    event_trace_proven: int
    event_trace_missing: int
    event_trace_invalid: int
    background_identity_proven: int
    background_identity_missing: int
    background_identity_invalid: int
    queue_worker_proven: int
    queue_worker_missing: int
    queue_worker_invalid: int
    async_worker_proven: int
    async_worker_missing: int
    async_worker_invalid: int
    cross_tenant_same_tenant_proven: int
    cross_tenant_rejected: int
    cross_tenant_allowed: int
    cross_tenant_unknown: int
    observability_carrier: int
    observability_validator: int
    observability_derived: int
    observability_authority: int
    observability_unknown: int
    extension_preserved: int
    extension_rejected: int
    extension_allowed: int
    extension_unknown: int
    substitution_illegal_present: int
    substitution_illegal_input_rejected: int
    substitution_legal_non_tenant: int
    substitution_unknown: int
    context_model_gap_proven: bool
    record_context_assembly_callers_total: int
    record_context_assembly_canonically_bound: int
    record_context_assembly_unsafe: int
    record_context_assembly_unknown_callers: int
    record_llm_call_callers_total: int
    record_llm_call_canonical_tenant: int
    record_llm_call_validated_explicit: int
    record_llm_call_untrusted_explicit: int
    record_llm_call_unknown: int
    e2e_chain_proven: int
    e2e_chain_missing: int
    e2e_chain_invalid: int
    attribution_optional_tenant_class: str
    b2b_nordea_verdict: str
    r4_ready_for_audit: bool
    remediation_children: tuple[str, ...]


def _verdict_counts(
    rows: tuple[object, ...],
    *,
    proven: set[object],
    missing: set[object] | None = None,
    invalid: set[object] | None = None,
) -> tuple[int, int, int]:
    missing = missing or set()
    invalid = invalid or set()
    p = sum(1 for r in rows if getattr(r, "verdict") in proven)
    m = sum(1 for r in rows if getattr(r, "verdict") in missing)
    i = sum(1 for r in rows if getattr(r, "verdict") in invalid)
    return p, m, i


def _context_model_gap_proven() -> bool:
    gap = next(g for g in TENANT_PROPAGATION_GAPS if g.gap_id == "gap.context_model_scope")
    return gap.status == PropagationEdgeStatus.PROVEN and context_model_chain_proven()


def build_r4_qualification_report() -> TenantXR4QualificationReport:
    families = build_r4_families()
    tenant_bound = sum(1 for f in families if f.binding == R4FamilyBinding.TENANT_BOUND)
    non_tenant = sum(1 for f in families if f.binding == R4FamilyBinding.NON_TENANT)
    unknown_families = len(r4_critical_unknown_families())

    cm_p, cm_m, cm_i = _verdict_counts(
        CONTEXT_MODEL_CONTINUITY_MATRIX,
        proven={R4ChainVerdict.PROVEN_CURRENT_HEAD},
        missing={R4ChainVerdict.MISSING},
        invalid={R4ChainVerdict.INVALID},
    )
    cv_p, cv_m, cv_i = _verdict_counts(
        CONTEXT_VIEW_ISOLATION_MATRIX,
        proven={R4ChainVerdict.PROVEN_CURRENT_HEAD},
        missing={R4ChainVerdict.MISSING},
        invalid={R4ChainVerdict.INVALID},
    )
    ev_p, ev_m, ev_i = _verdict_counts(
        EVIDENCE_TENANT_CONTINUITY_MATRIX,
        proven={R4ChainVerdict.PROVEN_CURRENT_HEAD},
        missing={R4ChainVerdict.MISSING},
        invalid={R4ChainVerdict.INVALID},
    )
    et_obs = {R4ObservabilityRole.CARRIER, R4ObservabilityRole.VALIDATOR, R4ObservabilityRole.DERIVED_OBSERVATION}
    et_p = sum(
        1
        for r in EVENT_TRACE_CONTINUITY_MATRIX
        if r.verdict in et_obs or r.verdict.value.endswith("EXECUTION")
    )
    et_m = sum(1 for r in EVENT_TRACE_CONTINUITY_MATRIX if r.verdict == R4ObservabilityRole.UNKNOWN)
    et_i = sum(
        1 for r in EVENT_TRACE_CONTINUITY_MATRIX if r.verdict == R4ObservabilityRole.ILLEGAL_AUTHORITY
    )

    bg_p, bg_m, bg_i = _verdict_counts(
        BACKGROUND_EXECUTION_IDENTITY_MATRIX,
        proven={R4ChainVerdict.PROVEN_CURRENT_HEAD, R4CrossTenantClass.CROSS_TENANT_REJECTED},
        missing={R4ChainVerdict.MISSING},
        invalid={R4ChainVerdict.INVALID},
    )
    qw_p, qw_m, qw_i = _verdict_counts(
        QUEUE_WORKER_CONTINUITY_MATRIX,
        proven={R4ChainVerdict.PROVEN_CURRENT_HEAD},
        missing={R4ChainVerdict.MISSING},
        invalid={R4ChainVerdict.INVALID},
    )
    ad_p, ad_m, ad_i = _verdict_counts(
        ASYNC_DOMAIN_WORKER_MATRIX,
        proven={R4ChainVerdict.PROVEN_CURRENT_HEAD, R4CrossTenantClass.CROSS_TENANT_REJECTED},
        missing={R4ChainVerdict.MISSING},
        invalid={R4ChainVerdict.INVALID},
    )

    ct_rows = CROSS_TENANT_R4_MATRIX
    ct_same = sum(1 for r in ct_rows if r.verdict == R4CrossTenantClass.SAME_TENANT_PROVEN)
    ct_rej = sum(1 for r in ct_rows if r.verdict == R4CrossTenantClass.CROSS_TENANT_REJECTED)
    ct_allow = sum(1 for r in ct_rows if r.verdict == R4CrossTenantClass.CROSS_TENANT_ALLOWED)
    ct_unk = sum(1 for r in ct_rows if r.verdict == R4CrossTenantClass.UNKNOWN)

    obs_rows = [r for r in EVENT_TRACE_CONTINUITY_MATRIX if isinstance(r.verdict, R4ObservabilityRole)]
    obs_carrier = sum(1 for r in obs_rows if r.verdict == R4ObservabilityRole.CARRIER)
    obs_validator = sum(1 for r in obs_rows if r.verdict == R4ObservabilityRole.VALIDATOR)
    obs_derived = sum(1 for r in obs_rows if r.verdict == R4ObservabilityRole.DERIVED_OBSERVATION)
    obs_authority = sum(1 for r in obs_rows if r.verdict == R4ObservabilityRole.ILLEGAL_AUTHORITY)
    obs_unk = sum(1 for r in obs_rows if r.verdict == R4ObservabilityRole.UNKNOWN)

    ext_rows = EXTENSION_SCOPE_MATRIX
    ext_preserved = sum(1 for r in ext_rows if r.verdict == R4ExtensionTenantClass.PRESERVED)
    ext_rejected = sum(1 for r in ext_rows if r.verdict == R4ExtensionTenantClass.REJECTED)
    ext_allowed = sum(1 for r in ext_rows if r.verdict == R4ExtensionTenantClass.ALLOWED)
    ext_unk = sum(1 for r in ext_rows if r.verdict == R4ExtensionTenantClass.UNKNOWN)

    sub_rows = MISSING_DEFAULT_GLOBAL_MATRIX
    sub_illegal = sum(1 for r in sub_rows if r.verdict == R4SubstitutionClass.ILLEGAL_SUBSTITUTION_PRESENT)
    sub_rej = sum(1 for r in sub_rows if r.verdict == R4SubstitutionClass.ILLEGAL_INPUT_REJECTED)
    sub_legal = sum(1 for r in sub_rows if r.verdict == R4SubstitutionClass.LEGAL_NON_TENANT)
    sub_unk = sum(1 for r in sub_rows if r.verdict == R4SubstitutionClass.UNKNOWN)

    optional_row = MODEL_ATTRIBUTION_OPTIONAL_TENANT_MATRIX[0]
    optional_class = str(optional_row.verdict.value)

    gap_proven = _context_model_gap_proven()
    chain_missing = r4_chain_missing_count()
    chain_invalid = r4_chain_invalid_count()
    chain_proven = r4_chain_proven_count()
    cm_chain = context_model_chain_counts()
    qw_chain = queue_worker_chain_counts()
    ev_chain = evidence_event_chain_counts()

    ctx_asm_inv = build_record_context_assembly_inventory()
    llm_inv = build_record_llm_call_inventory()

    remediation: list[str] = []
    if ctx_asm_inv.unknown or ctx_asm_inv.caller_controlled_unsafe:
        remediation.append("TENANT-X-R4-R1 - record_context_assembly production caller reconciliation")
    if llm_inv.unknown or llm_inv.untrusted_explicit_tenant:
        remediation.append("TENANT-X-R4-R1 - record_llm_call_runtime_event tenant param reconciliation")
    if unknown_families:
        remediation.append("TENANT-X-R4-R1 - critical R4 family UNKNOWN reconciliation")
    if not gap_proven or cm_m or cm_i:
        remediation.append("TENANT-X-R4-R2 - context→model continuity proof gaps")
    if chain_missing or chain_invalid:
        remediation.append("TENANT-X-R4-R3 - E2E chain structural gaps")
    if ct_allow or ct_unk:
        remediation.append("TENANT-X-R4-R4 - cross-tenant adversarial closure")
    if obs_authority or obs_unk:
        remediation.append("TENANT-X-R4-R5 - observability tenant authority reconciliation")
    if ext_allowed or ext_unk:
        remediation.append("TENANT-X-R4-R6 - extension tenant rewrite closure")
    if sub_illegal or sub_unk:
        remediation.append("TENANT-X-R4-R7 - illegal default/global substitution closure")
    if qw_m or qw_i or ad_m or ad_i or bg_m or bg_i:
        remediation.append("TENANT-X-R4-R8 - queue/worker continuity gaps")
    if optional_row.verdict == R4AttributionTenantOptionalClass.TENANT_REQUIRED_BUT_OPTIONAL:
        remediation.append("TENANT-X-R4-R9 - ModelContextAttributionEvidence tenant optionality blocker")

    ready = (
        unknown_families == 0
        and gap_proven
        and cm_m == 0
        and cm_i == 0
        and cm_chain[1] == 0
        and cm_chain[2] == 0
        and qw_chain[1] == 0
        and qw_chain[2] == 0
        and ev_chain[1] == 0
        and ev_chain[2] == 0
        and ctx_asm_inv.unknown == 0
        and ctx_asm_inv.caller_controlled_unsafe == 0
        and llm_inv.unknown == 0
        and llm_inv.untrusted_explicit_tenant == 0
        and chain_missing == 0
        and chain_invalid == 0
        and ct_allow == 0
        and ct_unk == 0
        and obs_authority == 0
        and obs_unk == 0
        and ext_allowed == 0
        and ext_unk == 0
        and sub_illegal == 0
        and sub_unk == 0
        and qw_m == 0
        and qw_i == 0
        and ad_m == 0
        and ad_i == 0
        and bg_m == 0
        and bg_i == 0
        and ev_m == 0
        and ev_i == 0
        and optional_row.verdict != R4AttributionTenantOptionalClass.TENANT_REQUIRED_BUT_OPTIONAL
    )

    return TenantXR4QualificationReport(
        r4_families_total=len(families),
        tenant_bound_families=tenant_bound,
        non_tenant_families=non_tenant,
        critical_unknown_families=unknown_families,
        context_model_proven=cm_p,
        context_model_missing=cm_m,
        context_model_invalid=cm_i,
        context_view_proven=cv_p,
        context_view_missing=cv_m,
        context_view_invalid=cv_i,
        evidence_proven=ev_p,
        evidence_missing=ev_m,
        evidence_invalid=ev_i,
        event_trace_proven=et_p,
        event_trace_missing=et_m,
        event_trace_invalid=et_i,
        background_identity_proven=bg_p,
        background_identity_missing=bg_m,
        background_identity_invalid=bg_i,
        queue_worker_proven=qw_p,
        queue_worker_missing=qw_m,
        queue_worker_invalid=qw_i,
        async_worker_proven=ad_p,
        async_worker_missing=ad_m,
        async_worker_invalid=ad_i,
        cross_tenant_same_tenant_proven=ct_same,
        cross_tenant_rejected=ct_rej,
        cross_tenant_allowed=ct_allow,
        cross_tenant_unknown=ct_unk,
        observability_carrier=obs_carrier,
        observability_validator=obs_validator,
        observability_derived=obs_derived,
        observability_authority=obs_authority,
        observability_unknown=obs_unk,
        extension_preserved=ext_preserved,
        extension_rejected=ext_rejected,
        extension_allowed=ext_allowed,
        extension_unknown=ext_unk,
        substitution_illegal_present=sub_illegal,
        substitution_illegal_input_rejected=sub_rej,
        substitution_legal_non_tenant=sub_legal,
        substitution_unknown=sub_unk,
        context_model_gap_proven=gap_proven,
        record_context_assembly_callers_total=ctx_asm_inv.production_callers_total,
        record_context_assembly_canonically_bound=ctx_asm_inv.canonically_bound,
        record_context_assembly_unsafe=ctx_asm_inv.caller_controlled_unsafe,
        record_context_assembly_unknown_callers=ctx_asm_inv.unknown,
        record_llm_call_callers_total=llm_inv.production_callers_total,
        record_llm_call_canonical_tenant=llm_inv.canonical_tenant,
        record_llm_call_validated_explicit=llm_inv.validated_explicit_tenant,
        record_llm_call_untrusted_explicit=llm_inv.untrusted_explicit_tenant,
        record_llm_call_unknown=llm_inv.unknown,
        e2e_chain_proven=chain_proven,
        e2e_chain_missing=chain_missing,
        e2e_chain_invalid=chain_invalid,
        attribution_optional_tenant_class=optional_class,
        b2b_nordea_verdict="READY WITH PLATFORM GAPS",
        r4_ready_for_audit=ready,
        remediation_children=tuple(remediation),
    )


def summarize_r4_metrics(report: TenantXR4QualificationReport) -> dict[str, int | str | bool | tuple[str, ...]]:
    cm_chain = context_model_chain_counts()
    qw_chain = queue_worker_chain_counts()
    ev_chain = evidence_event_chain_counts()
    return {
        "r4_families_total": report.r4_families_total,
        "tenant_bound_families": report.tenant_bound_families,
        "non_tenant_families": report.non_tenant_families,
        "critical_unknown_families": report.critical_unknown_families,
        "context_model_proven": report.context_model_proven,
        "context_model_missing": report.context_model_missing,
        "context_model_invalid": report.context_model_invalid,
        "context_view_proven": report.context_view_proven,
        "context_view_missing": report.context_view_missing,
        "context_view_invalid": report.context_view_invalid,
        "evidence_proven": report.evidence_proven,
        "evidence_missing": report.evidence_missing,
        "evidence_invalid": report.evidence_invalid,
        "event_trace_proven": report.event_trace_proven,
        "event_trace_missing": report.event_trace_missing,
        "event_trace_invalid": report.event_trace_invalid,
        "background_identity_proven": report.background_identity_proven,
        "background_identity_missing": report.background_identity_missing,
        "background_identity_invalid": report.background_identity_invalid,
        "queue_worker_proven": report.queue_worker_proven,
        "queue_worker_missing": report.queue_worker_missing,
        "queue_worker_invalid": report.queue_worker_invalid,
        "async_worker_proven": report.async_worker_proven,
        "async_worker_missing": report.async_worker_missing,
        "async_worker_invalid": report.async_worker_invalid,
        "cross_tenant_same_tenant_proven": report.cross_tenant_same_tenant_proven,
        "cross_tenant_rejected": report.cross_tenant_rejected,
        "cross_tenant_allowed": report.cross_tenant_allowed,
        "cross_tenant_unknown": report.cross_tenant_unknown,
        "observability_carrier": report.observability_carrier,
        "observability_validator": report.observability_validator,
        "observability_derived": report.observability_derived,
        "observability_authority": report.observability_authority,
        "observability_unknown": report.observability_unknown,
        "extension_preserved": report.extension_preserved,
        "extension_rejected": report.extension_rejected,
        "extension_allowed": report.extension_allowed,
        "extension_unknown": report.extension_unknown,
        "substitution_illegal_present": report.substitution_illegal_present,
        "substitution_illegal_input_rejected": report.substitution_illegal_input_rejected,
        "substitution_legal_non_tenant": report.substitution_legal_non_tenant,
        "substitution_unknown": report.substitution_unknown,
        "context_model_gap_proven": report.context_model_gap_proven,
        "record_context_assembly_callers_total": report.record_context_assembly_callers_total,
        "record_context_assembly_canonically_bound": report.record_context_assembly_canonically_bound,
        "record_context_assembly_unsafe": report.record_context_assembly_unsafe,
        "record_context_assembly_unknown_callers": report.record_context_assembly_unknown_callers,
        "record_llm_call_callers_total": report.record_llm_call_callers_total,
        "record_llm_call_canonical_tenant": report.record_llm_call_canonical_tenant,
        "record_llm_call_validated_explicit": report.record_llm_call_validated_explicit,
        "record_llm_call_untrusted_explicit": report.record_llm_call_untrusted_explicit,
        "record_llm_call_unknown": report.record_llm_call_unknown,
        "context_model_chain_proven": cm_chain[0],
        "context_model_chain_missing": cm_chain[1],
        "context_model_chain_invalid": cm_chain[2],
        "queue_worker_chain_proven": qw_chain[0],
        "queue_worker_chain_missing": qw_chain[1],
        "queue_worker_chain_invalid": qw_chain[2],
        "evidence_event_chain_proven": ev_chain[0],
        "evidence_event_chain_missing": ev_chain[1],
        "evidence_event_chain_invalid": ev_chain[2],
        "e2e_chain_proven": report.e2e_chain_proven,
        "e2e_chain_missing": report.e2e_chain_missing,
        "e2e_chain_invalid": report.e2e_chain_invalid,
        "e2e_chain_transitions": len(build_r4_e2e_chain()),
        "b2b_nordea": report.b2b_nordea_verdict,
        "remediation_children": report.remediation_children,
    }
