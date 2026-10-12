# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 authority mechanism reconciliation (22 P0 mechanisms → semantic scope + owner)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from tests.qualification.tenant_x._authority_semantics import (
    TenantAuthorityMechanism,
    build_authority_mechanisms,
    classify_authority_duplicate_mechanisms,
    is_positive_mint_rewrite,
)
from tests.qualification.tenant_x._inventory import get_tenant_x_inventory
from tests.qualification.tenant_x._r1_root_selectors import (
    build_r1_root_selector_inventory,
    sanctioned_root_selectors_from_inventory,
)
from tests.qualification.tenant_x._r1_types import (
    R1AuthorityDuplicateParticipation,
    R1AuthorityMechanismResolution,
    R1MechanismAuthorityClass,
)
from tests.qualification.tenant_x._types import (
    TenantAuthorityOwnerDisposition,
    TenantAuthorityScopeEvidence,
    TenantAuthoritySemanticRole,
    TenantSurfaceRecord,
)


def _p0_mint_surfaces(inventory: tuple[TenantSurfaceRecord, ...]) -> tuple[TenantSurfaceRecord, ...]:
    out: list[TenantSurfaceRecord] = []
    for record in inventory:
        role = record.authority_semantic_role
        if role == TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY and is_positive_mint_rewrite(
            role, record
        ):
            out.append(record)
    return tuple(out)


def _resolution_for(mechanism: TenantAuthorityMechanism) -> R1AuthorityMechanismResolution:
    mid = mechanism.mechanism_id
    decision = mechanism.decision_kind.value

    if "agents/external_contractor_adapter" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="TRACEABILITY",
            authority_scope="agent_adapter_observability_default",
            scope_evidence=TenantAuthorityScopeEvidence.STRUCTURAL_CURRENT_HEAD,
            owner_disposition=TenantAuthorityOwnerDisposition.OWNER_CONFIRMED,
            authority_class=R1MechanismAuthorityClass.OBSERVABILITY_NON_TENANT,
            canonical_contract="agents/external_contractor_adapter/external_work_adapter.py",
            notes="Default substitution for adapter telemetry — not production tenant admission",
            owner_evidence=("production_composition_boundary: agents tier-2 adapter telemetry only",),
            scope_evidence_refs=(
                "structural: default label on observability path — no admission authority",
            ),
            production_classification_evidence=(
                "not reachable as canonical root tenant admission — observability-only contract",
            ),
        )
    if mid.startswith("APPLICATION_BOUNDARY|"):
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="APPLICATION_BOUNDARY",
            authority_scope="application_demo_ingress_tenant",
            scope_evidence=TenantAuthorityScopeEvidence.EXPLICIT_CONTRACT,
            owner_disposition=TenantAuthorityOwnerDisposition.OWNER_CONFIRMED,
            authority_class=R1MechanismAuthorityClass.APPLICATION_LOCAL_INGRESS,
            canonical_contract="applications/*/host/* (Tier-3 ingress only)",
            notes="Application-local demo/offline literals — cannot override platform root admission",
            owner_evidence=("production_composition_boundary: Tier-3 application ingress",),
            scope_evidence_refs=("explicit: application host wiring bounded outside platform root",),
            production_classification_evidence=(
                "test/conformance-only and Tier-3 paths — not platform composition root",
            ),
        )
    if "decision_finalization_conformance.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="EXECUTION",
            authority_scope="conformance_harness_tenant_fixture",
            scope_evidence=TenantAuthorityScopeEvidence.EXPLICIT_CONTRACT,
            owner_disposition=TenantAuthorityOwnerDisposition.OWNER_CONFIRMED,
            authority_class=R1MechanismAuthorityClass.CONFORMANCE_OR_QUALIFICATION_ONLY,
            canonical_contract="intergrax/runtime/execution/decision_finalization_conformance.py",
            notes="Conformance factory defaults — isolated tenant-a/b fixtures, not runtime admission",
            owner_evidence=("qualification_historical: conformance module entrypoint only",),
            scope_evidence_refs=("explicit: conformance factory tenant-a/b fixtures",),
            production_classification_evidence=("conformance-only entrypoint — not production root",),
        )
    if "environment_wiring.py" in mid or "agent_platform_admin_routes.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="APPLICATION_BOUNDARY",
            authority_scope="application_environment_binding_tenant",
            scope_evidence=TenantAuthorityScopeEvidence.EXPLICIT_CONTRACT,
            owner_disposition=TenantAuthorityOwnerDisposition.OWNER_CONFIRMED,
            authority_class=R1MechanismAuthorityClass.APPLICATION_LOCAL_INGRESS,
            canonical_contract="intergrax/applications/_shared/*",
            notes="Harness/application environment wiring — bounded Tier-3 ingress",
            owner_evidence=("production_composition_boundary: intergrax/applications/_shared",),
            scope_evidence_refs=("explicit: environment binding for application harness",),
            production_classification_evidence=("application Tier-3 boundary outside platform root",),
        )
    if "intergrax/cli/external_work.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="EXECUTION",
            authority_scope="cli_operator_attestation_context",
            scope_evidence=TenantAuthorityScopeEvidence.EXPLICIT_CONTRACT,
            owner_disposition=TenantAuthorityOwnerDisposition.OWNER_CONFIRMED,
            authority_class=R1MechanismAuthorityClass.APPLICATION_LOCAL_INGRESS,
            canonical_contract="intergrax/cli/external_work.py",
            notes="CLI attestation retry context — operator tooling, not competing root selector",
            owner_evidence=("production_composition_boundary: CLI operator tooling",),
            scope_evidence_refs=("explicit: attestation retry context — subordinate to admitted execution",),
            production_classification_evidence=("operator CLI — not root admission port",),
        )
    if "agent_capability_intake.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="EXECUTION",
            authority_scope="task_envelope_derivation_from_task_context",
            scope_evidence=TenantAuthorityScopeEvidence.STRUCTURAL_CURRENT_HEAD,
            owner_disposition=TenantAuthorityOwnerDisposition.OWNER_CONFIRMED,
            authority_class=R1MechanismAuthorityClass.SANCTIONED_SEMANTIC_AUTHORITY,
            canonical_contract="intergrax/runtime/task/agent_capability_intake.py",
            notes="Derives task envelope tenant from admitted task context — subordinate to root admission",
            owner_evidence=("canonical_contract: intergrax/runtime/task/agent_capability_intake.py",),
            scope_evidence_refs=(
                "structural: envelope tenant derived from admitted task context — non-root derivation",
            ),
            production_classification_evidence=("NON_ROOT_DERIVATION — subordinate to root admission",),
        )
    if "integrations/_shared/p8/factories.py" in mid or "lancedb/opens.py" in mid:
        contract = (
            "intergrax/integrations/_shared/p8/factories.py::create_lancedb_vector_store"
            if "factories.py" in mid
            else "intergrax/integrations/providers/vector_store/lancedb/opens.py::_open_rag_store"
        )
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="INTEGRATIONS",
            authority_scope="vector_store_provider_local_scope_pin",
            scope_evidence=TenantAuthorityScopeEvidence.STRUCTURAL_CURRENT_HEAD,
            owner_disposition=TenantAuthorityOwnerDisposition.OWNER_CANDIDATE,
            authority_class=R1MechanismAuthorityClass.INTEGRATION_DEMO_OR_DEV_SEAM,
            canonical_contract=contract,
            notes="VectorStore client pins scope.tenant_id on first write — not platform root admission",
            owner_evidence=("structural: integration provider construction seam — INTEGRATIONS tier",),
            scope_evidence_refs=(
                "structural: _tenant_id assigned from VectorStoreScope.tenant_id on first add_records only",
                "structural: validate_scope/validate_records — cannot rewrite scope tenant independently",
            ),
            production_classification_evidence=(
                "not reachable from canonical production admission — provider-local vector scope only",
                "cannot select root tenant — no RootExecutionAuthorityAdmission port",
                "cannot override admitted tenant — validate_scope enforces scope tenant",
                "cannot create execution authority — VectorStore contract only",
                "cannot bypass Integration Catalog — factory returns catalog-wrapped RestVectorStoreIntegration",
            ),
        )
    if "allow_all.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="INTEGRATIONS",
            authority_scope="dev_auth_provider_compatibility",
            scope_evidence=TenantAuthorityScopeEvidence.STRUCTURAL_CURRENT_HEAD,
            owner_disposition=TenantAuthorityOwnerDisposition.OWNER_CONFIRMED,
            authority_class=R1MechanismAuthorityClass.INTEGRATION_DEMO_OR_DEV_SEAM,
            canonical_contract="intergrax/fastapi_core/auth/providers/allow_all.py",
            notes="Development auth provider — not production tenant mint authority",
            owner_evidence=("explicit dev-only provider registration: allow_all auth provider",),
            scope_evidence_refs=("structural: dev auth provider — no tenant mint",),
            production_classification_evidence=("explicit dev-only provider — not production admission",),
        )
    if "observability_bridge.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="TRACEABILITY",
            authority_scope="observability_tracking_default_label",
            scope_evidence=TenantAuthorityScopeEvidence.STRUCTURAL_CURRENT_HEAD,
            owner_disposition=TenantAuthorityOwnerDisposition.OWNER_CONFIRMED,
            authority_class=R1MechanismAuthorityClass.OBSERVABILITY_NON_TENANT,
            canonical_contract="**/tracking/observability_bridge.py",
            notes="Tracking label default — LEGAL_NON_TENANT on critical admission paths",
            owner_evidence=("production_composition_boundary: observability tracking only",),
            scope_evidence_refs=("structural: tracking default label — no execution authority",),
            production_classification_evidence=("observability-only — no execution/provider/state authority",),
        )
    if "rag/evaluation/load_soak.py" in mid or "rag/bootstrap/entry_point_load.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="PLATFORM_CORE",
            authority_scope="rag_evaluation_harness_tenant",
            scope_evidence=TenantAuthorityScopeEvidence.EXPLICIT_CONTRACT,
            owner_disposition=TenantAuthorityOwnerDisposition.OWNER_CONFIRMED,
            authority_class=R1MechanismAuthorityClass.EVALUATION_OR_LOAD_HARNESS,
            canonical_contract="intergrax/rag/evaluation/*",
            notes="RAG load/evaluation harness literals — not cross-tenant production authority",
            owner_evidence=("qualification_historical: RAG evaluation harness entrypoints",),
            scope_evidence_refs=("explicit: load/evaluation harness literals",),
            production_classification_evidence=("evaluation harness — not production composition root",),
        )
    if "l4_runtime_evidence.py" in mid or "maturity_gate_evidence.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="TRACEABILITY",
            authority_scope="qualification_evidence_harness_tenant",
            scope_evidence=TenantAuthorityScopeEvidence.EXPLICIT_CONTRACT,
            owner_disposition=TenantAuthorityOwnerDisposition.OWNER_CONFIRMED,
            authority_class=R1MechanismAuthorityClass.CONFORMANCE_OR_QUALIFICATION_ONLY,
            canonical_contract="intergrax/runtime/**/**_evidence.py",
            notes="Harness baseline evidence builders — qualification-only tenant literals",
            owner_evidence=("qualification_historical: maturity/l4 evidence builders",),
            scope_evidence_refs=("explicit: qualification evidence harness tenants",),
            production_classification_evidence=("qualification-only builders — not runtime admission",),
        )
    return R1AuthorityMechanismResolution(
        mechanism_id=mid,
        decision_kind=decision,
        semantic_owner=mechanism.semantic_owner,
        authority_scope=f"unresolved:{mechanism.authority_scope}",
        scope_evidence=TenantAuthorityScopeEvidence.UNKNOWN,
        owner_disposition=TenantAuthorityOwnerDisposition.OWNER_UNRESOLVED,
        authority_class=R1MechanismAuthorityClass.SANCTIONED_SEMANTIC_AUTHORITY,
        canonical_contract="UNKNOWN",
        notes="No R1 overlay — BLOCKER",
        owner_evidence=(),
        scope_evidence_refs=("unresolved: no bounded R1 evidence attached",),
        production_classification_evidence=(),
    )


def build_r1_authority_resolutions(
    inventory: tuple[TenantSurfaceRecord, ...] | None = None,
) -> tuple[R1AuthorityMechanismResolution, ...]:
    inv = inventory if inventory is not None else get_tenant_x_inventory()
    mechanisms = build_authority_mechanisms(_p0_mint_surfaces(inv))
    return tuple(_resolution_for(m) for m in mechanisms)


def apply_r1_mechanism_overlay(
    mechanism: TenantAuthorityMechanism,
    resolution: R1AuthorityMechanismResolution,
) -> TenantAuthorityMechanism:
    return replace(
        mechanism,
        semantic_owner=resolution.semantic_owner,
        authority_scope=resolution.authority_scope,
        authority_scope_evidence=resolution.scope_evidence,
        owner_disposition=resolution.owner_disposition,
    )


def build_r1_overlay_mechanisms(
    inventory: tuple[TenantSurfaceRecord, ...] | None = None,
) -> tuple[TenantAuthorityMechanism, ...]:
    inv = inventory if inventory is not None else get_tenant_x_inventory()
    base = build_authority_mechanisms(_p0_mint_surfaces(inv))
    resolutions = {r.mechanism_id: r for r in build_r1_authority_resolutions(inv)}
    return tuple(
        apply_r1_mechanism_overlay(m, resolutions[m.mechanism_id]) for m in base if m.mechanism_id in resolutions
    )


def build_sanctioned_root_tenant_selectors(
    inventory: tuple[TenantSurfaceRecord, ...] | None = None,
) -> tuple[str, ...]:
    inv = inventory if inventory is not None else get_tenant_x_inventory()
    _ = build_r1_root_selector_inventory(inv)
    return sanctioned_root_selectors_from_inventory(build_r1_root_selector_inventory(inv))


SANCTIONED_ROOT_TENANT_SELECTORS: Final[tuple[str, ...]] = build_sanctioned_root_tenant_selectors()

_SUBORDINATE_DERIVATION_MARKERS: Final[tuple[str, ...]] = (
    "NON_ROOT_DERIVATION",
    "subordinate to root admission",
    "subordinate to admitted",
)


def r1_authority_duplicate_participation(
    resolution: R1AuthorityMechanismResolution,
) -> R1AuthorityDuplicateParticipation:
    cls = resolution.authority_class
    if cls == R1MechanismAuthorityClass.SANCTIONED_SEMANTIC_AUTHORITY:
        evidence = " ".join(resolution.production_classification_evidence).lower()
        if any(marker.lower() in evidence for marker in _SUBORDINATE_DERIVATION_MARKERS):
            return R1AuthorityDuplicateParticipation.SUBORDINATE_DERIVATION
        if not resolution.production_classification_evidence:
            return R1AuthorityDuplicateParticipation.UNKNOWN
        return R1AuthorityDuplicateParticipation.CRITICAL_AUTHORITY
    if cls == R1MechanismAuthorityClass.APPLICATION_LOCAL_INGRESS:
        return R1AuthorityDuplicateParticipation.APPLICATION_LOCAL
    if cls in {
        R1MechanismAuthorityClass.CONFORMANCE_OR_QUALIFICATION_ONLY,
        R1MechanismAuthorityClass.EVALUATION_OR_LOAD_HARNESS,
    }:
        return R1AuthorityDuplicateParticipation.HARNESS_ONLY
    if cls == R1MechanismAuthorityClass.OBSERVABILITY_NON_TENANT:
        return R1AuthorityDuplicateParticipation.OBSERVABILITY_NON_AUTHORITY
    if cls == R1MechanismAuthorityClass.INTEGRATION_DEMO_OR_DEV_SEAM:
        if resolution.production_classification_evidence:
            return R1AuthorityDuplicateParticipation.DEV_COMPATIBILITY_SEAM
        return R1AuthorityDuplicateParticipation.UNKNOWN
    return R1AuthorityDuplicateParticipation.UNKNOWN


@dataclass(frozen=True, slots=True)
class R1ScopedDuplicateAuthorityReview:
    authority_mechanisms_total: int
    critical_authority_participant_ids: tuple[str, ...]
    non_authority_participant_ids: tuple[str, ...]
    unknown_participation_ids: tuple[str, ...]
    critical_duplicate_confirmed_ids: tuple[str, ...]
    critical_duplicate_unresolved_ids: tuple[str, ...]
    non_critical_owner_scope_uncertainty_ids: tuple[str, ...]


def _non_critical_owner_scope_uncertainty(
    resolutions: tuple[R1AuthorityMechanismResolution, ...],
    participation: dict[str, R1AuthorityDuplicateParticipation],
) -> tuple[str, ...]:
    out: list[str] = []
    for resolution in resolutions:
        part = participation[resolution.mechanism_id]
        if part in {
            R1AuthorityDuplicateParticipation.CRITICAL_AUTHORITY,
            R1AuthorityDuplicateParticipation.UNKNOWN,
        }:
            continue
        uncertain_owner = resolution.owner_disposition in {
            TenantAuthorityOwnerDisposition.OWNER_CANDIDATE,
            TenantAuthorityOwnerDisposition.OWNER_UNRESOLVED,
        }
        uncertain_scope = resolution.scope_evidence in {
            TenantAuthorityScopeEvidence.DOMAIN_FALLBACK,
            TenantAuthorityScopeEvidence.UNKNOWN,
        }
        if uncertain_owner or uncertain_scope:
            out.append(resolution.mechanism_id)
    return tuple(sorted(out))


def build_r1_scoped_duplicate_authority_review(
    inventory: tuple[TenantSurfaceRecord, ...] | None = None,
) -> R1ScopedDuplicateAuthorityReview:
    inv = inventory if inventory is not None else get_tenant_x_inventory()
    resolutions = build_r1_authority_resolutions(inv)
    mechanisms = build_r1_overlay_mechanisms(inv)
    participation = {r.mechanism_id: r1_authority_duplicate_participation(r) for r in resolutions}

    critical_ids: list[str] = []
    non_authority_ids: list[str] = []
    unknown_ids: list[str] = []
    for resolution in resolutions:
        part = participation[resolution.mechanism_id]
        if part == R1AuthorityDuplicateParticipation.CRITICAL_AUTHORITY:
            critical_ids.append(resolution.mechanism_id)
        elif part == R1AuthorityDuplicateParticipation.UNKNOWN:
            unknown_ids.append(resolution.mechanism_id)
        else:
            non_authority_ids.append(resolution.mechanism_id)

    critical_mechanisms = tuple(m for m in mechanisms if m.mechanism_id in critical_ids)
    duplicate_review = classify_authority_duplicate_mechanisms(critical_mechanisms)

    return R1ScopedDuplicateAuthorityReview(
        authority_mechanisms_total=len(resolutions),
        critical_authority_participant_ids=tuple(sorted(critical_ids)),
        non_authority_participant_ids=tuple(sorted(non_authority_ids)),
        unknown_participation_ids=tuple(sorted(unknown_ids)),
        critical_duplicate_confirmed_ids=duplicate_review.confirmed_duplicate_mechanism_ids,
        critical_duplicate_unresolved_ids=duplicate_review.unresolved_duplicate_mechanism_ids,
        non_critical_owner_scope_uncertainty_ids=_non_critical_owner_scope_uncertainty(
            resolutions, participation
        ),
    )


def r1_duplicate_authority_unresolved(
    mechanisms: tuple[TenantAuthorityMechanism, ...],
    inventory: tuple[TenantSurfaceRecord, ...] | None = None,
) -> tuple[str, ...]:
    _ = mechanisms
    review = build_r1_scoped_duplicate_authority_review(inventory)
    return review.critical_duplicate_unresolved_ids


def r1_duplicate_authority_confirmed(
    mechanisms: tuple[TenantAuthorityMechanism, ...],
    inventory: tuple[TenantSurfaceRecord, ...] | None = None,
) -> tuple[str, ...]:
    _ = mechanisms
    review = build_r1_scoped_duplicate_authority_review(inventory)
    return review.critical_duplicate_confirmed_ids


def r1_critical_scope_unresolved(
    resolutions: tuple[R1AuthorityMechanismResolution, ...],
) -> tuple[str, ...]:
    return tuple(
        r.mechanism_id
        for r in resolutions
        if r.scope_evidence
        in {TenantAuthorityScopeEvidence.DOMAIN_FALLBACK, TenantAuthorityScopeEvidence.UNKNOWN}
        and r.authority_class == R1MechanismAuthorityClass.SANCTIONED_SEMANTIC_AUTHORITY
    )


def r1_critical_owner_unresolved(
    resolutions: tuple[R1AuthorityMechanismResolution, ...],
) -> tuple[str, ...]:
    return tuple(
        r.mechanism_id
        for r in resolutions
        if r.owner_disposition
        in {
            TenantAuthorityOwnerDisposition.OWNER_UNRESOLVED,
            TenantAuthorityOwnerDisposition.OWNER_CANDIDATE,
        }
        and r.authority_class == R1MechanismAuthorityClass.SANCTIONED_SEMANTIC_AUTHORITY
    )
