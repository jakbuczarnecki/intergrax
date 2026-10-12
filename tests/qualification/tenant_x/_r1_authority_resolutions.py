# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 authority mechanism reconciliation (22 P0 mechanisms → semantic scope + owner)."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from tests.qualification.tenant_x._authority_semantics import (
    TenantAuthorityMechanism,
    build_authority_mechanisms,
    classify_authority_duplicate_mechanisms,
    is_positive_mint_rewrite,
)
from tests.qualification.tenant_x._inventory import get_tenant_x_inventory
from tests.qualification.tenant_x._r1_types import (
    R1AuthorityMechanismResolution,
    R1MechanismAuthorityClass,
    SANCTIONED_ROOT_TENANT_CONTRACT,
)
from tests.qualification.tenant_x._types import (
    TenantAuthorityOwnerDisposition,
    TenantAuthorityScopeEvidence,
    TenantAuthoritySemanticRole,
    TenantSurfaceRecord,
)

_SCOPE_EVIDENCE = TenantAuthorityScopeEvidence.EXPLICIT_QUALIFICATION_EVIDENCE
_OWNER = TenantAuthorityOwnerDisposition.OWNER_CONFIRMED


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
            semantic_owner="AGENTS_RUNTIME",
            authority_scope="agent_adapter_observability_default",
            scope_evidence=_SCOPE_EVIDENCE,
            owner_disposition=_OWNER,
            authority_class=R1MechanismAuthorityClass.OBSERVABILITY_NON_TENANT,
            canonical_contract="agents/external_contractor_adapter/external_work_adapter.py",
            notes="Default substitution for adapter telemetry — not production tenant admission",
        )
    if mid.startswith("APPLICATION_BOUNDARY|"):
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="APPLICATION_BOUNDARY",
            authority_scope="application_demo_ingress_tenant",
            scope_evidence=_SCOPE_EVIDENCE,
            owner_disposition=_OWNER,
            authority_class=R1MechanismAuthorityClass.APPLICATION_LOCAL_INGRESS,
            canonical_contract="applications/*/host/* (Tier-3 ingress only)",
            notes="Application-local demo/offline literals — cannot override platform root admission",
        )
    if "decision_finalization_conformance.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="EXECUTION",
            authority_scope="conformance_harness_tenant_fixture",
            scope_evidence=_SCOPE_EVIDENCE,
            owner_disposition=_OWNER,
            authority_class=R1MechanismAuthorityClass.CONFORMANCE_OR_QUALIFICATION_ONLY,
            canonical_contract="intergrax/runtime/execution/decision_finalization_conformance.py",
            notes="Conformance factory defaults — isolated tenant-a/b fixtures, not runtime admission",
        )
    if "environment_wiring.py" in mid or "agent_platform_admin_routes.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="APPLICATION_BOUNDARY",
            authority_scope="application_environment_binding_tenant",
            scope_evidence=_SCOPE_EVIDENCE,
            owner_disposition=_OWNER,
            authority_class=R1MechanismAuthorityClass.APPLICATION_LOCAL_INGRESS,
            canonical_contract="intergrax/applications/_shared/*",
            notes="Harness/application environment wiring — bounded Tier-3 ingress",
        )
    if "intergrax/cli/external_work.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="EXECUTION",
            authority_scope="cli_operator_attestation_context",
            scope_evidence=_SCOPE_EVIDENCE,
            owner_disposition=_OWNER,
            authority_class=R1MechanismAuthorityClass.APPLICATION_LOCAL_INGRESS,
            canonical_contract="intergrax/cli/external_work.py",
            notes="CLI attestation retry context — operator tooling, not competing root selector",
        )
    if "agent_capability_intake.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="EXECUTION",
            authority_scope="task_envelope_derivation_from_task_context",
            scope_evidence=TenantAuthorityScopeEvidence.STRUCTURAL_CURRENT_HEAD,
            owner_disposition=_OWNER,
            authority_class=R1MechanismAuthorityClass.SANCTIONED_SEMANTIC_AUTHORITY,
            canonical_contract="intergrax/runtime/task/agent_capability_intake.py",
            notes="Derives task envelope tenant from admitted task context — subordinate to root admission",
        )
    if "integrations/_shared/p8/factories.py" in mid or "lancedb/opens.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="INTEGRATIONS",
            authority_scope="integration_provider_factory_scope",
            scope_evidence=_SCOPE_EVIDENCE,
            owner_disposition=_OWNER,
            authority_class=R1MechanismAuthorityClass.INTEGRATION_DEMO_OR_DEV_SEAM,
            canonical_contract="intergrax/integrations/**",
            notes="Provider factory scope widening — integration-local; catalog remains canonical",
        )
    if "allow_all.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="INTEGRATIONS",
            authority_scope="dev_auth_provider_compatibility",
            scope_evidence=_SCOPE_EVIDENCE,
            owner_disposition=_OWNER,
            authority_class=R1MechanismAuthorityClass.INTEGRATION_DEMO_OR_DEV_SEAM,
            canonical_contract="intergrax/fastapi_core/auth/providers/allow_all.py",
            notes="Development auth provider — not production tenant mint authority",
        )
    if "observability_bridge.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="TRACEABILITY",
            authority_scope="observability_tracking_default_label",
            scope_evidence=_SCOPE_EVIDENCE,
            owner_disposition=_OWNER,
            authority_class=R1MechanismAuthorityClass.OBSERVABILITY_NON_TENANT,
            canonical_contract="**/tracking/observability_bridge.py",
            notes="Tracking label default — LEGAL_NON_TENANT on critical admission paths",
        )
    if "rag/evaluation/load_soak.py" in mid or "rag/bootstrap/entry_point_load.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="PLATFORM_CORE",
            authority_scope="rag_evaluation_harness_tenant",
            scope_evidence=_SCOPE_EVIDENCE,
            owner_disposition=_OWNER,
            authority_class=R1MechanismAuthorityClass.EVALUATION_OR_LOAD_HARNESS,
            canonical_contract="intergrax/rag/evaluation/*",
            notes="RAG load/evaluation harness literals — not cross-tenant production authority",
        )
    if "l4_runtime_evidence.py" in mid or "maturity_gate_evidence.py" in mid:
        return R1AuthorityMechanismResolution(
            mechanism_id=mid,
            decision_kind=decision,
            semantic_owner="TRACEABILITY",
            authority_scope="qualification_evidence_harness_tenant",
            scope_evidence=_SCOPE_EVIDENCE,
            owner_disposition=_OWNER,
            authority_class=R1MechanismAuthorityClass.CONFORMANCE_OR_QUALIFICATION_ONLY,
            canonical_contract="intergrax/runtime/**/**_evidence.py",
            notes="Harness baseline evidence builders — qualification-only tenant literals",
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


SANCTIONED_ROOT_TENANT_SELECTORS: Final[tuple[str, ...]] = (SANCTIONED_ROOT_TENANT_CONTRACT,)


def r1_duplicate_authority_unresolved(mechanisms: tuple[TenantAuthorityMechanism, ...]) -> tuple[str, ...]:
    review = classify_authority_duplicate_mechanisms(mechanisms)
    return review.unresolved_duplicate_mechanism_ids


def r1_duplicate_authority_confirmed(mechanisms: tuple[TenantAuthorityMechanism, ...]) -> tuple[str, ...]:
    review = classify_authority_duplicate_mechanisms(mechanisms)
    return review.confirmed_duplicate_mechanism_ids
