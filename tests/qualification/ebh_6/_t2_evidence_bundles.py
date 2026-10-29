# © Artur Czarnecki. All rights reserved.

"""T2-AUD — mechanism bundle SSOT + independent implementer review (replaces false scanner_effective T2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Final

from tests.qualification.ebh_6._aj_class_gate_scan import iter_applicable_layer_class_pairs

_W5_SINK = (
    "tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py"
    "::test_production_composition_creates_bounded_sink"
)
_RPLG = (
    "tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py"
    "::test_ebh5_rpl_g_config_x_active_blockers_remain_empty"
)
_EE_A1 = (
    "tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py"
    "::test_ee_a1_p0_inventory_reports_zero_production_bypass"
)
_SANCTIONED = (
    "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py"
    "::test_ebh6_exit_sanctioned_composition_roots_non_empty"
)
_GOV_H1 = (
    "tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py"
    "::test_npsc42_h1_governance_planes_do_not_mint_execution_identity"
)
_SQLITE = "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_sqlite_annex_accountability"
_NPSC5E = (
    "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"
    "::test_no_recovery_god_object_in_production"
)
_LIFECYCLE = (
    "tests/integration/agent_distribution/test_canonical_agent_lifecycle_architecture_gate.py"
    "::test_stage15_proof_files_avoid_forbidden_lifecycle_bypass_fragments"
)
_CFGX = (
    "tests/qualification/ebh_6/test_ebh_6_config_x_head_reconciliation.py"
    "::test_ebh6_config_x_all_concerns_mapped_at_head"
)
_EBH5_INV = "tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py::test_ebh5_exit_inventory_complete"
_EBH5_REACH_FACTORY = (
    "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py"
    "::test_ebh5_reach_factory_module_is_sanctioned_composition_owner"
)
_EBH5_REACH_RAG = (
    "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py"
    "::test_ebh5_reach_no_second_global_provider_registry_in_rag_graph"
)
_REG_PROHIBITED = (
    "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero"
)
_VBN_MATRIX = "docs/project/maintainers/qualification/VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md"


class EvidenceBundleId(StrEnum):
    INTEGRATION_FACTORY = "integration_factory"
    SANCTIONED_COMPOSITION = "sanctioned_composition"
    EXECUTION_BYPASS = "execution_bypass"
    NEXUS_REACHABILITY = "nexus_reachability"
    GOVERNANCE_BOUNDARY = "governance_boundary"
    PERSISTENCE_SQLITE = "persistence_sqlite"
    OBSERVABILITY_AUTHORITY = "observability_authority"
    RECOVERY_PLANE = "recovery_plane"
    AGENT_LIFECYCLE = "agent_lifecycle"
    RAG_BOOTSTRAP = "rag_bootstrap"
    CONFIG_HOSTING = "config_hosting"
    CONTRACTS_TYPING = "contracts_typing"
    PLATFORM_VENDOR_SCAN = "platform_vendor_scan"
    REPLACEABILITY_J = "replaceability_j"
    Q2D2_FINDING = "q2d2_finding"
    PLATFORM_FAIL_CLOSED = "platform_fail_closed"
    PLATFORM_HARDCODING = "platform_hardcoding"
    PLATFORM_TYPING_SCAN = "platform_typing_scan"
    L3_REACHABILITY = "l3_reachability"


class ImplementerReview(StrEnum):
    ACCEPTED = "ACCEPTED"
    INSUFFICIENT = "INSUFFICIENT"
    NA_WITH_EVIDENCE = "NA_WITH_EVIDENCE"


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    bundle_id: EvidenceBundleId
    invariant_ref: str
    primary_behavioral_gates: tuple[str, ...]
    implementer_review: ImplementerReview
    scope_limitation: str
    negative_test_ref: str | None = None


_BUNDLE_CATALOG: Final[dict[EvidenceBundleId, EvidenceBundle]] = {
    EvidenceBundleId.INTEGRATION_FACTORY: EvidenceBundle(
        EvidenceBundleId.INTEGRATION_FACTORY,
        "FRZ-CTR integration factory + EBH-5 reachability",
        (_EBH5_REACH_FACTORY, _EBH5_INV, _RPLG),
        ImplementerReview.ACCEPTED,
        "Integration factory + inventory — not live adapter construction audit.",
    ),
    EvidenceBundleId.SANCTIONED_COMPOSITION: EvidenceBundle(
        EvidenceBundleId.SANCTIONED_COMPOSITION,
        "TRACE-X sanctioned composition sites",
        (_SANCTIONED,),
        ImplementerReview.ACCEPTED,
        "Registry presence — not per-root wiring replay @ HEAD.",
    ),
    EvidenceBundleId.EXECUTION_BYPASS: EvidenceBundle(
        EvidenceBundleId.EXECUTION_BYPASS,
        "EE-A1 execution bypass inventory",
        (_EE_A1,),
        ImplementerReview.ACCEPTED,
        "EE-A1 certified surface — not whole-platform class-B census.",
    ),
    EvidenceBundleId.NEXUS_REACHABILITY: EvidenceBundle(
        EvidenceBundleId.NEXUS_REACHABILITY,
        "L3 nexus call chains + sanctioned roots",
        (_SANCTIONED,),
        ImplementerReview.ACCEPTED,
        "Mechanism-level L3 index — primary_gate nodeids @ HEAD.",
    ),
    EvidenceBundleId.GOVERNANCE_BOUNDARY: EvidenceBundle(
        EvidenceBundleId.GOVERNANCE_BOUNDARY,
        "NPSC42 governance non-execution identity",
        (_GOV_H1, _REG_PROHIBITED),
        ImplementerReview.ACCEPTED,
        "NPSC42 frozen subset — observability G uses non-authority scan elsewhere.",
    ),
    EvidenceBundleId.PERSISTENCE_SQLITE: EvidenceBundle(
        EvidenceBundleId.PERSISTENCE_SQLITE,
        "SQLite annex accountability",
        (_SQLITE, _REG_PROHIBITED),
        ImplementerReview.ACCEPTED,
        "Annex accountability — not all shadow-store class-E obligations.",
    ),
    EvidenceBundleId.OBSERVABILITY_AUTHORITY: EvidenceBundle(
        EvidenceBundleId.OBSERVABILITY_AUTHORITY,
        "W5-B2 bounded observability sink wiring",
        (_W5_SINK,),
        ImplementerReview.ACCEPTED,
        "Behavioral W5-B2 invoke — not global OBS qualification.",
        negative_test_ref=_REG_PROHIBITED,
    ),
    EvidenceBundleId.RECOVERY_PLANE: EvidenceBundle(
        EvidenceBundleId.RECOVERY_PLANE,
        "NPSC5e recovery plane qualification",
        (_NPSC5E,),
        ImplementerReview.ACCEPTED,
        "Sample L3 recovery admission — not full reachability replay.",
    ),
    EvidenceBundleId.AGENT_LIFECYCLE: EvidenceBundle(
        EvidenceBundleId.AGENT_LIFECYCLE,
        "Agent distribution lifecycle + HARDENING-3 boundary",
        (_LIFECYCLE,),
        ImplementerReview.ACCEPTED,
        "Boundary/lifecycle subset — not all agents/backend literals.",
    ),
    EvidenceBundleId.RAG_BOOTSTRAP: EvidenceBundle(
        EvidenceBundleId.RAG_BOOTSTRAP,
        "T-RAG-01 graph bootstrap / no shadow registry",
        (_EBH5_REACH_RAG, _REG_PROHIBITED),
        ImplementerReview.ACCEPTED,
        "RAG graph bootstrap gates — not full memory plane qual.",
    ),
    EvidenceBundleId.CONFIG_HOSTING: EvidenceBundle(
        EvidenceBundleId.CONFIG_HOSTING,
        "CONFIG-X structural reconciliation @ HEAD",
        (_CFGX, _EBH5_INV),
        ImplementerReview.ACCEPTED,
        "Mapping completeness — not host profile materialization audit.",
    ),
    EvidenceBundleId.CONTRACTS_TYPING: EvidenceBundle(
        EvidenceBundleId.CONTRACTS_TYPING,
        "FRZ-TYP public contract seams",
        (
            "tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py"
            "::test_ebh6_contracts_layer_frz_typ_surface_purity_passes",
            "tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py"
            "::test_ebh6_application_services_production_seam_typing_passes",
            "tests/unit/architecture/test_ebh_2i_final_rescan_gate.py::test_ebh_2i_contract_surface_purity_gate_passes",
        ),
        ImplementerReview.ACCEPTED,
        "EBH-2I CSP on closed-world contracts surface + direct Tier-3 wiring _shared "
        "seam gate with EBH6-DEBT-APP-TYP-* registry (11 rows @ HEAD).",
    ),
    EvidenceBundleId.PLATFORM_VENDOR_SCAN: EvidenceBundle(
        EvidenceBundleId.PLATFORM_VENDOR_SCAN,
        "Register prohibited/unclassified + layer closed-world scan",
        (_REG_PROHIBITED, _EBH5_INV),
        ImplementerReview.ACCEPTED,
        "Inventory + static vendor scan — not live adapter audit.",
    ),
    EvidenceBundleId.REPLACEABILITY_J: EvidenceBundle(
        EvidenceBundleId.REPLACEABILITY_J,
        "RPL-G CONFIG-X blockers + VBN matrix",
        (_RPLG, _VBN_MATRIX),
        ImplementerReview.ACCEPTED,
        "RPL-G behavioral invoke + matrix cite — 51 NYQ rows remain deferred.",
    ),
    EvidenceBundleId.Q2D2_FINDING: EvidenceBundle(
        EvidenceBundleId.Q2D2_FINDING,
        "EBH6-DEBT-Q2-D2 / FRZ-TYP-03",
        ("tests/qualification/ebh_6/test_ebh_6_b3_exit_gates.py::test_ebh6_b3_q2d2_semantic_core_f_is_finding",),
        ImplementerReview.ACCEPTED,
        "FINDING only — OD-EBH6-Q2D2 operator OPEN.",
    ),
    EvidenceBundleId.PLATFORM_FAIL_CLOSED: EvidenceBundle(
        EvidenceBundleId.PLATFORM_FAIL_CLOSED,
        "Fail-closed materialization markers",
        (_EBH5_REACH_FACTORY, _REG_PROHIBITED),
        ImplementerReview.ACCEPTED,
        "Static fail-closed markers — not shadow config paths platform-wide.",
    ),
    EvidenceBundleId.PLATFORM_HARDCODING: EvidenceBundle(
        EvidenceBundleId.PLATFORM_HARDCODING,
        "Class-B literal scan + register",
        (_REG_PROHIBITED,),
        ImplementerReview.ACCEPTED,
        "Sample literal scan — not exhaustive backend census.",
    ),
    EvidenceBundleId.PLATFORM_TYPING_SCAN: EvidenceBundle(
        EvidenceBundleId.PLATFORM_TYPING_SCAN,
        "Protocol/TYPE_CHECKING seam scan",
        (_REG_PROHIBITED,),
        ImplementerReview.ACCEPTED,
        "Static typing seam markers — not FRZ-TYP behavioral replay.",
    ),
    EvidenceBundleId.L3_REACHABILITY: EvidenceBundle(
        EvidenceBundleId.L3_REACHABILITY,
        "L3 mechanism primary_gate nodeids per layer",
        (_SANCTIONED,),
        ImplementerReview.ACCEPTED,
        "Primary_gate pytest nodeids from `_l3_reachability_evidence` — not entrypoint filename heuristic.",
    ),
}


def resolve_bundle_id(layer: str, detection_class: str) -> EvidenceBundleId:
    key = (layer, detection_class)
    if key == ("semantic_core", "F"):
        return EvidenceBundleId.Q2D2_FINDING
    if detection_class == "J":
        return EvidenceBundleId.INTEGRATION_FACTORY if layer == "integrations" else EvidenceBundleId.REPLACEABILITY_J
    if layer == "integrations" and detection_class in "ABCHIJ":
        return EvidenceBundleId.INTEGRATION_FACTORY
    if layer == "composition_roots" and detection_class == "C":
        return EvidenceBundleId.INTEGRATION_FACTORY
    if layer == "composition_roots" and detection_class == "H":
        return EvidenceBundleId.SANCTIONED_COMPOSITION
    if layer == "execution" and detection_class in ("B", "H"):
        return EvidenceBundleId.EXECUTION_BYPASS
    if layer == "semantic_core" and detection_class == "B":
        return EvidenceBundleId.EXECUTION_BYPASS
    if layer == "semantic_core" and detection_class == "H":
        return EvidenceBundleId.NEXUS_REACHABILITY
    if layer == "governance" and detection_class in ("G", "H"):
        return EvidenceBundleId.GOVERNANCE_BOUNDARY
    if layer == "state_persistence" and detection_class in ("E", "H"):
        return EvidenceBundleId.PERSISTENCE_SQLITE
    if layer == "observability_diagnostics" and detection_class == "G":
        return EvidenceBundleId.OBSERVABILITY_AUTHORITY
    if layer == "reliability_recovery" and detection_class == "H":
        return EvidenceBundleId.RECOVERY_PLANE
    if layer == "agent_distribution" and detection_class in ("B", "H"):
        return EvidenceBundleId.AGENT_LIFECYCLE
    if layer == "rag_memory" and detection_class in ("A", "D", "H"):
        return EvidenceBundleId.RAG_BOOTSTRAP
    if layer == "runtime_hosting" and detection_class in ("C", "I"):
        return EvidenceBundleId.CONFIG_HOSTING
    if key in (("contracts", "F"), ("application_services", "F")):
        return EvidenceBundleId.CONTRACTS_TYPING
    if detection_class in ("A", "D"):
        return EvidenceBundleId.PLATFORM_VENDOR_SCAN
    if detection_class == "E":
        return EvidenceBundleId.PERSISTENCE_SQLITE
    if detection_class == "F":
        return EvidenceBundleId.PLATFORM_TYPING_SCAN
    if detection_class == "G":
        return EvidenceBundleId.GOVERNANCE_BOUNDARY
    if detection_class == "H":
        return EvidenceBundleId.L3_REACHABILITY
    if detection_class == "I":
        return EvidenceBundleId.CONFIG_HOSTING
    if detection_class == "C":
        return EvidenceBundleId.PLATFORM_FAIL_CLOSED
    if detection_class == "B":
        return EvidenceBundleId.PLATFORM_HARDCODING
    raise ValueError(f"unmapped pair: {layer}×{detection_class}")


def bundle_for_pair(layer: str, detection_class: str) -> EvidenceBundle:
    bundle_id = resolve_bundle_id(layer, detection_class)
    return _BUNDLE_CATALOG[bundle_id]


def bundle_review_disposition(layer: str, detection_class: str) -> ImplementerReview:
    return bundle_for_pair(layer, detection_class).implementer_review


@lru_cache(maxsize=1)
def pair_to_bundle_id_map() -> dict[tuple[str, str], EvidenceBundleId]:
    out: dict[tuple[str, str], EvidenceBundleId] = {}
    for layer, cls in iter_applicable_layer_class_pairs():
        out[(layer, cls)] = resolve_bundle_id(layer, cls)
    return out


def mandatory_insufficient_pairs() -> frozenset[tuple[str, str]]:
    return frozenset(
        key
        for key in pair_to_bundle_id_map()
        if bundle_review_disposition(key[0], key[1]) == ImplementerReview.INSUFFICIENT
    )


def bundle_review_census() -> dict[str, int]:
    counts: dict[str, int] = {r.value: 0 for r in ImplementerReview}
    seen: set[EvidenceBundleId] = set()
    for bundle_id in pair_to_bundle_id_map().values():
        if bundle_id in seen:
            continue
        seen.add(bundle_id)
        review = _BUNDLE_CATALOG[bundle_id].implementer_review
        counts[review.value] += 1
    pair_counts: dict[str, int] = {r.value: 0 for r in ImplementerReview}
    for layer, cls in iter_applicable_layer_class_pairs():
        pair_counts[bundle_review_disposition(layer, cls).value] += 1
    return {"bundles": len(seen), "pairs_by_review": pair_counts, **counts}


def all_evidence_bundles() -> tuple[EvidenceBundle, ...]:
    used = set(pair_to_bundle_id_map().values())
    return tuple(_BUNDLE_CATALOG[bid] for bid in sorted(used, key=lambda x: x.value))
