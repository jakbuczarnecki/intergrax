# © Artur Czarnecki. All rights reserved.

"""T2-AUD — mechanism bundle SSOT + independent implementer review (replaces false scanner_effective T2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
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
_REG_UNCLASS = (
    "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_findings_unclassified_zero"
)
_REG_DEBT = (
    "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_tracked_debt_findings_ratified"
)
_EBH5_REACH_SLUG = (
    "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py::test_ebh5_reach_integration_unknown_slug_fail_closed"
)
_CONTRACTS_TYPING_CSP = (
    "tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py"
    "::test_ebh6_contracts_layer_frz_typ_surface_purity_passes"
)
_CONTRACTS_TYPING_APP = (
    "tests/qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py"
    "::test_ebh6_application_services_production_seam_typing_passes"
)
_EBH2I_CSP = "tests/unit/architecture/test_ebh_2i_final_rescan_gate.py::test_ebh_2i_contract_surface_purity_gate_passes"
_HARDENING3_BND = (
    "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py"
    "::test_hardening_3_contracts_do_not_import_runtime_except_allowlist"
)
_E01_HARD_B = (
    "tests/qualification/ebh_6/test_ebh_6_e01_e02_audit_findings_evidence.py"
    "::test_ebh6_e01_platform_hardcoding_invokes_class_b_scans"
)
_E01_SQLITE = (
    "tests/qualification/ebh_6/test_ebh_6_e01_e02_audit_findings_evidence.py"
    "::test_ebh6_e01_persistence_sqlite_closed_world_no_shadow_connects"
)
_E01_CMP = (
    "tests/qualification/ebh_6/test_ebh_6_e01_e02_audit_findings_evidence.py"
    "::test_ebh6_e01_sanctioned_composition_paths_resolvable_on_disk"
)
_E01_FAIL = (
    "tests/qualification/ebh_6/test_ebh_6_e01_e02_audit_findings_evidence.py"
    "::test_ebh6_e01_platform_fail_closed_reach_gates"
)
_EBH5_SQLITE_PROOF = (
    "tests/qualification/ebh_5/test_ebh_5_proof_integrity_gates.py::test_ebh5_proof_sqlite_closed_world_accountability"
)
_EBH5_REACH_OBS = (
    "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py::test_ebh5_reach_observability_fail_closed_without_role"
)
_VBN_MATRIX = "docs/project/maintainers/qualification/VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md"
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

_REGISTER_ONLY_GATE_MARKERS: Final[tuple[str, ...]] = (
    "hardcoding_register_integrity",
    "_findings_register.py",
)


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
        (_E01_CMP, _EBH5_REACH_FACTORY),
        ImplementerReview.ACCEPTED,
        "On-disk sanctioned roots + factory owner — not full wiring replay @ HEAD.",
        negative_test_ref=_REG_PROHIBITED,
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
        (_EE_A1, _SANCTIONED, _HARDENING3_BND),
        ImplementerReview.ACCEPTED,
        "Mechanism-level L3 index — primary_gate nodeids @ HEAD.",
        negative_test_ref=_REG_PROHIBITED,
    ),
    EvidenceBundleId.GOVERNANCE_BOUNDARY: EvidenceBundle(
        EvidenceBundleId.GOVERNANCE_BOUNDARY,
        "NPSC42 governance non-execution identity",
        (_GOV_H1,),
        ImplementerReview.ACCEPTED,
        "NPSC42 frozen subset — observability G uses non-authority scan elsewhere.",
        negative_test_ref=_REG_PROHIBITED,
    ),
    EvidenceBundleId.PERSISTENCE_SQLITE: EvidenceBundle(
        EvidenceBundleId.PERSISTENCE_SQLITE,
        "SQLite annex accountability",
        (_E01_SQLITE, _EBH5_SQLITE_PROOF, _SQLITE),
        ImplementerReview.ACCEPTED,
        "Closed-world sqlite3.connect census + class-E shadow scan — not live store semantics.",
        negative_test_ref=_REG_PROHIBITED,
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
        (_EBH5_REACH_RAG,),
        ImplementerReview.ACCEPTED,
        "RAG graph bootstrap gates — not full memory plane qual.",
        negative_test_ref=_REG_PROHIBITED,
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
        (_EBH5_REACH_FACTORY, _EBH5_REACH_SLUG, _EBH5_INV),
        ImplementerReview.ACCEPTED,
        "Inventory + static vendor scan — not live adapter audit.",
        negative_test_ref=_REG_UNCLASS,
    ),
    EvidenceBundleId.REPLACEABILITY_J: EvidenceBundle(
        EvidenceBundleId.REPLACEABILITY_J,
        "RPL-G CONFIG-X blockers + VBN matrix",
        (_RPLG, _EBH5_INV),
        ImplementerReview.ACCEPTED,
        "RPL-G behavioral invoke + matrix cite — 51 NYQ rows remain deferred.",
        negative_test_ref=_VBN_MATRIX,
    ),
    EvidenceBundleId.Q2D2_FINDING: EvidenceBundle(
        EvidenceBundleId.Q2D2_FINDING,
        "EBH6-DEBT-Q2-D2 / FRZ-TYP-03",
        (
            "tests/qualification/ebh_6/test_ebh_6_q2d2_dual_path.py"
            "::test_ebh6_q2d2_path_b_remediation_proves_registered_violation_removed",
            "tests/qualification/ebh_6/test_ebh_6_b3_exit_gates.py"
            "::test_ebh6_b3_q2d2_semantic_core_f_pending_independent_acceptance",
        ),
        ImplementerReview.ACCEPTED,
        "Path B remediated @ HEAD — PASS_CANDIDATE pending independent audit (not FINDING).",
    ),
    EvidenceBundleId.PLATFORM_FAIL_CLOSED: EvidenceBundle(
        EvidenceBundleId.PLATFORM_FAIL_CLOSED,
        "Fail-closed materialization markers",
        (_E01_FAIL, _EBH5_REACH_SLUG, _EBH5_REACH_OBS),
        ImplementerReview.ACCEPTED,
        "Reach fail-closed + applicable class-C scans — not every hosting profile path.",
        negative_test_ref=_REG_PROHIBITED,
    ),
    EvidenceBundleId.PLATFORM_HARDCODING: EvidenceBundle(
        EvidenceBundleId.PLATFORM_HARDCODING,
        "Class-B literal scan + register",
        (_E01_HARD_B,),
        ImplementerReview.ACCEPTED,
        "Closed-world class-B vendor literal scan on execution/semantic/agent layers.",
        negative_test_ref=_REG_PROHIBITED,
    ),
    EvidenceBundleId.PLATFORM_TYPING_SCAN: EvidenceBundle(
        EvidenceBundleId.PLATFORM_TYPING_SCAN,
        "Protocol/TYPE_CHECKING seam scan",
        (_CONTRACTS_TYPING_CSP, _CONTRACTS_TYPING_APP, _EBH2I_CSP),
        ImplementerReview.ACCEPTED,
        "Static typing seam markers — not FRZ-TYP behavioral replay.",
        negative_test_ref=_REG_DEBT,
    ),
    EvidenceBundleId.L3_REACHABILITY: EvidenceBundle(
        EvidenceBundleId.L3_REACHABILITY,
        "L3 mechanism primary_gate nodeids per layer",
        (_SANCTIONED, _LIFECYCLE, _NPSC5E, _HARDENING3_BND),
        ImplementerReview.ACCEPTED,
        "Primary_gate pytest nodeids from `_l3_reachability_evidence` — not entrypoint filename heuristic.",
        negative_test_ref=_REG_PROHIBITED,
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


def is_register_only_pytest_gate(node_id: str) -> bool:
    if "::" not in node_id:
        return False
    return any(marker in node_id for marker in _REGISTER_ONLY_GATE_MARKERS)


def pytest_primary_gates(bundle: EvidenceBundle) -> tuple[str, ...]:
    return tuple(g for g in bundle.primary_behavioral_gates if "::" in g)


def assert_resolvable_pytest_node_id(node_id: str) -> None:
    if "::" not in node_id:
        msg = f"not a pytest nodeid: {node_id!r}"
        raise AssertionError(msg)
    rel, func = node_id.split("::", 1)
    path = _REPO_ROOT / rel
    if not path.is_file():
        msg = f"missing gate module: {node_id}"
        raise AssertionError(msg)
    text = path.read_text(encoding="utf-8")
    func_base = func.split("[", 1)[0]
    if f"def {func_base}" not in text and f"async def {func_base}" not in text:
        msg = f"missing gate test function: {node_id}"
        raise AssertionError(msg)


def bundle_has_non_register_behavioral_gate(bundle: EvidenceBundle) -> bool:
    return any(not is_register_only_pytest_gate(g) for g in pytest_primary_gates(bundle))
