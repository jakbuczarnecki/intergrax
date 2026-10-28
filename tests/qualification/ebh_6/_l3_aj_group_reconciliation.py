# © Artur Czarnecki. All rights reserved.

"""Grouped A–J evidence reconciliation — P03-01 (no unconditional PASS promotion)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Final

from tests.qualification.ebh_6._closed_world_discovery import EBH6_AUDIT_LAYERS
from tests.qualification.ebh_6._layer_evidence import layer_aj_applicability


class AjControlOutcome(StrEnum):
    PASS_CANDIDATE = "PASS_CANDIDATE"
    PARTIAL_CANDIDATE = "PARTIAL_CANDIDATE"
    EVIDENCE_REQUIRED = "EVIDENCE_REQUIRED"
    FINDING = "FINDING"
    NA_WITH_EVIDENCE = "N/A — WITH EVIDENCE"


_REG_PROHIBITED = (
    "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero"
)
_REG_UNCLASS = (
    "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_findings_unclassified_zero"
)
_REG_DEBT = (
    "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_tracked_debt_findings_ratified"
)
_BND_GATE = (
    "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py"
    "::test_hardening_3_contracts_do_not_import_runtime_except_allowlist"
)
_CFGX = (
    "tests/qualification/ebh_6/test_ebh_6_config_x_head_reconciliation.py"
    "::test_ebh6_config_x_all_concerns_mapped_at_head"
)
_SANCTIONED = (
    "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py"
    "::test_ebh6_exit_sanctioned_composition_roots_non_empty"
)
_SQLITE = "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_sqlite_annex_accountability"
_EE_A1 = (
    "tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py"
    "::test_ee_a1_p0_inventory_reports_zero_production_bypass"
)
_GOV_H1 = (
    "tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py"
    "::test_npsc42_h1_governance_planes_do_not_mint_execution_identity"
)
_RPLG = (
    "tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py"
    "::test_ebh5_rpl_g_config_x_active_blockers_remain_empty"
)
_EBH5_INV = "tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py::test_ebh5_exit_inventory_complete"
_EBH5_REACH_SLUG = (
    "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py::test_ebh5_reach_integration_unknown_slug_fail_closed"
)
_EBH5_REACH_FACTORY = (
    "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py"
    "::test_ebh5_reach_factory_module_is_sanctioned_composition_owner"
)
_EBH5_REACH_ENV = (
    "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py"
    "::test_ebh5_reach_env_slug_is_canonical_merge_not_shadow_registry"
)
_EBH5_REACH_RAG_GRAPH = (
    "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py"
    "::test_ebh5_reach_no_second_global_provider_registry_in_rag_graph"
)
_NPSC5E = (
    "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"
    "::test_no_recovery_god_object_in_production"
)
_W5_SINK = (
    "tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py"
    "::test_production_composition_creates_bounded_sink"
)
_LIFECYCLE = "tests/integration/agent_distribution/test_canonical_agent_lifecycle_architecture_gate.py"
_REGISTER_SSOT = "tests/qualification/ebh_6/_findings_register.py"


@dataclass(frozen=True, slots=True)
class AjGroupReconciliation:
    group_id: str
    layer: str
    detection_class: str
    semantic_owner: str
    invariant_ref: str
    outcome: AjControlOutcome
    evidence_refs: tuple[str, ...]
    invariant_proves: str
    scope_limitation: str
    mandatory_blocker: str | None = None


_LAYER_OWNERS: Final[dict[str, str]] = {
    "contracts": "Platform contracts / composition materialization",
    "semantic_core": "Nexus / execution orchestration",
    "application_services": "Application _shared wiring",
    "composition_roots": "Sanctioned composition sites (TRACE-X)",
    "governance": "Governance plane (non-execution)",
    "execution": "ExecutionRuntime / orchestration",
    "integrations": "Integration factory / provider bundles",
    "state_persistence": "Integration persistence + SQLite annex",
    "reliability_recovery": "Recovery plane / resilience",
    "observability_diagnostics": "Observability export (non-authoritative)",
    "events_traceability": "Event delivery / trace spine",
    "agent_distribution": "Agent distribution / tier-2 agents",
    "rag_memory": "RAG / memory materialization",
    "runtime_hosting": "Hosting profile / bootstrap",
    "tools_skills_capabilities": "Tool registry / SPI",
    "background_workers_infra_adapters": "Worker / queue adapters",
}

_ADJUDICATION: Final[dict[tuple[str, str], tuple[AjControlOutcome, str, tuple[str, ...], str]]] = {
    ("execution", "B"): (
        AjControlOutcome.PASS_CANDIDATE,
        "EE-A1 P0 inventory: zero production execution bypass paths (static scan).",
        (_EE_A1,),
        "EE-A1 certified surface — not whole-platform class-B census.",
    ),
    ("execution", "H"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "L3 execution entrypoints + EE-A1 bypass inventory on execution layer samples.",
        (_EE_A1, _SANCTIONED),
        "Sample L3 mechanisms — not exhaustive reachability proof.",
    ),
    ("semantic_core", "B"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "EE-A1 bypass inventory intersecting nexus/core execution surfaces.",
        (_EE_A1,),
        "Execution bypass scan — not all semantic_core vendor literals.",
    ),
    ("semantic_core", "H"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "L3 nexus/execution call chains + sanctioned roots registry present.",
        (_SANCTIONED,),
        "Mechanism-level L3 index — not independent reachability re-run.",
    ),
    ("composition_roots", "H"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "TRACE-X sanctioned production composition sites registered non-empty.",
        (_SANCTIONED,),
        "Registry presence — not per-root wiring replay @ HEAD.",
    ),
    ("governance", "H"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "NPSC42 H1: governance planes do not mint execution identity.",
        (_GOV_H1,),
        "NPSC42 frozen subset — not full GOV-X2 replay.",
    ),
    ("state_persistence", "E"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "SQLite annex paths accountable vs EBH-5 touchpoint registry.",
        (_SQLITE,),
        "Annex accountability — not all shadow-store class-E obligations.",
    ),
    ("state_persistence", "H"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "Persistence L3 chains + sqlite annex gate.",
        (_SQLITE,),
        "Annex + samples — not canonical store behavioral qual.",
    ),
    ("integrations", "J"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "RPL-G: CONFIG-X active blocker list empty @ HEAD.",
        (_RPLG,),
        "RPL-G slice — 51 NYQ matrix rows remain deferred.",
    ),
    ("integrations", "A"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "EBH-5 vendor inventory completeness (qual mechanical) on integration scope.",
        (_EBH5_INV, _REG_PROHIBITED),
        "Inventory completeness — not live adapter construction audit.",
    ),
    ("integrations", "H"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "Integration bundle L3 call chains inventoried per mechanism records.",
        (_EBH5_INV,),
        "L3 annex index — not integration factory behavioral replay.",
    ),
    ("observability_diagnostics", "G"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "W5-B2: production composition wires bounded observability sink.",
        (_W5_SINK,),
        "Single wiring scenario — not global OBS qualification.",
    ),
    ("agent_distribution", "B"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "HARDENING-3 contracts/runtime boundary + agent_distribution lifecycle gate.",
        (_BND_GATE, _LIFECYCLE),
        "Boundary/lifecycle subset — not all agents/ backend literals.",
    ),
    ("agent_distribution", "H"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "Agent distribution L3 mechanisms + lifecycle architecture gate.",
        (_LIFECYCLE,),
        "Integration gate — not full tier-2 agent census.",
    ),
    ("contracts", "F"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "Register: Q2-D2 / debt rows ratified — not contract typing re-audit.",
        (_REG_DEBT,),
        "Register disposition — not FRZ-TYP behavioral proof.",
    ),
    ("semantic_core", "F"): (
        AjControlOutcome.FINDING,
        "EBH6-DEBT-Q2-D2 / FRZ-TYP-03 — tracked typing seam; "
        "ratification ≠ independent acceptance (OD-EBH6-Q2D2 OPEN).",
        (_REG_DEBT, _REGISTER_SSOT),
        "Single tracked debt — Q2-D2 remains operator decision; not prod PASS.",
    ),
    ("application_services", "F"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "Register ratified debt disposition for tracked seams.",
        (_REG_DEBT,),
        "Register-only for class F on app _shared.",
    ),
    ("integrations", "B"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "Production code: registry factory enforces PlatformIntegrationContract + category "
        "match; no alternate materialization path in factory.py.",
        (_EBH5_REACH_SLUG, _EBH5_REACH_FACTORY, _REG_PROHIBITED),
        "Factory module only — not all integration bundles statically rescanned @ HEAD.",
    ),
    ("integrations", "C"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "Production code: resolve() order explicit slug → profile → env → IntegrationConfigurationError (fail-closed).",
        (_EBH5_REACH_SLUG, _EBH5_REACH_FACTORY),
        "Single sanctioned factory — not shadow config paths platform-wide.",
    ),
    ("integrations", "I"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "Production code: IntegrationProfile + INTERGRAX_INTEGRATION_* env merge; "
        "application bridge attaches profile to RuntimeConfig without second selector.",
        (_EBH5_REACH_ENV, "intergrax/applications/_shared/integration_runtime_bridge.py"),
        "Config authority on integration slice — not all hosting/runtime_hosting class-I cells.",
    ),
    ("composition_roots", "C"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "EBH-5 reachability: integrations/registry/factory.py is sanctioned "
        "materialization owner (resolve_from_profile / resolve_slug).",
        (_EBH5_REACH_FACTORY, _SANCTIONED),
        "Integration factory root — not every TRACE-X composition site replayed.",
    ),
    ("reliability_recovery", "H"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "L3 recovery mechanisms + NPSC5e no recovery god-object gate on production paths.",
        (_NPSC5E,),
        "Sample L3 recovery admission — not full reachability replay.",
    ),
    ("runtime_hosting", "C"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "CONFIG-X 54/54 concern mapping @ HEAD (structural reconciliation).",
        (_CFGX,),
        "Mapping completeness — not host profile materialization audit.",
    ),
    ("runtime_hosting", "I"): (
        AjControlOutcome.PARTIAL_CANDIDATE,
        "CONFIG-X mapping + EBH-5 inventory for hosting-related concerns.",
        (_CFGX, _EBH5_INV),
        "Reconciliation inventory — not install-time default proof.",
    ),
}


def _default_blocker(layer: str, detection_class: str) -> str:
    return f"EBH6-P03-AJ-{layer}-{detection_class}"


def _merge_adjudication_table() -> dict[tuple[str, str], tuple[AjControlOutcome, str, tuple[str, ...], str]]:
    from tests.qualification.ebh_6._aj_class_gate_scan import class_gate_adjudication, class_gate_node_id

    table: dict[tuple[str, str], tuple[AjControlOutcome, str, tuple[str, ...], str]] = {}
    for layer in EBH6_AUDIT_LAYERS:
        for aj in layer_aj_applicability(layer):
            if not aj.applicable:
                continue
            key = (layer, aj.detection_class)
            if key in _ADJUDICATION:
                outcome, proves, refs, scope = _ADJUDICATION[key]
            else:
                cg = class_gate_adjudication(layer, aj.detection_class)
                outcome = AjControlOutcome(cg.outcome)
                proves = cg.invariant_proves
                refs = cg.evidence_refs
                scope = cg.scope_limitation
            gate = class_gate_node_id(layer, aj.detection_class)
            if gate not in refs:
                refs = (gate,) + refs
            table[key] = (outcome, proves, refs, scope)
    return table


@lru_cache(maxsize=1)
def _full_adjudication() -> dict[tuple[str, str], tuple[AjControlOutcome, str, tuple[str, ...], str]]:
    return _merge_adjudication_table()


def count_evidence_required_groups() -> int:
    return sum(1 for o, _, _, _ in _full_adjudication().values() if o == AjControlOutcome.EVIDENCE_REQUIRED)


def _adjudicate_layer_class(layer: str, detection_class: str) -> AjGroupReconciliation:
    owner = _LAYER_OWNERS[layer]
    key = (layer, detection_class)
    outcome, proves, refs, scope = _full_adjudication()[key]
    blocker = _default_blocker(layer, detection_class) if outcome == AjControlOutcome.EVIDENCE_REQUIRED else None
    return AjGroupReconciliation(
        f"AJG-{layer}-{detection_class}",
        layer,
        detection_class,
        owner,
        f"Roadmap §3.0.8 class {detection_class} @ {layer}",
        outcome,
        refs,
        proves,
        scope,
        blocker,
    )


def _build_group_table() -> dict[tuple[str, str], AjGroupReconciliation]:
    table: dict[tuple[str, str], AjGroupReconciliation] = {}
    for layer in EBH6_AUDIT_LAYERS:
        for aj in layer_aj_applicability(layer):
            if not aj.applicable:
                continue
            table[(layer, aj.detection_class)] = _adjudicate_layer_class(layer, aj.detection_class)
    return table


@lru_cache(maxsize=1)
def aj_group_by_layer_class() -> dict[tuple[str, str], AjGroupReconciliation]:
    return _build_group_table()


def format_group_evidence_ref(
    group: AjGroupReconciliation,
    mechanism_id: str,
    primary_gate: str,
) -> str:
    joined = " | ".join(group.evidence_refs[:5])
    if len(group.evidence_refs) > 5:
        joined += f" | +{len(group.evidence_refs) - 5} refs"
    return (
        f"group={group.group_id}; outcome={group.outcome.value}; "
        f"proves={group.invariant_proves}; mechanism={mechanism_id}; "
        f"L3_gate={primary_gate}; refs=[{joined}]"
    )


def summarize_aj_obligation_cells() -> dict[str, int]:
    from tests.qualification.ebh_6._l3_reachability_evidence import all_l3_reachability_records

    counts: dict[str, int] = {}
    for rec in all_l3_reachability_records():
        for ctrl in rec.aj_controls:
            key = ctrl.outcome.value
            counts[key] = counts.get(key, 0) + 1
    return counts


def assert_aj_group_reconciliation_complete() -> None:
    for layer in EBH6_AUDIT_LAYERS:
        for aj in layer_aj_applicability(layer):
            if aj.applicable:
                g = aj_group_by_layer_class()[(layer, aj.detection_class)]
                assert g.invariant_proves.strip()
                if g.outcome == AjControlOutcome.EVIDENCE_REQUIRED:
                    assert g.mandatory_blocker, g.group_id
