# © Artur Czarnecki. All rights reserved.

"""Grouped A–J evidence reconciliation — SCOPE-01 (no per-cell micro-iterations)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Final

from tests.qualification.ebh_6._closed_world_discovery import EBH6_AUDIT_LAYERS
from tests.qualification.ebh_6._layer_evidence import layer_aj_applicability, layer_gate_test_ids


class AjControlOutcome(StrEnum):
    PASS_CANDIDATE = "PASS_CANDIDATE"
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
_TIER0 = (
    "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py"
    "::test_tier0_must_not_import_agents_or_applications"
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
_W5_SINK = (
    "tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py"
    "::test_production_composition_creates_bounded_sink"
)
_VBN_AUDIT = "docs/project/maintainers/qualification/EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md"
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
    scope_note: str
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


def _class_evidence(layer: str, detection_class: str, layer_gates: tuple[str, ...]) -> tuple[str, ...]:
    base = (_REG_PROHIBITED, _REG_UNCLASS, _REGISTER_SSOT, _VBN_AUDIT)
    extra: tuple[str, ...] = ()
    if detection_class == "H":
        extra = (_SANCTIONED, _TIER0) + layer_gates
    elif detection_class == "A":
        extra = (_EBH5_INV,) if layer in ("integrations", "rag_memory", "tools_skills_capabilities") else ()
    elif detection_class == "B":
        extra = (_TIER0, _EE_A1) if layer in ("execution", "semantic_core") else (_TIER0,)
    elif detection_class == "C":
        extra = (_CFGX,)
    elif detection_class == "D":
        extra = (_EBH5_INV,) if layer in ("integrations", "application_services", "rag_memory") else ()
    elif detection_class == "E":
        extra = (_SQLITE,) if layer == "state_persistence" else ()
    elif detection_class == "F":
        extra = (_REG_DEBT,) if layer in ("semantic_core", "contracts", "application_services") else ()
    elif detection_class == "G":
        extra = (_W5_SINK,) if layer in ("observability_diagnostics", "execution", "governance") else ()
    elif detection_class == "I":
        if layer in ("integrations", "runtime_hosting", "background_workers_infra_adapters"):
            extra = (_CFGX, _EBH5_INV)
        else:
            extra = (_CFGX,)
    elif detection_class == "J":
        extra = (_RPLG, "EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX — 51 NYQ deferred EBH-7")
    refs = tuple(dict.fromkeys(base + extra + layer_gates))
    return refs


def _build_group_table() -> dict[tuple[str, str], AjGroupReconciliation]:
    table: dict[tuple[str, str], AjGroupReconciliation] = {}
    for layer in EBH6_AUDIT_LAYERS:
        gates = layer_gate_test_ids(layer)
        owner = _LAYER_OWNERS[layer]
        for aj in layer_aj_applicability(layer):
            if not aj.applicable:
                continue
            dc = aj.detection_class
            refs = _class_evidence(layer, dc, gates)
            blocker = None
            outcome = AjControlOutcome.PASS_CANDIDATE
            if dc == "J" and layer in ("rag_memory", "tools_skills_capabilities"):
                scope = "51 NYQ rows — partial replaceability evidence only"
            else:
                scope = "EBH-6 VBN mechanical slice — grouped closed-world + behavioral gate bundle"
            table[(layer, dc)] = AjGroupReconciliation(
                f"AJG-{layer}-{dc}",
                layer,
                dc,
                owner,
                f"Roadmap §3.0.8 class {dc} @ {layer}",
                outcome,
                refs,
                scope,
                blocker,
            )
    return table


@lru_cache(maxsize=1)
def aj_group_by_layer_class() -> dict[tuple[str, str], AjGroupReconciliation]:
    return _build_group_table()


def format_group_evidence_ref(
    group: AjGroupReconciliation,
    mechanism_id: str,
    primary_gate: str,
) -> str:
    joined = " | ".join(group.evidence_refs[:4])
    if len(group.evidence_refs) > 4:
        joined += f" | +{len(group.evidence_refs) - 4} refs"
    return f"group={group.group_id}; mechanism={mechanism_id}; L3_gate={primary_gate}; bundle=[{joined}]"


def assert_aj_group_reconciliation_complete() -> None:
    for layer in EBH6_AUDIT_LAYERS:
        for aj in layer_aj_applicability(layer):
            if aj.applicable:
                assert (layer, aj.detection_class) in aj_group_by_layer_class(), layer
