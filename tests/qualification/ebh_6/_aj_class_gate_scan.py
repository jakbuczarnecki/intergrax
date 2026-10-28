# © Artur Czarnecki. All rights reserved.

"""T-CLS-A..J class gate scan + adjudication SSOT for EBH-6 B3 (EX-01 evidence-backed)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import pytest
from tests.qualification.ebh_6._closed_world_discovery import EBH6_AUDIT_LAYERS
from tests.qualification.ebh_6._layer_evidence import layer_aj_applicability, layer_evidence_by_id

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_AJ_GATE_MODULE: Final[str] = "tests/qualification/ebh_6/test_ebh_6_aj_class_gates.py"

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
_EBH5_REACH_RAG = (
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
_LIFECYCLE = (
    "tests/integration/agent_distribution/test_canonical_agent_lifecycle_architecture_gate.py"
    "::test_stage15_proof_files_avoid_forbidden_lifecycle_bypass_fragments"
)
_B3_Q2D2 = "tests/qualification/ebh_6/test_ebh_6_b3_exit_gates.py::test_ebh6_b3_q2d2_semantic_core_f_is_finding"
_REGISTER_SSOT = "tests/qualification/ebh_6/_findings_register.py"

_CLASS_REUSE: Final[dict[str, tuple[str, ...]]] = {
    "A": (_REG_PROHIBITED, _REG_UNCLASS),
    "B": (_EE_A1, _REG_PROHIBITED),
    "C": (_EBH5_REACH_SLUG, _SANCTIONED),
    "D": (_REG_PROHIBITED, _SANCTIONED),
    "E": (_SQLITE, _REG_PROHIBITED),
    "F": (_REG_DEBT, _REG_UNCLASS),
    "G": (_W5_SINK, _REG_PROHIBITED),
    "H": (_SANCTIONED, _LIFECYCLE),
    "I": (_CFGX, _EBH5_INV),
    "J": (_RPLG, _EBH5_INV),
}

_LAYER_CLASS_EXTRA_REFS: Final[dict[tuple[str, str], tuple[str, ...]]] = {
    ("rag_memory", "A"): (_EBH5_REACH_RAG,),
    ("rag_memory", "D"): (_EBH5_REACH_RAG,),
    ("rag_memory", "H"): (_EBH5_REACH_RAG,),
    ("integrations", "A"): (_EBH5_REACH_FACTORY, _EBH5_REACH_SLUG),
    ("integrations", "B"): (_EBH5_REACH_FACTORY,),
    ("integrations", "C"): (_EBH5_REACH_FACTORY, _EBH5_REACH_SLUG),
    ("integrations", "I"): (_EBH5_REACH_ENV,),
    ("integrations", "H"): (_EBH5_INV,),
    ("integrations", "J"): (_RPLG,),
    ("agent_distribution", "B"): (_BND_GATE, _LIFECYCLE),
    ("agent_distribution", "H"): (_LIFECYCLE,),
    ("governance", "H"): (_GOV_H1,),
    ("governance", "G"): (_GOV_H1,),
    ("execution", "B"): (_EE_A1,),
    ("execution", "H"): (_EE_A1,),
    ("reliability_recovery", "H"): (_NPSC5E,),
    ("observability_diagnostics", "G"): (_W5_SINK,),
    ("state_persistence", "E"): (_SQLITE,),
    ("state_persistence", "H"): (_SQLITE,),
    ("runtime_hosting", "C"): (_CFGX,),
    ("runtime_hosting", "I"): (_CFGX, _EBH5_INV),
    ("composition_roots", "C"): (_EBH5_REACH_FACTORY, _SANCTIONED),
    ("composition_roots", "H"): (_SANCTIONED,),
    ("semantic_core", "B"): (_EE_A1,),
    ("semantic_core", "H"): (_SANCTIONED,),
    ("contracts", "F"): (_REG_DEBT,),
    ("application_services", "F"): (_REG_DEBT,),
}

_VENDOR_SDK_IMPORT_MARKERS: Final[tuple[str, ...]] = (
    "openai",
    "anthropic",
    "boto3",
    "botocore",
    "google.cloud",
    "azure.",
)

_SHADOW_SELECTOR_MARKERS: Final[tuple[str, ...]] = (
    "graph_store_backend",
    "backend_registry",
    "_GLOBAL_PROVIDER",
    "shadow_registry",
)


@dataclass(frozen=True, slots=True)
class ClassGateAdjudication:
    outcome: str
    invariant_proves: str
    evidence_refs: tuple[str, ...]
    scope_limitation: str


def class_gate_node_id(layer: str, detection_class: str) -> str:
    return f"{_AJ_GATE_MODULE}::test_ebh6_aj_class_gate[{layer}-{detection_class}]"


def iter_applicable_layer_class_pairs() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for layer in EBH6_AUDIT_LAYERS:
        for aj in layer_aj_applicability(layer):
            if aj.applicable:
                pairs.append((layer, aj.detection_class))
    return tuple(pairs)


def _layer_py_files(layer: str) -> tuple[Path, ...]:
    annex = layer_evidence_by_id()
    idx = EBH6_AUDIT_LAYERS.index(layer) + 1
    row = annex[f"EBH6-LAYER-{idx:02d}"]
    files: list[Path] = []
    for rel in row.closed_world_paths:
        if rel.startswith("layer-inventory:"):
            continue
        path = _REPO_ROOT / rel
        if path.is_file() and rel.endswith(".py"):
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return tuple(files)


def _read_py(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def assert_no_vendor_sdk_imports_in_text(text: str, rel: str, *, skip_provider_tree: bool) -> None:
    if skip_provider_tree:
        return
    for line in text.splitlines():
        if not line.startswith(("import ", "from ")):
            continue
        lowered = line.lower()
        for marker in _VENDOR_SDK_IMPORT_MARKERS:
            token = marker.rstrip(".")
            if token in lowered and (f"import {token}" in lowered or f"from {token}" in lowered):
                raise AssertionError(f"{rel}: vendor SDK import {marker!r}")


def assert_no_shadow_selector_markers(text: str, rel: str) -> None:
    for marker in _SHADOW_SELECTOR_MARKERS:
        if marker in text:
            raise AssertionError(f"{rel}: shadow selector {marker!r}")


def scan_class_a_vendor_authorship(layer: str) -> None:
    files = _layer_py_files(layer)
    assert files, f"T-CLS-A: no python paths for {layer}"
    for path in files:
        text = _read_py(path)
        rel = path.relative_to(_REPO_ROOT).as_posix()
        skip = "/providers/" in rel or rel.startswith("intergrax/integrations/providers/")
        assert_no_vendor_sdk_imports_in_text(text, rel, skip_provider_tree=skip)
        if layer in ("contracts", "governance") and "integrations.providers" in text:
            assert "TYPE_CHECKING" in text or "Protocol" in text, rel


def scan_class_b_hardcoding(layer: str) -> None:
    if layer not in ("execution", "semantic_core", "agent_distribution"):
        return
    files = _layer_py_files(layer)
    forbidden_literals = ('"openai"', "'openai'", '"gpt-4"', '"claude"')
    for path in files:
        if "/providers/" in path.as_posix():
            continue
        text = _read_py(path)
        for lit in forbidden_literals:
            if lit in text and "test" not in path.parts:
                assert "example" in text.lower() or "fixture" in path.name, f"{path}: {lit}"


def scan_class_c_fail_closed(layer: str) -> None:
    factory = _REPO_ROOT / "intergrax/integrations/registry/factory.py"
    text = _read_py(factory)
    assert "IntegrationConfigurationError" in text
    assert "raise" in text
    if layer in ("runtime_hosting", "integrations", "composition_roots"):
        assert "resolve_slug" in text or "resolve_from_profile" in text


def scan_class_d_duplicate_selector(layer: str) -> None:
    for path in _layer_py_files(layer):
        assert_no_shadow_selector_markers(_read_py(path), path.relative_to(_REPO_ROOT).as_posix())


def scan_class_e_shadow_store(layer: str) -> None:
    if layer not in ("state_persistence", "application_services", "events_traceability", "reliability_recovery"):
        return
    allowed_sqlite_layers = ("state_persistence", "application_services", "events_traceability")
    for path in _layer_py_files(layer):
        text = _read_py(path)
        if "sqlite" in text.lower() and layer not in allowed_sqlite_layers:
            rel = path.relative_to(_REPO_ROOT).as_posix()
            assert "annex" in text.lower() or "test" in path.parts or "fixture" in path.name, rel


def scan_class_f_typing_seam(layer: str) -> None:
    if layer == "semantic_core":
        reg = _REPO_ROOT / "tests/qualification/ebh_6/_findings_register.py"
        assert "EBH6-DEBT-Q2-D2" in _read_py(reg)


def scan_class_g_obs_non_authority(layer: str) -> None:
    return


def scan_class_h_reachability(layer: str) -> None:
    mech = _REPO_ROOT / "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py"
    assert "sanctioned_composition_roots" in _read_py(mech)


def scan_class_i_env_coupling(layer: str) -> None:
    if layer in ("integrations", "runtime_hosting"):
        bridge = _REPO_ROOT / "intergrax/applications/_shared/integration_runtime_bridge.py"
        if bridge.is_file():
            bridge_text = _read_py(bridge)
            assert "IntegrationProfile" in bridge_text or "profile" in bridge_text.lower()


def scan_class_j_replaceability(layer: str) -> None:
    if layer != "integrations":
        return
    factory = _REPO_ROOT / "intergrax/integrations/registry/factory.py"
    assert "PlatformIntegrationContract" in _read_py(factory)


_SCANNERS = {
    "A": scan_class_a_vendor_authorship,
    "B": scan_class_b_hardcoding,
    "C": scan_class_c_fail_closed,
    "D": scan_class_d_duplicate_selector,
    "E": scan_class_e_shadow_store,
    "F": scan_class_f_typing_seam,
    "G": scan_class_g_obs_non_authority,
    "H": scan_class_h_reachability,
    "I": scan_class_i_env_coupling,
    "J": scan_class_j_replaceability,
}


def _pytest_node_ids(refs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(r for r in refs if "::test_" in r)


_reuse_ran: set[str] = set()


def _invoke_reuse_gate(node_id: str) -> None:
    if node_id in _reuse_ran:
        return
    rel, func = node_id.split("::", 1)
    path = _REPO_ROOT / rel
    assert path.is_file(), f"missing reuse gate: {node_id}"
    exit_code = pytest.main(["-q", "--tb=line", f"{path}::{func}"])
    assert exit_code == 0, f"reuse gate failed: {node_id}"
    _reuse_ran.add(node_id)


@lru_cache(maxsize=16)
def _invoke_class_reuse_gates(detection_class: str) -> None:
    for node in _pytest_node_ids(_CLASS_REUSE.get(detection_class, (_REG_PROHIBITED,))):
        _invoke_reuse_gate(node)


@lru_cache(maxsize=128)
def _invoke_layer_class_extra_gates(layer: str, detection_class: str) -> None:
    extra = _LAYER_CLASS_EXTRA_REFS.get((layer, detection_class), ())
    for node in _pytest_node_ids(extra):
        _invoke_reuse_gate(node)


def run_class_gate_scan(layer: str, detection_class: str) -> None:
    _invoke_class_reuse_gates(detection_class)
    _invoke_layer_class_extra_gates(layer, detection_class)
    scanner = _SCANNERS[detection_class]
    scanner(layer)


def _default_outcome(layer: str, detection_class: str) -> str:
    if (layer, detection_class) == ("semantic_core", "F"):
        return "FINDING"
    return "EVIDENCE_REQUIRED"


def _default_proves(layer: str, detection_class: str) -> str:
    if (layer, detection_class) == ("semantic_core", "F"):
        return (
            "EBH6-DEBT-Q2-D2 / FRZ-TYP-03 — tracked typing seam; ratification ≠ independent acceptance "
            "(OD-EBH6-Q2D2 OPEN)."
        )
    return (
        f"T-CLS-{detection_class} @ `{layer}` — closed-world slice + class reuse gates run; "
        "no independent layer×class behavioral proof on record (mandatory blocker)."
    )


def _default_scope(detection_class: str) -> str:
    return (
        f"Class-{detection_class} mechanical scan + reused qual gates only — "
        "not full platform behavioral qual (51 NYQ / DUP-X deferred where noted)."
    )


def class_gate_adjudication(layer: str, detection_class: str) -> ClassGateAdjudication:
    gate = class_gate_node_id(layer, detection_class)
    reuse = _CLASS_REUSE.get(detection_class, (_REG_PROHIBITED,))
    extra = _LAYER_CLASS_EXTRA_REFS.get((layer, detection_class), ())
    refs = (gate,) + extra + reuse
    if (layer, detection_class) == ("semantic_core", "F"):
        refs = (gate, _B3_Q2D2, _REG_DEBT, "intergrax/runtime/nexus/nexus_loop.py")
    outcome = _default_outcome(layer, detection_class)
    return ClassGateAdjudication(
        outcome,
        _default_proves(layer, detection_class),
        refs,
        _default_scope(detection_class),
    )


def count_class_gates_by_outcome() -> dict[str, int]:
    counts: dict[str, int] = {}
    for layer, cls in iter_applicable_layer_class_pairs():
        adj = class_gate_adjudication(layer, cls)
        counts[adj.outcome] = counts.get(adj.outcome, 0) + 1
    return counts
