# © Artur Czarnecki. All rights reserved.

"""L9 closed-world layer evidence annex — path inventories, A–J matrix, L3 narrative (PL-01/PL-02)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.ebh_6._closed_world_discovery import (
    EBH6_AUDIT_LAYERS,
    discover_application_wiring_paths,
    discover_composition_root_paths,
    discover_integration_bundle_paths,
    discover_sanctioned_production_composition_roots,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

_AJ_CLASSES: Final[tuple[str, ...]] = tuple("ABCDEFGHIJ")


@dataclass(frozen=True, slots=True)
class AjApplicability:
    detection_class: str
    applicable: bool
    rationale: str


@dataclass(frozen=True, slots=True)
class LayerL9Evidence:
    finding_id: str
    layer: str
    closed_world_paths: tuple[str, ...]
    aj_applicability: tuple[AjApplicability, ...]
    sys_inv_refs: tuple[str, ...]
    l3_narrative: str
    production_effective_samples: tuple[str, ...]
    gate_test_ids: tuple[str, ...]


def _glob_paths(prefix: str, pattern: str = "**/*.py", *, limit: int = 512) -> tuple[str, ...]:
    root = _REPO_ROOT / prefix
    if not root.is_dir():
        return ()
    out: list[str] = []
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            out.append(path.relative_to(_REPO_ROOT).as_posix())
            if len(out) >= limit:
                break
    return tuple(out)


def _aj(*pairs: tuple[str, bool, str]) -> tuple[AjApplicability, ...]:
    return tuple(AjApplicability(c, y, r) for c, y, r in pairs)


# Roadmap §3.0.8 L9-A matrix (PL-01) — N/A requires rationale in annex.
_LAYER_AJ: Final[dict[str, tuple[AjApplicability, ...]]] = {
    "contracts": _aj(
        ("A", True, "Vendor construction in contract public seams"),
        ("B", False, "No runtime backend selection on pure contract layer"),
        ("C", False, "No implicit defaults without materialization"),
        ("D", False, "No parallel contract registries on this layer"),
        ("E", False, "No shadow persistence on contract-only surface"),
        ("F", True, "SDK/type leakage in public contract seams"),
        ("G", False, "No operational diagnostics owner here"),
        ("H", True, "Reachability from sanctioned wiring to contract imports"),
        ("I", False, "No install-time coupling on contracts alone"),
        ("J", True, "Extensibility claims vs replaceability rows"),
    ),
    "semantic_core": _aj(
        ("A", True, "Vendor construction in nexus/core loop"),
        ("B", True, "Hardcoded backend names in core"),
        ("C", True, "Implicit defaults in core init"),
        ("D", False, "No duplicate env registries owned by semantic core"),
        ("E", False, "No shadow persistence plane in nexus core"),
        ("F", True, "Typing pseudo-contracts (Q2-D2 tracked)"),
        ("G", False, "Observability not authoritative in core"),
        ("H", True, "Sanctioned host → nexus_loop reachability"),
        ("I", True, "Install-time defaults in core"),
        ("J", True, "Extensibility vs 51 NYQ limitation documented"),
    ),
    "application_services": _aj(
        ("A", True, "Default construction in shared Tier-3 helpers"),
        ("B", True, "Backend literals in _shared wiring"),
        ("C", True, "Implicit resolver defaults"),
        ("D", True, "Duplicate resolver risk in _shared"),
        ("E", False, "No shadow store owner in _shared"),
        ("F", True, "Typing/seam leakage"),
        ("G", False, "Not diagnostics owner"),
        ("H", True, "App wiring → _shared → contracts L3"),
        ("I", True, "Install coupling in shared services"),
        ("J", True, "Service extensibility boundary"),
    ),
    "composition_roots": _aj(
        ("A", True, "Factory/profile materialization"),
        ("B", True, "Backend names in composition hints"),
        ("C", True, "Implicit profile defaults"),
        ("D", True, "Shadow registries / parallel selectors"),
        ("E", False, "Not persistence shadow owner"),
        ("F", True, "Composition typing seams"),
        ("G", False, "Not observability owner"),
        ("H", True, "TRACE-X sanctioned root → hint modules"),
        ("I", True, "Install-time composition defaults"),
        ("J", True, "Replaceability at composition boundary"),
    ),
    "governance": _aj(
        ("A", False, "No vendor SDK construction in governance plane"),
        ("B", False, "No backend selection in governance"),
        ("C", True, "Policy/config implicit defaults"),
        ("D", True, "Shadow policy persistence risk"),
        ("E", True, "Shadow persistence of policy artifacts"),
        ("F", True, "Governance typing seams"),
        ("G", True, "Vendor-locked diagnostics in governance path"),
        ("H", True, "Production wiring → governance attach L3"),
        ("I", True, "Install defaults in governance middleware"),
        ("J", False, "Extensibility not primary on governance plane"),
    ),
    "execution": _aj(
        ("A", True, "Vendor in execution core"),
        ("B", True, "Hardcoded inference backend names"),
        ("C", True, "Implicit execution defaults"),
        ("D", False, "No duplicate scheduler registry on execution layer"),
        ("E", False, "Not persistence shadow owner"),
        ("F", True, "Execution typing / bypass seams"),
        ("G", True, "Operational coupling in execution"),
        ("H", True, "Host → ExecutionRequest → runtime L3"),
        ("I", True, "Install-time execution defaults"),
        ("J", True, "Execution extensibility vs guard matrix"),
    ),
    "integrations": _aj(
        ("A", True, "Adapter construction at integration boundary"),
        ("B", True, "Provider literals in integrations tree"),
        ("C", True, "Implicit integration defaults"),
        ("D", True, "Shadow provider registries (DUP-X overlap)"),
        ("E", True, "Shadow config/selection mechanisms"),
        ("F", True, "Contract leakage into integrations"),
        ("G", True, "Operational coupling at adapters"),
        ("H", True, "Wiring → factory → bundle L3"),
        ("I", True, "Install-time integration defaults"),
        ("J", True, "Replaceability inventory replay"),
    ),
    "state_persistence": _aj(
        ("A", True, "Store construction paths"),
        ("B", True, "Backend names in persistence"),
        ("C", True, "Silent sqlite/default store risk"),
        ("D", True, "Parallel store selectors"),
        ("E", True, "Shadow stores outside annex"),
        ("F", True, "Persistence typing seams"),
        ("G", False, "Not primary observability owner"),
        ("H", True, "L3 to sqlite annex + contract stores"),
        ("I", True, "Install defaults for stores"),
        ("J", True, "Persistence replaceability"),
    ),
    "reliability_recovery": _aj(
        ("A", True, "Recovery executor construction"),
        ("B", True, "Backend names in resilience"),
        ("C", True, "Implicit recovery defaults"),
        ("D", False, "No duplicate recovery registry primary owner elsewhere"),
        ("E", True, "Shadow recovery persistence"),
        ("F", True, "Recovery typing seams"),
        ("G", True, "Operational coupling in recovery"),
        ("H", True, "Execution host → recovery plane L3"),
        ("I", True, "Install-time recovery store defaults"),
        ("J", True, "Recovery extensibility"),
    ),
    "observability_diagnostics": _aj(
        ("A", False, "No authoritative decision construction here"),
        ("B", False, "No backend selection core owner"),
        ("C", True, "Implicit exporter defaults"),
        ("D", True, "Parallel export registries"),
        ("E", False, "Not primary shadow store layer"),
        ("F", True, "Export typing seams"),
        ("G", True, "Vendor-locked health/metrics (class G focus)"),
        ("H", True, "Production OTLP concern L3"),
        ("I", True, "Install exporter defaults"),
        ("J", True, "Observability replaceability rows"),
    ),
    "events_traceability": _aj(
        ("A", True, "Broker/client construction in delivery path"),
        ("B", True, "Hardcoded broker backends"),
        ("C", True, "Implicit delivery defaults"),
        ("D", True, "Duplicate provenance/delivery stores"),
        ("E", True, "Shadow event persistence"),
        ("F", True, "Event typing seams"),
        ("G", True, "Operational coupling in trace path"),
        ("H", True, "Sanctioned root → event delivery wiring L3"),
        ("I", True, "Install defaults in event plane"),
        ("J", True, "Trace extensibility"),
    ),
    "agent_distribution": _aj(
        ("A", True, "Agent assembly construction"),
        ("B", True, "Shadow registry names in agents/"),
        ("C", True, "Implicit registry defaults"),
        ("D", True, "Parallel agent registries"),
        ("E", False, "Not platform persistence owner"),
        ("F", True, "Agent distribution typing"),
        ("G", False, "Not observability owner"),
        ("H", True, "Hosted profile → registry L3"),
        ("I", True, "Install defaults in harness distribution"),
        ("J", True, "Agent extensibility boundary"),
    ),
    "rag_memory": _aj(
        ("A", True, "RAG/memory materialization"),
        ("B", True, "Vector backend literals"),
        ("C", True, "Implicit vector defaults (51 NYQ noted)"),
        ("D", True, "RAG-local backend registry risk"),
        ("E", True, "Shadow vector stores"),
        ("F", True, "RAG typing seams"),
        ("G", True, "Operational coupling in RAG path"),
        ("H", True, "App wiring → RAG → Integration L3"),
        ("I", True, "Install defaults in RAG"),
        ("J", True, "Extensibility vs EBH-7 NYQ scope"),
    ),
    "runtime_hosting": _aj(
        ("A", True, "Hosting adapter construction"),
        ("B", True, "Hosting provider literals"),
        ("C", True, "Implicit host profile defaults"),
        ("D", True, "Shadow host profile registries"),
        ("E", False, "Not persistence shadow primary"),
        ("F", True, "Hosting typing seams"),
        ("G", True, "Operational coupling in hosting"),
        ("H", True, "Host profile → bootstrap L3"),
        ("I", True, "Env shortcut host selection"),
        ("J", True, "Hosting replaceability"),
    ),
    "tools_skills_capabilities": _aj(
        ("A", True, "Tool provider construction"),
        ("B", True, "Tool backend literals"),
        ("C", True, "Implicit tool defaults"),
        ("D", True, "Duplicate tool dispatchers"),
        ("E", False, "Not persistence owner"),
        ("F", True, "Tool SPI typing"),
        ("G", True, "Operational coupling in tools plane"),
        ("H", True, "Wiring → tool registry → SPI L3"),
        ("I", True, "Install tool defaults"),
        ("J", True, "Tool backend swap requires core change check"),
    ),
    "background_workers_infra_adapters": _aj(
        ("A", True, "Worker/infra adapter construction"),
        ("B", True, "Queue/backend literals"),
        ("C", True, "Implicit queue defaults"),
        ("D", True, "Parallel worker registries"),
        ("E", True, "Shadow worker persistence"),
        ("F", True, "Worker typing seams"),
        ("G", True, "Operational coupling in workers"),
        ("H", True, "Worker host wiring L3 (if production-effective)"),
        ("I", True, "Silent queue fallback risk"),
        ("J", True, "Worker extensibility"),
    ),
}


def _layer_paths(layer: str) -> tuple[str, ...]:
    mapping: dict[str, tuple[str, ...]] = {
        "contracts": _glob_paths("intergrax/contracts") + _glob_paths("intergrax/integrations/contracts"),
        "semantic_core": _glob_paths("intergrax/runtime/nexus"),
        "application_services": _glob_paths("intergrax/applications/_shared"),
        "composition_roots": tuple(sorted(discover_composition_root_paths())),
        "governance": _glob_paths("intergrax/runtime/governance"),
        "execution": _glob_paths("intergrax/runtime/execution"),
        "integrations": _glob_paths("intergrax/integrations"),
        "state_persistence": _glob_paths("intergrax/integrations/providers")
        + tuple(sorted(discover_application_wiring_paths()))[:20],
        "reliability_recovery": _glob_paths("intergrax/runtime/resilience"),
        "observability_diagnostics": _glob_paths("intergrax/runtime/observability"),
        "events_traceability": _glob_paths("intergrax/applications/_shared")
        + _glob_paths("tests/qualification/trace_x"),
        "agent_distribution": _glob_paths("agents"),
        "rag_memory": _glob_paths("intergrax/rag") + _glob_paths("intergrax/memory"),
        "runtime_hosting": _glob_paths("intergrax/runtime/hosting") + _glob_paths("intergrax/runtime/bootstrap"),
        "tools_skills_capabilities": _glob_paths("intergrax/tools"),
        "background_workers_infra_adapters": _glob_paths("intergrax", "*worker*")
        + _glob_paths("intergrax/integrations/providers", "*background*"),
    }
    paths = mapping.get(layer, ())
    return paths if paths else (f"layer-inventory:{layer} — see EBH_6_ENTERPRISE_SURFACE_INVENTORY.md",)


def _l3_narrative(layer: str) -> str:
    roots = ", ".join(sorted(discover_sanctioned_production_composition_roots())[:3])
    bundles = len(discover_integration_bundle_paths())
    wiring = len(discover_application_wiring_paths())
    templates = {
        "contracts": (
            f"Sanctioned roots ({roots}) import Tier-0 contracts only; no direct provider SDK in contract modules."
        ),
        "semantic_core": ("Sanctioned roots → nexus host composition → nexus_loop; profile/catalog indirection only."),
        "application_services": (
            f"App *_wiring.py ({wiring} paths) → intergrax/applications/_shared helpers → contracts."
        ),
        "composition_roots": (
            f"TRACE-X sanctioned sites → composition hints ({len(discover_composition_root_paths())} paths)."
        ),
        "governance": "Production wiring attaches governance middleware; fail-closed samples in qual gates.",
        "execution": "Sanctioned execution host → ExecutionRequest → canonical ExecutionRuntime path.",
        "integrations": f"Wiring → Integration factory → {bundles} bundle.py adapter boundaries.",
        "state_persistence": "Integration profile → persistence contract → provider; SQLite annex 38 paths @ HEAD.",
        "reliability_recovery": "Execution host → recovery plane; single recovery owner (guard matrix #5).",
        "observability_diagnostics": "Non-authoritative OTLP export via integration profile / qualified rows.",
        "events_traceability": "Sanctioned root → runtime_event_delivery_wiring → TRACE-X registry spine.",
        "agent_distribution": "Tier-2 agents/ registry; tier import gate blocks apps → private agent internals.",
        "rag_memory": "App wiring → RAG services → Integration materialization (no RAG-local backend registry).",
        "runtime_hosting": "Host profile → runtime bootstrap via configuration contracts.",
        "tools_skills_capabilities": "tools/registry/wiring.py → tool registry → SPI (plugin guard #11).",
        "background_workers_infra_adapters": (
            "Worker composition via integration contracts; ingress census in qual annex."
        ),
    }
    return templates.get(layer, f"L3 trace documented for `{layer}` @ sanctioned roots.")


def _sys_inv(layer: str) -> tuple[str, ...]:
    table: dict[str, tuple[str, ...]] = {
        "contracts": ("SYS-INV-BND-01", "SYS-INV-CTR-01", "SYS-INV-VBN-01"),
        "semantic_core": ("SYS-INV-OWN-01", "SYS-INV-VBN-02", "SYS-INV-TYP-01"),
        "application_services": ("SYS-INV-BND-02", "SYS-INV-CTR-02", "SYS-INV-VBN-01"),
        "composition_roots": ("SYS-INV-OWN-02", "SYS-INV-CFG-01", "SYS-INV-VBN-03"),
        "governance": ("SYS-INV-GOV-01", "SYS-INV-SEC-01", "SYS-INV-VBN-04"),
        "execution": ("SYS-INV-EXE-01", "SYS-INV-GOV-02", "SYS-INV-VBN-01"),
        "integrations": ("SYS-INV-VBN-01", "SYS-INV-RPL-01", "SYS-INV-PLG-01"),
        "state_persistence": ("SYS-INV-STA-01", "SYS-INV-RPL-02", "FRZ-STA"),
        "reliability_recovery": ("SYS-INV-REC-01", "SYS-INV-REL-01", "SYS-INV-VBN-02"),
        "observability_diagnostics": ("SYS-INV-OBS-01", "SYS-INV-VBN-05", "FRZ-OBS"),
        "events_traceability": ("SYS-INV-TRC-01", "SYS-INV-VBN-03", "FRZ-TRC"),
        "agent_distribution": ("SYS-INV-BND-03", "SYS-INV-HRN-01", "FRZ-BND"),
        "rag_memory": ("SYS-INV-RPL-03", "SYS-INV-VBN-01", "FRZ-RPL"),
        "runtime_hosting": ("SYS-INV-CFG-02", "SYS-INV-VBN-02", "FRZ-CFG"),
        "tools_skills_capabilities": ("SYS-INV-PLG-02", "SYS-INV-VBN-04", "FRZ-PLG"),
        "background_workers_infra_adapters": ("SYS-INV-GOV-03", "SYS-INV-VBN-06", "FRZ-SEC"),
    }
    return table.get(layer, ("SYS-INV-VBN-01",))


def _gates(layer: str) -> tuple[str, ...]:
    reg_gate = (
        "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero"
    )
    common = (reg_gate,)
    mech = "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py"
    extra: dict[str, tuple[str, ...]] = {
        "contracts": ("tests/unit/contracts/",),
        "semantic_core": (
            "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py"
            "::test_ebh6_tracked_debt_findings_ratified",
        ),
        "composition_roots": (f"{mech}::test_ebh6_exit_sanctioned_composition_roots_non_empty",),
        "integrations": (
            "tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py::test_ebh5_exit_inventory_complete",
        ),
        "state_persistence": (f"{mech}::test_ebh6_exit_sqlite_annex_accountability",),
        "execution": ("tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py",),
        "agent_distribution": (
            "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py"
            "::test_tier0_must_not_import_agents_or_applications",
        ),
    }
    return common + extra.get(layer, ())


def layer_gate_test_ids(layer: str) -> tuple[str, ...]:
    return _gates(layer)


def layer_aj_applicability(layer: str) -> tuple[AjApplicability, ...]:
    return _LAYER_AJ[layer]


@lru_cache(maxsize=1)
def all_layer_evidence() -> tuple[LayerL9Evidence, ...]:
    from tests.qualification.ebh_6._l3_reachability_evidence import (
        l3_records_by_layer,
        mechanism_ids_for_layer,
    )

    rows: list[LayerL9Evidence] = []
    for idx, layer in enumerate(EBH6_AUDIT_LAYERS, start=1):
        paths = _layer_paths(layer)
        l3_records = l3_records_by_layer(layer)
        mech_ids = mechanism_ids_for_layer(layer)
        prod_samples = tuple(r.entrypoint for r in l3_records if r.production_effective)
        if not prod_samples:
            prod_samples = tuple(r.entrypoint for r in l3_records)
        l3_index = (
            f"{len(l3_records)} L3 mechanisms ({', '.join(mech_ids[:3])}"
            f"{'…' if len(mech_ids) > 3 else ''}) — "
            "annex tests/qualification/ebh_6/_l3_reachability_evidence.py"
        )
        rows.append(
            LayerL9Evidence(
                finding_id=f"EBH6-LAYER-{idx:02d}",
                layer=layer,
                closed_world_paths=paths,
                aj_applicability=_LAYER_AJ[layer],
                sys_inv_refs=_sys_inv(layer),
                l3_narrative=l3_index,
                production_effective_samples=prod_samples[:3] if len(prod_samples) > 3 else prod_samples,
                gate_test_ids=_gates(layer),
            ),
        )
    return tuple(rows)


def layer_evidence_by_id() -> dict[str, LayerL9Evidence]:
    return {row.finding_id: row for row in all_layer_evidence()}


def assert_l9_layer_evidence_complete() -> None:
    from tests.qualification.ebh_6._l3_reachability_evidence import assert_l3_reachability_complete

    assert_l3_reachability_complete()
    evidence = all_layer_evidence()
    assert len(evidence) == 16
    for row in evidence:
        assert row.closed_world_paths, row.finding_id
        assert len(row.aj_applicability) == 10, row.finding_id
        for aj in row.aj_applicability:
            assert aj.detection_class in _AJ_CLASSES
            if not aj.applicable:
                assert aj.rationale.strip(), row.finding_id
        assert row.l3_narrative.strip()
        assert "L3 mechanisms" in row.l3_narrative
        assert len(row.production_effective_samples) >= 1
        assert row.sys_inv_refs
