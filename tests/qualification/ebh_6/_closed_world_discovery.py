# © Artur Czarnecki. All rights reserved.

"""Closed-world discovery for EBH-6 — extends EBH-5 composition/sqlite patterns."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

from tests.qualification.ebh_5._closed_world_discovery import (
    COMPOSITION_ROOT_HINTS,
    discover_composition_root_paths,
    discover_integration_category_enum_size,
    discover_sqlite_connect_paths_under_intergrax,
)
from tests.qualification.trace_x._trace_x_p4_production_composition_registry import (
    PRODUCTION_COMPOSITION_SITE_REGISTRY,
    ProductionCompositionSiteClassification,
)

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

# Roadmap §3.0.8 universal hardcoding audit layers (mandatory current HEAD).
EBH6_AUDIT_LAYERS: Final[tuple[str, ...]] = (
    "contracts",
    "semantic_core",
    "application_services",
    "composition_roots",
    "governance",
    "execution",
    "integrations",
    "state_persistence",
    "reliability_recovery",
    "observability_diagnostics",
    "events_traceability",
    "agent_distribution",
    "rag_memory",
    "runtime_hosting",
    "tools_skills_capabilities",
    "background_workers_infra_adapters",
)

_APPLICATION_WIRING_GLOB: Final[str] = "applications/**/*_wiring.py"


@lru_cache(maxsize=1)
def discover_sanctioned_production_composition_roots() -> frozenset[str]:
    return frozenset(
        row.path
        for row in PRODUCTION_COMPOSITION_SITE_REGISTRY
        if row.classification
        in (
            ProductionCompositionSiteClassification.SANCTIONED_PRODUCTION_ROOT,
            ProductionCompositionSiteClassification.SANCTIONED_PRODUCTION_ROUTER,
        )
    )


@lru_cache(maxsize=1)
def discover_application_wiring_paths() -> frozenset[str]:
    paths: set[str] = set()
    for path in (_REPO_ROOT / "applications").rglob("*_wiring.py"):
        if path.is_file():
            paths.add(path.relative_to(_REPO_ROOT).as_posix())
    return frozenset(paths)


@lru_cache(maxsize=1)
def discover_integration_bundle_paths() -> frozenset[str]:
    root = _REPO_ROOT / "intergrax" / "integrations" / "providers"
    paths: set[str] = set()
    for bundle in root.rglob("bundle.py"):
        if bundle.is_file():
            paths.add(bundle.relative_to(_REPO_ROOT).as_posix())
    return frozenset(paths)


def assert_ebh6_closed_world_surfaces_non_empty() -> None:
    assert discover_composition_root_paths(), "composition root hints must resolve"
    assert discover_sanctioned_production_composition_roots(), "TRACE-X sanctioned roots required"
    assert discover_application_wiring_paths(), "application wiring paths required"
    assert discover_integration_bundle_paths(), "integration bundle.py paths required"
    assert len(EBH6_AUDIT_LAYERS) == 16, "audit layer catalog must remain closed-world complete"


def assert_ebh6_sqlite_annex_accountability() -> None:
    from tests.qualification.ebh_5._sqlite_touchpoint_registry import (
        assert_closed_world_sqlite_accountability,
    )

    assert_closed_world_sqlite_accountability()


# Batch B2 — T-LAY-* package → roadmap layer mapping (PL-02).
T_LAY_PACKAGE_LAYERS: Final[dict[str, tuple[str, ...]]] = {
    "T-LAY-CTR-SMC": ("contracts", "semantic_core"),
    "T-LAY-APP-CMP": ("application_services", "composition_roots"),
    "T-LAY-GOV-EXE": ("governance", "execution"),
    "T-LAY-INT-STA": ("integrations", "state_persistence"),
    "T-LAY-REC-OBS": ("reliability_recovery", "observability_diagnostics"),
    "T-LAY-TRC-WRK": ("events_traceability", "background_workers_infra_adapters"),
    "T-LAY-AGD-RAG": ("agent_distribution", "rag_memory"),
    "T-LAY-HST-TLC": ("runtime_hosting", "tools_skills_capabilities"),
}


@lru_cache(maxsize=1)
def discover_contracts_surface_paths() -> frozenset[str]:
    root = _REPO_ROOT / "intergrax" / "contracts"
    paths: set[str] = set()
    if root.is_dir():
        for path in root.rglob("*.py"):
            if path.is_file():
                paths.add(path.relative_to(_REPO_ROOT).as_posix())
    ic = _REPO_ROOT / "intergrax" / "integrations" / "contracts"
    if ic.is_dir():
        for path in ic.rglob("*.py"):
            if path.is_file():
                paths.add(path.relative_to(_REPO_ROOT).as_posix())
    return frozenset(paths)


@lru_cache(maxsize=1)
def discover_semantic_core_paths() -> frozenset[str]:
    root = _REPO_ROOT / "intergrax" / "runtime" / "nexus"
    paths: set[str] = set()
    if root.is_dir():
        for path in root.rglob("*.py"):
            if path.is_file():
                paths.add(path.relative_to(_REPO_ROOT).as_posix())
    return frozenset(paths)


@lru_cache(maxsize=1)
def discover_governance_execution_paths() -> frozenset[str]:
    paths: set[str] = set()
    for sub in ("governance", "execution"):
        root = _REPO_ROOT / "intergrax" / "runtime" / sub
        if root.is_dir():
            for path in root.rglob("*.py"):
                if path.is_file():
                    paths.add(path.relative_to(_REPO_ROOT).as_posix())
    return frozenset(paths)


@lru_cache(maxsize=1)
def discover_resilience_observability_paths() -> frozenset[str]:
    paths: set[str] = set()
    for sub in ("resilience", "observability", "diagnostics"):
        root = _REPO_ROOT / "intergrax" / "runtime" / sub
        if root.is_dir():
            for path in root.rglob("*.py"):
                if path.is_file():
                    paths.add(path.relative_to(_REPO_ROOT).as_posix())
    return frozenset(paths)


@lru_cache(maxsize=1)
def discover_rag_graph_bootstrap_paths() -> frozenset[str]:
    path = _REPO_ROOT / "intergrax" / "rag" / "graph" / "bootstrap" / "graph_store_bootstrap.py"
    if path.is_file():
        return frozenset((path.relative_to(_REPO_ROOT).as_posix(),))
    return frozenset()


@lru_cache(maxsize=1)
def discover_agent_distribution_paths() -> frozenset[str]:
    root = _REPO_ROOT / "agents"
    paths: set[str] = set()
    if root.is_dir():
        for path in root.rglob("*.py"):
            if path.is_file():
                paths.add(path.relative_to(_REPO_ROOT).as_posix())
    ad = _REPO_ROOT / "intergrax" / "agent_distribution"
    if ad.is_dir():
        for path in ad.rglob("*.py"):
            if path.is_file():
                paths.add(path.relative_to(_REPO_ROOT).as_posix())
    return frozenset(paths)


@lru_cache(maxsize=1)
def discover_tools_skills_runtime_paths() -> frozenset[str]:
    paths: set[str] = set()
    tools = _REPO_ROOT / "intergrax" / "tools"
    if tools.is_dir():
        for path in tools.rglob("*.py"):
            if path.is_file():
                paths.add(path.relative_to(_REPO_ROOT).as_posix())
    skills = _REPO_ROOT / "intergrax" / "runtime" / "skills"
    if skills.is_dir():
        for path in skills.rglob("*.py"):
            if path.is_file():
                paths.add(path.relative_to(_REPO_ROOT).as_posix())
    return frozenset(paths)


@lru_cache(maxsize=1)
def discover_runtime_hosting_paths() -> frozenset[str]:
    paths: set[str] = set()
    for prefix in (
        "intergrax/runtime/hosting",
        "intergrax/runtime/bootstrap",
        "intergrax/contracts/runtime",
        "intergrax/applications/_shared",
    ):
        root = _REPO_ROOT / prefix
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if path.is_file():
                rel = path.relative_to(_REPO_ROOT).as_posix()
                if prefix.endswith("_shared") and "runtime_config" not in rel and "host" not in rel.lower():
                    if "environment_wiring" not in rel and "context_wiring" not in rel:
                        continue
                paths.add(rel)
    plugins = _REPO_ROOT / "intergrax" / "runtime" / "plugins" / "bootstrap.py"
    if plugins.is_file():
        paths.add(plugins.relative_to(_REPO_ROOT).as_posix())
    bridge = _REPO_ROOT / "intergrax" / "applications" / "_shared" / "runtime_config_bridge.py"
    if bridge.is_file():
        paths.add(bridge.relative_to(_REPO_ROOT).as_posix())
    return frozenset(paths)


def assert_ebh6_b2_t_lay_package_surfaces() -> None:
    """B2 exit — each T-LAY package must resolve a non-empty closed-world census @ HEAD."""
    assert discover_contracts_surface_paths(), "T-LAY-CTR-SMC contracts surface"
    assert discover_semantic_core_paths(), "T-LAY-CTR-SMC semantic_core surface"
    assert discover_sanctioned_production_composition_roots(), "T-LAY-APP-CMP TRACE-X roots"
    assert discover_application_wiring_paths(), "T-LAY-APP-CMP wiring census"
    assert discover_composition_root_paths(), "T-LAY-APP-CMP composition hints"
    assert discover_governance_execution_paths(), "T-LAY-GOV-EXE paths"
    assert discover_integration_bundle_paths(), "T-LAY-INT-STA bundles"
    assert discover_resilience_observability_paths(), "T-LAY-REC-OBS paths"
    assert discover_rag_graph_bootstrap_paths(), "T-LAY-AGD-RAG graph_store_bootstrap"
    assert discover_agent_distribution_paths(), "T-LAY-AGD-RAG agent distribution"
    assert discover_runtime_hosting_paths(), "T-LAY-HST-TLC hosting"
    assert discover_tools_skills_runtime_paths(), "T-LAY-HST-TLC tools/skills"
    covered = frozenset(layer for layers in T_LAY_PACKAGE_LAYERS.values() for layer in layers)
    assert covered == frozenset(EBH6_AUDIT_LAYERS), "T-LAY packages must cover all 16 layers"


def assert_ebh6_b2_layer_l3_inventory() -> None:
    from tests.qualification.ebh_6._layer_evidence import assert_l9_layer_evidence_complete

    assert_l9_layer_evidence_complete()


__all__ = [
    "COMPOSITION_ROOT_HINTS",
    "EBH6_AUDIT_LAYERS",
    "T_LAY_PACKAGE_LAYERS",
    "assert_ebh6_b2_layer_l3_inventory",
    "assert_ebh6_b2_t_lay_package_surfaces",
    "assert_ebh6_closed_world_surfaces_non_empty",
    "assert_ebh6_sqlite_annex_accountability",
    "discover_agent_distribution_paths",
    "discover_contracts_surface_paths",
    "discover_governance_execution_paths",
    "discover_rag_graph_bootstrap_paths",
    "discover_resilience_observability_paths",
    "discover_runtime_hosting_paths",
    "discover_semantic_core_paths",
    "discover_tools_skills_runtime_paths",
    "discover_application_wiring_paths",
    "discover_composition_root_paths",
    "discover_integration_bundle_paths",
    "discover_integration_category_enum_size",
    "discover_sanctioned_production_composition_roots",
    "discover_sqlite_connect_paths_under_intergrax",
]
