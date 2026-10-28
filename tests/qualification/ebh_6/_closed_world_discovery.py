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


__all__ = [
    "COMPOSITION_ROOT_HINTS",
    "EBH6_AUDIT_LAYERS",
    "assert_ebh6_closed_world_surfaces_non_empty",
    "assert_ebh6_sqlite_annex_accountability",
    "discover_application_wiring_paths",
    "discover_composition_root_paths",
    "discover_integration_bundle_paths",
    "discover_integration_category_enum_size",
    "discover_sanctioned_production_composition_roots",
    "discover_sqlite_connect_paths_under_intergrax",
]
