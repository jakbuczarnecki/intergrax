# © Artur Czarnecki. All rights reserved.

"""COMPAT-X mechanical closed-world discovery (current HEAD)."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.compat_x._compat_x_closed_world import (
    build_closed_world_report,
    build_closed_world_report_with_extra_candidates,
    discover_candidates_from_source,
)
from tests.qualification.compat_x._compat_x_classifiers import classify_migration_module, classify_shim_module
from tests.qualification.compat_x._compat_x_shim_authority import build_shim_authority_scope_reconciliation
from tests.qualification.compat_x._compat_x_types import DiscoveredCompatSurface, DiscoveryCandidate, ShimAuthorityScopeReconciliation

__all__ = [
    "DiscoveredCompatSurface",
    "discover_all_compat_surfaces",
    "discover_all_compat_surface_ids",
    "discover_migration_mechanism_surfaces",
    "discover_shim_surfaces",
    "classify_synthetic_unclassified_surface",
    "discover_class_field_versions_from_source",
    "classify_migration_from_source",
    "classify_shim_from_source",
    "parity_gate_failed_if_surface_omitted",
    "closed_world_parity_holds",
    "discover_candidates_from_source",
    "build_shim_authority_scope_reconciliation",
    "ShimAuthorityScopeReconciliation",
]


@lru_cache(maxsize=1)
def discover_all_compat_surfaces() -> frozenset[DiscoveredCompatSurface]:
    report = build_closed_world_report()
    return frozenset(report.semantic_surfaces)


def discover_all_compat_surface_ids() -> frozenset[str]:
    return frozenset(surface.surface_id for surface in discover_all_compat_surfaces())


@lru_cache(maxsize=1)
def discover_migration_mechanism_surfaces() -> frozenset[DiscoveredCompatSurface]:
    return frozenset(
        s for s in discover_all_compat_surfaces() if s.discovery_kind == "migration.mechanism"
    )


@lru_cache(maxsize=1)
def discover_shim_surfaces() -> frozenset[DiscoveredCompatSurface]:
    return frozenset(s for s in discover_all_compat_surfaces() if s.discovery_kind == "compat.shim")


def classify_synthetic_unclassified_surface(surface_id: str) -> str:
    report = build_closed_world_report()
    if surface_id in report.unclassified_candidate_ids:
        return "UNCLASSIFIED"
    known = frozenset(s.surface_id for s in report.semantic_surfaces)
    if surface_id in known:
        return "CLASSIFIED"
    return "UNCLASSIFIED"


def discover_class_field_versions_from_source(module_path: str, source: str) -> frozenset[DiscoveryCandidate]:
    return frozenset(
        c for c in discover_candidates_from_source(module_path, source) if c.discovery_kind == "class.field.version"
    )


def classify_migration_from_source(module_path: str, source: str) -> str:
    return classify_migration_module(module_path, source).value


def classify_shim_from_source(module_path: str, source: str) -> str:
    return classify_shim_module(module_path, source).value


def parity_gate_failed_if_surface_omitted(omitted_semantic_identity: str) -> bool:
    """Return True when a discovered semantic identity is missing from classification."""
    report = build_closed_world_report()
    present = frozenset(s.semantic_identity for s in report.semantic_surfaces)
    return omitted_semantic_identity not in present


def closed_world_parity_holds(extra: tuple[DiscoveryCandidate, ...] = ()) -> bool:
    if extra:
        report = build_closed_world_report_with_extra_candidates(extra)
    else:
        report = build_closed_world_report()
    if report.unclassified_candidate_ids:
        return False
    return len(report.resolved_candidate_ids) == len(report.raw_candidates) and len(report.raw_candidates) > 0
