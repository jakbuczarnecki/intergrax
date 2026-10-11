# © Artur Czarnecki. All rights reserved.

"""Mechanical persisted durable-boundary discovery and closed-world parity (R2-P0-R1)."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.compat_x._compat_x_persisted_classifier import (
    agent_distribution_table_parity_holds,
    classify_durable_candidates,
    classify_is_unclassified,
    mechanical_agent_distribution_table_family_map,
    semantic_family_ids_from_classifications,
)
from tests.qualification.compat_x._compat_x_persisted_mechanical_scan import (
    discover_all_production_durable_candidates,
    discover_durable_candidates_in_source,
    discovery_root_paths,
)
from tests.qualification.compat_x._compat_x_persisted_types import (
    DurableBoundaryCandidate,
    PersistedMechanicalDiscoveryReport,
)


def _build_report(
    raw: tuple[DurableBoundaryCandidate, ...],
) -> PersistedMechanicalDiscoveryReport:
    classifications = classify_durable_candidates(raw)
    unclassified = frozenset(
        c.candidate.candidate_id for c in classifications if classify_is_unclassified(c)
    )
    semantic_ids = semantic_family_ids_from_classifications(classifications)
    from tests.qualification.compat_x._compat_x_persisted_types import PersistedCandidateDisposition

    persisted_members = sum(
        1
        for c in classifications
        if c.disposition == PersistedCandidateDisposition.PERSISTED_FAMILY_MEMBER
    )
    exclusions = sum(1 for c in classifications if c.exclusion_reason is not None)
    return PersistedMechanicalDiscoveryReport(
        discovery_roots=discovery_root_paths(),
        raw_candidates=raw,
        classifications=classifications,
        unclassified_candidate_ids=unclassified,
        semantic_persisted_family_ids=semantic_ids,
        exclusion_count=exclusions,
        persisted_member_count=persisted_members,
    )


@lru_cache(maxsize=1)
def build_persisted_mechanical_discovery_report() -> PersistedMechanicalDiscoveryReport:
    return _build_report(discover_all_production_durable_candidates())


def build_persisted_mechanical_discovery_report_with_extra(
    extra_source: tuple[tuple[str, str], ...],
) -> PersistedMechanicalDiscoveryReport:
    base = list(discover_all_production_durable_candidates())
    seen = {c.candidate_id for c in base}
    for module_path, source in extra_source:
        for candidate in discover_durable_candidates_in_source(module_path, source):
            if candidate.candidate_id not in seen:
                seen.add(candidate.candidate_id)
                base.append(candidate)
    return _build_report(tuple(base))


def inventory_family_ids() -> frozenset[str]:
    from tests.qualification.compat_x._compat_x_persisted_inventory import (
        PERSISTED_SCHEMA_FAMILIES,
    )

    return frozenset(f.family_id for f in PERSISTED_SCHEMA_FAMILIES)


@lru_cache(maxsize=1)
def mechanical_closed_world_family_ids() -> frozenset[str]:
    """Semantic families evidenced by mechanical discovery + classification (not inventory injection)."""
    report = build_persisted_mechanical_discovery_report()
    return report.semantic_persisted_family_ids


def mechanical_agent_distribution_family_ids() -> frozenset[str]:
    table_map = mechanical_agent_distribution_table_family_map()
    ids = set(table_map.values())
    ids.add("agent_distribution.sqlite_container")
    ids.add("agent_distribution.package_agent_contract_authority_record")
    return frozenset(ids)


def persisted_closed_world_parity_holds(
    extra_source: tuple[tuple[str, str], ...] = (),
) -> bool:
    if extra_source:
        report = build_persisted_mechanical_discovery_report_with_extra(extra_source)
    else:
        report = build_persisted_mechanical_discovery_report()
    if report.unclassified_candidate_ids:
        return False
    if len(report.raw_candidates) != report.persisted_member_count + report.exclusion_count:
        return False
    if not agent_distribution_table_parity_holds():
        return False
    inventory = inventory_family_ids()
    if report.semantic_persisted_family_ids != inventory:
        return False
    return True
