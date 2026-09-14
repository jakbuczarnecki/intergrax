# © Artur Czarnecki. All rights reserved.

"""Catalog structural validation."""

from __future__ import annotations

from testing_support.execution_qualification.catalog.contracts import (
    CompiledCatalogProfile,
)
from testing_support.execution_qualification.catalog.profile_builders import (
    PROFILE_BUILDERS,
)


def validate_catalog_profile_ids() -> None:
    seen: set[str] = set()
    for profile_id in PROFILE_BUILDERS:
        if profile_id in seen:
            raise ValueError(f"duplicate profile_id: {profile_id}")
        seen.add(profile_id)


def validate_compiled_profile(profile: CompiledCatalogProfile) -> None:
    suite_ids = {suite.suite_id for suite in profile.graph.run_manifest.suites}
    gate_ids = {gate.gate_id for gate in profile.graph.gates}
    overlap = suite_ids & gate_ids
    if overlap:
        raise ValueError(f"suite/gate id collision: {sorted(overlap)}")
    if len(profile.plan.leaf_suite_ids) != len(set(profile.plan.leaf_suite_ids)):
        raise ValueError("duplicate leaf suite_id in compiled plan")
