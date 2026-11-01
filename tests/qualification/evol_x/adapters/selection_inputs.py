# © Artur Czarnecki. All rights reserved.

"""Assemble ``SelectionInputs`` from the real QUAL-X catalog, EVOL VL3 manifest and exclusions (T1/T2/T3)."""

from __future__ import annotations

from dataclasses import replace

from testing_support.change_safety.contracts.impact import ImpactAssessmentResult
from testing_support.change_safety.data.loaders import load_evol_x_data_document
from testing_support.change_safety.selection.ci_manifest import load_pinned_manifest
from testing_support.change_safety.selection.policy import load_conservative_exclusions
from testing_support.change_safety.selection.selector import SelectionInputs
from tests.qualification.evol_x.adapters.qual_x import QualXSuiteCatalogAdapter


def build_selection_inputs(impact: ImpactAssessmentResult | None, **overrides: object) -> SelectionInputs:
    catalog = QualXSuiteCatalogAdapter()
    base = SelectionInputs(
        impact=impact,
        catalog_node_ids=catalog.collectable_node_ids(),
        mandatory_node_ids=catalog.mandatory_min_node_ids(),
        pop_a_node_ids=catalog.pop_a_collectable_node_ids(),
        evol_manifest_node_ids=frozenset(load_pinned_manifest()["pytest_node_ids"]),
        adjacent_node_ids=frozenset(load_evol_x_data_document("evol_adjacent_regression_nodes.v1.json")["node_ids"]),
        exclusions=load_conservative_exclusions(),
    )
    return replace(base, **overrides) if overrides else base  # type: ignore[arg-type]
