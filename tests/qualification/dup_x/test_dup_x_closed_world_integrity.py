# © Artur Czarnecki. All rights reserved.

"""DUP-X closed-world integrity gates (Wave A exit)."""

from __future__ import annotations

import pytest
from tests.qualification.dup_x._closed_world_inventory import (
    DUP_X_RESPONSIBILITY_FAMILIES,
    assert_dup_x_closed_world_integrity,
    class_coverage_playbook,
    discover_normative_composition_root_union,
    discover_production_tools_base_imports,
    discover_registry_class_census,
    discover_resolver_class_census,
)
from tests.qualification.dup_x._findings_register import (
    DUP_CLASSES,
    DUP_X_CANDIDATE_RECORDS,
    count_unclassified_candidates,
    retained_parallel_without_dossier,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_dup_x_closed_world_integrity() -> None:
    assert_dup_x_closed_world_integrity()


def test_dup_x_fifteen_responsibility_families() -> None:
    ids = {row.family_id for row in DUP_X_RESPONSIBILITY_FAMILIES}
    assert len(ids) == 15


def test_dup_x_fourteen_dup_classes_catalog() -> None:
    assert class_coverage_playbook().keys() == set(DUP_CLASSES)


def test_dup_x_composition_root_union_non_empty() -> None:
    union = discover_normative_composition_root_union()
    assert "intergrax/integrations/registry/factory.py" in union
    assert any("wiring.py" in path for path in union)


def test_dup_x_mechanical_registry_census_non_empty() -> None:
    assert len(discover_registry_class_census()) >= 5
    assert len(discover_resolver_class_census()) >= 5


def test_dup_x_candidates_fifteen_rows_no_orphans() -> None:
    assert len(DUP_X_CANDIDATE_RECORDS) == 15
    assert count_unclassified_candidates() == 0


def test_dup_x_retained_parallel_delta_dossiers_present() -> None:
    assert not retained_parallel_without_dossier()


def test_dup_x_tools_base_no_production_importers() -> None:
    assert not discover_production_tools_base_imports()
