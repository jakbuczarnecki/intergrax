# © Artur Czarnecki. All rights reserved.

"""EE-B4-C — incident taxonomy catalog gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.operations.incident_taxonomy import (
    EXECUTION_ENGINE_INCIDENT_CATALOG,
    IncidentCategoryId,
    REQUIRED_RUNBOOK_IDS,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_MODEL = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_PRODUCTION_OPERATIONS_INCIDENT_MODEL.md"
)


def test_ee_b4_c_incident_catalog_has_fourteen_categories() -> None:
    assert len(EXECUTION_ENGINE_INCIDENT_CATALOG) == 14
    ids = {entry.incident_id for entry in EXECUTION_ENGINE_INCIDENT_CATALOG}
    assert ids == set(IncidentCategoryId)


def test_ee_b4_c_each_incident_maps_to_listed_runbook() -> None:
    allowed = set(REQUIRED_RUNBOOK_IDS)
    for entry in EXECUTION_ENGINE_INCIDENT_CATALOG:
        assert entry.primary_runbook_id in allowed


def test_ee_b4_c_model_lists_all_incident_ids() -> None:
    text = _MODEL.read_text(encoding="utf-8")
    for incident in IncidentCategoryId:
        assert incident.value in text
