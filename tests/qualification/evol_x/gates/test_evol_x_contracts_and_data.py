# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult
from testing_support.change_safety.contracts.impact import ImpactAssessmentResult
from testing_support.change_safety.contracts.selection import SelectionDecision
from testing_support.change_safety.data.loaders import load_evol_x_data_document

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_data_documents_have_source_authority() -> None:
    for name in (
        "ruleset.v1.json",
        "owner_rules.v1.json",
        "frozen_core_registry.v1.json",
        "selection_policy.v1.json",
    ):
        doc = load_evol_x_data_document(name)
        assert doc["source_authority"].startswith("docs/")


def test_evol_x_contract_models_roundtrip() -> None:
    sample = ImpactAssessmentResult(
        source_commit_sha="0" * 40,
        graph_digest="a" * 64,
        inventory_digest="b" * 64,
        ruleset_version="ruleset.v1",
        status="COMPLETE",
    )
    restored = ImpactAssessmentResult.model_validate(sample.model_dump())
    assert restored == sample
    sel = SelectionDecision(
        decision_id="d1",
        status="SELECTED",
        graph_digest="a" * 64,
        inventory_digest="b" * 64,
        ruleset_version="ruleset.v1",
    )
    assert SelectionDecision.model_validate(sel.model_dump()) == sel


def test_evol_x_contract_diff_incompatible_on_required_removal() -> None:
    from testing_support.change_safety.assess.contract_diff import diff_json_field_schemas

    before = {"properties": {"x": {"type": "string"}}, "required": ["x"]}
    after = {"properties": {}, "required": []}
    assert diff_json_field_schemas(before, after) is CompatibilityAtChangeResult.INCOMPATIBLE
