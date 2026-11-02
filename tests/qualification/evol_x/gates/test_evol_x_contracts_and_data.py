# © Artur Czarnecki. All rights reserved.

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from testing_support.change_safety.contracts.artifacts import (
    ArtifactDigests,
    EvolJunitOutcomes,
    EvolMetricsArtifact,
    ExperimentVerdict,
    FrzEvidence,
    P03Manifest,
    TestExecutionManifest,
)
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
        "exclusion_register.v1.json",
        "evol_vl3_execution_manifest.v1.json",
        "critical_consumer_register.v1.json",
    ):
        doc = load_evol_x_data_document(name)
        assert doc["source_authority"].startswith("docs/")


def test_evol_x_data_loader_rejects_missing_source_authority(tmp_path: Path) -> None:
    (tmp_path / "x.v1.json").write_text(json.dumps({"schema_version": "x.v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="source_authority"):
        load_evol_x_data_document("x.v1.json", data_dir=tmp_path)


def test_evol_x_data_loader_rejects_invalid_owner_row(tmp_path: Path) -> None:
    doc = {"schema_version": "owner_rules.v1", "source_authority": "docs/x", "owners": [{"owner_id": "X"}]}
    (tmp_path / "owner_rules.v1.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid owner"):
        load_evol_x_data_document("owner_rules.v1.json", data_dir=tmp_path)


def test_evol_x_data_loader_rejects_invalid_exclusion_status(tmp_path: Path) -> None:
    doc = {"schema_version": "exclusion_register.v1", "source_authority": "docs/x", "entries": [{"status": "MAYBE"}]}
    (tmp_path / "exclusion_register.v1.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid exclusion"):
        load_evol_x_data_document("exclusion_register.v1.json", data_dir=tmp_path)


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


def test_evol_x_artifact_models_roundtrip_and_forbid_extra() -> None:
    sha = "1" * 40
    models = [
        TestExecutionManifest(source_commit_sha=sha, node_ids=("a::b",), manifest_content_digest="d" * 64),
        EvolJunitOutcomes(source_commit_sha=sha, outcomes={"a::b": "passed"}),
        EvolMetricsArtifact(source_commit_sha=sha, readings=()),
        ExperimentVerdict(scenario_id="S1", arm="positive", passed=True, source_commit_sha=sha, detail="x"),
        FrzEvidence(frz_id="FRZ-EVOL-01", behavioral_test="t::x", passed=True, source_commit_sha=sha),
        P03Manifest(
            plan_publication_sha=sha,
            test_execution_sha=sha,
            implementation_sha=sha,
            independent_replay_sha=sha,
            pytest_profile="evol_x_p3_l3",
            uv_extra="dev-ci",
            executed_command="uv run pytest",
            artifact_paths=("a.json",),
        ),
        ArtifactDigests(digests={"a.json": "e" * 64}),
    ]
    for model in models:
        assert type(model).model_validate(model.model_dump()) == model
        with pytest.raises(ValidationError):
            type(model).model_validate({**model.model_dump(), "unexpected_field": 1})


def test_evol_x_p03_manifest_forbids_self_referential_evidence_sha() -> None:
    sha = "1" * 40
    payload = {
        "plan_publication_sha": sha,
        "test_execution_sha": sha,
        "implementation_sha": sha,
        "independent_replay_sha": sha,
        "pytest_profile": "evol_x_p3_l3",
        "uv_extra": "dev-ci",
        "executed_command": "uv run pytest",
        "artifact_paths": [],
        "evidence_publication_sha": sha,
    }
    with pytest.raises(ValidationError):
        P03Manifest.model_validate(payload)


def test_evol_x_contract_diff_incompatible_on_required_removal() -> None:
    from testing_support.change_safety.assess.contract_diff import diff_json_field_schemas

    before = {"properties": {"x": {"type": "string"}}, "required": ["x"]}
    after = {"properties": {}, "required": []}
    assert diff_json_field_schemas(before, after) is CompatibilityAtChangeResult.INCOMPATIBLE
