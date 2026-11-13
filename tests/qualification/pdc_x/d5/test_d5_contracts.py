# © Artur Czarnecki. All rights reserved.

"""PDC-D5-R001..R004: typed dataset, scenario and run contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from intergrax.contracts.evaluation_contracts import (
    EvalDataset,
    EvalRevisionRef,
    EvalRun,
    EvalScenario,
)
from intergrax.eval.eval_result import EvalResult
from intergrax.runtime.architecture.evaluation_assets import GoldenDatasetAsset
from intergrax.runtime.architecture.evaluation_scenario_loader import (
    load_scenario_library_dataset,
)
from pydantic import ValidationError
from tests.qualification.pdc_x.d5._support import build_run, make_dataset

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_r001_dataset_schema_and_persistence_roundtrip() -> None:
    dataset = make_dataset()
    restored = EvalDataset.model_validate_json(dataset.model_dump_json())
    assert restored == dataset
    assert restored.content_digest == dataset.content_digest
    assert dataset.content_digest.startswith("sha256:")
    assert dataset.version == "1.0.0"


def test_r001_run_schema_and_persistence_roundtrip() -> None:
    run = build_run()
    restored = EvalRun.model_validate_json(run.model_dump_json())
    assert restored == run
    assert restored.result_digest == run.result_digest
    assert restored.run_id == run.run_id


def test_r001_existing_eval_result_still_valid() -> None:
    result = EvalResult(
        case_id="c1",
        success=True,
        final_answer="x",
        total_tokens=1,
        total_cost=1.0,
        tool_calls_count=0,
    )
    assert result.error is None


def test_r002_declared_digest_mismatch_is_rejected() -> None:
    dataset = make_dataset()
    with pytest.raises(ValidationError, match="digest"):
        EvalDataset(
            dataset_id=dataset.dataset_id,
            version=dataset.version,
            scenarios=dataset.scenarios,
            declared_content_digest="sha256:" + "0" * 64,
        )
    accepted = EvalDataset(
        dataset_id=dataset.dataset_id,
        version=dataset.version,
        scenarios=dataset.scenarios,
        declared_content_digest=dataset.content_digest,
    )
    assert accepted.content_digest == dataset.content_digest


def test_r002_digest_tracks_content_not_order_or_version() -> None:
    dataset = make_dataset()
    reordered = EvalDataset(
        dataset_id=dataset.dataset_id,
        version="2.0.0",
        scenarios=tuple(reversed(dataset.scenarios)),
    )
    assert reordered.content_digest == dataset.content_digest
    changed = EvalScenario(scenario_id="s-math", input_payload={"q": "2+2"}, expected={"output": "5"})
    mutated = EvalDataset(
        dataset_id=dataset.dataset_id,
        version=dataset.version,
        scenarios=(changed, *dataset.scenarios[1:]),
    )
    assert mutated.content_digest != dataset.content_digest


def test_r002_golden_dataset_asset_binds_version_to_content() -> None:
    content = b'{"cases": []}'
    asset = GoldenDatasetAsset(dataset_id="g", version="1", storage_ref="x").bind_content(content)
    asset.verify_content(content)
    with pytest.raises(ValueError, match="digest mismatch"):
        asset.verify_content(b'{"cases": [1]}')
    unbound = GoldenDatasetAsset(dataset_id="g", version="1", storage_ref="x")
    with pytest.raises(ValueError, match="no bound content digest"):
        unbound.verify_content(content)


def test_r003_scenario_identity_is_stable() -> None:
    first = EvalScenario(
        scenario_id="s1",
        description="first wording",
        input_payload={"b": 2, "a": 1},
        expected={"output": "x"},
        risk_tags=("t2", "t1"),
    )
    second = EvalScenario(
        scenario_id="s1",
        description="reworded",
        input_payload={"a": 1, "b": 2},
        expected={"output": "x"},
        risk_tags=("t1", "t2"),
    )
    assert first.identity == second.identity
    other = EvalScenario(scenario_id="s1", input_payload={"a": 1, "b": 2}, expected={"output": "y"})
    assert other.scenario_id == first.scenario_id
    assert other.identity != first.identity
    with pytest.raises(ValidationError):
        EvalScenario(scenario_id="  ")


def test_r003_scenario_library_loads_as_digested_dataset() -> None:
    first = load_scenario_library_dataset(REPO_ROOT)
    second = load_scenario_library_dataset(REPO_ROOT)
    assert first.content_digest == second.content_digest
    assert {s.scenario_id for s in first.scenarios} >= {"sync_intake_smoke"}


def test_r003_duplicate_scenario_ids_rejected() -> None:
    scenario = EvalScenario(scenario_id="dup")
    with pytest.raises(ValidationError, match="duplicate"):
        EvalDataset(dataset_id="d", version="1", scenarios=(scenario, scenario))


def test_r004_run_provenance_is_complete() -> None:
    run = build_run()
    payload = json.loads(run.model_dump_json())
    for key in (
        "tenant_id",
        "agent",
        "model",
        "prompt",
        "profile",
        "code_sha",
        "dataset_digest",
        "seed",
    ):
        assert payload[key] not in (None, "", {})
    assert run.dataset_digest == make_dataset().content_digest
    assert run.model.revision == "m-1"


@pytest.mark.parametrize("missing", ["agent", "model", "prompt", "profile", "tenant_id"])
def test_r004_run_without_provenance_field_is_rejected(missing: str) -> None:
    payload = json.loads(build_run().model_dump_json())
    del payload[missing]
    with pytest.raises(ValidationError):
        EvalRun.model_validate(payload)


def test_r004_blank_revision_and_bad_code_sha_rejected() -> None:
    with pytest.raises(ValidationError):
        EvalRevisionRef(ref_id="model:x", revision=" ")
    payload = json.loads(build_run().model_dump_json())
    payload["code_sha"] = "not-a-sha"
    with pytest.raises(ValidationError, match="code_sha"):
        EvalRun.model_validate(payload)
