from __future__ import annotations

import pytest

from intergrax.runtime.architecture.evaluation_assets import (
    EvaluationAssetBundle,
    GoldenDatasetAsset,
    ScenarioCase,
    ScenarioLibraryAsset,
)


def test_evaluation_assets_reject_unknown_scenario_reference() -> None:
    with pytest.raises(ValueError, match="unknown scenario_id"):
        EvaluationAssetBundle(
            datasets=[
                GoldenDatasetAsset(
                    dataset_id="golden.core.v1",
                    version="1.0.0",
                    storage_ref="datasets/golden/core.jsonl",
                    scenario_ids=["scn.unknown"],
                )
            ],
            scenario_libraries=[
                ScenarioLibraryAsset(
                    library_id="core",
                    version="1.0.0",
                    scenarios=[
                        ScenarioCase(
                            scenario_id="scn.safe_tool_usage",
                            description="Tool-safe path",
                        )
                    ],
                )
            ],
        )


def test_evaluation_assets_accept_valid_bundle() -> None:
    bundle = EvaluationAssetBundle(
        datasets=[
            GoldenDatasetAsset(
                dataset_id="golden.core.v1",
                version="1.0.0",
                storage_ref="datasets/golden/core.jsonl",
                scenario_ids=["scn.safe_tool_usage"],
            )
        ],
        scenario_libraries=[
            ScenarioLibraryAsset(
                library_id="core",
                version="1.0.0",
                scenarios=[
                    ScenarioCase(
                        scenario_id="scn.safe_tool_usage",
                        description="Tool-safe path",
                    )
                ],
            )
        ],
    )
    assert bundle.datasets[0].dataset_id == "golden.core.v1"
