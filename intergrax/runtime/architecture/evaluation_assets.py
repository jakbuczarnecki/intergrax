# © Artur Czarnecki. All rights reserved.

"""Evaluation asset contracts for golden datasets and scenario libraries (Phase V-EVAL.2)."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, model_validator


def content_digest_of(content: bytes) -> str:
    """Return ``sha256:<hex>`` of dataset content bytes."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


class GoldenDatasetAsset(BaseModel):
    dataset_id: str
    version: str
    storage_ref: str
    scenario_ids: list[str] = Field(default_factory=list)
    # Optional for backward compatibility; when set, the version is bound to content.
    content_digest: str | None = None

    def bind_content(self, content: bytes) -> GoldenDatasetAsset:
        """Return a copy whose version is bound to the content digest of ``content``."""
        return self.model_copy(update={"content_digest": content_digest_of(content)})

    def verify_content(self, content: bytes) -> None:
        """Reject content whose digest differs from the bound digest (or no bound digest)."""
        if not self.content_digest:
            raise ValueError(f"Dataset {self.dataset_id}@{self.version} has no bound content digest")
        actual = content_digest_of(content)
        if actual != self.content_digest:
            raise ValueError(
                f"Dataset {self.dataset_id}@{self.version} content digest mismatch: "
                f"expected {self.content_digest}, got {actual}"
            )


class ScenarioCase(BaseModel):
    scenario_id: str
    description: str
    risk_tags: list[str] = Field(default_factory=list)


class ScenarioLibraryAsset(BaseModel):
    library_id: str
    version: str
    scenarios: list[ScenarioCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_scenarios(self) -> ScenarioLibraryAsset:
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("Scenario library contains duplicate scenario_id values")
        return self


class EvaluationAssetBundle(BaseModel):
    schema_version: str = "1.0.0"
    datasets: list[GoldenDatasetAsset] = Field(default_factory=list)
    scenario_libraries: list[ScenarioLibraryAsset] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> EvaluationAssetBundle:
        known_scenario_ids = {
            scenario.scenario_id for library in self.scenario_libraries for scenario in library.scenarios
        }
        for dataset in self.datasets:
            for scenario_id in dataset.scenario_ids:
                if scenario_id not in known_scenario_ids:
                    raise ValueError(f"Dataset references unknown scenario_id: {dataset.dataset_id} -> {scenario_id}")
        return self
