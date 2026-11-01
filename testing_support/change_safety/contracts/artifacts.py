# © Artur Czarnecki. All rights reserved.

"""W2-01: v1 artifact contracts for evidence / P03 (Pydantic, extra=forbid)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SHA_LEN = 40


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TestExecutionManifest(_Strict):
    __test__ = False
    schema_version: str = "evol_test_execution_manifest.v1"
    source_commit_sha: str = Field(description="TEST_EXECUTION_SHA only")
    node_ids: tuple[str, ...]
    manifest_content_digest: str
    excluded_populations: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()


class EvolJunitOutcomes(_Strict):
    schema_version: str = "evol_junit_outcomes.v1"
    source_commit_sha: str
    outcomes: dict[str, Literal["passed", "failed", "skipped", "error"]]


class MetricReadingArtifact(_Strict):
    metric_id: str
    numerator: int
    denominator: int
    status: Literal["MEASURED", "NOT_MEASURED"]
    threshold: float | None = None
    passed: bool | None = None
    population_digest: str | None = None
    note: str = ""


class EvolMetricsArtifact(_Strict):
    schema_version: str = "evol_metrics.v1"
    source_commit_sha: str
    readings: tuple[MetricReadingArtifact, ...]


class ExperimentVerdict(_Strict):
    schema_version: str = "evol_experiment_verdict.v1"
    scenario_id: str
    arm: Literal["positive", "negative"]
    passed: bool
    source_commit_sha: str
    detail: str = ""


class FrzEvidence(_Strict):
    schema_version: str = "evol_frz_evidence.v1"
    frz_id: str
    behavioral_test: str
    passed: bool
    source_commit_sha: str
    scope_footnote: str = ""


class P03Manifest(_Strict):
    """MANIFEST.json (written @ EVIDENCE_PUBLICATION_SHA). Self-referential evidence SHA is **forbidden** (Part V-A)."""

    schema_version: str = "evol_p03_manifest.v1"
    stage: Literal["EVOL-X"] = "EVOL-X"
    plan_publication_sha: str
    test_execution_sha: str
    implementation_sha: str
    independent_replay_sha: str
    pytest_profile: str
    uv_extra: str
    executed_command: str
    dirty_tree: Literal[False] = False
    parent_test_execution_tree: str | None = None
    artifact_paths: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_shas(self) -> P03Manifest:
        for name in ("plan_publication_sha", "test_execution_sha", "implementation_sha", "independent_replay_sha"):
            if len(getattr(self, name)) != _SHA_LEN:
                raise ValueError(f"{name} must be a 40-char git sha")
        return self


class ArtifactDigests(_Strict):
    schema_version: str = "evol_artifact_digests.v1"
    digests: dict[str, str]


class M09DrillCaseRegistryEntry(_Strict):
    case_id: Literal["C1", "C2", "C3"]
    changed_path: str
    input_digest: str = Field(min_length=64, max_length=64)
    expected_cause_identities: tuple[str, ...]


class M09DrillCaseRegistry(_Strict):
    schema_version: str = "evol_m09_drill_case_registry.v1"
    bound_test_execution_sha: str
    cases: tuple[M09DrillCaseRegistryEntry, ...]

    @model_validator(mode="after")
    def _check_cases(self) -> M09DrillCaseRegistry:
        if len(self.cases) != 3:
            raise ValueError("M09 drill registry requires exactly 3 cases (C1..C3)")
        ids = [c.case_id for c in self.cases]
        if ids != ["C1", "C2", "C3"]:
            raise ValueError("M09 drill registry cases must be ordered C1, C2, C3")
        if len({c.changed_path for c in self.cases}) != 3:
            raise ValueError("M09 drill registry changed_path values must be distinct")
        if len(self.bound_test_execution_sha) != _SHA_LEN:
            raise ValueError("bound_test_execution_sha must be a 40-char git sha")
        return self


class M09HumanDrillCaseResult(_Strict):
    case_id: Literal["C1", "C2", "C3"]
    correct: bool
    tool_seconds: float = Field(ge=0.0)
    attributed_cause_identities: tuple[str, ...]


class M09HumanDrillArtifact(_Strict):
    schema_version: str = "evol_m09_human_drill.v1"
    source_commit_sha: str
    test_execution_sha: str
    registry_digest: str = Field(min_length=64, max_length=64)
    cases: tuple[M09HumanDrillCaseResult, ...]
    median_tool_seconds: float = Field(ge=0.0)
    adversarial_control_passed: bool
    drill_started_at_utc: str
    drill_completed_at_utc: str
    evaluator_id: str = Field(min_length=1)
    evaluator_display_name: str = Field(min_length=1)
    worksheet_digest: str | None = None

    @model_validator(mode="after")
    def _check_cases(self) -> M09HumanDrillArtifact:
        if len(self.cases) != 3:
            raise ValueError("M09 human drill requires exactly 3 case results")
        if [c.case_id for c in self.cases] != ["C1", "C2", "C3"]:
            raise ValueError("M09 human drill cases must be ordered C1, C2, C3")
        for name in ("source_commit_sha", "test_execution_sha"):
            if len(getattr(self, name)) != _SHA_LEN:
                raise ValueError(f"{name} must be a 40-char git sha")
        return self
