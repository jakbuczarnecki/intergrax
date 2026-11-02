# © Artur Czarnecki. All rights reserved.

"""W2-01: v1 artifact contracts for evidence / P03 (Pydantic, extra=forbid)."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

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
    certification_profile: Literal["HUMAN_FULL", "AUTOMATION_FIRST_SCOPED"] | None = None

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


def _finite_nonnegative(name: str, value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


class M09HumanDrillCaseResult(_Strict):
    """Per-case v2 result. ``tool_seconds`` (tool runtime) and ``attribution_seconds`` (human clock) are separate."""

    case_id: Literal["C1", "C2", "C3"]
    correct: bool
    tool_seconds: float
    attribution_seconds: float
    attributed_cause_identities: tuple[str, ...]

    @field_validator("tool_seconds", "attribution_seconds")
    @classmethod
    def _check_seconds(cls, value: float, info: ValidationInfo) -> float:
        return _finite_nonnegative(str(info.field_name), value)


class M09HumanDrillArtifact(_Strict):
    schema_version: Literal["evol_m09_human_drill.v2"]
    measurement_policy_id: Literal["OD-EVOL-08-M1"]
    source_commit_sha: str
    test_execution_sha: str
    registry_digest: str = Field(min_length=64, max_length=64)
    cases: tuple[M09HumanDrillCaseResult, ...]
    median_tool_seconds: float
    median_attribution_seconds: float
    adversarial_control_passed: bool
    drill_started_at_utc: str
    drill_completed_at_utc: str
    evaluator_id: str = Field(min_length=1)
    evaluator_display_name: str = Field(min_length=1)
    worksheet_digest: str | None = None

    @field_validator("median_tool_seconds", "median_attribution_seconds")
    @classmethod
    def _check_medians(cls, value: float, info: ValidationInfo) -> float:
        return _finite_nonnegative(str(info.field_name), value)

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


class M09AutomatedQualificationCase(_Strict):
    case_id: Literal["C1", "C2", "C3"]
    seed_changed_path: str
    input_digest: str = Field(min_length=64, max_length=64)
    tool_seconds: float
    tool_sla_met: bool
    direct_node_id: str
    direct_node_kind: str
    derived_cause_identities: tuple[str, ...]
    cause_identity_digest: str = Field(min_length=64, max_length=64)
    registry_agreement: bool
    diagnostic_output_digest: str = Field(min_length=64, max_length=64)
    diagnostic_output_relpath: str

    @field_validator("tool_seconds")
    @classmethod
    def _check_tool_seconds(cls, value: float) -> float:
        return _finite_nonnegative("tool_seconds", value)


class M09AutomatedAdversarialControl(_Strict):
    control_id: str
    passed: bool | None
    detail: str
    automated_only: bool


class M09AutomatedQualificationSummary(_Strict):
    cases_passed: bool
    median_tool_seconds: float
    automated_adversarial_controls_passed: bool


class M09AutomatedDiagnosticQualification(_Strict):
    schema_version: Literal["evol_m09_automated_diagnostic_qualification.v1"]
    measurement_mode: Literal["AUTOMATED_DIAGNOSTIC"]
    qualification_policy_id: Literal["OD-EVOL-13"]
    operator_decision_id: Literal["OD-EVOL-13"]
    source_commit_sha: str
    test_execution_sha: str
    registry_path: str
    registry_digest: str = Field(min_length=64, max_length=64)
    run_id: str = Field(min_length=8)
    execution_started_at_utc: str
    execution_finished_at_utc: str
    cases: tuple[M09AutomatedQualificationCase, ...]
    summary: M09AutomatedQualificationSummary
    adversarial_controls: tuple[M09AutomatedAdversarialControl, ...]
    certification_disposition: Literal["AUTOMATED_SCOPED_PASS", "AUTOMATED_SCOPED_FAIL"]
    human_m09h_measured: Literal[False] = False
    artifact_digest: str | None = Field(default=None, min_length=64, max_length=64)

    @model_validator(mode="after")
    def _check_cases(self) -> M09AutomatedDiagnosticQualification:
        if len(self.cases) != 3:
            raise ValueError("automated M-09A qualification requires exactly 3 cases")
        if [c.case_id for c in self.cases] != ["C1", "C2", "C3"]:
            raise ValueError("automated M-09A cases must be ordered C1, C2, C3")
        for name in ("source_commit_sha", "test_execution_sha"):
            if len(getattr(self, name)) != _SHA_LEN:
                raise ValueError(f"{name} must be a 40-char git sha")
        if self.certification_disposition != "AUTOMATED_SCOPED_PASS":
            raise ValueError("bundled automated qualification must disposition AUTOMATED_SCOPED_PASS")
        if not self.summary.cases_passed:
            raise ValueError("summary.cases_passed must be true")
        if not self.summary.automated_adversarial_controls_passed:
            raise ValueError("summary.automated_adversarial_controls_passed must be true")
        return self
