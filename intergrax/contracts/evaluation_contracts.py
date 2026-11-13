# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed evaluation contracts: dataset, scenario, run, comparison (ADR-PDC-05).

Additive to ``intergrax.eval.eval_result.EvalResult`` (unchanged). These contracts are
advisory inputs to governance-owned promotion: nothing here authorizes a deployment.

Reproducibility rule: ``EvalRun.result_digest`` covers pinned inputs and verdicts only.
Telemetry (cost, latency, tokens) is recorded next to the verdicts but is excluded from
the digest, because wall-clock latency is not reproducible by construction.

Model, prompt and profile revisions are neutral ``EvalRevisionRef`` values, so this
module does not depend on the owners of those artifacts.
"""

from __future__ import annotations

import hashlib
import math
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.canonical_payload_hash import stable_payload_hash

EVAL_CONTRACT_SCHEMA_VERSION = "eval_contracts.v1"

_CODE_SHA_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class EvalContractError(ValueError):
    """Base error for evaluation contract violations."""


class EvalDigestMismatchError(EvalContractError):
    """A declared content digest does not match the recomputed digest."""


class EvalTenantBoundaryError(EvalContractError):
    """An evaluation artifact was read or compared across tenants."""


def text_digest(text: str) -> str:
    """Return ``sha256:<hex>`` of UTF-8 text (used for output digests, no raw output kept)."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _non_blank(value: str, *, field: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field} must not be blank")
    return value


class EvalRevisionRef(BaseModel):
    """Neutral typed reference to a revisioned artifact (agent, model, prompt, profile)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref_id: str
    revision: str

    @field_validator("ref_id")
    @classmethod
    def _ref_id(cls, value: str) -> str:
        return _non_blank(value, field="ref_id")

    @field_validator("revision")
    @classmethod
    def _revision(cls, value: str) -> str:
        return _non_blank(value, field="revision")


class EvalScenario(BaseModel):
    """One evaluation scenario with a stable identity.

    Identity is ``scenario_id`` plus a content digest over the scenario id, input,
    expectation and sorted risk tags. The free-text description is cosmetic and does
    not change identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    description: str = ""
    input_payload: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    risk_tags: tuple[str, ...] = ()
    declared_content_digest: str | None = None

    @field_validator("scenario_id")
    @classmethod
    def _scenario_id(cls, value: str) -> str:
        return _non_blank(value, field="scenario_id")

    @model_validator(mode="after")
    def _verify_declared_digest(self) -> EvalScenario:
        declared = self.declared_content_digest
        if declared is not None and declared != self.content_digest:
            raise EvalDigestMismatchError(
                f"scenario {self.scenario_id}: declared digest {declared} does not match "
                f"recomputed digest {self.content_digest}"
            )
        return self

    @property
    def content_digest(self) -> str:
        return stable_payload_hash(
            {
                "scenario_id": self.scenario_id,
                "input_payload": self.input_payload,
                "expected": self.expected,
                "risk_tags": sorted(self.risk_tags),
            }
        )

    @property
    def identity(self) -> str:
        return f"{self.scenario_id}@{self.content_digest}"


class EvalDataset(BaseModel):
    """Versioned dataset with a content digest independent of scenario order and version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    version: str
    scenarios: tuple[EvalScenario, ...]
    declared_content_digest: str | None = None

    @field_validator("dataset_id")
    @classmethod
    def _dataset_id(cls, value: str) -> str:
        return _non_blank(value, field="dataset_id")

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        return _non_blank(value, field="version")

    @model_validator(mode="after")
    def _validate(self) -> EvalDataset:
        if not self.scenarios:
            raise EvalContractError("dataset must contain at least one scenario")
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(set(ids)) != len(ids):
            raise EvalContractError("dataset contains duplicate scenario_id values")
        declared = self.declared_content_digest
        if declared is not None and declared != self.content_digest:
            raise EvalDigestMismatchError(
                f"dataset {self.dataset_id}@{self.version}: declared digest {declared} "
                f"does not match recomputed digest {self.content_digest}"
            )
        return self

    @property
    def content_digest(self) -> str:
        entries = sorted((scenario.scenario_id, scenario.content_digest) for scenario in self.scenarios)
        return stable_payload_hash({"dataset_id": self.dataset_id, "scenarios": [list(item) for item in entries]})

    @property
    def identity(self) -> str:
        return f"{self.dataset_id}@{self.version}#{self.content_digest}"

    def scenario(self, scenario_id: str) -> EvalScenario:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(scenario_id)


class EvaluatorKind(StrEnum):
    DETERMINISTIC = "deterministic"
    MODEL_ASSISTED = "model_assisted"
    HUMAN = "human"
    SAFETY = "safety"


class EvaluatorOutcomeStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"


class EvalSubjectOutput(BaseModel):
    """What the system under evaluation produced for one scenario (executor result)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_text: str = ""
    tool_calls: tuple[str, ...] = ()
    cited_ids: tuple[str, ...] = ()
    cost: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    tokens: int = Field(default=0, ge=0)
    error: str | None = None


class EvalSubject(BaseModel):
    """Input handed to one evaluator. Carries no outcome of any other evaluator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario: EvalScenario
    run_ref: str
    output: EvalSubjectOutput
    seed: int


class EvaluatorOutcome(BaseModel):
    """Verdict of one evaluator on one scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator_id: str
    kind: EvaluatorKind
    scenario_id: str
    status: EvaluatorOutcomeStatus
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reasons: tuple[str, ...] = ()
    judge_model: EvalRevisionRef | None = None

    @model_validator(mode="after")
    def _validate_kind_rules(self) -> EvaluatorOutcome:
        if self.kind is EvaluatorKind.MODEL_ASSISTED and self.judge_model is None:
            raise EvalContractError("model-assisted outcome must name its judge_model")
        if self.kind is not EvaluatorKind.MODEL_ASSISTED and self.judge_model is not None:
            raise EvalContractError("only model-assisted outcomes may carry a judge_model")
        if self.status is EvaluatorOutcomeStatus.PENDING and self.kind is not EvaluatorKind.HUMAN:
            raise EvalContractError("only human outcomes may be pending")
        return self

    @property
    def passed(self) -> bool:
        return self.status is EvaluatorOutcomeStatus.PASSED


class EvalScenarioTelemetry(BaseModel):
    """Per-scenario execution record. Excluded from ``EvalRun.result_digest``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    output_digest: str
    cost: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    tokens: int = Field(default=0, ge=0)
    error: str | None = None


class EvalRunMetrics(BaseModel):
    """Aggregates derived from a run. Judge results are kept apart from rule results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_count: int
    deterministic_pass_rate: float | None
    safety_pass_rate: float | None
    judge_pass_rate: float | None
    human_pass_rate: float | None
    human_pending: int
    successes: int
    total_cost: float
    cost_to_success: float | None
    mean_latency_ms: float | None
    p95_latency_ms: float | None

    def metric_values(self) -> dict[str, float | None]:
        return {
            "deterministic_pass_rate": self.deterministic_pass_rate,
            "safety_pass_rate": self.safety_pass_rate,
            "judge_pass_rate": self.judge_pass_rate,
            "human_pass_rate": self.human_pass_rate,
            "cost_to_success": self.cost_to_success,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
        }


_RULE_KINDS = (EvaluatorKind.DETERMINISTIC, EvaluatorKind.SAFETY)


def _rate(outcomes: list[EvaluatorOutcome]) -> float | None:
    decided = [item for item in outcomes if item.status is not EvaluatorOutcomeStatus.PENDING]
    if not decided:
        return None
    return sum(1 for item in decided if item.passed) / len(decided)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


class EvalRun(BaseModel):
    """One evaluation run with complete provenance.

    ``result_digest`` is identical for a rerun with pinned inputs (same dataset digest,
    seed, code SHA and revisions) that yields the same verdicts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["eval_contracts.v1"] = EVAL_CONTRACT_SCHEMA_VERSION
    tenant_id: str
    agent: EvalRevisionRef
    model: EvalRevisionRef
    prompt: EvalRevisionRef
    profile: EvalRevisionRef
    code_sha: str
    dataset_id: str
    dataset_version: str
    dataset_digest: str
    seed: int
    outcomes: tuple[EvaluatorOutcome, ...] = ()
    telemetry: tuple[EvalScenarioTelemetry, ...] = ()

    @field_validator("tenant_id")
    @classmethod
    def _tenant_id(cls, value: str) -> str:
        return _non_blank(value, field="tenant_id")

    @field_validator("code_sha")
    @classmethod
    def _code_sha(cls, value: str) -> str:
        if not _CODE_SHA_PATTERN.match(value):
            raise ValueError("code_sha must be a lowercase hex git SHA (7-64 chars)")
        return value

    @field_validator("dataset_digest")
    @classmethod
    def _dataset_digest(cls, value: str) -> str:
        if not _DIGEST_PATTERN.match(value):
            raise ValueError("dataset_digest must be sha256:<64 hex>")
        return value

    @property
    def inputs_digest(self) -> str:
        return stable_payload_hash(
            {
                "tenant_id": self.tenant_id,
                "agent": self.agent.model_dump(),
                "model": self.model.model_dump(),
                "prompt": self.prompt.model_dump(),
                "profile": self.profile.model_dump(),
                "code_sha": self.code_sha,
                "dataset_id": self.dataset_id,
                "dataset_version": self.dataset_version,
                "dataset_digest": self.dataset_digest,
                "seed": self.seed,
            }
        )

    @property
    def run_id(self) -> str:
        return f"evalrun:{self.inputs_digest[len('sha256:') :][:24]}"

    @property
    def result_digest(self) -> str:
        outcomes = sorted(
            (
                item.scenario_id,
                item.evaluator_id,
                item.kind.value,
                item.status.value,
                item.score,
                list(item.reasons),
                item.judge_model.model_dump() if item.judge_model else None,
            )
            for item in self.outcomes
        )
        outputs = sorted((item.scenario_id, item.output_digest) for item in self.telemetry)
        return stable_payload_hash(
            {
                "inputs_digest": self.inputs_digest,
                "outcomes": [list(item) for item in outcomes],
                "outputs": [list(item) for item in outputs],
            }
        )

    @property
    def rule_outcomes(self) -> tuple[EvaluatorOutcome, ...]:
        return tuple(item for item in self.outcomes if item.kind in _RULE_KINDS)

    @property
    def judge_outcomes(self) -> tuple[EvaluatorOutcome, ...]:
        return tuple(item for item in self.outcomes if item.kind is EvaluatorKind.MODEL_ASSISTED)

    def scenario_successes(self) -> dict[str, bool]:
        """Per-scenario success from rule outcomes only; judge and human never count."""
        by_scenario: dict[str, list[EvaluatorOutcome]] = {}
        for item in self.rule_outcomes:
            by_scenario.setdefault(item.scenario_id, []).append(item)
        errors = {item.scenario_id for item in self.telemetry if item.error}
        return {
            scenario_id: (scenario_id not in errors and all(o.passed for o in items))
            for scenario_id, items in by_scenario.items()
        }

    def metrics(self) -> EvalRunMetrics:
        def of(kind: EvaluatorKind) -> list[EvaluatorOutcome]:
            return [item for item in self.outcomes if item.kind is kind]

        humans = of(EvaluatorKind.HUMAN)
        successes = sum(1 for ok in self.scenario_successes().values() if ok)
        total_cost = sum(item.cost for item in self.telemetry)
        latencies = [item.latency_ms for item in self.telemetry]
        return EvalRunMetrics(
            scenario_count=len({item.scenario_id for item in self.telemetry}),
            deterministic_pass_rate=_rate(of(EvaluatorKind.DETERMINISTIC)),
            safety_pass_rate=_rate(of(EvaluatorKind.SAFETY)),
            judge_pass_rate=_rate(of(EvaluatorKind.MODEL_ASSISTED)),
            human_pass_rate=_rate(humans),
            human_pending=sum(1 for item in humans if item.status is EvaluatorOutcomeStatus.PENDING),
            successes=successes,
            total_cost=total_cost,
            cost_to_success=(total_cost / successes) if successes else None,
            mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
            p95_latency_ms=_percentile(latencies, 0.95) if latencies else None,
        )


class EvalThresholds(BaseModel):
    """Absolute floors/ceilings and regression tolerances for a comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_deterministic_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_safety_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_judge_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_cost_to_success: float | None = Field(default=None, ge=0.0)
    max_mean_latency_ms: float | None = Field(default=None, ge=0.0)
    rate_regression_tolerance: float = Field(default=0.0, ge=0.0, le=1.0)
    cost_latency_regression_ratio: float = Field(default=0.0, ge=0.0)


class EvalMetricDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    higher_is_better: bool
    baseline: float | None
    candidate: float | None
    delta: float | None
    regressed: bool


class EvalThresholdBreach(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    limit: float
    observed: float | None
    detail: str


class EvalComparison(BaseModel):
    """Advisory baseline-vs-candidate comparison. It can inform promotion, never grant it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["eval_contracts.v1"] = EVAL_CONTRACT_SCHEMA_VERSION
    advisory: Literal[True] = True
    tenant_id: str
    dataset_digest: str
    baseline_run_id: str
    candidate_run_id: str
    baseline_result_digest: str
    candidate_result_digest: str
    thresholds: EvalThresholds
    metric_deltas: tuple[EvalMetricDelta, ...]
    scenario_regressions: tuple[str, ...] = ()
    scenario_improvements: tuple[str, ...] = ()
    threshold_breaches: tuple[EvalThresholdBreach, ...] = ()
    baseline_cost_to_success: float | None = None
    candidate_cost_to_success: float | None = None
    baseline_mean_latency_ms: float | None = None
    candidate_mean_latency_ms: float | None = None

    @property
    def regressed(self) -> bool:
        """Regression on rule-based evidence; the judge metric is reported apart."""
        return bool(self.scenario_regressions) or any(
            item.regressed for item in self.metric_deltas if item.metric != "judge_pass_rate"
        )

    @property
    def judge_regressed(self) -> bool:
        return any(item.regressed for item in self.metric_deltas if item.metric == "judge_pass_rate")

    @property
    def threshold_breached(self) -> bool:
        return bool(self.threshold_breaches)

    @property
    def clean(self) -> bool:
        return not self.regressed and not self.judge_regressed and not self.threshold_breached

    @property
    def comparison_digest(self) -> str:
        return stable_payload_hash(self.model_dump(mode="json"))
