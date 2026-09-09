# © Artur Czarnecki. All rights reserved.

"""Multi-run decision reliability qualification harness (DS-E2E-14.3b)."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from intergrax.contracts.execution_identity import RunId
from intergrax.decision_system.qualification.reliability import (
    DecisionReliabilitySummary,
    aggregate_decision_reliability,
)
from intergrax.decision_system.qualification.axis_outcome import DecisionQualificationAxisOutcome
from intergrax.decision_system.qualification.run_result import DecisionQualificationRunResult
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureCategory,
    DecisionFailureReason,
)

from testing_support.decision_e2e.ai_incident_qualification_run import (
    AiIncidentQualificationRunOutcome,
    AiIncidentQualificationRunSignals,
)
from testing_support.decision_e2e.env_bootstrap import (
    QualificationEnvBootstrapReport,
    bootstrap_qualification_environment,
)


class DecisionQualificationRunExecutor(Protocol):
    async def execute(self, *, run_index: int) -> AiIncidentQualificationRunOutcome:
        """Execute one qualification run and return typed outcome."""


@dataclass(frozen=True, slots=True)
class DecisionReliabilityQualificationPlan:
    run_count: int
    provider_id: str
    model_id: str
    scenario_id: str
    scenario_input_identity: str


@dataclass(frozen=True, slots=True)
class QualificationConfigurationFingerprint:
    provider: str
    model: str
    scenario_id: str
    scenario_input_identity: str
    taxonomy_version_sha: str
    git_head: str


@dataclass(frozen=True, slots=True)
class DecisionReliabilityQualificationProvenance:
    qualification_id: str
    git_sha: str
    started_at: str
    completed_at: str
    env_bootstrap: QualificationEnvBootstrapReport
    fingerprint: QualificationConfigurationFingerprint


@dataclass(frozen=True, slots=True)
class DecisionReliabilityQualificationRunRecord:
    run_index: int
    run_id: RunId | None
    valid_model_trial: bool
    environment_event: bool
    completed: bool
    run_result: DecisionQualificationRunResult | None
    signals: AiIncidentQualificationRunSignals | None
    block_reason: str | None

    @property
    def platform_passed(self) -> bool:
        if self.run_result is None:
            return False
        return self.run_result.platform_outcome is DecisionQualificationAxisOutcome.PASS

    @property
    def model_passed(self) -> bool:
        if self.run_result is None:
            return False
        return self.run_result.model_outcome is DecisionQualificationAxisOutcome.PASS

    @property
    def evaluator_passed(self) -> bool:
        if self.run_result is None:
            return False
        return self.run_result.evaluator_outcome is DecisionQualificationAxisOutcome.PASS

    @property
    def provider_failed(self) -> bool:
        if self.run_result is None or self.run_result.classification is None:
            return False
        return (
            self.run_result.classification.category
            is DecisionFailureCategory.PROVIDER_INFRASTRUCTURE
        )

    @property
    def environment_failed(self) -> bool:
        if self.run_result is None or self.run_result.classification is None:
            return False
        return self.run_result.classification.category is DecisionFailureCategory.ENVIRONMENT

    @property
    def observability_complete(self) -> bool:
        if self.signals is None:
            return False
        return self.signals.trace_readback_pass


@dataclass(frozen=True, slots=True)
class DecisionReliabilityQualificationResult:
    plan: DecisionReliabilityQualificationPlan
    provenance: DecisionReliabilityQualificationProvenance
    runs: tuple[DecisionReliabilityQualificationRunRecord, ...]
    summary: DecisionReliabilitySummary
    environment_failure_count: int
    valid_model_trial_count: int
    completed_run_count: int
    session_complete: bool
    stop_reason: str | None


class QualificationSessionIntegrityError(RuntimeError):
    """Fail-closed stop for corrupted qualification sessions."""


def validate_run_result_consistency(record: DecisionReliabilityQualificationRunRecord) -> None:
    if record.run_result is None:
        return
    classification = record.run_result.classification
    if classification is None:
        return
    if (
        classification.is_model_failure
        and record.run_result.model_outcome is not DecisionQualificationAxisOutcome.FAIL
    ):
        raise QualificationSessionIntegrityError(
            f"run {record.run_index}: model failure classification with model_outcome!=FAIL"
        )
    if (
        classification.is_platform_failure
        and record.run_result.platform_outcome is not DecisionQualificationAxisOutcome.FAIL
    ):
        raise QualificationSessionIntegrityError(
            f"run {record.run_index}: platform failure classification with platform_outcome!=FAIL"
        )


def _assert_no_run_id_collision(records: tuple[DecisionReliabilityQualificationRunRecord, ...]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.run_id is None:
            continue
        key = str(record.run_id)
        if key in seen:
            raise QualificationSessionIntegrityError(f"run identity collision: {key}")
        seen.add(key)


def _assert_configuration_stable(
    fingerprint: QualificationConfigurationFingerprint,
    plan: DecisionReliabilityQualificationPlan,
) -> None:
    if fingerprint.provider != plan.provider_id:
        raise QualificationSessionIntegrityError("provider configuration drift detected")
    if fingerprint.model != plan.model_id:
        raise QualificationSessionIntegrityError("model configuration drift detected")
    if fingerprint.scenario_id != plan.scenario_id:
        raise QualificationSessionIntegrityError("scenario configuration drift detected")
    if fingerprint.scenario_input_identity != plan.scenario_input_identity:
        raise QualificationSessionIntegrityError("scenario input drift detected")


async def execute_reliability_qualification(
    plan: DecisionReliabilityQualificationPlan,
    executor: DecisionQualificationRunExecutor,
    *,
    git_sha: str,
    qualification_id: str | None = None,
    env_bootstrap: QualificationEnvBootstrapReport | None = None,
) -> DecisionReliabilityQualificationResult:
    started_at = datetime.now(tz=UTC).isoformat()
    bootstrap = env_bootstrap or bootstrap_qualification_environment()
    session_id = qualification_id or f"ds-e2e-14.3b-{uuid.uuid4().hex[:12]}"
    fingerprint = QualificationConfigurationFingerprint(
        provider=plan.provider_id,
        model=plan.model_id,
        scenario_id=plan.scenario_id,
        scenario_input_identity=plan.scenario_input_identity,
        taxonomy_version_sha=git_sha,
        git_head=git_sha,
    )
    _assert_configuration_stable(fingerprint, plan)

    records: list[DecisionReliabilityQualificationRunRecord] = []
    stop_reason: str | None = None
    for run_index in range(plan.run_count):
        outcome = await executor.execute(run_index=run_index)
        record = DecisionReliabilityQualificationRunRecord(
            run_index=run_index,
            run_id=outcome.run_id,
            valid_model_trial=outcome.valid_model_trial,
            environment_event=outcome.environment_event,
            completed=True,
            run_result=outcome.run_result,
            signals=outcome.signals,
            block_reason=outcome.block_reason,
        )
        validate_run_result_consistency(record)
        if (
            record.valid_model_trial
            and record.run_result is not None
            and record.run_result.classification is not None
            and record.run_result.classification.category is DecisionFailureCategory.UNCLASSIFIED
        ):
            raise QualificationSessionIntegrityError(
                f"taxonomy returned UNCLASSIFIED for valid run {run_index}"
            )
        records.append(record)
        _assert_no_run_id_collision(tuple(records))

    completed_at = datetime.now(tz=UTC).isoformat()
    provenance = DecisionReliabilityQualificationProvenance(
        qualification_id=session_id,
        git_sha=git_sha,
        started_at=started_at,
        completed_at=completed_at,
        env_bootstrap=bootstrap,
        fingerprint=fingerprint,
    )

    valid_records = tuple(record for record in records if record.valid_model_trial)
    valid_run_results = tuple(
        record.run_result
        for record in valid_records
        if record.run_result is not None
    )
    summary = aggregate_decision_reliability(valid_run_results)
    environment_failure_count = sum(1 for record in records if record.environment_event)
    session_complete = len(valid_records) == plan.run_count
    if not session_complete and environment_failure_count > 0:
        stop_reason = "environment blocked one or more planned runs"

    return DecisionReliabilityQualificationResult(
        plan=plan,
        provenance=provenance,
        runs=tuple(records),
        summary=summary,
        environment_failure_count=environment_failure_count,
        valid_model_trial_count=len(valid_records),
        completed_run_count=len(records),
        session_complete=session_complete,
        stop_reason=stop_reason,
    )


@dataclass(frozen=True, slots=True)
class CallableDecisionQualificationRunExecutor:
    _callable: Callable[[int], Awaitable[AiIncidentQualificationRunOutcome]]

    async def execute(self, *, run_index: int) -> AiIncidentQualificationRunOutcome:
        return await self._callable(run_index)


def count_distribution(
    records: tuple[DecisionReliabilityQualificationRunRecord, ...],
    *,
    valid_only: bool = True,
) -> dict[str, int]:
    selected = records
    if valid_only:
        selected = tuple(record for record in records if record.valid_model_trial)
    return {str(len(selected)): len(selected)}


def failure_category_distribution(
    records: tuple[DecisionReliabilityQualificationRunRecord, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if not record.valid_model_trial or record.run_result is None:
            continue
        classification = record.run_result.classification
        if classification is None:
            key = "NONE"
        else:
            key = classification.category.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def failure_reason_distribution(
    records: tuple[DecisionReliabilityQualificationRunRecord, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if not record.valid_model_trial or record.run_result is None:
            continue
        classification = record.run_result.classification
        if classification is None:
            continue
        key = classification.reason.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def failure_boundary_distribution(
    records: tuple[DecisionReliabilityQualificationRunRecord, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if not record.valid_model_trial or record.run_result is None:
            continue
        classification = record.run_result.classification
        if classification is None:
            continue
        key = classification.boundary.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def failure_owner_distribution(
    records: tuple[DecisionReliabilityQualificationRunRecord, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if not record.valid_model_trial or record.run_result is None:
            continue
        classification = record.run_result.classification
        if classification is None:
            continue
        key = classification.owner.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def tool_selection_distribution(
    records: tuple[DecisionReliabilityQualificationRunRecord, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.signals is None:
            continue
        for tool_id in record.signals.selected_tool_ids:
            counts[tool_id] = counts.get(tool_id, 0) + 1
    return counts


def tool_execution_distribution(
    records: tuple[DecisionReliabilityQualificationRunRecord, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.signals is None:
            continue
        for tool_id in record.signals.executed_tool_ids:
            counts[tool_id] = counts.get(tool_id, 0) + 1
    return counts


def tool_depth_distribution(
    records: tuple[DecisionReliabilityQualificationRunRecord, ...],
) -> dict[str, int]:
    buckets = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for record in records:
        if record.signals is None:
            continue
        depth = record.signals.tool_invocation_count
        if depth == 0:
            buckets["0"] += 1
        elif depth == 1:
            buckets["1"] += 1
        elif depth == 2:
            buckets["2"] += 1
        else:
            buckets["3+"] += 1
    return buckets


def planner_round_distribution(
    records: tuple[DecisionReliabilityQualificationRunRecord, ...],
) -> dict[str, int]:
    buckets = {"1": 0, "2": 0, "3": 0, "4+": 0}
    for record in records:
        if record.signals is None:
            continue
        rounds = record.signals.planner_round_count
        if rounds <= 1:
            buckets["1"] += 1
        elif rounds == 2:
            buckets["2"] += 1
        elif rounds == 3:
            buckets["3"] += 1
        else:
            buckets["4+"] += 1
    return buckets


def terminal_outcome_distribution(
    records: tuple[DecisionReliabilityQualificationRunRecord, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.signals is None or record.signals.terminal_outcome is None:
            key = "PRE_TERMINAL_FAILURE"
        else:
            key = record.signals.terminal_outcome
        counts[key] = counts.get(key, 0) + 1
    return counts


def top_model_failure_reasons(
    records: tuple[DecisionReliabilityQualificationRunRecord, ...],
    *,
    limit: int = 3,
) -> tuple[tuple[DecisionFailureReason, int], ...]:
    counts: dict[DecisionFailureReason, int] = {}
    for record in records:
        if not record.valid_model_trial or record.run_result is None:
            continue
        classification = record.run_result.classification
        if classification is None:
            continue
        if classification.category is not DecisionFailureCategory.MODEL_BEHAVIOR:
            continue
        counts[classification.reason] = counts.get(classification.reason, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0].value))
    return tuple(ranked[:limit])
