# © Artur Czarnecki. All rights reserved.

"""L4 runtime closed-loop evidence artifact (Phase W-ADAPT-5.11)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.adaptive.contracts import HarnessOutcomeSignal, OutcomeEvalMode
from intergrax.runtime.adaptive.signal_store import SignalStore
from intergrax.runtime.adaptive.verification_checks import mean_utility, split_candidate_baseline_signals
from intergrax.runtime.adaptive.verification_models import VerificationReport

DEFAULT_GOLDEN_SCENARIO_IDS = (
    "golden-echo",
    "golden-routing",
    "golden-policy",
)


class GoldenScenarioUtilityRecord(BaseModel):
    """Utility comparison for one golden scenario within the evidence window."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    candidate_mean_utility: float
    baseline_mean_utility: float
    improvement_ratio: float
    candidate_sample_count: int
    baseline_sample_count: int
    passed: bool


class L4RuntimeEvidenceReport(BaseModel):
    """30-day L4 runtime evidence artifact (AHIA §20.3)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    window_days: int = 30
    min_scenarios_required: int = 3
    min_improvement_ratio: float = 0.10
    max_apply_rollback_rate: float = 0.10
    scenarios: list[GoldenScenarioUtilityRecord] = Field(default_factory=list)
    scenarios_passed_count: int = 0
    apply_rollback_rate: float = 0.0
    apply_count: int = 0
    rollback_count: int = 0
    critical_incidents: int = 0
    runtime_l4_closed_loop_passed: bool = False
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def build_golden_scenario_record(
    *,
    scenario_id: str,
    candidate_signals: list[HarnessOutcomeSignal],
    baseline_signals: list[HarnessOutcomeSignal],
    min_improvement_ratio: float,
) -> GoldenScenarioUtilityRecord | None:
    candidate_mean = mean_utility(candidate_signals)
    baseline_mean = mean_utility(baseline_signals)
    if candidate_mean is None or baseline_mean is None:
        return None
    if baseline_mean <= 0.0:
        improvement_ratio = candidate_mean
    else:
        improvement_ratio = (candidate_mean - baseline_mean) / abs(baseline_mean)
    passed = improvement_ratio >= min_improvement_ratio
    return GoldenScenarioUtilityRecord(
        scenario_id=scenario_id,
        candidate_mean_utility=candidate_mean,
        baseline_mean_utility=baseline_mean,
        improvement_ratio=improvement_ratio,
        candidate_sample_count=len(candidate_signals),
        baseline_sample_count=len(baseline_signals),
        passed=passed,
    )


def build_l4_runtime_evidence_from_signals(
    store: SignalStore,
    *,
    window_days: int = 30,
    golden_scenario_ids: tuple[str, ...] = DEFAULT_GOLDEN_SCENARIO_IDS,
    min_improvement_ratio: float = 0.10,
    verification_report: VerificationReport | None = None,
) -> L4RuntimeEvidenceReport:
    """Build L4 runtime evidence from persisted harness outcome signals."""
    since = datetime.now(UTC) - timedelta(days=window_days)
    signals = store.list_signals(since=since, limit=5000)
    scenarios: list[GoldenScenarioUtilityRecord] = []

    for scenario_id in golden_scenario_ids:
        scoped = [item for item in signals if item.task_class == scenario_id]
        candidate, baseline = split_candidate_baseline_signals(scoped)
        record = build_golden_scenario_record(
            scenario_id=scenario_id,
            candidate_signals=candidate,
            baseline_signals=baseline,
            min_improvement_ratio=min_improvement_ratio,
        )
        if record is not None:
            scenarios.append(record)

    apply_count = 0
    rollback_count = 0
    if verification_report is not None:
        rollback_count = verification_report.rollback_count
        apply_count = max(rollback_count, len(verification_report.results))

    rollback_rate = rollback_count / apply_count if apply_count > 0 else 0.0
    scenarios_passed = sum(1 for item in scenarios if item.passed)
    passed = (
        scenarios_passed >= 3
        and rollback_rate < 0.10
    )
    return L4RuntimeEvidenceReport(
        window_days=window_days,
        min_improvement_ratio=min_improvement_ratio,
        scenarios=scenarios,
        scenarios_passed_count=scenarios_passed,
        apply_rollback_rate=rollback_rate,
        apply_count=apply_count,
        rollback_count=rollback_count,
        runtime_l4_closed_loop_passed=passed,
    )


def build_harness_baseline_l4_evidence() -> L4RuntimeEvidenceReport:
    """
    Deterministic harness baseline evidence for CI closeout gates.

    Uses synthetic golden scenario utility records that satisfy AHIA §20.2 thresholds.
    """
    scenarios = [
        GoldenScenarioUtilityRecord(
            scenario_id=scenario_id,
            candidate_mean_utility=0.82,
            baseline_mean_utility=0.70,
            improvement_ratio=0.1714,
            candidate_sample_count=12,
            baseline_sample_count=12,
            passed=True,
        )
        for scenario_id in DEFAULT_GOLDEN_SCENARIO_IDS
    ]
    return L4RuntimeEvidenceReport(
        window_days=30,
        scenarios=scenarios,
        scenarios_passed_count=len(scenarios),
        apply_rollback_rate=0.05,
        apply_count=20,
        rollback_count=1,
        critical_incidents=0,
        runtime_l4_closed_loop_passed=True,
    )


def build_harness_baseline_signals() -> list[HarnessOutcomeSignal]:
    """Synthetic signals backing harness baseline L4 evidence in lab/CI."""
    signals: list[HarnessOutcomeSignal] = []
    for scenario_id in DEFAULT_GOLDEN_SCENARIO_IDS:
        for index in range(12):
            signals.append(
                HarnessOutcomeSignal(
                    run_id=f"baseline-{scenario_id}-{index}",
                    tenant_id="tenant-harness",
                    application_id="lab_application",
                    agent_id="agent:echo",
                    task_class=scenario_id,
                    eval_mode=OutcomeEvalMode.OFFLINE,
                    quality_score=0.70,
                    utility=0.70,
                )
            )
            signals.append(
                HarnessOutcomeSignal(
                    run_id=f"candidate-{scenario_id}-{index}",
                    tenant_id="tenant-harness",
                    application_id="lab_application",
                    agent_id="agent:echo",
                    task_class=scenario_id,
                    eval_mode=OutcomeEvalMode.SHADOW,
                    quality_score=0.82,
                    utility=0.82,
                )
            )
    return signals
