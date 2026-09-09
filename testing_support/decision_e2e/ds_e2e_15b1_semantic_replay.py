# © Artur Czarnecki. All rights reserved.

"""Semantic reinterpretation replay for DS-E2E-15A.1 dataset (DS-E2E-15B.1)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from intergrax.decision_system.qualification.classifier import classify_decision_failure
from intergrax.decision_system.qualification.taxonomy import DecisionFailureReason
from testing_support.decision_e2e.failure_observation_adapter import (
    observation_from_ai_incident_evaluation,
)

PROXY_ONLY_FAILURE_ID = "tool_runtime_not_exercised"
PROXY_ONLY_RUN_INDICES: frozenset[int] = frozenset({2, 3, 5, 6, 9, 10, 13, 16, 17, 18, 19})
GENUINE_SEMANTIC_RUN_INDICES: frozenset[int] = frozenset({8, 14})
SUCCESS_RUN_INDICES: frozenset[int] = frozenset({0, 4, 7, 11, 12, 15})
UNSUPPORTED_COMPLETION_RUN_INDEX = 1


class ReplayRunRecord(TypedDict):
    run_index: int
    before_evaluator_passed: bool
    after_evaluator_passed: bool
    before_failures: tuple[str, ...]
    after_failures: tuple[str, ...]
    before_reason: str | None
    after_reason: str | None


class SemanticReplayReport(TypedDict):
    source_dataset_path: str
    source_dataset_sha256: str
    implementation_commit_sha: str
    replay_timestamp: str
    before_distribution: dict[str, int]
    after_distribution: dict[str, int]
    proxy_only_runs: tuple[int, ...]
    genuine_semantic_runs: tuple[int, ...]
    success_runs: tuple[int, ...]
    unsupported_completion_run: int
    runs: tuple[ReplayRunRecord, ...]


def _strip_proxy_tool_count_failure(failures: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(failure for failure in failures if failure != PROXY_ONLY_FAILURE_ID)


def corrected_evaluator_failures(failures: tuple[str, ...]) -> tuple[str, ...]:
    """Remove non-authoritative tool-count proxy failure from persisted evaluator facts."""
    return _strip_proxy_tool_count_failure(failures)


def corrected_evaluator_passed(failures: tuple[str, ...]) -> bool:
    return len(corrected_evaluator_failures(failures)) == 0


def _classification_reason(
    *,
    failures: tuple[str, ...],
    evaluator_passed: bool,
) -> str | None:
    if evaluator_passed:
        return None
    observation = observation_from_ai_incident_evaluation(
        failures=failures,
        evaluator_passed=False,
    )
    classification = classify_decision_failure(observation)
    if classification is None:
        return None
    return classification.reason.value


def _count_reasons(runs: tuple[ReplayRunRecord, ...], *, field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        reason = run[field]  # type: ignore[literal-required]
        if reason is None:
            continue
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    return counts


def replay_ai_incident_qualification_dataset(
    *,
    runs_json_path: Path,
    implementation_commit_sha: str,
) -> SemanticReplayReport:
    payload = json.loads(runs_json_path.read_text(encoding="utf-8"))
    source_sha256 = hashlib.sha256(runs_json_path.read_bytes()).hexdigest()
    replay_runs: list[ReplayRunRecord] = []

    for run in payload["runs"]:
        run_index = int(run["run_index"])
        signals = run.get("signals") or {}
        before_failures = tuple(str(item) for item in signals.get("evaluator_failures", ()))
        before_evaluator_passed = bool(run.get("evaluator_passed"))
        before_classification = (run.get("run_result") or {}).get("classification")
        before_reason = (
            str(before_classification["reason"]) if before_classification is not None else None
        )

        if before_failures:
            after_failures = corrected_evaluator_failures(before_failures)
            after_evaluator_passed = corrected_evaluator_passed(before_failures)
            after_reason = _classification_reason(
                failures=after_failures,
                evaluator_passed=after_evaluator_passed,
            )
        else:
            after_failures = before_failures
            after_evaluator_passed = before_evaluator_passed
            after_reason = before_reason

        replay_runs.append(
            ReplayRunRecord(
                run_index=run_index,
                before_evaluator_passed=before_evaluator_passed,
                after_evaluator_passed=after_evaluator_passed,
                before_failures=before_failures,
                after_failures=after_failures,
                before_reason=before_reason,
                after_reason=after_reason,
            )
        )

    runs_tuple = tuple(replay_runs)
    return SemanticReplayReport(
        source_dataset_path=str(runs_json_path),
        source_dataset_sha256=source_sha256,
        implementation_commit_sha=implementation_commit_sha,
        replay_timestamp=datetime.now(tz=UTC).isoformat(),
        before_distribution=_count_reasons(runs_tuple, field="before_reason"),
        after_distribution=_count_reasons(runs_tuple, field="after_reason"),
        proxy_only_runs=tuple(sorted(PROXY_ONLY_RUN_INDICES)),
        genuine_semantic_runs=tuple(sorted(GENUINE_SEMANTIC_RUN_INDICES)),
        success_runs=tuple(sorted(SUCCESS_RUN_INDICES)),
        unsupported_completion_run=UNSUPPORTED_COMPLETION_RUN_INDEX,
        runs=runs_tuple,
    )


def write_semantic_replay_artifacts(
    report: SemanticReplayReport,
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "semantic_replay.json"
    md_path = output_dir / "semantic_replay.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    proxy_corrected = sum(
        1
        for run in report["runs"]
        if run["run_index"] in PROXY_ONLY_RUN_INDICES
        and run["before_evaluator_passed"] is False
        and run["after_evaluator_passed"] is True
    )
    before_tool_use = report["before_distribution"].get(
        DecisionFailureReason.TOOL_USE_DEFICIENCY.value,
        0,
    )
    after_tool_use = report["after_distribution"].get(
        DecisionFailureReason.TOOL_USE_DEFICIENCY.value,
        0,
    )
    md_lines = [
        "# DS-E2E-15B.1 semantic replay",
        "",
        "Semantic reinterpretation of DS-E2E-15A.1 dataset — not a new live reliability baseline.",
        "",
        f"- source dataset: `{report['source_dataset_path']}`",
        f"- source dataset SHA256: `{report['source_dataset_sha256']}`",
        f"- implementation commit: `{report['implementation_commit_sha']}`",
        f"- replay timestamp: `{report['replay_timestamp']}`",
        "",
        "## Before classification distribution",
        "",
        *[f"- `{key}`: {value}" for key, value in sorted(report["before_distribution"].items())],
        "",
        "## After classification distribution",
        "",
        *[f"- `{key}`: {value}" for key, value in sorted(report["after_distribution"].items())],
        "",
        "## Cohort matrix",
        "",
        "| Cohort | Before | After |",
        "| --- | ---: | ---: |",
        f"| Existing success ({len(SUCCESS_RUN_INDICES)} runs) | pass | pass |",
        f"| Proxy-only deficient ({len(PROXY_ONLY_RUN_INDICES)} runs) | fail | corrected ({proxy_corrected} pass) |",
        f"| Genuine semantic deficient ({len(GENUINE_SEMANTIC_RUN_INDICES)} runs) | fail | fail |",
        f"| Unsupported completion (run {UNSUPPORTED_COMPLETION_RUN_INDEX}) | fail | unchanged |",
        "",
        f"- false negatives removed: {before_tool_use - after_tool_use} tool-use classifications",
        f"- proxy-only evaluator passes: {proxy_corrected}/{len(PROXY_ONLY_RUN_INDICES)}",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path
