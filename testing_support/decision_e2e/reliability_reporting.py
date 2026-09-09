# © Artur Czarnecki. All rights reserved.

"""Artifact serialization and human report rendering for DS-E2E-14.3b."""

from __future__ import annotations

import json
from pathlib import Path

from intergrax.decision_system.qualification.serialization import (
    reliability_summary_to_dict,
    run_result_to_dict,
)

from testing_support.decision_e2e.reliability_qualification import (
    DecisionReliabilityQualificationResult,
    failure_boundary_distribution,
    failure_category_distribution,
    failure_owner_distribution,
    failure_reason_distribution,
    planner_round_distribution,
    terminal_outcome_distribution,
    tool_depth_distribution,
    tool_execution_distribution,
    tool_selection_distribution,
    top_model_failure_reasons,
)


def _pct(count: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{count / total:.1%}"


def _axis_outcome_label(outcome_value: str | None) -> str:
    if outcome_value is None:
        return "n/a"
    if outcome_value == "pass":
        return "PASS"
    if outcome_value == "fail":
        return "FAIL"
    return "NOT_EVALUABLE"


def _run_record_to_dict(
    result: DecisionReliabilityQualificationResult,
    record_index: int,
) -> dict[str, object]:
    record = result.runs[record_index]
    payload: dict[str, object] = {
        "run_index": record.run_index,
        "run_id": str(record.run_id) if record.run_id is not None else None,
        "valid_model_trial": record.valid_model_trial,
        "environment_event": record.environment_event,
        "completed": record.completed,
        "platform_passed": record.platform_passed,
        "model_passed": record.model_passed,
        "evaluator_passed": record.evaluator_passed,
        "provider_failed": record.provider_failed,
        "environment_failed": record.environment_failed,
        "observability_complete": record.observability_complete,
        "block_reason": record.block_reason,
        "run_result": None,
        "signals": None,
    }
    if record.run_result is not None:
        payload["run_result"] = run_result_to_dict(record.run_result)
    if record.signals is not None:
        payload["signals"] = {
            "selected_tool_ids": list(record.signals.selected_tool_ids),
            "executed_tool_ids": list(record.signals.executed_tool_ids),
            "tool_invocation_count": record.signals.tool_invocation_count,
            "planner_round_count": record.signals.planner_round_count,
            "evidence_node_count": record.signals.evidence_node_count,
            "initial_evidence_count": record.signals.initial_evidence_count,
            "follow_up_evidence_count": record.signals.follow_up_evidence_count,
            "evidence_gathering_stop_reason": record.signals.evidence_gathering_stop_reason,
            "terminal_outcome": record.signals.terminal_outcome,
            "model_completion_intent": record.signals.model_completion_intent,
            "reconciliation_result": record.signals.reconciliation_result,
            "critic_verdict_passed": record.signals.critic_verdict_passed,
            "evaluator_failures": list(record.signals.evaluator_failures),
            "validation_error_categories": list(record.signals.validation_error_categories),
            "strict_tool_capability": record.signals.strict_tool_capability,
            "trace_readback_pass": record.signals.trace_readback_pass,
            "trace_event_count": record.signals.trace_event_count,
            "route": record.signals.route,
            "stop_reason": record.signals.stop_reason,
            "reconciliation_error_reason": record.signals.reconciliation_error_reason,
            "reconciliation_model_intent": record.signals.reconciliation_model_intent,
            "reconciliation_has_supported_diagnosis": (
                record.signals.reconciliation_has_supported_diagnosis
            ),
            "reconciliation_validation_errors": list(
                record.signals.reconciliation_validation_errors
            ),
        }
    return payload


def qualification_result_to_summary_dict(
    result: DecisionReliabilityQualificationResult,
) -> dict[str, object]:
    summary = result.summary
    evaluator_fail_count = summary.evaluator_fail_count
    return {
        "qualification_id": result.provenance.qualification_id,
        "git_sha": result.provenance.git_sha,
        "plan": {
            "run_count": result.plan.run_count,
            "provider_id": result.plan.provider_id,
            "model_id": result.plan.model_id,
            "scenario_id": result.plan.scenario_id,
            "scenario_input_identity": result.plan.scenario_input_identity,
        },
        "provenance": {
            "started_at": result.provenance.started_at,
            "completed_at": result.provenance.completed_at,
            "fingerprint": {
                "provider": result.provenance.fingerprint.provider,
                "model": result.provenance.fingerprint.model,
                "scenario_id": result.provenance.fingerprint.scenario_id,
                "scenario_input_identity": result.provenance.fingerprint.scenario_input_identity,
                "taxonomy_version_sha": result.provenance.fingerprint.taxonomy_version_sha,
                "git_head": result.provenance.fingerprint.git_head,
            },
            "env_bootstrap": {
                "dotenv_discovered": result.provenance.env_bootstrap.dotenv_discovered,
                "dotenv_loaded": result.provenance.env_bootstrap.dotenv_loaded,
                "provider": result.provenance.env_bootstrap.provider,
                "model": result.provenance.env_bootstrap.model,
                "qualification_enabled": result.provenance.env_bootstrap.qualification_enabled,
                "credential_available": result.provenance.env_bootstrap.credential_available,
            },
        },
        "session_complete": result.session_complete,
        "planned_runs": result.plan.run_count,
        "valid_model_trials": result.valid_model_trial_count,
        "completed_runs": result.completed_run_count,
        "environment_failure_count": result.environment_failure_count,
        "reliability": reliability_summary_to_dict(summary),
        "evaluator_fail_count": evaluator_fail_count,
        "failure_category_distribution": failure_category_distribution(result.runs),
        "failure_reason_distribution": failure_reason_distribution(result.runs),
        "failure_boundary_distribution": failure_boundary_distribution(result.runs),
        "failure_owner_distribution": failure_owner_distribution(result.runs),
        "tool_selection_distribution": tool_selection_distribution(result.runs),
        "tool_execution_distribution": tool_execution_distribution(result.runs),
        "tool_depth_distribution": tool_depth_distribution(result.runs),
        "planner_round_distribution": planner_round_distribution(result.runs),
        "terminal_outcome_distribution": terminal_outcome_distribution(result.runs),
        "top_model_failure_reasons": [
            {"reason": reason.value, "count": count}
            for reason, count in top_model_failure_reasons(result.runs)
        ],
    }


def write_qualification_artifacts(
    result: DecisionReliabilityQualificationResult,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "runs.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    runs_payload = {
        "qualification_id": result.provenance.qualification_id,
        "git_sha": result.provenance.git_sha,
        "runs": [_run_record_to_dict(result, index) for index in range(len(result.runs))],
    }
    summary_payload = qualification_result_to_summary_dict(result)

    runs_path.write_text(json.dumps(runs_payload, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(render_qualification_report(result), encoding="utf-8")
    return runs_path, summary_path, report_path


def render_qualification_report(result: DecisionReliabilityQualificationResult) -> str:
    summary = result.summary
    total = summary.total_runs
    evaluator_fail = summary.evaluator_fail_count
    top_failures = top_model_failure_reasons(result.runs)
    lines = [
        "# DS-E2E-14.3b — Model Reliability Qualification",
        "",
        "## Config / Provenance",
        f"- qualification_id: `{result.provenance.qualification_id}`",
        f"- git_sha: `{result.provenance.git_sha}`",
        f"- provider: `{result.plan.provider_id}`",
        f"- model: `{result.plan.model_id}`",
        f"- scenario: `{result.plan.scenario_id}`",
        f"- scenario_input_identity: `{result.plan.scenario_input_identity}`",
        f"- planned_runs: `{result.plan.run_count}`",
        f"- dotenv_discovered: `{result.provenance.env_bootstrap.dotenv_discovered}`",
        f"- dotenv_loaded: `{result.provenance.env_bootstrap.dotenv_loaded}`",
        f"- qualification_enabled: `{result.provenance.env_bootstrap.qualification_enabled}`",
        f"- credential_available: `{result.provenance.env_bootstrap.credential_available}`",
        "",
        "## Metric Summary",
        "",
        "| Metric | Count / Rate |",
        "|---|---|",
        f"| Platform pass | {summary.platform_pass_count}/{summary.platform_evaluable_count} ({_pct(summary.platform_pass_count, summary.platform_evaluable_count)}) |",
        f"| Platform fail | {summary.platform_failure_count}/{summary.platform_evaluable_count} ({_pct(summary.platform_failure_count, summary.platform_evaluable_count)}) |",
        f"| Platform not evaluable | {summary.platform_not_evaluable_count}/{total} |",
        f"| Platform reliability | {_pct(summary.platform_pass_count, summary.platform_evaluable_count)} |",
        f"| Platform coverage | {_pct(summary.platform_evaluable_count, total)} |",
        f"| Model pass | {summary.model_pass_count}/{summary.model_evaluable_count} ({_pct(summary.model_pass_count, summary.model_evaluable_count)}) |",
        f"| Model fail | {summary.model_failure_count}/{summary.model_evaluable_count} ({_pct(summary.model_failure_count, summary.model_evaluable_count)}) |",
        f"| Model not evaluable | {summary.model_not_evaluable_count}/{total} |",
        f"| Model reliability | {_pct(summary.model_pass_count, summary.model_evaluable_count)} |",
        f"| Model coverage | {_pct(summary.model_evaluable_count, total)} |",
        f"| Evaluator pass | {summary.evaluator_pass_count}/{summary.evaluator_evaluable_count} ({_pct(summary.evaluator_pass_count, summary.evaluator_evaluable_count)}) |",
        f"| Evaluator fail | {evaluator_fail}/{summary.evaluator_evaluable_count} ({_pct(evaluator_fail, summary.evaluator_evaluable_count)}) |",
        f"| Evaluator not evaluable | {summary.evaluator_not_evaluable_count}/{total} |",
        f"| Evaluator pass rate | {_pct(summary.evaluator_pass_count, summary.evaluator_evaluable_count)} |",
        f"| Evaluator coverage | {_pct(summary.evaluator_evaluable_count, total)} |",
        f"| Provider infra failures | {summary.provider_infra_failure_count} |",
        f"| Environment failures | {result.environment_failure_count} |",
        f"| Observability gaps | {summary.observability_gap_count} |",
        "",
        "## Failure Distributions",
        "",
        "### Category",
    ]
    for key, value in sorted(failure_category_distribution(result.runs).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "### Reason"])
    for key, value in sorted(failure_reason_distribution(result.runs).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "### Boundary"])
    for key, value in sorted(failure_boundary_distribution(result.runs).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "### Owner"])
    for key, value in sorted(failure_owner_distribution(result.runs).items()):
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Run Matrix",
            "",
            "| Run | RunId | Platform | Model | Evaluator | Category | Reason | Boundary | Owner | Tools | Evidence | Stop reason | Outcome |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for record in result.runs:
        classification = (
            record.run_result.classification if record.run_result is not None else None
        )
        platform_label = "n/a"
        model_label = "n/a"
        evaluator_label = "n/a"
        if record.run_result is not None:
            platform_label = _axis_outcome_label(record.run_result.platform_outcome.value)
            model_label = _axis_outcome_label(record.run_result.model_outcome.value)
            evaluator_label = _axis_outcome_label(record.run_result.evaluator_outcome.value)
        tools = ""
        evidence = ""
        stop_reason = ""
        outcome = ""
        if record.signals is not None:
            tools = str(record.signals.tool_invocation_count)
            evidence = str(record.signals.evidence_node_count)
            stop_reason = record.signals.stop_reason or ""
            outcome = record.signals.terminal_outcome or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.run_index),
                    str(record.run_id) if record.run_id is not None else "n/a",
                    platform_label,
                    model_label,
                    evaluator_label,
                    classification.category.value if classification else "NONE",
                    classification.reason.value if classification else "",
                    classification.boundary.value if classification else "",
                    classification.owner.value if classification else "",
                    tools,
                    evidence,
                    stop_reason,
                    outcome,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Historical Comparison",
            "- DS-E2E-14.1b: platform=100%, model=0%",
            f"- DS-E2E-14.3b: platform={_pct(summary.platform_pass_count, summary.platform_evaluable_count)}, model={_pct(summary.model_pass_count, summary.model_evaluable_count)}",
            "- directional: insufficient evidence for strict statistical comparison at n=20 vs n=5",
            "",
            "## Top Model Failure Modes",
        ]
    )
    for index, (reason, count) in enumerate(top_failures, start=1):
        lines.append(f"{index}. {reason.value} ({count})")
    if not top_failures:
        lines.append("1. none observed")

    harness_pass = result.completed_run_count == result.plan.run_count
    qualification_status = "PASS" if result.session_complete and harness_pass else "BLOCKED"
    lines.extend(
        [
            "",
            "## Conclusion",
            f"- QUALIFICATION HARNESS: {'PASS' if harness_pass else 'FAIL'}",
            f"- SESSION COMPLETE: {result.session_complete}",
            f"- PLATFORM RELIABILITY: {summary.platform_pass_count}/{summary.platform_evaluable_count}",
            f"- MODEL RELIABILITY: {summary.model_pass_count}/{summary.model_evaluable_count}",
            f"- EVALUATOR PASS RATE: {summary.evaluator_pass_count}/{summary.evaluator_evaluable_count}",
            f"- STATUS: {qualification_status}",
        ]
    )
    return "\n".join(lines) + "\n"
