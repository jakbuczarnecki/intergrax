#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Generate Phase V governance artifacts for V-AM.2, V-ALG.2, and V-EVAL.1."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Protocol

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from intergrax.runtime.architecture import (
    AutomatedEvaluationReport,
    EvaluationSignal,
    ArchitectureDebtReviewPolicy,
    DebtReviewCadence,
    EvaluationReleaseSnapshot,
    AgentLifecycleState,
    AgentLifecycleTransitionRequest,
    AgentCertificationEvaluation,
    EvaluationAssetBundle,
    EvaluationMode,
    EvaluationModeRequest,
    EvaluationModeResult,
    GoldenDatasetAsset,
    ProductionOwnerMetadata,
    ProductionOwnershipEvidence,
    PromotionEvidenceBundle,
    PromotionStage,
    ScenarioCase,
    ScenarioLibraryAsset,
    UnifiedEvaluationReport,
    ArchitectureMetricsSnapshot,
    build_catalog_capability_graph,
    compute_architecture_coverage,
    build_evaluation_registry_trend_report,
    build_metrics_pipeline_report,
    compute_architecture_metrics,
    evaluate_automated_results,
    evaluate_architecture_debt_governance,
    evaluate_agent_lifecycle_transition,
    evaluate_agent_promotion,
    evaluate_production_ownership,
)


class ReportWriter(Protocol):
    def write(self, *, output_path: Path, payload: BaseModel) -> None:
        ...


class JsonReportWriter:
    def write(self, *, output_path: Path, payload: BaseModel) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def _build_eval_mode_report() -> UnifiedEvaluationReport:
    requests = [
        EvaluationModeRequest(
            run_id="eval-offline-baseline",
            target_id="agent:echo",
            mode=EvaluationMode.OFFLINE,
            dataset_ref="datasets/eval/offline_baseline.jsonl",
        ),
        EvaluationModeRequest(
            run_id="eval-online-canary",
            target_id="agent:research",
            mode=EvaluationMode.ONLINE,
            traffic_slice_ref="traffic/canary/research-5pct",
        ),
        EvaluationModeRequest(
            run_id="eval-shadow-research",
            target_id="agent:research",
            mode=EvaluationMode.SHADOW,
            traffic_slice_ref="traffic/shadow/research-100pct",
        ),
        EvaluationModeRequest(
            run_id="eval-human-signoff",
            target_id="agent:legal",
            mode=EvaluationMode.HUMAN,
            reviewer_ref="human-review/legal-oncall",
        ),
    ]
    results = [
        EvaluationModeResult(
            run_id=request.run_id,
            target_id=request.target_id,
            mode=request.mode,
            success=True,
            score=0.90,
            evidence_refs=[f"evidence/{request.run_id}.json"],
        )
        for request in requests
    ]
    return UnifiedEvaluationReport(requests=requests, results=results)


def _build_automated_evaluation_report(
    evaluation_report: UnifiedEvaluationReport,
) -> AutomatedEvaluationReport:
    rule_signals_by_run_id: dict[str, list[EvaluationSignal]] = {
        "eval-offline-baseline": [
            EvaluationSignal(signal_id="format.valid_json", value=1.0, threshold=1.0),
            EvaluationSignal(signal_id="policy.no_forbidden_tool", value=1.0, threshold=1.0),
        ],
        "eval-online-canary": [
            EvaluationSignal(signal_id="latency.p95", value=0.84, threshold=0.80),
            EvaluationSignal(signal_id="safety.block_rate", value=0.90, threshold=0.80),
        ],
        "eval-shadow-research": [
            EvaluationSignal(signal_id="consistency.answer_shape", value=0.88, threshold=0.80),
            EvaluationSignal(signal_id="trace.completeness", value=0.95, threshold=0.90),
        ],
        "eval-human-signoff": [
            EvaluationSignal(signal_id="review.pass_ratio", value=0.92, threshold=0.85),
            EvaluationSignal(signal_id="escalation.correctness", value=0.90, threshold=0.85),
        ],
    }
    llm_judge_scores_by_run_id: dict[str, float] = {
        "eval-offline-baseline": 0.90,
        "eval-online-canary": 0.84,
        "eval-shadow-research": 0.88,
        "eval-human-signoff": 0.91,
    }
    return evaluate_automated_results(
        mode_results=evaluation_report.results,
        rule_signals_by_run_id=rule_signals_by_run_id,
        llm_judge_scores_by_run_id=llm_judge_scores_by_run_id,
    )


def _build_promotion_bundle() -> PromotionEvidenceBundle:
    return PromotionEvidenceBundle(
        agent_id="agent:research",
        agent_version="1.0.0",
        source_stage=PromotionStage.DEV,
        target_stage=PromotionStage.STAGING,
        certification=AgentCertificationEvaluation(
            agent_id="agent:research",
            agent_version="1.0.0",
            eligible=True,
            reasons=[],
        ),
        evaluation_report_refs=["build/architecture_hardening/unified_evaluation_report.json"],
        rollback_plan_ref="runbook/promotions/research_rollback.md",
        change_ticket_ref="CHG-2026-0602-001",
    )


def _build_lifecycle_transition_request() -> AgentLifecycleTransitionRequest:
    return AgentLifecycleTransitionRequest(
        agent_id="agent:research",
        agent_version="1.0.0",
        current_state=AgentLifecycleState.PRODUCTION,
        target_state=AgentLifecycleState.DEPRECATED,
        migration_window_days=30,
        migration_guide_ref="runbook/migrations/research_agent.md",
        deprecation_notice_ref="notices/agents/research_deprecation.md",
    )


def _build_production_ownership_evidence() -> ProductionOwnershipEvidence:
    return ProductionOwnershipEvidence(
        agent_id="agent:research",
        agent_version="1.0.0",
        production_eligible=True,
        owner=ProductionOwnerMetadata(
            team="harness-platform",
            owner="research-owner",
            on_call="research-oncall",
            escalation_channel="#harness-oncall",
        ),
        runbook_ref="runbook/agents/research_ops.md",
    )


def _build_evaluation_assets() -> EvaluationAssetBundle:
    scenario_library = ScenarioLibraryAsset(
        library_id="core-governance",
        version="1.0.0",
        scenarios=[
            ScenarioCase(
                scenario_id="scn.safe_tool_usage",
                description="Agent uses allowed tools only.",
                risk_tags=["policy", "tooling"],
            ),
            ScenarioCase(
                scenario_id="scn.hitl_escalation",
                description="Agent escalates to human when confidence is low.",
                risk_tags=["human", "safety"],
            ),
        ],
    )
    dataset = GoldenDatasetAsset(
        dataset_id="golden.core.v1",
        version="1.0.0",
        storage_ref="datasets/golden/core_v1.jsonl",
        scenario_ids=["scn.safe_tool_usage", "scn.hitl_escalation"],
    )
    return EvaluationAssetBundle(
        datasets=[dataset],
        scenario_libraries=[scenario_library],
    )


def main() -> int:
    output_dir = REPO_ROOT / "build" / "architecture_hardening"
    writer: ReportWriter = JsonReportWriter()

    graph = build_catalog_capability_graph()
    previous_snapshot = ArchitectureMetricsSnapshot(
        snapshot_id="baseline",
        report=compute_architecture_metrics(graph),
    )
    current_snapshot = ArchitectureMetricsSnapshot(
        snapshot_id="current",
        report=compute_architecture_metrics(graph),
    )
    metrics_pipeline_report = build_metrics_pipeline_report(
        snapshots=[previous_snapshot, current_snapshot]
    )
    coverage_report = compute_architecture_coverage(graph)
    debt_governance_report = evaluate_architecture_debt_governance(
        metrics_report=current_snapshot.report,
        policy=ArchitectureDebtReviewPolicy(
            cadence=DebtReviewCadence.BIWEEKLY,
            max_debt_index=0.50,
            owner_team="harness-architecture",
            runbook_ref="runbook/architecture/debt_review.md",
        ),
    )
    promotion_decision = evaluate_agent_promotion(_build_promotion_bundle())
    lifecycle_decision = evaluate_agent_lifecycle_transition(_build_lifecycle_transition_request())
    ownership_decision = evaluate_production_ownership(_build_production_ownership_evidence())
    evaluation_report = _build_eval_mode_report()
    automated_evaluation_report = _build_automated_evaluation_report(evaluation_report)
    evaluation_trend_report = build_evaluation_registry_trend_report(
        snapshots=[
            EvaluationReleaseSnapshot(
                release_id="2026.05",
                automated_report=automated_evaluation_report.model_copy(deep=True),
            ),
            EvaluationReleaseSnapshot(
                release_id="2026.06",
                automated_report=automated_evaluation_report,
            ),
        ]
    )
    evaluation_assets = _build_evaluation_assets()

    writer.write(
        output_path=output_dir / "architecture_metrics_pipeline_report.json",
        payload=metrics_pipeline_report,
    )
    writer.write(
        output_path=output_dir / "architecture_coverage_report.json",
        payload=coverage_report,
    )
    writer.write(
        output_path=output_dir / "architecture_debt_governance_report.json",
        payload=debt_governance_report,
    )
    writer.write(
        output_path=output_dir / "agent_promotion_decision_report.json",
        payload=promotion_decision,
    )
    writer.write(
        output_path=output_dir / "agent_lifecycle_decision_report.json",
        payload=lifecycle_decision,
    )
    writer.write(
        output_path=output_dir / "production_ownership_decision_report.json",
        payload=ownership_decision,
    )
    writer.write(
        output_path=output_dir / "unified_evaluation_report.json",
        payload=evaluation_report,
    )
    writer.write(
        output_path=output_dir / "automated_evaluation_report.json",
        payload=automated_evaluation_report,
    )
    writer.write(
        output_path=output_dir / "evaluation_registry_trend_report.json",
        payload=evaluation_trend_report,
    )
    writer.write(
        output_path=output_dir / "evaluation_assets_report.json",
        payload=evaluation_assets,
    )

    print("phase-v governance report: OK")
    print(f"artifacts: {output_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
