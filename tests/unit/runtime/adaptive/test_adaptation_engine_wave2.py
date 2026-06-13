# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-2: adaptation engine, governance pipeline, and scheduler tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.runtime.adaptive.adaptation_engine import AdaptationEngine
from intergrax.runtime.adaptive.adaptation_models import AdaptationEngineContext
from intergrax.runtime.adaptive.adaptation_scheduler import AdaptationScheduler
from intergrax.runtime.adaptive.bandit_state_store import InMemoryBanditStateStore
from intergrax.runtime.adaptive.contracts import HarnessOutcomeSignal
from intergrax.runtime.adaptive.cost_anomaly_bridge import proposals_from_cost_anomalies
from intergrax.runtime.adaptive.evaluation_feedback_engine import EvaluationFeedbackEngine
from intergrax.runtime.adaptive.execution_strategy_engine import ExecutionStrategyEngine
from intergrax.runtime.adaptive.governance_pipeline import (
    AdaptationGovernancePipeline,
    validate_evaluation_assets_bundle,
)
from intergrax.runtime.adaptive.policy_learning_engine import PolicyLearningEngine
from intergrax.runtime.adaptive.proposal_builder import ProposalBuilder
from intergrax.runtime.adaptive.proposal_cooldown_store import InMemoryProposalCooldownStore
from intergrax.runtime.adaptive.proposal_store import InMemoryProposalStore, SQLiteProposalStore
from intergrax.runtime.adaptive.routing_tuning_engine import RoutingTuningEngine
from intergrax.runtime.adaptive.signal_store import InMemorySignalStore
from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)
from intergrax.runtime.architecture.cost_forecast import CostAnomalyRecord, CostAnomalySeverity
from intergrax.runtime.architecture.evaluation_assets import (
    EvaluationAssetBundle,
    GoldenDatasetAsset,
    ScenarioCase,
    ScenarioLibraryAsset,
)
from intergrax.runtime.architecture.evaluation_registry_trends import (
    EvaluationComparisonSummary,
    EvaluationRegistryTrendReport,
)
from scripts.phase_w_adapt_report import build_proposal_report

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

REPO_ROOT = Path(__file__).resolve().parents[4]


def _low_utility_signal(*, task_class: str = "echo.basic") -> HarnessOutcomeSignal:
    return HarnessOutcomeSignal(
        run_id="run_low",
        tenant_id="tenant_a",
        application_id="lab",
        agent_id="echo",
        task_class=task_class,
        utility=0.2,
        step_count=4,
    )


def _build_engine(
    *,
    sub_engines: list | None = None,
    proposal_store: InMemoryProposalStore | None = None,
) -> AdaptationEngine:
    bandit_store = InMemoryBanditStateStore()
    governance = AdaptationGovernancePipeline()
    builder = ProposalBuilder(governance)
    engines = sub_engines or [RoutingTuningEngine(bandit_store)]
    return AdaptationEngine(
        sub_engines=engines,
        proposal_builder=builder,
        bandit_store=bandit_store,
        cooldown_store=InMemoryProposalCooldownStore(),
        proposal_store=proposal_store,
    )


def test_routing_tuning_engine_proposes_when_utility_low() -> None:
    bandit_store = InMemoryBanditStateStore()
    engine = RoutingTuningEngine(bandit_store, utility_threshold=0.45)
    context = AdaptationEngineContext(
        tenant_id="tenant_a",
        task_class="echo.basic",
        signals=[_low_utility_signal()],
    )
    candidates = engine.propose(context)
    assert len(candidates) == 1
    assert candidates[0].source_engine == "routing_tuning"
    assert candidates[0].proposal.envelope.kind.value == "routing_tuning"


def test_routing_tuning_engine_skips_when_utility_high() -> None:
    bandit_store = InMemoryBanditStateStore()
    engine = RoutingTuningEngine(bandit_store, utility_threshold=0.45)
    signal = _low_utility_signal()
    signal = signal.model_copy(update={"utility": 0.9})
    context = AdaptationEngineContext(
        tenant_id="tenant_a",
        task_class="echo.basic",
        signals=[signal],
    )
    assert engine.propose(context) == []


def test_execution_strategy_engine_proposes_on_step_explosion() -> None:
    engine = ExecutionStrategyEngine(step_count_threshold=12)
    signal = _low_utility_signal().model_copy(
        update={"step_count": 20, "regression_flags": ["step_explosion"]}
    )
    context = AdaptationEngineContext(
        tenant_id="tenant_a",
        task_class="echo.basic",
        signals=[signal],
    )
    candidates = engine.propose(context)
    assert len(candidates) == 1
    assert candidates[0].source_engine == "execution_strategy"


def test_policy_learning_engine_requires_human_approver_in_proposal() -> None:
    engine = PolicyLearningEngine()
    signal = _low_utility_signal().model_copy(
        update={"regression_flags": ["tool_usage_drop"]}
    )
    context = AdaptationEngineContext(
        tenant_id="tenant_a",
        task_class="echo.basic",
        signals=[signal],
        default_human_approver_id="owner:ops",
    )
    candidates = engine.propose(context)
    assert len(candidates) == 1
    assert candidates[0].proposal.human_approver_id == "owner:ops"


def test_evaluation_feedback_engine_proposes_on_regression_trend() -> None:
    engine = EvaluationFeedbackEngine()
    trend = EvaluationRegistryTrendReport(
        comparisons=[
            EvaluationComparisonSummary(
                release_from="r1",
                release_to="r2",
                pass_rate_from=0.9,
                pass_rate_to=0.7,
                delta=-0.2,
            )
        ]
    )
    context = AdaptationEngineContext(
        tenant_id="tenant_a",
        task_class="echo.basic",
        signals=[_low_utility_signal()],
        evaluation_trend=trend,
    )
    candidates = engine.propose(context)
    assert len(candidates) == 1
    assert candidates[0].proposal.envelope.authority.value == "observe_only"


def test_cost_anomaly_bridge_emits_routing_candidate() -> None:
    context = AdaptationEngineContext(
        tenant_id="tenant_a",
        task_class="echo.basic",
        signals=[_low_utility_signal()],
        cost_anomalies=[
            CostAnomalyRecord(
                scope_id="tenant_a",
                severity=CostAnomalySeverity.WARNING,
                spend_delta_ratio=0.3,
                token_delta_ratio=0.1,
                reasons=["spend drift"],
            )
        ],
    )
    candidates = proposals_from_cost_anomalies(context)
    assert len(candidates) == 1
    assert candidates[0].source_engine == "cost_anomaly_bridge"


def test_governance_pipeline_rejects_incompatible_capability_graph() -> None:
    previous = CapabilityGraph(
        nodes=[
            CapabilityNode(node_id="integration:sqlite", node_type=CapabilityNodeType.INTEGRATION),
            CapabilityNode(node_id="tool:rag.retrieve", node_type=CapabilityNodeType.TOOL),
        ],
        edges=[
            CapabilityEdge(
                source_node_id="tool:rag.retrieve",
                target_node_id="integration:sqlite",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            )
        ],
    )
    current = CapabilityGraph(
        nodes=[CapabilityNode(node_id="integration:sqlite", node_type=CapabilityNodeType.INTEGRATION)],
        edges=[],
    )
    engine = _build_engine()
    context = AdaptationEngineContext(
        tenant_id="tenant_a",
        task_class="echo.basic",
        signals=[_low_utility_signal()],
        capability_graph_previous=previous,
        capability_graph_candidate=current,
    )
    result = engine.run(context)
    assert result.packages
    assert result.packages[0].capability_gate_passed is False
    assert result.packages[0].passed_all_gates is False


def test_governance_pipeline_rejects_low_golden_scenario_pass_rate() -> None:
    engine = _build_engine()
    context = AdaptationEngineContext(
        tenant_id="tenant_a",
        task_class="echo.basic",
        signals=[_low_utility_signal()],
        golden_scenario_pass_rate=0.5,
        golden_scenario_min_pass_rate=0.7,
    )
    result = engine.run(context)
    assert result.packages
    assert result.packages[0].golden_scenario_gate_passed is False


def test_adaptation_engine_applies_cooldown() -> None:
    engine = _build_engine()
    context = AdaptationEngineContext(
        tenant_id="tenant_a",
        task_class="echo.basic",
        signals=[_low_utility_signal()],
    )
    first = engine.run(context)
    assert first.packages
    second = engine.run(context)
    assert second.packages == []
    assert second.skipped_cooldown_loop_ids


def test_adaptation_engine_persists_runs_in_proposal_store(tmp_path) -> None:
    store = SQLiteProposalStore(db_path=tmp_path / "proposals.db")
    engine = _build_engine(proposal_store=store)
    context = AdaptationEngineContext(
        tenant_id="tenant_a",
        task_class="echo.basic",
        signals=[_low_utility_signal()],
    )
    engine.run(context)
    runs = store.list_runs(tenant_id="tenant_a")
    assert len(runs) == 1
    assert runs[0].packages


def test_adaptation_scheduler_groups_signals_by_tenant_and_task_class() -> None:
    signal_store = InMemorySignalStore()
    signal_store.append(_low_utility_signal(task_class="echo.basic"))
    signal_store.append(
        _low_utility_signal(task_class="echo.advanced").model_copy(update={"run_id": "run_b"})
    )
    engine = _build_engine()
    scheduler = AdaptationScheduler(engine=engine, signal_store=signal_store)
    results = scheduler.run_adaptation_engine()
    assert len(results) == 2
    task_classes = {result.task_class for result in results}
    assert task_classes == {"echo.basic", "echo.advanced"}


def test_validate_evaluation_assets_bundle_accepts_consistent_bundle() -> None:
    bundle = EvaluationAssetBundle(
        scenario_libraries=[
            ScenarioLibraryAsset(
                library_id="lib1",
                version="1.0.0",
                scenarios=[ScenarioCase(scenario_id="s1", description="smoke")],
            )
        ],
        datasets=[
            GoldenDatasetAsset(
                dataset_id="ds1",
                version="1.0.0",
                storage_ref="s3://bucket/ds1",
                scenario_ids=["s1"],
            )
        ],
    )
    assert validate_evaluation_assets_bundle(bundle) is True


def test_build_proposal_report_summarizes_gate_results(tmp_path) -> None:
    store = SQLiteProposalStore(db_path=tmp_path / "proposals.db")
    engine = _build_engine(proposal_store=store)
    engine.run(
        AdaptationEngineContext(
            tenant_id="tenant_a",
            task_class="echo.basic",
            signals=[_low_utility_signal()],
        )
    )
    report = build_proposal_report(store)
    assert report.run_count == 1
    assert report.proposal_count >= 1
    assert report.passed_gate_count >= 0


def test_phase_w_adapt_report_cli_writes_proposals_json(tmp_path) -> None:
    proposals_db = tmp_path / "proposals.db"
    store = SQLiteProposalStore(db_path=proposals_db)
    engine = _build_engine(proposal_store=store)
    engine.run(
        AdaptationEngineContext(
            tenant_id="tenant_cli",
            task_class="echo.basic",
            signals=[
                HarnessOutcomeSignal(
                    run_id="run_cli_prop",
                    tenant_id="tenant_cli",
                    application_id="lab",
                    agent_id="echo",
                    task_class="echo.basic",
                    utility=0.15,
                )
            ],
        )
    )
    output = tmp_path / "proposals.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "phase_w_adapt_report.py"),
            "--proposals-db-path",
            str(proposals_db),
            "--proposals-output",
            str(output),
            "--skip-signals",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run_count"] == 1
    assert payload["proposal_count"] >= 1
