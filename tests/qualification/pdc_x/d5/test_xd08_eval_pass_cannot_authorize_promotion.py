# © Artur Czarnecki. All rights reserved.

"""PDC-D5-R014 (xd_08): an evaluation PASS can never authorize promotion."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest
from intergrax.contracts.evaluation_contracts import EvalSubjectOutput, EvalThresholds
from intergrax.eval.eval_comparison import compare_eval_runs
from intergrax.runtime.architecture.agent_certification import AgentCertificationEvaluation
from intergrax.runtime.architecture.agent_promotion import (
    PromotionEvidenceBundle,
    PromotionStage,
    evaluate_agent_promotion,
)
from tests.qualification.pdc_x.d5._support import ANSWERS, build_run

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
GATE_SCRIPT = REPO_ROOT / "scripts" / "gates" / "check_mvp_promotion_gates.py"


def _bundle(**overrides: object) -> PromotionEvidenceBundle:
    base: dict[str, object] = {
        "agent_id": "agent:research",
        "agent_version": "1.0.0",
        "source_stage": PromotionStage.STAGING,
        "target_stage": PromotionStage.PRODUCTION,
        "certification": AgentCertificationEvaluation(
            agent_id="agent:research", agent_version="1.0.0", eligible=True, reasons=[]
        ),
        "evaluation_report_refs": ["eval/report.json"],
        "rollback_plan_ref": "runbook/rollback",
        "change_ticket_ref": "CHG-1",
    }
    base.update(overrides)
    return PromotionEvidenceBundle(**base)  # type: ignore[arg-type]


def _clean_comparison():
    return compare_eval_runs(build_run(), build_run())


def test_xd08_eval_pass_alone_never_authorizes() -> None:
    decision = evaluate_agent_promotion(
        PromotionEvidenceBundle(
            agent_id="agent:research",
            agent_version="1.0.0",
            source_stage=PromotionStage.STAGING,
            target_stage=PromotionStage.PRODUCTION,
            certification=AgentCertificationEvaluation(
                agent_id="agent:research", agent_version="1.0.0", eligible=False, reasons=["no"]
            ),
            evaluation_report_refs=["eval/report.json"],
            evaluation_comparisons=[_clean_comparison()],
        )
    )
    assert decision.approved is False
    assert decision.governance_authorized is False
    assert decision.evaluation_advisory_only is True


@pytest.mark.parametrize(
    "missing",
    [
        {
            "certification": AgentCertificationEvaluation(
                agent_id="agent:research", agent_version="1.0.0", eligible=False, reasons=["x"]
            )
        },
        {"change_ticket_ref": ""},
        {"change_ticket_ref": "   "},
        {"rollback_plan_ref": ""},
        {"target_stage": PromotionStage.DEV},
    ],
)
def test_xd08_each_governance_input_is_required_despite_a_clean_eval(
    missing: dict[str, object],
) -> None:
    decision = evaluate_agent_promotion(_bundle(evaluation_comparisons=[_clean_comparison()], **missing))
    assert decision.approved is False
    assert decision.governance_authorized is False


def test_xd08_governance_decision_is_independent_of_evaluation_input() -> None:
    with_eval = evaluate_agent_promotion(_bundle(evaluation_comparisons=[_clean_comparison()]))
    without_refs = evaluate_agent_promotion(_bundle(evaluation_report_refs=[]))
    blocked = evaluate_agent_promotion(
        _bundle(evaluation_comparisons=[compare_eval_runs(build_run(), _regressed_run())])
    )
    assert with_eval.governance_authorized is True
    assert without_refs.governance_authorized is True
    assert blocked.governance_authorized is True
    assert with_eval.approved is True
    assert without_refs.approved is False  # evidence is required, but never sufficient alone
    assert blocked.approved is False
    assert with_eval.advisory_notes and "does not authorize" in with_eval.advisory_notes[0]


def _regressed_run():
    answers = dict(ANSWERS)
    answers["s-math"] = EvalSubjectOutput(output_text="5")
    return build_run(answers=answers)


def test_xd08_regression_and_threshold_breach_block_promotion() -> None:
    regressed = compare_eval_runs(build_run(), _regressed_run())
    decision = evaluate_agent_promotion(_bundle(evaluation_comparisons=[regressed]))
    assert decision.approved is False
    assert any("regression" in reason for reason in decision.reasons)

    breached = compare_eval_runs(build_run(), build_run(), thresholds=EvalThresholds(max_cost_to_success=0.5))
    decision = evaluate_agent_promotion(_bundle(evaluation_comparisons=[breached]))
    assert decision.approved is False
    assert any("threshold breach" in reason for reason in decision.reasons)


def test_xd08_blank_evaluation_ref_does_not_count_as_evidence() -> None:
    decision = evaluate_agent_promotion(_bundle(evaluation_report_refs=["", "  "]))
    assert decision.approved is False
    assert "Missing evaluation report evidence" in decision.reasons


def test_xd08_comparison_contract_cannot_claim_authority() -> None:
    comparison = _clean_comparison()
    assert comparison.advisory is True
    with pytest.raises(ValueError):
        type(comparison).model_validate({**comparison.model_dump(), "advisory": False})
    assert not hasattr(comparison, "approved")
    assert not hasattr(comparison, "authorized")


def _load_gate_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_mvp_promotion_gates_under_test", GATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _copy_required_files(gates: ModuleType, destination: Path) -> None:
    paths: list[str] = [
        "intergrax/scaffold/cli.py",
        "intergrax/cli/doctor.py",
        "intergrax/cli/run.py",
        "applications/lab_application/host/factory.py",
        "intergrax/runtime/architecture/evaluation_automation.py",
        "intergrax/runtime/architecture/online_evaluation_registry.py",
        "scripts/maintenance/check_harness_evaluation_wiring.py",
        "intergrax/contracts/resilience_policy.py",
        "intergrax/contracts/autonomy_level.py",
        "intergrax/runtime/policy/autonomy_resolver.py",
        "scripts/maintenance/check_harness_resilience_policy.py",
    ]
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)


def test_xd08_gate_script_is_not_satisfied_by_file_existence_alone(tmp_path: Path) -> None:
    gates = _load_gate_module()
    _copy_required_files(gates, tmp_path)
    ok_report = gates.run_gates(None, tmp_path)
    assert ok_report.ready
    # Replace the evaluation modules by empty files: existence stays, definitions vanish.
    for relative in (
        "intergrax/runtime/architecture/evaluation_automation.py",
        "intergrax/runtime/architecture/online_evaluation_registry.py",
    ):
        (tmp_path / relative).write_text("# empty\n", encoding="utf-8")
    hollow = gates.run_gates(None, tmp_path)
    assert not hollow.ready
    assert "G1" in hollow.structural_failures
    assert "lack required definitions" in " ".join(hollow.lines)


def test_xd08_gate_script_never_authorizes_from_evaluation_and_readiness_alone(
    tmp_path: Path,
) -> None:
    gates = _load_gate_module()
    _copy_required_files(gates, tmp_path)
    report = gates.run_gates(None, tmp_path)
    assert report.ready  # every gate, including the evaluation baseline, is green
    assert report.authorized is False


def test_xd08_gate_script_authorizes_only_from_a_governance_ticket(tmp_path: Path) -> None:
    gates = _load_gate_module()
    _copy_required_files(gates, tmp_path)
    tickets = tmp_path / "governance" / "tickets"
    tickets.mkdir(parents=True)
    (tickets / "T-1.md").write_text("## Approved\n\nApproved: operator @ 2026-11-12T10:00:00+01:00\n", encoding="utf-8")
    (tickets / "T-2.md").write_text("## Approved\n\nApproved: pending\n", encoding="utf-8")
    (tickets / "T-3.md").write_text("no approval here\n", encoding="utf-8")
    (tickets / "T-4.md").write_text("Approved: operator-delegate @ 2026-11-12T10:00:00+01:00\n", encoding="utf-8")
    (tmp_path / "outside.md").write_text("Approved: operator @ now\n", encoding="utf-8")

    assert gates.run_gates("governance/tickets/T-1.md", tmp_path).authorized is True
    for rejected in (
        "governance/tickets/T-2.md",
        "governance/tickets/T-3.md",
        "governance/tickets/T-4.md",
        "governance/tickets/missing.md",
        "outside.md",
        "../outside.md",
    ):
        assert gates.run_gates(rejected, tmp_path).authorized is False, rejected


def test_xd08_gate_script_real_repo_is_ready_but_not_authorized() -> None:
    gates = _load_gate_module()
    report = gates.run_gates(None, REPO_ROOT)
    assert report.ready, report.lines
    assert report.authorized is False
