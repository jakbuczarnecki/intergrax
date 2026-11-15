"""IA-D5 adversarial probes A (contracts, evidence, comparison, promotion). Read-only against product code."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "D:/Projekty/intergrax/.claude/worktrees/pdc-x-ia-d5-fb28f3")
sys.path.insert(0, "D:/Projekty/intergrax/.claude/worktrees/pdc-x-ia-d5-fb28f3/tests/qualification/pdc_x/d5")

from _support import ANSWERS, build_run, make_dataset, provenance, executor_for  # noqa: E402
from intergrax.contracts.evaluation_contracts import *  # noqa: E402,F403
from intergrax.contracts.evaluation_contracts import EvalRun, EvalSubjectOutput, EvalScenario, EvalDataset, EvalRevisionRef, EvalScenarioTelemetry  # noqa: E402
from intergrax.eval.evaluators import EvaluatorRegistry, register_existing_evaluators, GuardrailSafetyEvaluator  # noqa: E402
from intergrax.eval.typed_run import execute_eval_run  # noqa: E402
from intergrax.eval.eval_comparison import compare_eval_runs  # noqa: E402
from intergrax.runtime.evidence.eval_evidence_contracts import (  # noqa: E402
    build_eval_run_evidence_report, read_eval_run_evidence, build_eval_comparison_evidence_result)
from intergrax.integrations.contracts.llm_guardrail import GuardrailScanResult  # noqa: E402


def p(name, val):
    print(f"[{name}] {val}")


# P1: executor error + safety-only registry -> run status
class Safe:
    slug = "fake"
    def scan_output(self, text, *, context=None):
        return GuardrailScanResult(allowed=True)
    def scan_input(self, *a, **k):
        return GuardrailScanResult(allowed=True)

reg = EvaluatorRegistry()
reg.register(GuardrailSafetyEvaluator(Safe()))
ds = EvalDataset(dataset_id="d", version="1", scenarios=(EvalScenario(scenario_id="s1"),))
def boom(sc, seed):
    raise RuntimeError("provider down")
run_err = execute_eval_run(dataset=ds, executor=boom, registry=reg, provenance=provenance())
rep = build_eval_run_evidence_report(run_err)
p("P1 executor-failed run evidence status", rep.results[0].status)
p("P1 scenario_successes", run_err.scenario_successes())
p("P1 telemetry error", run_err.telemetry[0].error)

# P2: empty-outcome runs compare clean
empty = EvalRun(tenant_id="tenant-a", agent=EvalRevisionRef(ref_id="a", revision="1"), model=EvalRevisionRef(ref_id="m", revision="1"),
                prompt=EvalRevisionRef(ref_id="p", revision="1"), profile=EvalRevisionRef(ref_id="pr", revision="1"),
                code_sha="abcdef1", dataset_id="d", dataset_version="1", dataset_digest=make_dataset().content_digest, seed=1)
cmp_ = compare_eval_runs(empty, empty)
p("P2 empty-vs-empty clean", cmp_.clean)
cmp2 = compare_eval_runs(build_run(), empty)
p("P2 good baseline vs empty candidate: clean/regressed", (cmp2.clean, cmp2.regressed, cmp2.scenario_regressions))


# P3: digest ignores telemetry error -> same result_digest, different successes
good = build_run()
tele = tuple(t.model_copy(update={"error": "executor failed: X"}) if t.scenario_id == "s-math" else t for t in good.telemetry)
bad_tel = good.model_copy(update={"telemetry": tele})
p("P3 result_digest equal despite telemetry error", good.result_digest == bad_tel.result_digest)
p("P3 successes good/bad", (good.metrics().successes, bad_tel.metrics().successes))
cost_tel = tuple(t.model_copy(update={"cost": 999.0}) for t in good.telemetry)
cost_run = good.model_copy(update={"telemetry": cost_tel})
rep2 = build_eval_run_evidence_report(cost_run)
p("P3 tampered cost accepted by read_eval_run_evidence", len(read_eval_run_evidence(rep2, tenant_id="tenant-a")) == 1)

# P4: code_sha trailing newline
try:
    e = EvalRun(**{**empty.model_dump(), "code_sha": "abcdef1\n"})
    p("P4 code_sha with trailing newline accepted", repr(e.code_sha))
except Exception as exc:
    p("P4 rejected", type(exc).__name__)
try:
    e = EvalRun(**{**empty.model_dump(), "dataset_digest": ds.content_digest + "\n"})
    p("P4 dataset_digest with trailing newline accepted", True)
except Exception as exc:
    p("P4b rejected", type(exc).__name__)

# P5: mutable payload inside frozen model changes digest after construction
sc = EvalScenario(scenario_id="x", input_payload={"a": 1})
d1 = sc.content_digest
sc.input_payload["a"] = 2
p("P5 frozen scenario digest changes after in-place mutation", d1 != sc.content_digest)
ds2 = EvalDataset(dataset_id="d", version="1", scenarios=(sc,))
dig_before = ds2.content_digest
sc.input_payload["a"] = 3
p("P5 dataset digest drifts after construction", dig_before != ds2.content_digest)

# P6: default=str hashing of set payload (nondeterminism across hash seeds) printed for caller to compare
sc_set = EvalScenario(scenario_id="x", input_payload={"tags": {"alpha", "beta", "gamma", "delta"}})
print("P6 set-payload digest:", sc_set.content_digest)

# P7: run/outcomes not validated against dataset (extra scenario ids)
rogue = good.model_copy(update={"outcomes": good.outcomes + (EvaluatorOutcome(evaluator_id="x", kind=EvaluatorKind.DETERMINISTIC, scenario_id="ghost", status=EvaluatorOutcomeStatus.PASSED),)})
p("P7 run with outcome for scenario outside dataset accepted (model_copy skips validation; direct ctor below)", True)
try:
    EvalRun(**{**good.model_dump(), "outcomes": rogue.outcomes})
    p("P7 direct ctor accepted ghost scenario", True)
except Exception as exc:
    p("P7 rejected", type(exc).__name__)

# P8: promotion decision with free-text refs
from intergrax.runtime.architecture.agent_promotion import PromotionEvidenceBundle, PromotionStage, evaluate_agent_promotion  # noqa: E402
from intergrax.runtime.architecture.agent_certification import AgentCertificationEvaluation  # noqa: E402
bundle = PromotionEvidenceBundle(agent_id="a", agent_version="1", source_stage=PromotionStage.STAGING, target_stage=PromotionStage.PRODUCTION,
    certification=AgentCertificationEvaluation(agent_id="OTHER", agent_version="9", eligible=True, reasons=[]),
    evaluation_report_refs=["x"], rollback_plan_ref="x", change_ticket_ref="x",
    evaluation_comparisons=[compare_eval_runs(good, good)])
dec = evaluate_agent_promotion(bundle)
p("P8 approved with free-text refs, mismatched certification agent, unrelated comparison", (dec.approved, dec.governance_authorized))
