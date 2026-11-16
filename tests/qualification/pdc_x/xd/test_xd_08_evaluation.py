# © Artur Czarnecki. All rights reserved.

"""XD-08 (Evaluation <-> Model/Agent): reruns are digest-identical; an evaluation PASS authorizes nothing.

PDC-D5-R009 (reproducible runs) and PDC-D5-R014 (evaluation stays advisory).

Integrated version of the lane stand-ins in ``d5/test_xd08_*``: the system under evaluation is a real
model call. A provider is registered in the real ``LLMAdapterRegistry``, the adapter is created from an
``LLMProfile`` through ``create_adapter`` and every eval scenario is answered by
``adapter.generate_messages``; the real ``execute_eval_run`` and ``compare_eval_runs`` produce the
digests, and the real ``evaluate_agent_promotion`` consumes the comparison. Fakes: the provider
(deterministic canned answers, no network) only.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from intergrax.contracts.evaluation_contracts import (
    EvalRevisionRef,
    EvalRun,
    EvalScenario,
    EvalSubjectOutput,
)
from intergrax.eval.eval_comparison import compare_eval_runs
from intergrax.eval.evaluators import EvaluatorRegistry, register_existing_evaluators
from intergrax.eval.typed_run import EvalRunProvenance, execute_eval_run
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.profile import LLMProfile, create_adapter
from intergrax.runtime.architecture.agent_certification import AgentCertificationEvaluation
from intergrax.runtime.architecture.agent_promotion import (
    PromotionEvidenceBundle,
    PromotionStage,
    evaluate_agent_promotion,
)
from tests.qualification.pdc_x.d5._support import CODE_SHA, make_dataset

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[4]
PROVIDER = "xd08_fake"

_GOOD = {"2+2": "4", "capital of France": "Paris", "tools": "done"}
_REGRESSED = {"2+2": "5", "capital of France": "Paris", "tools": "done"}


class _CannedAdapter(BaseLLMAdapter):
    """External provider stand-in: deterministic answers keyed by the prompt (no randomness, no network)."""

    provider = PROVIDER

    def __init__(self, *, model: str = "m-1", answers: dict[str, str] | None = None, **_: object) -> None:
        super().__init__()
        self.model = model
        self._answers = answers if answers is not None else _GOOD
        self.calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def generate_messages(self, messages: Sequence[ChatMessage], **_: object) -> LLMAdapterResponse:
        self.calls += 1
        prompt = messages[-1].content or ""
        return LLMAdapterResponse(content=self._answers[prompt], provider=self.provider, model=self.model)


def _register() -> None:
    def factory(**kwargs: Any) -> _CannedAdapter:
        return _CannedAdapter(**kwargs)

    LLMAdapterRegistry.register(PROVIDER, factory, override=True)


def _adapter(model: str, *, regressed: bool = False) -> _CannedAdapter:
    _register()
    adapter = create_adapter(
        LLMProfile(provider=PROVIDER, model=model),
        answers=_REGRESSED if regressed else _GOOD,  # type: ignore[arg-type]
    )
    assert isinstance(adapter, _CannedAdapter) or hasattr(adapter, "generate_messages")
    return adapter  # type: ignore[return-value]


def _run(adapter, *, revision: str, seed: int = 7, tenant: str = "tenant-a") -> EvalRun:  # type: ignore[no-untyped-def]
    def executor(scenario: EvalScenario, run_seed: int) -> EvalSubjectOutput:
        prompt = str(scenario.input_payload["q"])
        response = adapter.generate_messages([ChatMessage(role="user", content=prompt)])
        return EvalSubjectOutput(
            output_text=response.content or "",
            tool_calls=("search", "fetch") if prompt == "tools" else (("search",) if "France" in prompt else ()),
            cited_ids=("doc-1",) if "France" in prompt else (),
            cost=1.0,
            latency_ms=7.0 * run_seed,  # noisy telemetry must not reach the digest
            tokens=10,
        )

    provenance = EvalRunProvenance(
        tenant_id=tenant,
        agent=EvalRevisionRef(ref_id="agent:research", revision="a-1"),
        model=EvalRevisionRef(ref_id=f"model:{adapter.provider}/{adapter.model}", revision=revision),
        prompt=EvalRevisionRef(ref_id="prompt:research", revision="p-1"),
        profile=EvalRevisionRef(ref_id="profile:default", revision="pr-1"),
        code_sha=CODE_SHA,
        seed=seed,
    )
    return execute_eval_run(
        dataset=make_dataset(),
        executor=executor,
        registry=register_existing_evaluators(EvaluatorRegistry()),
        provenance=provenance,
    )


# ----------------------------------------------------------------------- reproducibility (D5-R009)


def test_xd08_baseline_and_candidate_rerun_of_the_same_model_are_digest_identical() -> None:
    baseline = _run(_adapter("m-1"), revision="rev-1")
    candidate = _run(_adapter("m-1"), revision="rev-1")  # a freshly created adapter from the same profile
    assert baseline.run_id == candidate.run_id
    assert baseline.inputs_digest == candidate.inputs_digest
    assert baseline.result_digest == candidate.result_digest
    assert baseline.model_dump_json() == candidate.model_dump_json()
    comparison = compare_eval_runs(baseline, candidate)
    assert comparison.advisory is True and comparison.clean


def test_xd08_the_eval_run_pins_the_model_that_was_really_called() -> None:
    adapter = _adapter("m-1")
    run = _run(adapter, revision="rev-1")
    assert adapter.calls == 3, "one model call per scenario"
    assert run.model.ref_id == f"model:{adapter.provider}/{adapter.model}"
    other = _run(_adapter("m-2"), revision="rev-1")
    assert other.inputs_digest != run.inputs_digest, "a different model changes the pinned inputs"


def test_xd08_telemetry_noise_does_not_change_the_digest() -> None:
    quiet = _run(_adapter("m-1"), revision="rev-1", seed=7)
    noisy = _run(_adapter("m-1"), revision="rev-1", seed=7)
    assert quiet.result_digest == noisy.result_digest


@pytest.mark.parametrize("change", ["seed", "model_revision", "answer"])
def test_xd08_a_changed_pinned_input_changes_the_digest(change: str) -> None:
    base = _run(_adapter("m-1"), revision="rev-1")
    if change == "seed":
        other = _run(_adapter("m-1"), revision="rev-1", seed=8)
    elif change == "model_revision":
        other = _run(_adapter("m-1"), revision="rev-2")
    else:
        other = _run(_adapter("m-1", regressed=True), revision="rev-1")
    assert other.result_digest != base.result_digest


# ----------------------------------------------------------------------- advisory only (D5-R014)


def _bundle(*comparisons, eligible: bool = True, **overrides: Any) -> PromotionEvidenceBundle:  # type: ignore[no-untyped-def]
    base: dict[str, object] = {
        "agent_id": "agent:research",
        "agent_version": "1.0.0",
        "source_stage": PromotionStage.STAGING,
        "target_stage": PromotionStage.PRODUCTION,
        "certification": AgentCertificationEvaluation(
            agent_id="agent:research", agent_version="1.0.0", eligible=eligible, reasons=[] if eligible else ["no"]
        ),
        "evaluation_report_refs": ["eval/report.json"],
        "rollback_plan_ref": "runbook/rollback",
        "change_ticket_ref": "CHG-1",
        "evaluation_comparisons": list(comparisons),
    }
    base.update(overrides)
    return PromotionEvidenceBundle(**base)  # type: ignore[arg-type]


def test_xd08_a_passing_model_evaluation_alone_never_authorizes_promotion() -> None:
    clean = compare_eval_runs(_run(_adapter("m-1"), revision="rev-1"), _run(_adapter("m-1"), revision="rev-1"))
    decision = evaluate_agent_promotion(_bundle(clean, eligible=False, change_ticket_ref="", rollback_plan_ref=""))
    assert decision.approved is False
    assert decision.governance_authorized is False
    assert decision.evaluation_advisory_only is True


def test_xd08_a_regressed_model_candidate_blocks_promotion_even_with_every_governance_input() -> None:
    regressed = compare_eval_runs(
        _run(_adapter("m-1"), revision="rev-1"),
        _run(_adapter("m-2", regressed=True), revision="rev-2"),
    )
    assert regressed.regressed and regressed.scenario_regressions
    decision = evaluate_agent_promotion(_bundle(regressed))
    assert decision.governance_authorized is True, "governance is independent of the evaluation"
    assert decision.approved is False
    assert any("regression" in reason for reason in decision.reasons)


def test_xd08_governance_input_is_required_even_when_the_evaluation_is_clean() -> None:
    clean = compare_eval_runs(_run(_adapter("m-1"), revision="rev-1"), _run(_adapter("m-1"), revision="rev-1"))
    approved = evaluate_agent_promotion(_bundle(clean))
    assert approved.approved is True and approved.evaluation_advisory_only is True
    assert approved.advisory_notes and "does not authorize" in approved.advisory_notes[0]
    cases: list[dict[str, Any]] = [
        {"change_ticket_ref": ""},
        {"rollback_plan_ref": ""},
        {"target_stage": PromotionStage.DEV},
    ]
    for missing in cases:
        assert evaluate_agent_promotion(_bundle(clean, **missing)).approved is False


def test_xd08_the_evaluation_packages_hold_no_promotion_or_activation_authority() -> None:
    """Static boundary: nothing under intergrax/eval or the evaluation runtime imports a decision owner."""
    forbidden = ("agent_promotion", "agent_distribution", "agent_governance", "ActivationService")
    offenders: list[str] = []
    for root in ("intergrax/eval", "intergrax/runtime/evaluation"):
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or "", *(alias.name for alias in node.names)]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                if any(token in name for name in names for token in forbidden):
                    offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{getattr(node, 'lineno', 0)}")
    assert offenders == []
