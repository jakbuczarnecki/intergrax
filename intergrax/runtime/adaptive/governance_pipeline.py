# © Artur Czarnecki. All rights reserved.

"""Multi-stage governance pipeline for adaptive proposals (Phase W-ADAPT-2.8–2.9)."""

from __future__ import annotations

from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationEngineContext,
    AdaptationProposalCandidate,
    AdaptationProposalPackage,
)
from intergrax.runtime.architecture.adaptive_governance import evaluate_bounded_adaptive_loop
from intergrax.runtime.architecture.capability_graph_compatibility import (
    evaluate_capability_graph_compatibility,
)
from intergrax.runtime.architecture.evaluation_assets import EvaluationAssetBundle


class AdaptationGovernancePipeline:
    """Runs envelope, capability graph, and golden-scenario gates."""

    def evaluate(
        self,
        candidate: AdaptationProposalCandidate,
        *,
        context: AdaptationEngineContext,
    ) -> AdaptationProposalPackage:
        reasons: list[str] = []
        envelope_gate = evaluate_bounded_adaptive_loop(candidate.proposal)
        if not envelope_gate.passed:
            reasons.extend(envelope_gate.reasons)

        capability_gate_passed = True
        capability_report = None
        if (
            candidate.profile_draft is not None
            and context.capability_graph_previous is not None
            and context.capability_graph_candidate is not None
        ):
            capability_report = evaluate_capability_graph_compatibility(
                previous=context.capability_graph_previous,
                current=context.capability_graph_candidate,
            )
            capability_gate_passed = capability_report.compatible
            if not capability_gate_passed:
                reasons.extend(issue.message for issue in capability_report.issues)

        golden_gate_passed = self._evaluate_golden_scenario_gate(context)
        if not golden_gate_passed:
            reasons.append(
                "Golden scenario pass rate below minimum threshold "
                f"({context.golden_scenario_pass_rate} < {context.golden_scenario_min_pass_rate})"
            )

        passed_all = envelope_gate.passed and capability_gate_passed and golden_gate_passed
        return AdaptationProposalPackage(
            candidate=candidate,
            envelope_gate=envelope_gate,
            capability_gate_passed=capability_gate_passed,
            capability_report=capability_report,
            golden_scenario_gate_passed=golden_gate_passed,
            passed_all_gates=passed_all,
            gate_reasons=reasons,
        )

    def _evaluate_golden_scenario_gate(self, context: AdaptationEngineContext) -> bool:
        if context.golden_scenario_pass_rate is None:
            return True
        return context.golden_scenario_pass_rate >= context.golden_scenario_min_pass_rate


def validate_evaluation_assets_bundle(bundle: EvaluationAssetBundle) -> bool:
    """Stage-4 helper: ensure golden asset bundle references are internally consistent."""
    try:
        EvaluationAssetBundle.model_validate(bundle.model_dump())
    except ValueError:
        return False
    return bool(bundle.datasets or bundle.scenario_libraries)
