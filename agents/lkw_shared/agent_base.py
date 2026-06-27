# © Artur Czarnecki. All rights reserved.

"""Shared LKW Reflex agent wiring for typed domain diagnostics."""

from __future__ import annotations

from abc import abstractmethod

from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.authoring.patterns.types import AgentEvaluation, CognitiveEvaluation
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_run import AgentRunError
from intergrax.contracts.agent_run_enums import AgentRunErrorCode, TerminalReason
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


class LkwReflexAgent(ReflexAgent):
    """Reflex agent base that propagates typed LKW domain diagnostics."""

    @abstractmethod
    def build_diagnostic_payloads(self, output: dict[str, object]) -> list[DiagnosticPayload]:
        raise NotImplementedError

    def _evaluation_to_outcome(
        self,
        evaluation: AgentEvaluation,
        output: dict[str, object],
        state_delta: dict[str, object],
    ) -> StepOutcome:
        payloads = self.build_diagnostic_payloads(output)
        if evaluation.verdict == CognitiveEvaluation.COMPLETE:
            return StepOutcome.complete(
                output=output,
                terminal_reason=TerminalReason.GOAL_MET,
                state_delta=state_delta,
                confidence=evaluation.confidence,
                diagnostic_payloads=payloads,
            )
        if evaluation.verdict == CognitiveEvaluation.CONTINUE:
            return StepOutcome.continue_with(state_delta, diagnostic_payloads=payloads)
        if evaluation.verdict == CognitiveEvaluation.REPLAN:
            return StepOutcome.replan(
                state_delta,
                diagnostics={"reason": evaluation.reason},
                diagnostic_payloads=payloads,
            )
        if evaluation.verdict == CognitiveEvaluation.HUMAN:
            return StepOutcome.pause_hitl(
                evaluation.reason or "human_required",
                state_delta=state_delta,
                diagnostic_payloads=payloads,
            )
        return StepOutcome.fail(
            errors=[
                AgentRunError(
                    code=AgentRunErrorCode.INTERNAL_ERROR,
                    message=evaluation.reason or CognitiveEvaluation.FAIL.value,
                )
            ],
            terminal_reason=TerminalReason.ERROR,
            state_delta=state_delta,
            diagnostic_payloads=payloads,
        )
