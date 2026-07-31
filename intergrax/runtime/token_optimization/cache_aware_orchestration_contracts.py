# © Artur Czarnecki. All rights reserved.

"""Cache-aware orchestration contracts (TOKEN-10D-1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.runtime.token_optimization.contracts import (
    CacheAwareCompactionDecision,
    CacheAwareCompactionTimingDecision,
    CacheAwareCompactionTimingInput,
    TokenOptimizationPipelineResult,
)
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationLLMRouterRequest,
    TokenOptimizationLLMRouterResult,
    TokenOptimizationRouterStatus,
)


class CacheAwareTokenOptimizationOrchestrationStatus(StrEnum):
    """Outcome of cache-aware orchestration between router and pipeline."""

    EXECUTED = "executed"
    DEFERRED = "deferred"
    BYPASSED = "bypassed"
    REVIEW_REQUIRED = "review_required"
    ROUTER_TERMINAL = "router_terminal"


@dataclass(frozen=True, slots=True)
class CacheAwareTokenOptimizationOrchestrationRequest:
    """Caller-supplied router request and cache-aware timing input."""

    router_request: TokenOptimizationLLMRouterRequest
    timing_input: CacheAwareCompactionTimingInput


@dataclass(frozen=True, slots=True)
class CacheAwareTokenOptimizationOrchestrationResult:
    """Cache-aware orchestration outcome (no raw content)."""

    router_result: TokenOptimizationLLMRouterResult
    timing_decision: CacheAwareCompactionTimingDecision | None
    orchestration_status: CacheAwareTokenOptimizationOrchestrationStatus
    pipeline_result: TokenOptimizationPipelineResult | None
    executed: bool
    review_required: bool

    def __post_init__(self) -> None:
        status = self.orchestration_status
        timing = self.timing_decision
        pipeline = self.pipeline_result

        if status is CacheAwareTokenOptimizationOrchestrationStatus.EXECUTED:
            if timing is None:
                raise ValueError("EXECUTED requires timing_decision")
            if timing.decision is not CacheAwareCompactionDecision.RUN:
                raise ValueError("EXECUTED requires timing_decision.decision=RUN")
            if pipeline is None:
                raise ValueError("EXECUTED requires pipeline_result")
            if not self.executed:
                raise ValueError("EXECUTED requires executed=True")
            if self.review_required:
                raise ValueError("EXECUTED requires review_required=False")
            return

        if status is CacheAwareTokenOptimizationOrchestrationStatus.DEFERRED:
            if timing is None:
                raise ValueError("DEFERRED requires timing_decision")
            if timing.decision is not CacheAwareCompactionDecision.DEFER:
                raise ValueError("DEFERRED requires timing_decision.decision=DEFER")
            if pipeline is not None:
                raise ValueError("DEFERRED requires pipeline_result=None")
            if self.executed:
                raise ValueError("DEFERRED requires executed=False")
            if self.review_required:
                raise ValueError("DEFERRED requires review_required=False")
            return

        if status is CacheAwareTokenOptimizationOrchestrationStatus.BYPASSED:
            if timing is None:
                raise ValueError("BYPASSED requires timing_decision")
            if timing.decision is not CacheAwareCompactionDecision.BYPASS:
                raise ValueError("BYPASSED requires timing_decision.decision=BYPASS")
            if pipeline is not None:
                raise ValueError("BYPASSED requires pipeline_result=None")
            if self.executed:
                raise ValueError("BYPASSED requires executed=False")
            if self.review_required:
                raise ValueError("BYPASSED requires review_required=False")
            return

        if status is CacheAwareTokenOptimizationOrchestrationStatus.REVIEW_REQUIRED:
            if timing is None:
                raise ValueError("REVIEW_REQUIRED requires timing_decision")
            if timing.decision is not CacheAwareCompactionDecision.REQUIRE_MANUAL_REVIEW:
                raise ValueError(
                    "REVIEW_REQUIRED requires timing_decision.decision=REQUIRE_MANUAL_REVIEW"
                )
            if pipeline is not None:
                raise ValueError("REVIEW_REQUIRED requires pipeline_result=None")
            if self.executed:
                raise ValueError("REVIEW_REQUIRED requires executed=False")
            if not self.review_required:
                raise ValueError("REVIEW_REQUIRED requires review_required=True")
            return

        if status is CacheAwareTokenOptimizationOrchestrationStatus.ROUTER_TERMINAL:
            if self.router_result.status is TokenOptimizationRouterStatus.ROUTED:
                raise ValueError("ROUTER_TERMINAL requires router status != ROUTED")
            if pipeline is not None:
                raise ValueError("ROUTER_TERMINAL requires pipeline_result=None")
            if self.executed:
                raise ValueError("ROUTER_TERMINAL requires executed=False")
            if self.review_required:
                raise ValueError("ROUTER_TERMINAL requires review_required=False")
