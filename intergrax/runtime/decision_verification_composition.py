# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production composition for Decision Verification eval capabilities (DS-VER-PROD-COMP).

Neutral adapters bridge typed Verification ports to canonical Tier-0 eval services
via ``ToolWiringContext``. No Critic, Nexus, or provider-specific coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.decision_verification_stage import (
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.contracts.semantic_verification import (
    SemanticContentProvider,
    SemanticJudge,
    SemanticRubricRef,
    SemanticRubricResolver,
    SemanticVerificationIndependenceConfig,
)
from intergrax.contracts.trajectory_verification import (
    TrajectoryAgentIdProvider,
    TrajectoryEvaluator,
    TrajectoryVerificationStageConfig,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.decision_verification_observability import VerificationObserver
from intergrax.runtime.decision_verification_stages.semantic import (
    SEMANTIC_VERIFICATION_STAGE_KIND,
    SemanticVerificationStage,
)
from intergrax.runtime.decision_verification_stages.trajectory import (
    TRAJECTORY_VERIFICATION_STAGE_KIND,
    TrajectoryVerificationStage,
)
from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalJudgeOutput
from intergrax.tools.providers.eval.contracts import EvalTrajectoryInput, EvalTrajectoryOutput
from intergrax.tools.providers.eval.judge import eval_judge
from intergrax.tools.providers.eval.trajectory import eval_trajectory
from intergrax.tools.registry.wiring import ToolWiringContext

T = TypeVar("T")


def _semantic_judge_available(ctx: ToolWiringContext) -> bool:
    adapter = ctx.extras.get("llm_adapter")
    return adapter is not None and isinstance(adapter, LLMAdapter)


def _trajectory_evaluator_available(ctx: ToolWiringContext) -> bool:
    return ctx.trace_reader is not None


@dataclass(frozen=True, slots=True)
class ToolWiringSemanticJudge:
    """Production ``SemanticJudge`` over canonical ``eval_judge``."""

    ctx: ToolWiringContext

    def is_available(self) -> bool:
        return _semantic_judge_available(self.ctx)

    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
        return eval_judge(self.ctx, params)


@dataclass(frozen=True, slots=True)
class ToolWiringTrajectoryEvaluator:
    """Production ``TrajectoryEvaluator`` over canonical ``eval_trajectory``."""

    ctx: ToolWiringContext

    def is_available(self) -> bool:
        return _trajectory_evaluator_available(self.ctx)

    def evaluate(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
        return eval_trajectory(self.ctx, params)


@dataclass(frozen=True, slots=True)
class ToolWiringEvalVerificationBridge:
    """Shared bridge binding one ``ToolWiringContext`` to eval verification ports."""

    ctx: ToolWiringContext
    semantic_judge: ToolWiringSemanticJudge
    trajectory_evaluator: ToolWiringTrajectoryEvaluator


def compose_tool_wiring_eval_verification(
    ctx: ToolWiringContext,
) -> ToolWiringEvalVerificationBridge:
    """Compose neutral eval verification adapters from one wiring context."""
    if type(ctx) is not ToolWiringContext:
        raise TypeError("ctx must be ToolWiringContext")
    return ToolWiringEvalVerificationBridge(
        ctx=ctx,
        semantic_judge=ToolWiringSemanticJudge(ctx=ctx),
        trajectory_evaluator=ToolWiringTrajectoryEvaluator(ctx=ctx),
    )


@dataclass(frozen=True, slots=True)
class VerificationProductionCapabilities(Generic[T]):
    """Immutable production capability bundle for semantic and trajectory stages."""

    semantic_judge: SemanticJudge
    trajectory_evaluator: TrajectoryEvaluator
    rubric_resolver: SemanticRubricResolver | None
    semantic_content_provider: SemanticContentProvider[T] | None
    trajectory_agent_id_provider: TrajectoryAgentIdProvider[T] | None


@dataclass(frozen=True, slots=True)
class SemanticProductionStageSpec(Generic[T]):
    """Explicit semantic stage production configuration."""

    rubric_ref: SemanticRubricRef
    rubric_resolver: SemanticRubricResolver
    content_provider: SemanticContentProvider[T]
    independence: SemanticVerificationIndependenceConfig
    judge: SemanticJudge | None = None
    required: bool = True


@dataclass(frozen=True, slots=True)
class TrajectoryProductionStageSpec(Generic[T]):
    """Explicit trajectory stage production configuration."""

    agent_id_provider: TrajectoryAgentIdProvider[T]
    evaluator: TrajectoryEvaluator | None = None
    config: TrajectoryVerificationStageConfig = TrajectoryVerificationStageConfig()
    required: bool = True


@dataclass(frozen=True, slots=True)
class DecisionVerificationPipelineBuildSpec(Generic[T]):
    """Host-level immutable specification for one verification pipeline build."""

    eval_bridge: ToolWiringEvalVerificationBridge | None = None
    semantic: SemanticProductionStageSpec[T] | None = None
    trajectory: TrajectoryProductionStageSpec[T] | None = None
    extra_registrations: tuple[VerificationStageRegistration[T], ...] = ()
    observer: VerificationObserver[T] | None = None


def _resolve_semantic_judge(
    *,
    spec: SemanticProductionStageSpec[T],
    bridge: ToolWiringEvalVerificationBridge | None,
) -> SemanticJudge:
    if spec.judge is not None:
        return spec.judge
    if bridge is None:
        raise ValueError(
            "semantic stage requires SemanticJudge or ToolWiringEvalVerificationBridge",
        )
    return bridge.semantic_judge


def _resolve_trajectory_evaluator(
    *,
    spec: TrajectoryProductionStageSpec[T],
    bridge: ToolWiringEvalVerificationBridge | None,
) -> TrajectoryEvaluator:
    if spec.evaluator is not None:
        return spec.evaluator
    if bridge is None:
        raise ValueError(
            "trajectory stage requires TrajectoryEvaluator or "
            "ToolWiringEvalVerificationBridge",
        )
    return bridge.trajectory_evaluator


def build_decision_verification_pipeline(
    spec: DecisionVerificationPipelineBuildSpec[T],
) -> VerificationPipeline[T]:
    """Build one immutable verification pipeline from explicit stage configuration."""
    if type(spec) is not DecisionVerificationPipelineBuildSpec:
        raise TypeError("spec must be DecisionVerificationPipelineBuildSpec")
    registrations: list[VerificationStageRegistration[T]] = list(spec.extra_registrations)
    if spec.semantic is not None:
        semantic_spec = spec.semantic
        judge = _resolve_semantic_judge(spec=semantic_spec, bridge=spec.eval_bridge)
        semantic_stage = SemanticVerificationStage(
            rubric_ref=semantic_spec.rubric_ref,
            rubric_resolver=semantic_spec.rubric_resolver,
            content_provider=semantic_spec.content_provider,
            judge=judge,
            independence=semantic_spec.independence,
        )
        registrations.append(
            VerificationStageRegistration(
                kind=SEMANTIC_VERIFICATION_STAGE_KIND,
                stage=semantic_stage,
                required=semantic_spec.required,
            ),
        )
    if spec.trajectory is not None:
        trajectory_spec = spec.trajectory
        evaluator = _resolve_trajectory_evaluator(
            spec=trajectory_spec,
            bridge=spec.eval_bridge,
        )
        trajectory_stage = TrajectoryVerificationStage(
            evaluator=evaluator,
            agent_id_provider=trajectory_spec.agent_id_provider,
            config=trajectory_spec.config,
        )
        registrations.append(
            VerificationStageRegistration(
                kind=TRAJECTORY_VERIFICATION_STAGE_KIND,
                stage=trajectory_stage,
                required=trajectory_spec.required,
            ),
        )
    registry = verification_stage_registry(tuple(registrations))
    return VerificationPipeline(registry=registry, observer=spec.observer)
