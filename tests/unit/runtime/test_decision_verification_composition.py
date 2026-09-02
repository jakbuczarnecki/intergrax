# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifact,
    DecisionVersionLineage,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationStageOutcome,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    VerificationStageUnavailableError,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.semantic_verification import (
    ResolvedSemanticRubric,
    SemanticRubricRef,
    VerifierIndependenceMode,
    resolved_semantic_rubric,
    semantic_rubric_ref,
    semantic_verification_independence_config,
)
from intergrax.contracts.trajectory_verification import (
    TrajectoryAgentId,
    trajectory_verification_stage_config,
)
from intergrax.runtime.decision_verification_composition import (
    DecisionVerificationPipelineBuildSpec,
    SemanticProductionStageSpec,
    ToolWiringSemanticJudge,
    ToolWiringTrajectoryEvaluator,
    TrajectoryProductionStageSpec,
    build_decision_verification_pipeline,
    compose_tool_wiring_eval_verification,
)
from intergrax.runtime.decision_verification_stages.semantic import (
    SEMANTIC_VERIFICATION_STAGE_KIND,
)
from intergrax.runtime.decision_verification_stages.structural import (
    STRUCTURAL_VERIFICATION_STAGE_KIND,
    NonEmptyTextStructuralValidator,
    StructuralVerificationStage,
)
from intergrax.runtime.decision_verification_stages.trajectory import (
    TRAJECTORY_VERIFICATION_STAGE_KIND,
)
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunMetadata, RunStats
from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalTrajectoryInput
from intergrax.tools.providers.eval.judge import _JudgeLLMResult, eval_judge
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MODULE_PATH = Path("intergrax/runtime/decision_verification_composition.py")
_FORBIDDEN_FRAGMENTS = (
    "runtime.critic",
    "runtime.nexus",
    "L1Gateway",
    "CriticEvalToolClient",
    "CriticOrchestrator",
    "openai",
    "anthropic",
    "Any",
    "cast(",
    "type: ignore",
    "pyright: ignore",
    "getattr",
    "setattr",
    "hasattr",
    "inspect",
    "exec(",
    "eval(",
    "object.__setattr__",
    "dict[str, Any]",
)


@dataclass(frozen=True, slots=True)
class Payload:
    text: str


@dataclass(frozen=True, slots=True)
class TextContentProvider:
    def extract(self, candidate: CandidateDecision[Payload]) -> str:
        return candidate.artifact.content.text


@dataclass(frozen=True, slots=True)
class TextFieldExtractor:
    def extract(self, content: Payload) -> str:
        return content.text


@dataclass(frozen=True, slots=True)
class FixedAgentIdProvider:
    agent_id: TrajectoryAgentId

    def resolve(self, candidate: CandidateDecision[Payload]) -> TrajectoryAgentId | None:
        _ = candidate
        return self.agent_id


@dataclass(frozen=True, slots=True)
class InMemoryRubricResolver:
    rubric: ResolvedSemanticRubric

    def is_available(self) -> bool:
        return True

    def resolve(self, ref: SemanticRubricRef) -> ResolvedSemanticRubric:
        _ = ref
        return self.rubric


class _TraceReader:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self._events = events

    def read_run(self, run_id: str, tenant_id: str) -> PersistedRun:
        metadata = RunMetadata(
            run_id=run_id,
            session_id="s1",
            user_id="u1",
            tenant_id=tenant_id,
            started_at_utc="2026-06-07T00:00:00Z",
            stats=RunStats(duration_ms=100, llm_usage={}),
        )
        return PersistedRun(metadata=metadata, events=self._events)


def _candidate(text: str = "answer") -> CandidateDecision[Payload]:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="demo", subject="case-1"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    return CandidateDecision(
        identity=identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("demo.payload"),
            content=Payload(text=text),
        ),
        lineage=DecisionVersionLineage(
            current=decision_lineage_ref(identity.version),
        ),
    )


def _rubric() -> ResolvedSemanticRubric:
    ref = semantic_rubric_ref(rubric_id="demo.rubric", version=1)
    return resolved_semantic_rubric(
        ref=ref,
        criteria=("complete",),
        min_score=0.75,
        provenance_ref="prov/demo/1",
    )


def _structural_stage() -> StructuralVerificationStage[Payload]:
    return StructuralVerificationStage(
        validators=(
            NonEmptyTextStructuralValidator(
                extractor=TextFieldExtractor(),
                field_label="text",
            ),
        ),
    )


def test_module_forbidden_patterns() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for fragment in _FORBIDDEN_FRAGMENTS:
        assert fragment not in source


    source = _MODULE_PATH.read_text(encoding="utf-8")
    for fragment in _FORBIDDEN_FRAGMENTS:
        assert fragment not in source


def test_semantic_adapter_invokes_eval_judge_with_preserved_input() -> None:
    captured: list[EvalJudgeInput] = []
    llm = FakeLLMAdapter(
        fake_structured_data=_JudgeLLMResult(score=0.95, passed=True, reasons=["ok"]),
    )
    ctx = ToolWiringContext(extras={"llm_adapter": llm})
    bridge = compose_tool_wiring_eval_verification(ctx)
    assert bridge.semantic_judge.is_available() is True
    params = EvalJudgeInput(
        output_text="candidate",
        rubric_id="demo.rubric",
        criteria=["complete"],
        min_score=0.75,
        run_id="run-1",
    )
    original = EvalJudgeInput.model_validate(params.model_dump())
    out = bridge.semantic_judge.judge(params)
    captured.append(params)
    assert captured[0].model_dump() == original.model_dump()
    assert out.passed is True
    assert out.score == pytest.approx(0.95)


def test_trajectory_adapter_invokes_eval_trajectory_with_preserved_input() -> None:
    events = [
        {
            "step": "tool_invocation_start",
            "message": "invoke",
            "payload": {"tool_name": "search"},
        },
    ]
    ctx = ToolWiringContext(trace_reader=_TraceReader(events))
    bridge = compose_tool_wiring_eval_verification(ctx)
    assert bridge.trajectory_evaluator.is_available() is True
    params = EvalTrajectoryInput(
        run_id="run-traj",
        tenant_id="tenant-a",
        min_score=0.5,
        agent_id="agent-1",
    )
    original = EvalTrajectoryInput.model_validate(params.model_dump())
    out = bridge.trajectory_evaluator.evaluate(params)
    assert params.model_dump() == original.model_dump()
    assert out.passed is True


def test_availability_reflects_wiring_context_capabilities() -> None:
    empty = compose_tool_wiring_eval_verification(ToolWiringContext())
    assert empty.semantic_judge.is_available() is False
    assert empty.trajectory_evaluator.is_available() is False
    llm_ctx = ToolWiringContext(extras={"llm_adapter": FakeLLMAdapter()})
    assert compose_tool_wiring_eval_verification(llm_ctx).semantic_judge.is_available() is True
    trace_ctx = ToolWiringContext(trace_reader=_TraceReader([]))
    assert compose_tool_wiring_eval_verification(trace_ctx).trajectory_evaluator.is_available() is True


def test_integration_boundary_through_real_eval_services() -> None:
    llm = FakeLLMAdapter(
        fake_structured_data=_JudgeLLMResult(score=0.9, passed=True, reasons=["good"]),
    )
    ctx = ToolWiringContext(extras={"llm_adapter": llm})
    judge_out = eval_judge(
        ctx,
        EvalJudgeInput(
            output_text="done",
            rubric_id="demo",
            criteria=["ok"],
            min_score=0.75,
        ),
    )
    bridge = compose_tool_wiring_eval_verification(ctx)
    adapter_out = bridge.semantic_judge.judge(
        EvalJudgeInput(
            output_text="done",
            rubric_id="demo",
            criteria=["ok"],
            min_score=0.75,
        ),
    )
    assert adapter_out.passed == judge_out.passed
    assert adapter_out.score == judge_out.score


def test_pipeline_factory_registers_selected_stages_only() -> None:
    structural = _structural_stage()
    extra = VerificationStageRegistration(
        kind=STRUCTURAL_VERIFICATION_STAGE_KIND,
        stage=structural,
        required=True,
    )
    pipeline = build_decision_verification_pipeline(
        DecisionVerificationPipelineBuildSpec(extra_registrations=(extra,)),
    )
    kinds = {registration.kind for registration in pipeline.registry.registrations}
    assert kinds == {STRUCTURAL_VERIFICATION_STAGE_KIND}


def test_pipeline_factory_preserves_required_and_execution_class() -> None:
    rubric = _rubric()
    resolver = InMemoryRubricResolver(rubric)
    llm = FakeLLMAdapter(
        fake_structured_data=_JudgeLLMResult(score=0.2, passed=False, reasons=["weak"]),
    )
    bridge = compose_tool_wiring_eval_verification(
        ToolWiringContext(extras={"llm_adapter": llm}),
    )
    pipeline = build_decision_verification_pipeline(
        DecisionVerificationPipelineBuildSpec(
            eval_bridge=bridge,
            semantic=SemanticProductionStageSpec(
                rubric_ref=rubric.ref,
                rubric_resolver=resolver,
                content_provider=TextContentProvider(),
                independence=semantic_verification_independence_config(
                    mode=VerifierIndependenceMode.SHARED_PROFILE,
                    producer_profile_id="producer",
                    verifier_profile_id="producer",
                ),
                required=False,
            ),
        ),
    )
    registration = pipeline.registry.registrations[0]
    assert registration.kind == SEMANTIC_VERIFICATION_STAGE_KIND
    assert registration.required is False
    assert registration.stage.execution_class is VerificationStageExecutionClass.PROBABILISTIC


def test_semantic_disabled_does_not_require_resolver_or_judge() -> None:
    pipeline = build_decision_verification_pipeline(
        DecisionVerificationPipelineBuildSpec(),
    )
    assert pipeline.registry.registrations == ()


def test_trajectory_disabled_does_not_require_evaluator_or_provider() -> None:
    pipeline = build_decision_verification_pipeline(
        DecisionVerificationPipelineBuildSpec(),
    )
    assert pipeline.registry.registrations == ()


@pytest.mark.asyncio
async def test_enabled_semantic_missing_capability_raises_unavailable() -> None:
    rubric = _rubric()
    pipeline = build_decision_verification_pipeline(
        DecisionVerificationPipelineBuildSpec(
            semantic=SemanticProductionStageSpec(
                rubric_ref=rubric.ref,
                rubric_resolver=InMemoryRubricResolver(rubric),
                content_provider=TextContentProvider(),
                independence=semantic_verification_independence_config(
                    mode=VerifierIndependenceMode.SHARED_PROFILE,
                    producer_profile_id="producer",
                    verifier_profile_id="producer",
                ),
                judge=ToolWiringSemanticJudge(ctx=ToolWiringContext(), available=False),
            ),
        ),
    )
    stage = pipeline.registry.registrations[0].stage
    with pytest.raises(VerificationStageUnavailableError):
        await stage.verify(_candidate())


@pytest.mark.asyncio
async def test_enabled_trajectory_missing_capability_raises_unavailable() -> None:
    pipeline = build_decision_verification_pipeline(
        DecisionVerificationPipelineBuildSpec(
            trajectory=TrajectoryProductionStageSpec(
                agent_id_provider=FixedAgentIdProvider(TrajectoryAgentId("agent-1")),
                evaluator=ToolWiringTrajectoryEvaluator(
                    ctx=ToolWiringContext(),
                    available=False,
                ),
            ),
        ),
    )
    stage = pipeline.registry.registrations[0].stage
    with pytest.raises(VerificationStageUnavailableError):
        await stage.verify(_candidate())


def test_custom_domain_stage_can_be_added_without_factory_core_changes() -> None:
    structural = _structural_stage()
    pipeline = build_decision_verification_pipeline(
        DecisionVerificationPipelineBuildSpec(
            extra_registrations=(
                VerificationStageRegistration(
                    kind=STRUCTURAL_VERIFICATION_STAGE_KIND,
                    stage=structural,
                    required=True,
                ),
            ),
        ),
    )
    assert len(pipeline.registry.registrations) == 1
    assert pipeline.registry.registrations[0].kind == STRUCTURAL_VERIFICATION_STAGE_KIND


def test_independent_compositions_do_not_share_state() -> None:
    bridge_a = compose_tool_wiring_eval_verification(
        ToolWiringContext(extras={"llm_adapter": FakeLLMAdapter()}),
    )
    bridge_b = compose_tool_wiring_eval_verification(ToolWiringContext())
    assert bridge_a.semantic_judge.is_available() is True
    assert bridge_b.semantic_judge.is_available() is False
    assert bridge_a.ctx is not bridge_b.ctx


@pytest.mark.asyncio
async def test_full_pipeline_semantic_and_trajectory_via_factory() -> None:
    rubric = _rubric()
    llm = FakeLLMAdapter(
        fake_structured_data=_JudgeLLMResult(score=0.95, passed=True, reasons=["ok"]),
    )
    events = [
        {
            "step": "tool_invocation_start",
            "message": "invoke",
            "payload": {"tool_name": "search"},
        },
    ]
    ctx = ToolWiringContext(
        extras={"llm_adapter": llm},
        trace_reader=_TraceReader(events),
    )
    bridge = compose_tool_wiring_eval_verification(ctx)
    structural = _structural_stage()
    pipeline = build_decision_verification_pipeline(
        DecisionVerificationPipelineBuildSpec(
            eval_bridge=bridge,
            extra_registrations=(
                VerificationStageRegistration(
                    kind=STRUCTURAL_VERIFICATION_STAGE_KIND,
                    stage=structural,
                    required=True,
                ),
            ),
            semantic=SemanticProductionStageSpec(
                rubric_ref=rubric.ref,
                rubric_resolver=InMemoryRubricResolver(rubric),
                content_provider=TextContentProvider(),
                independence=semantic_verification_independence_config(
                    mode=VerifierIndependenceMode.SHARED_PROFILE,
                    producer_profile_id="producer",
                    verifier_profile_id="producer",
                ),
            ),
            trajectory=TrajectoryProductionStageSpec(
                agent_id_provider=FixedAgentIdProvider(TrajectoryAgentId("agent-1")),
                config=trajectory_verification_stage_config(min_score=0.5),
            ),
        ),
    )
    result = await pipeline.verify(_candidate("complete answer"))
    assert result.disposition is VerificationDisposition.PASSED
    assert all(
        record.outcome is VerificationStageOutcome.PASSED
        for record in result.stage_records
    )


def test_build_without_bridge_or_judge_raises_for_semantic() -> None:
    rubric = _rubric()
    with pytest.raises(ValueError, match="SemanticJudge"):
        build_decision_verification_pipeline(
            DecisionVerificationPipelineBuildSpec(
                semantic=SemanticProductionStageSpec(
                    rubric_ref=rubric.ref,
                    rubric_resolver=InMemoryRubricResolver(rubric),
                    content_provider=TextContentProvider(),
                    independence=semantic_verification_independence_config(
                        mode=VerifierIndependenceMode.SHARED_PROFILE,
                        producer_profile_id="producer",
                        verifier_profile_id="producer",
                    ),
                ),
            ),
        )


def test_bridge_types_are_immutable() -> None:
    bridge = compose_tool_wiring_eval_verification(ToolWiringContext())
    with pytest.raises(Exception):
        bridge.ctx = ToolWiringContext()  # type: ignore[misc]


def test_no_global_mutable_registry_in_factory() -> None:
    pipeline_one = build_decision_verification_pipeline(
        DecisionVerificationPipelineBuildSpec(),
    )
    pipeline_two = build_decision_verification_pipeline(
        DecisionVerificationPipelineBuildSpec(),
    )
    assert pipeline_one.registry is not pipeline_two.registry
