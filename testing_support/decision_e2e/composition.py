# © Artur Czarnecki. All rights reserved.

"""Execution + Decision composition helpers for DS-E2E."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import validate_decision_artifact_kind
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.single_model_strategy import (
    SingleModelDeliberationInput,
    SingleModelInferenceConfiguration,
    single_model_candidate_decision,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionFlowGateCapabilities,
    DecisionFlowIdentitySeed,
    DecisionFlowRequest,
    DecisionFlowScope,
)
from intergrax.runtime.decision_lifecycle_observability import (
    CanonicalRuntimeEventDecisionLifecycleObserver,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.decision_verification import VerificationPipeline
from intergrax.runtime.execution.active_execution_work_port import (
    ActiveExecutionWorkPortBinding,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost
from intergrax.runtime.execution.execution_work_port import (
    ExecutionWorkPort,
    child_execution_work_port,
)
from intergrax.runtime.execution.facade import Execution
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.inference_profile import (
    InferenceProfileCatalog,
    InferenceProfileId,
    validate_inference_profile_id,
)
from intergrax.runtime.execution.request import ExecutionRequest
from intergrax.runtime.execution.result import ExecutionResult, ExecutionStatus
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionOptions
from intergrax.runtime.execution.single_model_deliberation import (
    single_model_inference_execution_request,
)
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.execution.sqlite_decision_checkpoint_persistence import (
    SQLiteDecisionCheckpointPersistence,
)
from intergrax.runtime.execution.sqlite_decision_finalization_persistence import (
    SQLiteDecisionFinalizationPersistence,
)
from intergrax.runtime.execution.decision_finalization_conformance import (
    IncidentDecisionPayload,
    conformance_artifact_payload_codec_registry,
)
from intergrax.tools.registry.wiring import ToolWiringContext

from testing_support.decision_e2e.environment import QualificationEnvironment
from testing_support.decision_e2e.payloads import QualificationRecommendation

_ARTIFACT_KIND = validate_decision_artifact_kind("decision_e2e_qualification")
_PROFILE_PRODUCER = validate_inference_profile_id("profile-producer")
_PROFILE_VERIFIER = validate_inference_profile_id("profile-verifier")
_PROFILE_B = validate_inference_profile_id("profile-b")
_PROFILE_C = validate_inference_profile_id("profile-c")


@dataclass(slots=True)
class InferenceExecutionWorkPort:
    """Maps canonical inference child work to structured payload output."""

    _delegate: ExecutionWorkPort[
        tuple[ChatMessage, ...],
        QualificationRecommendation,
        ExecutionResult[QualificationRecommendation],
    ]
    invocation_count: int = 0

    async def execute(
        self,
        request: ExecutionRequest[tuple[ChatMessage, ...], QualificationRecommendation],
    ) -> QualificationRecommendation:
        self.invocation_count += 1
        result = await self._delegate.execute(request)
        if result.status is not ExecutionStatus.COMPLETED or result.output is None:
            raise RuntimeError(f"inference work failed with status {result.status}")
        return result.output


@dataclass(frozen=True, slots=True)
class QualificationPersistenceBundle:
    checkpoint: SQLiteDecisionCheckpointPersistence[IncidentDecisionPayload]
    finalization: SQLiteDecisionFinalizationPersistence[IncidentDecisionPayload]
    db_dir: Path


@dataclass(frozen=True, slots=True)
class QualificationComposition:
    """Canonical DS-E2E composition root."""

    environment: QualificationEnvironment
    profile_catalog: InferenceProfileCatalog
    work_port: InferenceExecutionWorkPort
    execution: Execution[
        ExecutionRequest[tuple[ChatMessage, ...], QualificationRecommendation],
        ExecutionResult[QualificationRecommendation],
    ]
    lifecycle_host: CanonicalDecisionLifecycleHost
    lifecycle_observer: CanonicalRuntimeEventDecisionLifecycleObserver
    event_bus: RuntimeEventBus

    def lifecycle_for_identity(
        self,
        identity: DecisionIdentity,
    ) -> tuple[CanonicalDecisionLifecycleHost, RuntimeEventBus]:
        bus = RuntimeEventBus()
        from testing_support.runtime_events import emit_context_test_identity

        emit_ctx = emit_context_test_identity(
            task_id=identity.execution.task_id,
            run_id=identity.execution.run_id,
            attempt_id=identity.execution.attempt_id,
            execution_id=identity.execution.execution_id,
            tenant_id=identity.tenant_id,
            bus=bus,
        )
        host = CanonicalDecisionLifecycleHost(
            observer=CanonicalRuntimeEventDecisionLifecycleObserver(ctx=emit_ctx),
        )
        return host, bus
    tool_wiring: ToolWiringContext
    persistence: QualificationPersistenceBundle | None = None

    def build_flow_gate(
        self,
        *,
        pipeline: VerificationPipeline[QualificationRecommendation],
        revision_policy,
        human_review_port=None,
        governance_spec=None,
    ) -> CanonicalDecisionFlowGate[QualificationRecommendation]:
        return CanonicalDecisionFlowGate(
            capabilities=DecisionFlowGateCapabilities(
                verification_pipeline=pipeline,
                revision_policy=revision_policy,
                scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
                human_review_port=human_review_port,
                governance_spec=governance_spec,
            ),
        )


def build_profile_catalog(environment: QualificationEnvironment) -> InferenceProfileCatalog:
    return InferenceProfileCatalog(
        (
            (_PROFILE_PRODUCER, environment.producer_adapter),
            (_PROFILE_VERIFIER, environment.verifier_adapter),
            (_PROFILE_B, environment.council_adapter_b),
            (_PROFILE_C, environment.council_adapter_c),
        ),
    )


def build_sqlite_persistence(tmp_dir: Path) -> QualificationPersistenceBundle:
    codecs = conformance_artifact_payload_codec_registry()
    return QualificationPersistenceBundle(
        checkpoint=SQLiteDecisionCheckpointPersistence(
            db_path=tmp_dir / "checkpoint.db",
            payload_codecs=codecs,
        ),
        finalization=SQLiteDecisionFinalizationPersistence(
            db_path=tmp_dir / "finalization.db",
            payload_codecs=codecs,
        ),
        db_dir=tmp_dir,
    )


def build_qualification_composition(
    environment: QualificationEnvironment,
    *,
    run_budget: RunBudget | None = None,
    persistence: QualificationPersistenceBundle | None = None,
) -> QualificationComposition:
    catalog = build_profile_catalog(environment)
    inference_executor = InferenceExecutor(
        environment.producer_adapter,
        profile_resolver=catalog,
    )
    router = StrategyExecutionRouter[
        tuple[ChatMessage, ...],
        QualificationRecommendation,
        ExecutionResult[QualificationRecommendation],
    ](inference_executor=inference_executor)
    budget_ledger = create_execution_budget_ledger(run_budget)
    child_port = child_execution_work_port(router, ledger=budget_ledger)
    work_port = InferenceExecutionWorkPort(_delegate=child_port)
    placeholder_bus = RuntimeEventBus()
    placeholder_ctx = EmitContext(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        bus=placeholder_bus,
    )
    lifecycle_host = CanonicalDecisionLifecycleHost(
        observer=CanonicalRuntimeEventDecisionLifecycleObserver(ctx=placeholder_ctx),
    )
    runtime = ExecutionRuntime[
        ExecutionRequest[tuple[ChatMessage, ...], QualificationRecommendation],
        ExecutionResult[QualificationRecommendation],
    ](
        router,
        run_budget=run_budget,
        decision_lifecycle_host=lifecycle_host,
        decision_checkpoint_persistence=(
            persistence.checkpoint if persistence is not None else None
        ),
        decision_finalization_persistence=(
            persistence.finalization if persistence is not None else None
        ),
        execution_work_port_binding=ActiveExecutionWorkPortBinding.for_port(work_port),
    )
    tool_wiring = ToolWiringContext(extras={"llm_adapter": environment.verifier_adapter})
    return QualificationComposition(
        environment=environment,
        profile_catalog=catalog,
        work_port=work_port,
        execution=Execution(runtime),
        lifecycle_host=lifecycle_host,
        lifecycle_observer=CanonicalRuntimeEventDecisionLifecycleObserver(ctx=placeholder_ctx),
        tool_wiring=tool_wiring,
        persistence=persistence,
        event_bus=placeholder_bus,
    )


def mint_qualification_identity(
    *,
    tenant_id: str = "tenant-decision-e2e",
    namespace: str = "decision_e2e",
    subject: str = "qualification-case",
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace=namespace, subject=subject),
        tenant_id=tenant_id,
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )


def identity_seed_from_identity(identity: DecisionIdentity) -> DecisionFlowIdentitySeed:
    return DecisionFlowIdentitySeed(
        scope=identity.scope,
        tenant_id=identity.tenant_id,
        execution=identity.execution,
        decision_id=identity.decision_id,
    )


async def run_single_model_producer(
    composition: QualificationComposition,
    *,
    identity: DecisionIdentity,
    task_message: str,
    profile_id: InferenceProfileId = _PROFILE_PRODUCER,
) -> tuple[QualificationRecommendation, int]:
    deliberation_input = SingleModelDeliberationInput(
        messages=(ChatMessage(role="user", content=task_message),),
        output_type=QualificationRecommendation,
        artifact_kind=_ARTIFACT_KIND,
    )
    inference = SingleModelInferenceConfiguration(inference_profile_id=profile_id)
    request = single_model_inference_execution_request(
        deliberation_input,
        inference=inference,
    )
    calls_before = composition.work_port.invocation_count
    result = await composition.execution.execute(
        request,
        options=RootExecutionOptions(
            authority=ParentExecutionAuthority.unrestricted_root(),
            tenant_id=identity.tenant_id,
            run_id=identity.execution.run_id,
            attempt_id=identity.execution.attempt_id,
        ),
    )
    if result.status is not ExecutionStatus.COMPLETED or result.output is None:
        raise RuntimeError("single-model producer inference failed")
    invocations = composition.work_port.invocation_count - calls_before
    if invocations == 0 and result.output.recommendation:
        invocations = 1
    return result.output, invocations


def candidate_from_producer_output(
    *,
    identity: DecisionIdentity,
    payload: QualificationRecommendation,
) -> object:
    return single_model_candidate_decision(
        identity=identity,
        artifact_kind=_ARTIFACT_KIND,
        payload=payload,
    )


async def evaluate_decision_flow(
    composition: QualificationComposition,
    gate: CanonicalDecisionFlowGate[QualificationRecommendation],
    *,
    identity: DecisionIdentity,
    payload: QualificationRecommendation,
):
    return await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=identity_seed_from_identity(identity),
            artifact_kind=_ARTIFACT_KIND,
            payload=payload,
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )


def lifecycle_stage_reached(stage: DecisionLifecycleStage, lifecycle_state) -> bool:
    return lifecycle_state.stage is stage
