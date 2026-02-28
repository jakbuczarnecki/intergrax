# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from datetime import datetime
from intergrax.fastapi_core.runs.models import RunResponse, RunStatus
from intergrax.fastapi_core.runs.store_base import RunStore
from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.governance.execution_guard import ExecutionGuard, GovernanceEvaluation
from intergrax.runtime.governance.service import GovernanceService
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.pipelines.no_planner_pipeline import NoPlannerPipeline
from intergrax.runtime.nexus.planning.engine_plan_models import EngineNextStep, PlanIntent, PlannerPromptConfig
from intergrax.runtime.nexus.planning.plan_loop_models import PlanLoopPolicy
from intergrax.runtime.nexus.planning.plan_sources import PlanSpec, ScriptedPlanSource
from intergrax.runtime.nexus.planning.step_executor_models import StepExecutorConfig
from intergrax.runtime.nexus.planning.step_planner import StepPlannerConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.replay.metrics import ExecutionMetrics
from intergrax.runtime.replay.policy import PolicyDecision, PolicyDecisionType
from intergrax.runtime.replay.regression import RegressionSignals
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.tools.core.contracts import ToolContract


class FakeLLMAdapter(LLMAdapter):
    """
    Deterministic LLM adapter for CI-safe tests.

    Goals:
    - no network
    - stable output
    - still exercises CoreLLMStep / finalization path
    """

    provider = "fake"
    model = "fake"

    @property
    def context_window_tokens(self) -> int:
        # Large enough for tests; avoids truncation logic influencing results.
        return 128_000


    def __init__(self, *, fixed_text: str = "OK") -> None:
        super().__init__()
        self._fixed_text = fixed_text

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> str:
        # Deterministic response for tests.
        # Keep it simple: do NOT depend on message content.
        call = self.usage.begin_call(run_id=run_id)
        try:
            return self._fixed_text
        finally:
            # Tokens are fake here; that's fine for tests.
            self.usage.end_call(
                call,
                input_tokens=0,
                output_tokens=len(self._fixed_text),
                success=True,
            )

class DummyExecutionGuard(ExecutionGuard):
    """
    Minimal execution guard for production-trace tests.

    It does NOT:
    - reconstruct execution
    - compute real metrics
    - evaluate history
    - execute actions

    It only returns a deterministic ALLOW decision.
    """

    def __init__(self) -> None:
        # We intentionally do NOT call super().__init__
        # because we do not need replay/metrics/policy engines.
        pass

    def evaluate_run(
        self,
        run_id: str,
        agent_id: str,
    ) -> GovernanceEvaluation:
        decision = PolicyDecision(
            decision=PolicyDecisionType.ALLOW,
            reasons=["dummy-allow"],
        )

        # Minimal dummy objects (empty but correctly typed)
        metrics = ExecutionMetrics()
        regression = RegressionSignals()

        return GovernanceEvaluation(
            decision=decision,
            metrics=metrics,
            regression=regression,
        )


@dataclass(frozen=True)
class DeterministicRuntimeHarness:
    """
    What integration tests need:
    - engine
    - config (to inspect/adjust in tests)
    - session manager (optional, for direct history assertions later)
    """
    engine: RuntimeEngine
    config: RuntimeConfig
    session_manager: SessionManager


def build_in_memory_session_manager() -> SessionManager:
    storage = InMemorySessionStorage()
    return SessionManager(storage)


def build_runtime_config_deterministic(
    *,
    pipeline: RuntimePipeline | None = None,
    plan_specs: Optional[Sequence[PlanSpec]] = None,
    llm_text: str = "OK",
    plan_loop_policy: Optional[PlanLoopPolicy] = None,
    idempotency_store: Optional[IdempotencyStore] = None,
) -> RuntimeConfig:
    """
    Deterministic RuntimeConfig for CI:
    - no RAG/vectorstore/web/tools unless explicitly enabled later
    - scripted plan source
    - required planner/step configs present (fail-fast validations pass)
    """
    llm = FakeLLMAdapter(fixed_text=llm_text)

    if plan_specs is None:
        plan_specs = [
            PlanSpec(
                version="1",
                intent=PlanIntent.GENERIC,
                next_step=EngineNextStep.FINALIZE,
                reasoning_summary="test: minimal plan spec for deterministic harness",
                ask_clarifying_question=False,
                clarifying_question=None,
                use_websearch=False,
                use_user_longterm_memory=False,
                use_rag=False,
                use_tools=False,
                debug=None,
            )
        ]
    

    cfg = RuntimeConfig(
        llm_adapter=llm,
        embedding_manager=None,
        vectorstore_manager=None,
        tenant_id="test-tenant",
        workspace_id="test-workspace",
        websearch_executor=None,
        websearch_config=None,
        tools_agent=None,
        pipeline=pipeline if pipeline is not None else NoPlannerPipeline(),
        step_planner_cfg=StepPlannerConfig(),
        step_executor_cfg=StepExecutorConfig(),
        planner_prompt_config=PlannerPromptConfig(),
        plan_loop_policy=plan_loop_policy or PlanLoopPolicy(),
        plan_source=ScriptedPlanSource(plans=plan_specs),
        enable_rag=False,
        enable_websearch=False,
        enable_org_profile_memory=False,
        tools_mode="off",
        idempotency_store=idempotency_store,
        production_mode=False,
    )

    # If RuntimeConfig exposes validate(), keep it enabled (enterprise style).
    # This makes test failures immediate and readable.
    cfg.validate()

    return cfg


def build_engine_harness_production_trace(
    *,
    trace_db_path: Path,
    llm_text: str = "OK",
) -> DeterministicRuntimeHarness:

    if trace_db_path is None:
        raise ValueError("trace_db_path must be provided.")

    llm = FakeLLMAdapter(fixed_text=llm_text)

    cfg = RuntimeConfig(
        llm_adapter=llm,
        embedding_manager=None,
        vectorstore_manager=None,
        tenant_id="test-tenant",
        workspace_id="test-workspace",
        websearch_executor=None,
        websearch_config=None,
        tools_agent=None,
        pipeline=NoPlannerPipeline(),
        step_planner_cfg=StepPlannerConfig(),
        step_executor_cfg=StepExecutorConfig(),
        planner_prompt_config=PlannerPromptConfig(),
        plan_loop_policy=PlanLoopPolicy(),
        plan_source=ScriptedPlanSource(
            plans=[
                PlanSpec(
                    version="1",
                    intent=PlanIntent.GENERIC,
                    next_step=EngineNextStep.FINALIZE,
                    reasoning_summary="test: production trace harness",
                    ask_clarifying_question=False,
                    clarifying_question=None,
                    use_websearch=False,
                    use_user_longterm_memory=False,
                    use_rag=False,
                    use_tools=False,
                    debug=None,
                )
            ]
        ),
        enable_rag=False,
        enable_websearch=False,
        enable_org_profile_memory=False,
        tools_mode="off",
        production_mode=True,
        trace_db_path=str(trace_db_path),
    )

    cfg.validate()

    session_manager = build_in_memory_session_manager()

    governance_service = GovernanceService(
        guard=DummyExecutionGuard()
    )

    ctx = RuntimeContext.build(
        config=cfg,
        session_manager=session_manager,
        ingestion_service=None,
        context_builder=None,
        rag_prompt_builder=None,
        user_longterm_memory_prompt_builder=None,
        websearch_prompt_builder=None,
        history_prompt_builder=None,
        prompt_registry=None,
        governance_service=governance_service,
    )

    engine = RuntimeEngine(context=ctx)

    return DeterministicRuntimeHarness(
        engine=engine,
        config=cfg,
        session_manager=session_manager,
    )



def build_engine_harness(
    *,
    cfg: RuntimeConfig,
    session_manager: Optional[SessionManager] = None,
) -> DeterministicRuntimeHarness:
    sm = session_manager or build_in_memory_session_manager()

    ctx = RuntimeContext.build(
        config=cfg,
        session_manager=sm,
        ingestion_service=None,
        context_builder=None,
        rag_prompt_builder=None,
        user_longterm_memory_prompt_builder=None,
        websearch_prompt_builder=None,
        history_prompt_builder=None,
    )

    engine = RuntimeEngine(context=ctx)
    return DeterministicRuntimeHarness(engine=engine, config=cfg, session_manager=sm)


def build_runtime_state_for_tests(*, run_id: str) -> RuntimeState:
    """
    Minimal RuntimeState builder for unit tests that only need tracing.
    No engine, no pipeline, no planner — just state + trace_event support.
    """

    request = RuntimeRequest(
        tenant_id="test-tenant",
        agent_id="agent_test",
        user_id="test-user",
        session_id="test-session",
        message="test",
    )

    cfg = RuntimeConfig(
        llm_adapter=None,
        embedding_manager=None,
        vectorstore_manager=None,
        tenant_id="test-tenant",
        workspace_id="test-workspace",
        websearch_executor=None,
        websearch_config=None,
        tools_agent=None,
        pipeline=None,
        step_planner_cfg=None,
        step_executor_cfg=None,
        planner_prompt_config=None,
        plan_loop_policy=None,
        plan_source=None,
        enable_rag=False,
        enable_websearch=False,
        enable_org_profile_memory=False,
        tools_mode="off",
        production_mode=False,
    )

    sm = SessionManager(storage=InMemorySessionStorage())

    ctx = RuntimeContext.build(
        config=cfg,
        session_manager=sm,
        ingestion_service=None,
        context_builder=None,
        rag_prompt_builder=None,
        user_longterm_memory_prompt_builder=None,
        websearch_prompt_builder=None,
        history_prompt_builder=None,
    )

    return RuntimeState(context=ctx, run_id=run_id, request=request)


class DummyRunStore(RunStore):
    def __init__(self) -> None:
        self._runs: dict[str, RunResponse] = {}

    def create(self) -> RunResponse:
        run_id = "r1"
        run = RunResponse(
            run_id=run_id,
            status=RunStatus.PENDING,
        )
        self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> RunResponse:
        return self._runs[run_id]

    def cancel(self, run_id: str) -> RunResponse:
        raise AssertionError("Should not reach store.cancel() if transition invalid")

    def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        duration_ms: int | None = None,
        result_payload: dict | None = None,
    ) -> RunResponse:
        current = self._runs[run_id]

        updated = RunResponse(
            run_id=current.run_id,
            status=status,
            error_type=error_type if error_type is not None else current.error_type,
            error_message=error_message if error_message is not None else current.error_message,
            started_at=started_at if started_at is not None else current.started_at,
            finished_at=finished_at if finished_at is not None else current.finished_at,
            duration_ms=duration_ms if duration_ms is not None else current.duration_ms,
            result_payload=result_payload if result_payload is not None else current.result_payload,
        )

        self._runs[run_id] = updated
        return updated


def tools_agent_make_contract(tool_id: str, input_model, output_model):
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description=f"{tool_id} description",
        input_schema=input_model,
        output_schema=output_model,
        error_mapping={},
        side_effects=False,
    )

def prepare_sqlite_db(name:str)->Path:
    db_path = Path(f"documents/{name}")
    if db_path.exists():
        db_path.unlink()
    return db_path