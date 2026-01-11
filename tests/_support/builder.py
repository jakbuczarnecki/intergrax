# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig, StepPlanningStrategy
from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.planning.engine_plan_models import PlannerPromptConfig
from intergrax.runtime.nexus.planning.plan_loop_models import PlanLoopPolicy
from intergrax.runtime.nexus.planning.plan_sources import PlanSpec, ScriptedPlanSource
from intergrax.runtime.nexus.planning.step_executor_models import StepExecutorConfig
from intergrax.runtime.nexus.planning.step_planner import StepPlannerConfig
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager


class FakeLLMAdapter(LLMAdapter):
    """
    Deterministic LLM adapter for CI-safe tests.

    Goals:
    - no network
    - stable output
    - still exercises CoreLLMStep / finalization path
    """

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
        return self._fixed_text

    def context_window_tokens(self) -> int:
        # Large enough for tests; avoids truncation logic influencing results.
        return 128_000


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
    step_planning_strategy: StepPlanningStrategy,
    plan_specs: Sequence[PlanSpec],
    llm_text: str = "OK",
    plan_loop_policy: Optional[PlanLoopPolicy] = None,
) -> RuntimeConfig:
    """
    Deterministic RuntimeConfig for CI:
    - no RAG/vectorstore/web/tools unless explicitly enabled later
    - scripted plan source
    - required planner/step configs present (fail-fast validations pass)
    """
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
        step_planning_strategy=step_planning_strategy,
        step_planner_cfg=StepPlannerConfig(),
        step_executor_cfg=StepExecutorConfig(),
        planner_prompt_config=PlannerPromptConfig(),
        plan_loop_policy=plan_loop_policy or PlanLoopPolicy(),
        plan_source=ScriptedPlanSource(plans=plan_specs),
    )

    # If RuntimeConfig exposes validate(), keep it enabled (enterprise style).
    # This makes test failures immediate and readable.
    cfg.validate()

    return cfg


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
