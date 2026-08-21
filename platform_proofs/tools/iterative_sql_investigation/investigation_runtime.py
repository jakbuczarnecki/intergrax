# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3C).

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.profile import LLMProfile, llm_profile_from_env
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.config_types import ToolsContextScope
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tools.tool_loop import run_bounded_tool_loop
from intergrax.runtime.nexus.tools.tool_planning_config import ToolPlanningConfig
from intergrax.runtime.nexus.tools.tool_planning_service import ToolPlanningService
from intergrax.tools.core.tool_plan import ToolCallPlan
from intergrax.tools.core.tool_plan_decision import ToolPlanDecision

from platform_proofs.tools.iterative_sql_investigation.contracts import PLATFORM_PROOF_SQL_QUERY_TOOL_ID
from platform_proofs.tools.iterative_sql_investigation.evaluator import (
    build_execution_snapshot,
    evaluate_scenario,
)
from platform_proofs.tools.iterative_sql_investigation.model_context import build_investigation_messages
from platform_proofs.tools.iterative_sql_investigation.proof_result import (
    ModelProviderIdentity,
    ScenarioRunResult,
)
from platform_proofs.tools.iterative_sql_investigation.runtime import ProofSqlRuntime
from platform_proofs.tools.iterative_sql_investigation.scenarios import (
    MAX_TOOL_CALLS_PER_ROUND,
    MAX_TOOL_ITERATIONS,
    InvestigationScenario,
)


def _build_session_manager() -> SessionManager:
    return SessionManager(InMemorySessionStorage())


class RecordingToolPlanner:
    """Proof-local delegate that preserves canonical ToolPlanningService behavior."""

    def __init__(self, inner: ToolPlanningService) -> None:
        self._inner = inner
        self.last_final_answer: str = ""

    def attach_routing_runtime_config(self, config: object) -> None:
        self._inner.attach_routing_runtime_config(config)

    def plan_tools(
        self,
        input_data: str | list[ChatMessage],
        context: Any = None,
        *,
        run_id: str,
        allowed_tool_ids: Sequence[str] | None = None,
    ) -> ToolPlanDecision:
        return self._inner.plan_tools(
            input_data,
            context,
            run_id=run_id,
            allowed_tool_ids=allowed_tool_ids,
        )

    def plan_native_round(
        self,
        messages: list[ChatMessage],
        *,
        allowed_tool_ids: Sequence[str] | None = None,
        run_id: str,
        tool_choice: Any = None,
    ) -> tuple[LLMAdapterResponse, ToolCallPlan]:
        result, plan = self._inner.plan_native_round(
            messages,
            allowed_tool_ids=allowed_tool_ids,
            run_id=run_id,
            tool_choice=tool_choice,
        )
        if result.content and not plan.calls:
            self.last_final_answer = result.content
        return result, plan


class ProofConfigurationError(RuntimeError):
    """Missing or invalid proof runtime configuration."""


class ProofProviderUnavailableError(RuntimeError):
    """Real LLM provider unavailable for canonical proof execution."""


def resolve_llm_profile_from_env() -> LLMProfile:
    return llm_profile_from_env()


def build_real_llm_adapter(profile: LLMProfile | None = None) -> LLMAdapter:
    from intergrax.llm_adapters.llm_provider_registry import LLMAdapterDependencyError

    resolved = profile or resolve_llm_profile_from_env()
    warnings = resolved.validate_runtime()
    if warnings:
        slug = resolved.provider.value if hasattr(resolved.provider, "value") else str(resolved.provider)
        if any("api_key" in warning for warning in warnings) and slug not in {
            "ollama",
            "vllm",
            "llama_cpp",
        }:
            raise ProofProviderUnavailableError(
                f"provider credentials unavailable: {'; '.join(warnings)}"
            )
    try:
        adapter = resolved.create_adapter()
    except LLMAdapterDependencyError as exc:
        raise ProofProviderUnavailableError(str(exc)) from exc
    if not adapter.supports_tools():
        raise ProofConfigurationError(
            "selected LLM adapter does not support native tool calling"
        )
    return adapter


def model_provider_identity(llm: LLMAdapter, profile: LLMProfile) -> ModelProviderIdentity:
    provider = profile.provider.value if hasattr(profile.provider, "value") else str(profile.provider)
    model = profile.model or getattr(llm, "model", "") or "unknown"
    return ModelProviderIdentity(
        provider=provider,
        model=str(model),
        supports_native_tools=bool(llm.supports_tools()),
    )


def build_proof_runtime_state(*, llm: LLMAdapter) -> RuntimeState:
    run_id = mint_run_id()
    config = RuntimeConfig(
        llm_adapter=llm,
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tools_context_scope=ToolsContextScope.CURRENT_MESSAGE_ONLY,
        max_tool_iterations=MAX_TOOL_ITERATIONS,
        max_tool_calls_per_round=MAX_TOOL_CALLS_PER_ROUND,
        max_identical_tool_call_repeats=2,
        tool_invoker=None,
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=_build_session_manager(),
        prompt_registry=MagicMock(),
    )
    return RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id="tools-sql-proof-agent",
            user_id="tools-sql-proof-user",
            session_id="tools-sql-proof-session",
            tenant_id="tools-sql-proof-tenant",
            message="investigate",
            task_id=mint_task_id(),
            run_id=run_id,
        ),
        run_id=str(run_id),
        messages_for_llm=[],
    )


def build_canonical_tool_planner(
    *,
    llm: LLMAdapter,
    proof_runtime: ProofSqlRuntime,
) -> RecordingToolPlanner:
    service = ToolPlanningService(
        llm=llm,
        tools=proof_runtime.registry,
        config=ToolPlanningConfig.default(),
    )
    return RecordingToolPlanner(service)


def run_investigation_scenario(
    *,
    scenario: InvestigationScenario,
    llm: LLMAdapter,
    proof_runtime: ProofSqlRuntime,
) -> ScenarioRunResult:
    state = build_proof_runtime_state(llm=llm)
    planner = build_canonical_tool_planner(llm=llm, proof_runtime=proof_runtime)
    planner.attach_routing_runtime_config(state.context.config)
    messages = build_investigation_messages(question=scenario.question)
    loop_result = run_bounded_tool_loop(
        state=state,
        invoker=proof_runtime.invoker,
        tool_planner=planner,
        planner_input=messages,
        allowed_tool_ids=(PLATFORM_PROOF_SQL_QUERY_TOOL_ID,),
        max_iterations=MAX_TOOL_ITERATIONS,
    )
    snapshot = build_execution_snapshot(
        traces=loop_result.tool_traces,
        investigation_proof=loop_result.investigation_proof,
        stop_reason=loop_result.stop_reason,
        final_answer=planner.last_final_answer,
    )
    return evaluate_scenario(scenario.scenario_id, snapshot)
