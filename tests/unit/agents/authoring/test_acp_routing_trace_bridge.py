# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.agents.acp_routing_trace_bridge import record_acp_routing_rule_evaluation
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    BudgetBelowRule,
    LLMRoutingEvaluator,
    LLMRoutingProfile,
    RoutingContext,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.kernel.step_kernel import StepKernelContext
from testing_support.builder import canonical_governed_execution_scope


@pytest.mark.unit
@pytest.mark.gate
def test_record_acp_routing_rule_evaluation_emits_plane_a_event() -> None:
    run_id = mint_run_id()
    task_id = mint_task_id()
    with canonical_governed_execution_scope(run_id):
        kernel_ctx = StepKernelContext(
            agent_id="agent-1",
            run_id=run_id,
            task_id=task_id,
            tenant_id="tenant-1",
        )
        evaluation = LLMRoutingEvaluator().evaluate(
            LLMRoutingProfile(
                default_profile=LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B"),
                allowed_profiles=(
                    LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini"),
                    LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B"),
                ),
                rules=(
                    BudgetBelowRule(
                        threshold=0.2,
                        profile=LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B"),
                    ),
                ),
            ),
            RoutingContext(budget_remaining_ratio=0.1),
        )

        record_acp_routing_rule_evaluation(kernel_ctx, evaluation)

        assert len(kernel_ctx.routing_rule_evaluations) == 1
        assert kernel_ctx.routing_rule_evaluations[0]["model"] == "meta-llama/Llama-3.1-8B"
        assert len(kernel_ctx.events) == 1
        event = kernel_ctx.events[0]
        assert event.event_type == RuntimeEventType.LLM_CALL
        assert event.payload["model"] == "meta-llama/Llama-3.1-8B"
        assert event.payload.get("trace_step") == "llm_routing_rule"
        assert event.task_id == task_id
        assert event.run_id == run_id
        assert kernel_ctx.routing_rule_evaluations[0]["matched_rule_id"] == "builtin.budget_below"
