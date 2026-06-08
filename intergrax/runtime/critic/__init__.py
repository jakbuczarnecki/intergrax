# © Artur Czarnecki. All rights reserved.

"""Critic & Verification Layer (CVL) — Phase CRIT-V-1."""

from intergrax.runtime.critic.contracts import (
    CriticAction,
    CriticLayer,
    CriticRequest,
    CriticScope,
    CriticVerdict,
    LayerVerdict,
    RubricSpec,
    build_critic_request,
)
from intergrax.runtime.critic.critic_orchestrator import CriticOrchestrator
from intergrax.runtime.critic.critic_wiring import (
    CriticGraphHooks,
    CriticHookConfig,
    build_critic_graph_hooks,
    validate_final_with_critic,
    validate_node_with_critic,
)
from intergrax.runtime.critic.eval_tool_client import CriticEvalToolClient
from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec
from intergrax.runtime.critic.l0_gateway import L0Gateway
from intergrax.runtime.critic.l1_gateway import L1Gateway
from intergrax.runtime.critic.trace import (
    CriticTraceEmitter,
    build_critic_trace_emitter,
)
from intergrax.runtime.critic.trace_steps import (
    CRITIC_STEP_FINAL_VERDICT,
    CRITIC_STEP_L0_FAILED,
    CRITIC_STEP_L1_JUDGE,
    CRITIC_STEP_TRAJECTORY,
)

__all__ = [
    "CriticAction",
    "CriticEvalToolClient",
    "CriticLayer",
    "CriticGraphHooks",
    "CriticHookConfig",
    "CriticOrchestrator",
    "CriticTraceEmitter",
    "CriticRequest",
    "CriticScope",
    "CriticVerdict",
    "EvaluatorLoopSpec",
    "L0Gateway",
    "L1Gateway",
    "LayerVerdict",
    "RubricSpec",
    "build_critic_graph_hooks",
    "build_critic_request",
    "build_critic_trace_emitter",
    "CRITIC_STEP_FINAL_VERDICT",
    "CRITIC_STEP_L0_FAILED",
    "CRITIC_STEP_L1_JUDGE",
    "CRITIC_STEP_TRAJECTORY",
    "validate_final_with_critic",
    "validate_node_with_critic",
]
