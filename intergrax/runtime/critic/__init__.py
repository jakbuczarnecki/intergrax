# © Artur Czarnecki. All rights reserved.

"""Legacy Critic & Verification Layer (CVL) — migration/tests only after DS-MIG-02."""

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
    validate_final_with_critic_detail,
    validate_node_with_critic,
    validate_node_with_critic_detail,
    validate_uaep_step_with_critic_detail,
    critic_completion_blocked,
)
from intergrax.runtime.critic.eval_tool_client import CriticEvalToolClient
from intergrax.runtime.critic.evaluator_loop_executor import (
    EvaluatorLoopDecision,
    EvaluatorLoopExecutor,
    EvaluatorLoopIterationState,
    EvaluatorLoopOutcome,
)
from intergrax.runtime.critic.evaluator_loop_metadata import (
    COORDINATION_PATTERN_KEY,
    EVALUATOR_LOOP_ITERATION_KEY,
    EVALUATOR_LOOP_SPEC_KEY,
    evaluator_loop_spec_from_node,
    tag_node_evaluator_loop,
)
from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec
from intergrax.runtime.critic.l0_gateway import L0Gateway
from intergrax.runtime.critic.l1_gateway import L1Gateway
from intergrax.runtime.critic.policy_bridge import (
    critic_governance_from_fragment,
    resolve_critic_action,
)
from intergrax.runtime.critic.tool_registry_client import ToolRegistryCriticEvalClient
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
    "COORDINATION_PATTERN_KEY",
    "EVALUATOR_LOOP_ITERATION_KEY",
    "EVALUATOR_LOOP_SPEC_KEY",
    "EvaluatorLoopDecision",
    "EvaluatorLoopExecutor",
    "EvaluatorLoopIterationState",
    "EvaluatorLoopOutcome",
    "EvaluatorLoopSpec",
    "evaluator_loop_spec_from_node",
    "tag_node_evaluator_loop",
    "critic_completion_blocked",
    "L0Gateway",
    "L1Gateway",
    "ToolRegistryCriticEvalClient",
    "critic_governance_from_fragment",
    "resolve_critic_action",
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
    "validate_final_with_critic_detail",
    "validate_node_with_critic",
    "validate_node_with_critic_detail",
    "validate_uaep_step_with_critic_detail",
]
