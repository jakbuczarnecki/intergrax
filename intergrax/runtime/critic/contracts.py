# © Artur Czarnecki. All rights reserved.

"""CVL contracts — critic requests, layer verdicts, rubric specs (Phase CRIT-V-1.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer


class CriticScope(str, Enum):
    """Where in the execution lifecycle verification runs."""

    NODE_PARTIAL = "node_partial"
    GRAPH_FINAL = "graph_final"
    UAEP_STEP = "uaep_step"
    OFFLINE_CASE = "offline_case"


class CriticLayer(str, Enum):
    """Verification layer in the L0 / L1 / L2 stack."""

    L0_DETERMINISTIC = "l0_deterministic"
    L1_SEMANTIC = "l1_semantic"
    L1_TRAJECTORY = "l1_trajectory"
    L2_HUMAN = "l2_human"


class CriticAction(str, Enum):
    """Recommended orchestration action after a critic verdict."""

    CONTINUE = "continue"
    RETRY = "retry"
    REVISE = "revise"
    ESCALATE_HITL = "escalate_hitl"
    FAIL = "fail"


class RubricSpec(BaseModel):
    """Domain-authored rubric reference for L1 semantic judges."""

    model_config = ConfigDict(extra="forbid")

    rubric_id: str
    prompt_registry_ref: str | None = None
    criteria: list[str] = Field(default_factory=list)
    min_score: float = Field(default=0.75, ge=0.0, le=1.0)
    reference_context: str | None = None


class LayerVerdict(BaseModel):
    """Result of a single critic layer."""

    model_config = ConfigDict(extra="forbid")

    layer: CriticLayer
    passed: bool
    score: float | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CriticVerdict(BaseModel):
    """Combined verdict across enabled critic layers."""

    model_config = ConfigDict(extra="forbid")

    scope: CriticScope
    passed: bool
    layers: list[LayerVerdict] = Field(default_factory=list)
    recommended_action: CriticAction = CriticAction.CONTINUE
    failure_reasons: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CriticRequest:
    """Runtime input to the critic orchestrator (Phase CRIT-V-3)."""

    scope: CriticScope
    run_id: str
    agent_id: str
    enabled_layers: tuple[CriticLayer, ...] = (CriticLayer.L0_DETERMINISTIC,)
    execution: AgentExecutionResult | None = None
    answer: RuntimeAnswer | None = None
    rubric: RubricSpec | None = None
    context: dict[str, Any] = field(default_factory=dict)


def build_critic_request(
    *,
    scope: CriticScope,
    run_id: str,
    agent_id: str,
    enabled_layers: tuple[CriticLayer, ...] | None = None,
    execution: AgentExecutionResult | None = None,
    answer: RuntimeAnswer | None = None,
    rubric: RubricSpec | None = None,
    context: dict[str, Any] | None = None,
) -> CriticRequest:
    """Construct a critic request with safe defaults."""
    return CriticRequest(
        scope=scope,
        run_id=run_id,
        agent_id=agent_id,
        enabled_layers=enabled_layers or (CriticLayer.L0_DETERMINISTIC,),
        execution=execution,
        answer=answer,
        rubric=rubric,
        context=dict(context or {}),
    )
