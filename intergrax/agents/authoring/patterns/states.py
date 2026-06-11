# © Artur Czarnecki. All rights reserved.

"""Typed session state subclasses per cognitive pattern (architecture §25.2)."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from intergrax.contracts.acp_state import AcpSessionState
from intergrax.contracts.agent_run_enums import CognitivePattern


class ReActSessionState(AcpSessionState):
    model_config = ConfigDict(extra="forbid")

    pattern: CognitivePattern = CognitivePattern.REACT
    react_iterations_used: int = Field(default=0, ge=0)
    max_react_iterations: int = Field(default=8, ge=1)
    last_thought: str | None = None


class PlanExecuteSessionState(AcpSessionState):
    model_config = ConfigDict(extra="forbid")

    pattern: CognitivePattern = CognitivePattern.PLAN_EXECUTE
    phase: str = "plan"


class DecompositionSessionState(AcpSessionState):
    model_config = ConfigDict(extra="forbid")

    pattern: CognitivePattern = CognitivePattern.DECOMPOSITION
    pending_sub_questions: list[str] = Field(default_factory=list)
    answered: dict[str, str] = Field(default_factory=dict)
    max_sub_questions: int = Field(default=8, ge=1)


class ReflectionSessionState(AcpSessionState):
    model_config = ConfigDict(extra="forbid")

    pattern: CognitivePattern = CognitivePattern.REFLECTION
    phase: str = "draft"
    draft: str | None = None
    critique: str | None = None
