# © Artur Czarnecki. All rights reserved.

"""Cognitive pattern library (architecture §24–§26 · Wave 5)."""

from intergrax.agents.authoring.patterns.base import CognitiveAgent, PATTERN_VERSION
from intergrax.agents.authoring.patterns.decomposition import DecompositionAgent
from intergrax.agents.authoring.patterns.plan_execute import PlanExecuteAgent
from intergrax.agents.authoring.patterns.react import ReActAgent
from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.authoring.patterns.reflection import ReflectionAgent
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)

PATTERN_AGENT_BY_ID = {
    ReflexAgent.cognitive_pattern: ReflexAgent,
    ReActAgent.cognitive_pattern: ReActAgent,
    PlanExecuteAgent.cognitive_pattern: PlanExecuteAgent,
    DecompositionAgent.cognitive_pattern: DecompositionAgent,
    ReflectionAgent.cognitive_pattern: ReflectionAgent,
}

__all__ = [
    "AgentEvaluation",
    "CognitiveAgent",
    "CognitiveEvaluation",
    "DecompositionAgent",
    "Observation",
    "PATTERN_AGENT_BY_ID",
    "PATTERN_VERSION",
    "PlanExecuteAgent",
    "ReActAgent",
    "ReasoningResult",
    "ReflexAgent",
    "ReflectionAgent",
]
