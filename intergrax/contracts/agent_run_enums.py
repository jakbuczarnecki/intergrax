# © Artur Czarnecki. All rights reserved.

"""Controlled enums for agent run and step contracts (architecture §37.4–§37.5)."""

from __future__ import annotations

from enum import Enum


class AgentRunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class PrincipalType(str, Enum):
    USER = "user"
    SERVICE = "service"
    ORG_SYSTEM = "org_system"


class AgentRunAutonomyLevel(str, Enum):
    STRICT = "strict"
    BALANCED = "balanced"
    EXPLORATORY = "exploratory"


class SideEffectMode(str, Enum):
    IMMEDIATE = "immediate"
    DECLARATIVE = "declarative"


class StepNextAction(str, Enum):
    CONTINUE = "continue"
    PAUSE_HITL = "pause_hitl"
    FAIL = "fail"
    REPLAN = "replan"


class AgentRunErrorCode(str, Enum):
    POLICY_DENIED = "policy_denied"
    TOOL_FAILED = "tool_failed"
    LLM_FAILED = "llm_failed"
    RAG_FAILED = "rag_failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    VALIDATION_FAILED = "validation_failed"
    HITL_REQUIRED = "hitl_required"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


class TerminalReason(str, Enum):
    GOAL_MET = "goal_met"
    BEST_EFFORT = "best_effort"
    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    HUMAN_REQUIRED = "human_required"
    POLICY_DENIED = "policy_denied"
    VALIDATION_FAILED = "validation_failed"
    CANCELLED = "cancelled"
    ERROR = "error"
    REPLANNED = "replanned"
    DELEGATED = "delegated"


class CognitivePattern(str, Enum):
    REFLEX = "reflex"
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    DECOMPOSITION = "decomposition"
    REFLECTION = "reflection"
    CUSTOM = "custom"
