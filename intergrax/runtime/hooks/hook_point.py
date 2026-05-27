# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hook interception points (architecture §42.3, §42.42)."""

from __future__ import annotations

from enum import Enum


class HookPoint(str, Enum):
    BEFORE_TASK_INTAKE = "before_task_intake"
    AFTER_TASK_INTAKE = "after_task_intake"
    BEFORE_CLASSIFICATION = "before_classification"
    AFTER_CLASSIFICATION = "after_classification"
    BEFORE_PLANNING = "before_planning"
    AFTER_PLANNING = "after_planning"
    BEFORE_AGENT_SELECTION = "before_agent_selection"
    AFTER_AGENT_SELECTION = "after_agent_selection"
    BEFORE_CONTEXT_BUILD = "before_context_build"
    AFTER_CONTEXT_BUILD = "after_context_build"
    BEFORE_STEP = "before_step"
    AFTER_STEP = "after_step"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_VALIDATION = "before_validation"
    AFTER_VALIDATION = "after_validation"
    BEFORE_DECISION = "before_decision"
    AFTER_DECISION = "after_decision"
    BEFORE_INTERRUPT = "before_interrupt"
    AFTER_INTERRUPT = "after_interrupt"
    BEFORE_HUMAN_APPROVAL = "before_human_approval"
    AFTER_HUMAN_APPROVAL = "after_human_approval"
    BEFORE_RETRY = "before_retry"
    AFTER_RETRY = "after_retry"
    BEFORE_HANDOFF = "before_handoff"
    AFTER_HANDOFF = "after_handoff"
    BEFORE_FINALIZATION = "before_finalization"
    AFTER_FINALIZATION = "after_finalization"
    BEFORE_TRACE_PERSIST = "before_trace_persist"
    AFTER_TRACE_PERSIST = "after_trace_persist"
