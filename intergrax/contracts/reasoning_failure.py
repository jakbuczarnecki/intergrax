# © Artur Czarnecki. All rights reserved.

"""Reasoning failure taxonomy (COG-6.1)."""

from __future__ import annotations

from enum import Enum


class ReasoningFailureKind(str, Enum):
    """Planning/classification failure kinds for trace payloads."""

    PLANNER_PARSE_FAILED = "planner_parse_failed"
    PLANNER_FALLBACK = "planner_fallback"
    PLANNER_VALIDATION_FAILED = "planner_validation_failed"
    PLANNER_POLICY_BLOCKED = "planner_policy_blocked"
    CLASSIFIER_FALLBACK = "classifier_fallback"
    CLASSIFIER_UNSUPPORTED = "classifier_unsupported"
