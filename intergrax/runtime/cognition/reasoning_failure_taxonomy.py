# © Artur Czarnecki. All rights reserved.

"""Reasoning failure taxonomy (IDEAL-7.2)."""

from __future__ import annotations

from enum import Enum


class ReasoningFailureFamily(str, Enum):
    PLANNING = "planning"
    CLASSIFICATION = "classification"
    TOOL_SELECTION = "tool_selection"
    CONTEXT_OVERFLOW = "context_overflow"
    POLICY_BLOCK = "policy_block"
    QUALITY = "quality"
