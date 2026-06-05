# © Artur Czarnecki. All rights reserved.

"""Prompt governance enums shared by schema and runtime contracts."""

from __future__ import annotations

from enum import Enum


class PromptRiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
