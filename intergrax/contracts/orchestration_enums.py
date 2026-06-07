# © Artur Czarnecki. All rights reserved.

"""Orchestration profile enums (Phase FLOW)."""

from __future__ import annotations

from enum import Enum


class MergeStrategy(str, Enum):
    """Final response composition strategy for multi-agent graph runs."""

    CONCAT = "concat"
    LAST_WINS = "last_wins"
    STRUCTURED_JSON = "structured_json"


class MultiAgentOrder(str, Enum):
    """Deterministic ordering for auto-generated multi-agent plans."""

    REGISTRY = "registry"
    STABLE_ALPHA = "stable_alpha"
    PRIORITY = "priority"
