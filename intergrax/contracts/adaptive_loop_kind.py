# © Artur Czarnecki. All rights reserved.

"""Adaptive loop kind enum — public declarative contract (Phase V-L4)."""

from __future__ import annotations

from enum import Enum


class AdaptiveLoopKind(str, Enum):
    ROUTING_TUNING = "routing_tuning"
    EXECUTION_STRATEGY_TUNING = "execution_strategy_tuning"
    POLICY_LEARNING = "policy_learning"
    EVALUATION_FEEDBACK = "evaluation_feedback"
