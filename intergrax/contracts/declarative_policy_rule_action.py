# © Artur Czarnecki. All rights reserved.

"""Declarative policy rule action vocabulary (shared contracts / runtime policy)."""

from __future__ import annotations

from enum import Enum


class PolicyRuleAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_HITL = "require_hitl"
