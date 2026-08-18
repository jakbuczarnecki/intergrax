# © Artur Czarnecki. All rights reserved.

"""Low-level policy action vocabulary — no agent-decision dependencies."""

from __future__ import annotations

from enum import Enum


class PolicyAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    MODIFY = "modify"
    ESCALATE = "escalate"
    REQUIRE_HUMAN = "require_human"


class EnforcementLevel(str, Enum):
    ADVISORY = "advisory"
    MANDATORY = "mandatory"
