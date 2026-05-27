# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Parse human approval/rejection/escalation response tokens."""

from __future__ import annotations

from intergrax.runtime.human.models import HumanResponseVerdict

_APPROVE_TOKENS = frozenset({"approve", "approved", "yes", "accept", "ok"})
_REJECT_TOKENS = frozenset({"reject", "rejected", "no", "deny", "denied", "decline"})
_ESCALATE_TOKENS = frozenset({"escalate", "escalated", "forward", "admin", "supervisor"})


def parse_human_response(response: str) -> HumanResponseVerdict:
    normalized = response.strip().lower()
    if normalized in _APPROVE_TOKENS:
        return HumanResponseVerdict.APPROVE
    if normalized in _REJECT_TOKENS:
        return HumanResponseVerdict.REJECT
    if normalized in _ESCALATE_TOKENS:
        return HumanResponseVerdict.ESCALATE
    return HumanResponseVerdict.UNKNOWN
