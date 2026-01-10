# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

class OnMaxReplansPolicy(str, Enum):
    RAISE = "raise"
    HITL = "hitl"

@dataclass(frozen=True)
class PlanLoopPolicy:
    max_replans: int = 1
    # If replanning produces the same engine plan repeatedly, fail fast / escalate.
    max_same_plan_repeats: int = 1

    on_max_replans: OnMaxReplansPolicy = OnMaxReplansPolicy.RAISE

    # Builder for escalation HITL message
    hitl_escalation_message_builder: Callable[[Optional[str]], str] = (
        lambda reason: (
            "I need one clarification to continue."
            + (f" Replan reason: {reason}" if reason else "")
            + " Please clarify what you want me to do next."
        )
    )