# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PlannerStaticEnginePlanProducedDiagV1(DiagnosticPayload):
    fingerprint: str

    intent: str
    next_step: Optional[str]

    ask_clarifying_question: bool

    use_websearch: bool
    use_user_longterm_memory: bool
    use_rag: bool
    use_tools: bool

    same_plan_repeats: int


    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.planner.static_engine_plan_produced"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "intent": self.intent,
            "next_step": self.next_step,
            "ask_clarifying_question": self.ask_clarifying_question,
            "use_websearch": self.use_websearch,
            "use_user_longterm_memory": self.use_user_longterm_memory,
            "use_rag": self.use_rag,
            "use_tools": self.use_tools,
            "same_plan_repeats": self.same_plan_repeats,
        }
