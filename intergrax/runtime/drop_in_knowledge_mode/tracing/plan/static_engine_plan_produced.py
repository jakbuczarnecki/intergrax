# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.drop_in_knowledge_mode.tracing.trace_models import DiagnosticPayload


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

    capability_clamp: Optional[str]

    planner_forced_plan_used: Optional[bool]
    planner_forced_plan_hash: Optional[str]

    planner_replan_ctx_present: Optional[bool]
    planner_replan_ctx_hash: Optional[str]

    @property
    def schema_id(self) -> str:
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
            "capability_clamp": self.capability_clamp,
            "planner_forced_plan_used": self.planner_forced_plan_used,
            "planner_forced_plan_hash": self.planner_forced_plan_hash,
            "planner_replan_ctx_present": self.planner_replan_ctx_present,
            "planner_replan_ctx_hash": self.planner_replan_ctx_hash,
        }
