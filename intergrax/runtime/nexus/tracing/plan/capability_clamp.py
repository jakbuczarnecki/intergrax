# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PlannerCapabilityClampDiagV1(DiagnosticPayload):
    before_use_websearch: bool
    before_use_user_longterm_memory: bool
    before_use_rag: bool
    before_use_tools: bool

    available_websearch: bool
    available_user_ltm: bool
    available_rag: bool
    available_tools: bool

    after_use_websearch: bool
    after_use_user_longterm_memory: bool
    after_use_rag: bool
    after_use_tools: bool

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.planner.capability_clamp"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "before_use_websearch": self.before_use_websearch,
            "before_use_user_longterm_memory": self.before_use_user_longterm_memory,
            "before_use_rag": self.before_use_rag,
            "before_use_tools": self.before_use_tools,
            "available_websearch": self.available_websearch,
            "available_user_ltm": self.available_user_ltm,
            "available_rag": self.available_rag,
            "available_tools": self.available_tools,
            "after_use_websearch": self.after_use_websearch,
            "after_use_user_longterm_memory": self.after_use_user_longterm_memory,
            "after_use_rag": self.after_use_rag,
            "after_use_tools": self.after_use_tools,
        }
