# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class RuntimeRunEndDiagV1(DiagnosticPayload):
    strategy: str = ""
    used_rag: bool = False
    used_websearch: bool = False
    used_tools: bool = False
    used_user_longterm_memory: bool = False
    run_id: str = ""

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.runtime.run_end"

    @classmethod
    def schema_version(cls) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "used_rag": self.used_rag,
            "used_websearch": self.used_websearch,
            "used_tools": self.used_tools,
            "used_user_longterm_memory": self.used_user_longterm_memory,
            "run_id": self.run_id,
        }

