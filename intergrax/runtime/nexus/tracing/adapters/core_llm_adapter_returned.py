# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class CoreLLMAdapterReturnedDiagV1(DiagnosticPayload):
    used_tools_answer: bool
    adapter_return_type: str
    answer_len: int
    answer_is_empty: bool

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.engine.core_llm.adapter_returned"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "used_tools_answer": self.used_tools_answer,
            "adapter_return_type": self.adapter_return_type,
            "answer_len": self.answer_len,
            "answer_is_empty": self.answer_is_empty,
        }
