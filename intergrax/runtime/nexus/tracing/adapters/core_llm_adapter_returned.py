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
    finish_reason: str
    input_tokens: int
    output_tokens: int
    answer_len: int
    answer_is_empty: bool

    def redact(self) -> CoreLLMAdapterReturnedDiagV1:
        """
        This diagnostic payload contains only LLM adapter execution metadata
        (flags and answer size information).
        It does not include any user content or model output
        and is considered PII-safe.
        """
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.engine.core_llm.adapter_returned"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "used_tools_answer": self.used_tools_answer,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "answer_len": self.answer_len,
            "answer_is_empty": self.answer_is_empty,
        }
