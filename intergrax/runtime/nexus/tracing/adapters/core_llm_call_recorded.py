# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class CoreLLMCallRecordedDiagV1(DiagnosticPayload):
    """PII-safe per-call LLM metadata for replay and metrics (M-LLM-R.7.2)."""

    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    response_id: str | None
    has_refusal: bool
    has_tool_calls: bool

    def redact(self) -> CoreLLMCallRecordedDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.engine.core_llm.call_recorded"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "response_id": self.response_id,
            "has_refusal": self.has_refusal,
            "has_tool_calls": self.has_tool_calls,
        }

    def replay_payload(self) -> Dict[str, Any]:
        """Payload shape consumed by ``ReplayEngine`` for ``LLM_CALL`` events."""
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
        }
