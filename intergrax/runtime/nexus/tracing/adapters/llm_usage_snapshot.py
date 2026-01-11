# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class LLMUsageSnapshotDiag(DiagnosticPayload):
    run_id: str = ""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    errors: int = 0

    # small, stable extras
    adapters_registered: int = 0
    provider_model_groups: int = 0

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.llm_usage_snapshot"

    @classmethod
    def schema_version(cls) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "adapters_registered": self.adapters_registered,
            "provider_model_groups": self.provider_model_groups,
        }

