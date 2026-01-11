# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class LLMUsageFinalizeDiag(DiagnosticPayload):
    run_id: str = ""
    session_id: str = ""
    user_id: str = ""
    aborted: bool = False

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.llm_usage_finalize"

    @classmethod
    def schema_version(cls) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "aborted": self.aborted,
        }

