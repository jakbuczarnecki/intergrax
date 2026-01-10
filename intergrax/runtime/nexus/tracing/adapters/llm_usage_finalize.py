# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class LLMUsageFinalizeDiag(DiagnosticPayload):
    schema_id: str = "intergrax.diag.llm_usage_finalize"
    schema_version: int = 1

    run_id: str = ""
    session_id: str = ""
    user_id: str = ""
    aborted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "aborted": self.aborted,
        }
