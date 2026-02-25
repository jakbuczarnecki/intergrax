# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DEFAULT_REDACTED_TEXT, DiagnosticPayload


@dataclass(frozen=True)
class LLMUsageFinalizeDiag(DiagnosticPayload):
    run_id: str = ""
    session_id: str = ""
    user_id: str = ""
    aborted: bool = False

    def redact(self) -> LLMUsageFinalizeDiag:
        """
        This diagnostic payload contains user-level identifier (user_id)
        which must not be persisted in raw form in production traces.
        We preserve technical execution identifiers but remove user_id.
        """
        return LLMUsageFinalizeDiag(
            run_id=self.run_id,
            session_id=self.session_id,
            user_id=DEFAULT_REDACTED_TEXT,
            aborted=self.aborted,
        )

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

