# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DEFAULT_REDACTED_TEXT, DiagnosticPayload


@dataclass(frozen=True)
class RuntimeStepFailedDiagV1(DiagnosticPayload):
    step_name: str
    error_type: str
    error_message: str
    error_repr: Optional[str]

    def redact(self) -> RuntimeStepFailedDiagV1:
        """
        This diagnostic payload may contain raw exception messages
        and representations which can include user content, prompts,
        tool outputs or other sensitive data.
        In production, error details must not be persisted.
        We preserve structural metadata but remove raw error content.
        """
        return RuntimeStepFailedDiagV1(
            step_name=self.step_name,
            error_type=self.error_type,
            error_message=DEFAULT_REDACTED_TEXT,
            error_repr=DEFAULT_REDACTED_TEXT if self.error_repr is not None else None,
        )

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.runtime.step_failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_repr": self.error_repr,
        }
