# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class RuntimeStepFailedDiagV1(DiagnosticPayload):
    step_name: str
    error_type: str
    error_message: str
    error_repr: Optional[str]

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
