# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class RuntimeStepFinishedDiagV1(DiagnosticPayload):
    step_name: str

    def redact(self) -> RuntimeStepFinishedDiagV1:
        """
        This diagnostic payload contains only the technical name
        of a runtime step.
        It does not include user content or identifying information
        and is considered PII-safe.
        """
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.runtime.step_finished"

    def to_dict(self) -> Dict[str, Any]:
        return {"step_name": self.step_name}
