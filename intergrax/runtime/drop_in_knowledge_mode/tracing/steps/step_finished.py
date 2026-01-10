# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.drop_in_knowledge_mode.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class RuntimeStepFinishedDiagV1(DiagnosticPayload):
    step_name: str

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.runtime.step_finished"

    def to_dict(self) -> Dict[str, Any]:
        return {"step_name": self.step_name}
