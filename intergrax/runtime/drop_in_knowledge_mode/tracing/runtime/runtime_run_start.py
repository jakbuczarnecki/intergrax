# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.drop_in_knowledge_mode.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class RuntimeRunStartDiagV1(DiagnosticPayload):
    schema_id: str = "intergrax.diag.runtime.run_start"
    schema_version: int = 1

    session_id: str = ""
    user_id: str = ""
    tenant_id: str = ""
    run_id: str = ""
    step_planning_strategy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "run_id": self.run_id,
            "step_planning_strategy": self.step_planning_strategy,
        }
