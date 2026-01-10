# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class RuntimeRunAbortDiagV1(DiagnosticPayload):
    schema_id: str = "intergrax.diag.runtime.run_abort"
    schema_version: int = 1

    run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"run_id": self.run_id}
