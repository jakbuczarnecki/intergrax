# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class BudgetExceededDiagV1(DiagnosticPayload):
    """
    Emitted when a runtime budget is exceeded.
    """
    run_id: str
    budget_name: str
    limit: Optional[float]
    actual: Optional[float]
    enforcement_mode: str

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.runtime.budget_exceeded"

    @classmethod
    def schema_version(cls) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "budget_name": self.budget_name,
            "limit": self.limit,
            "actual": self.actual,
            "enforcement_mode": self.enforcement_mode,
        }
