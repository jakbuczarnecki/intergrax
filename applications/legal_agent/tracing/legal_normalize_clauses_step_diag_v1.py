# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class LegalNormalizeClausesStepDiagV1(DiagnosticPayload):
    step_name: str
    outcome: str
    input_clauses_count: int
    output_clauses_count: int
    duplicates_removed_count: int

    def redact(self) -> LegalNormalizeClausesStepDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.legal.normalize_clauses.step.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "outcome": self.outcome,
            "input_clauses_count": self.input_clauses_count,
            "output_clauses_count": self.output_clauses_count,
            "duplicates_removed_count": self.duplicates_removed_count,
        }
