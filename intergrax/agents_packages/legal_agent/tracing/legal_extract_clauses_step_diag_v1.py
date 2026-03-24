# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

LegalExtractClausesOutcome = Literal["no_hits", "extracted"]


@dataclass(frozen=True)
class LegalExtractClausesStepDiagV1(DiagnosticPayload):
    step_name: str
    outcome: LegalExtractClausesOutcome
    retrieval_chunks_count: int
    clauses_extracted_count: int
    llm_calls_count: int
    attachments_ingested: bool
    pre_flagged_sensitive_clauses_count: int

    def redact(self) -> LegalExtractClausesStepDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.legal.extract_clauses.step.v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "outcome": self.outcome,
            "retrieval_chunks_count": self.retrieval_chunks_count,
            "clauses_extracted_count": self.clauses_extracted_count,
            "llm_calls_count": self.llm_calls_count,
            "attachments_ingested": self.attachments_ingested,
            "pre_flagged_sensitive_clauses_count": self.pre_flagged_sensitive_clauses_count,
        }
