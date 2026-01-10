# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class PlannerRawPlanParseFailedDiagV1(DiagnosticPayload):
    planner_source_kind: Optional[str]
    planner_source_detail: Optional[str]

    raw_len: int
    raw_hash: str
    raw_preview: str
    raw_tail_preview: str

    error_type: str
    error_message: str

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.planner.raw_plan_parse_failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "planner_source_kind": self.planner_source_kind,
            "planner_source_detail": self.planner_source_detail,
            "raw_len": self.raw_len,
            "raw_hash": self.raw_hash,
            "raw_preview": self.raw_preview,
            "raw_tail_preview": self.raw_tail_preview,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
