# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class InstructionsSummaryDiagV1(DiagnosticPayload):
    has_instructions: bool
    source_request: bool
    source_user_profile: bool
    source_org_profile: bool

    @property
    def schema_id(self) -> str:
        return "intergrax.diag.instructions.summary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_instructions": self.has_instructions,
            "sources": {
                "request": self.source_request,
                "user_profile": self.source_user_profile,
                "organization_profile": self.source_org_profile,
            },
        }
