# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class SessionConsolidationDiagV1(DiagnosticPayload):
    entries_count: int
    entry_types: Dict[str, int]

    def redact(self) -> SessionConsolidationDiagV1:
        """
        This diagnostic payload contains only aggregation metadata
        about session entry counts and types.
        It does not include user content or identifying information
        and is considered PII-safe.
        """
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.nexus.session.consolidation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries_count": int(self.entries_count),
            "entry_types": dict(self.entry_types),
        }
