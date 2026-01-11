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

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.nexus.session.consolidation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries_count": int(self.entries_count),
            "entry_types": dict(self.entry_types),
        }
