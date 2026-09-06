# © Artur Czarnecki. All rights reserved.

"""Model-visible semantic evidence references (runtime/evidence boundary)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelVisibleEvidenceReference:
    """One semantic evidence identity exposed to the model outside native tool JSON."""

    evidence_reference: str
    provenance_tool_call_id: str | None = None
    acquisition_id: str | None = None

    def binding_id(self) -> str:
        """Runtime provenance key used by ENG-6 binding (native tool_call_id or acquisition id)."""
        if self.provenance_tool_call_id is not None:
            return self.provenance_tool_call_id
        if self.acquisition_id is not None:
            return self.acquisition_id
        raise ValueError("ModelVisibleEvidenceReference requires provenance identity")
