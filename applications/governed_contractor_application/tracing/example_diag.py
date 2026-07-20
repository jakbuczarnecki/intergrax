# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload
from intergrax.runtime.observability.extension_sdk import application_diagnostic_schema_id


@dataclass(frozen=True)
class HostLifecycleDiagV1(DiagnosticPayload):
    """Example application diagnostic payload — replace with product semantics."""

    phase: str
    status: str

    @classmethod
    def schema_id(cls) -> str:
        return application_diagnostic_schema_id("governed_contractor", "host_lifecycle")

    def to_dict(self) -> Dict[str, Any]:
        return {"phase": self.phase, "status": self.status}

    def redact(self) -> HostLifecycleDiagV1:
        return self
