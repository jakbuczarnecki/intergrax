# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from dataclasses import dataclass
from typing import Dict, Any

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class ExecutionConcurrencyDiagV1(DiagnosticPayload):
    tenant_id: str
    run_id: str
    action: str  # "acquire_success" | "acquire_rejected" | "release"

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.execution.concurrency"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "run_id": self.run_id,
            "action": self.action,
        }

    def redact(self) -> "ExecutionConcurrencyDiagV1":
        # No PII stored in this payload
        return self