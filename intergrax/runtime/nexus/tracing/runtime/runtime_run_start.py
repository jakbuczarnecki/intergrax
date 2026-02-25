# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DEFAULT_REDACTED_TEXT, DiagnosticPayload


@dataclass(frozen=True)
class RuntimeRunStartDiagV1(DiagnosticPayload):
    session_id: str = ""
    user_id: str = ""
    tenant_id: str = ""
    run_id: str = ""

    pipeline_name: str = ""

    def redact(self) -> RuntimeRunStartDiagV1:
        """
        This diagnostic payload contains runtime identifiers including
        user_id and tenant_id, which must not be persisted in raw form
        in production traces.
        We preserve technical execution identifiers but remove
        user- and tenant-level identifiers.
        """
        return RuntimeRunStartDiagV1(
            session_id=self.session_id,
            user_id=DEFAULT_REDACTED_TEXT,
            tenant_id=DEFAULT_REDACTED_TEXT,
            run_id=self.run_id,
            pipeline_name=self.pipeline_name,
        )

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.runtime.run_start"

    @classmethod
    def schema_version(cls) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
        }

