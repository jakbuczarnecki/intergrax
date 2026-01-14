# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class RuntimeRunRetryDiagV1(DiagnosticPayload):
    run_id: str = ""
    attempt: int = 0
    max_retries: int = 0
    error_code: RuntimeErrorCode = RuntimeErrorCode.INTERNAL_ERROR

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.runtime.run_retry"

    @classmethod
    def schema_version(cls) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "attempt": self.attempt,
            "max_retries": self.max_retries,
            "error_code": self.error_code.value,
        }
