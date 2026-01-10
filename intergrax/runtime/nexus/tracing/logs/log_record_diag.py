# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class LogRecordDiagV1(DiagnosticPayload):
    schema_id: str = "intergrax.diag.log_record"
    schema_version: int = 1

    logger_name: str = ""
    pathname: str = ""
    lineno: int = 0
    func_name: str = ""
    levelname: str = ""

    # Structured extra context from logging (already pre-validated upstream)
    data: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.data is None:
            object.__setattr__(self, "data", {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logger_name": self.logger_name,
            "pathname": self.pathname,
            "lineno": self.lineno,
            "func_name": self.func_name,
            "levelname": self.levelname,
            "data": self.data,
        }
