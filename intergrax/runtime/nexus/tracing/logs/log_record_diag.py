# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class LogRecordDiagV1(DiagnosticPayload):
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

    def redact(self) -> LogRecordDiagV1:
        """
        This diagnostic payload may contain arbitrary structured logging data
        which can include user content, prompts, tool inputs/outputs
        or external API responses.
        In production, log data must not be persisted in trace.
        """
        return LogRecordDiagV1(
            logger_name=self.logger_name,
            pathname=self.pathname,
            lineno=self.lineno,
            func_name=self.func_name,
            levelname=self.levelname,
            data={"_redacted": True},
        )

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.log_record"

    @classmethod
    def schema_version(cls) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logger_name": self.logger_name,
            "pathname": self.pathname,
            "lineno": self.lineno,
            "func_name": self.func_name,
            "levelname": self.levelname,
            "data": self.data,
        }

