# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid

from intergrax.utils.time_provider import SystemTimeProvider


class TraceLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TraceComponent(str, Enum):
    RUNTIME = "runtime"
    ENGINE = "engine"
    PIPELINE = "pipeline"
    STEP = "step"
    TOOLS = "tools"
    WEBSEARCH = "websearch"
    RAG = "rag"
    MEMORY = "memory"
    PLANNER = "planner"


class DiagnosticPayload(ABC):
    """
    Typed diagnostic payload contract (production).

    Rules:
    - schema_id: stable identifier (never reused for different semantics)
    - schema_version: bump only when schema changes
    - to_dict(): MUST return JSON-serializable dict
    """

    @classmethod
    @abstractmethod
    def schema_id(cls) -> str:
        raise NotImplementedError

    @classmethod
    def schema_version(cls) -> int:
        return 1

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError



def utc_now_iso() -> str:
    return SystemTimeProvider.utc_now().isoformat()


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    run_id: str
    seq: int
    ts_utc: str

    level: TraceLevel
    component: TraceComponent
    step: str
    message: str

    payload: Optional[DiagnosticPayload] = None

    tags: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())
    

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-safe serialization for notebooks/tests/log export.

        Notes:
        - Enums are serialized to their `.value`.
        - payload is exported as:
            - payload_schema_id / payload_schema_version computed from payload classmethods
            - payload = payload.to_dict()
        - tags is kept as-is (must be JSON-safe by convention).
        """
        payload_schema_id: Optional[str] = None
        payload_schema_version: Optional[int] = None
        payload_dict: Optional[Dict[str, Any]] = None

        if self.payload is not None:
            payload_schema_id = self.payload.__class__.schema_id()
            payload_schema_version = self.payload.__class__.schema_version()
            payload_dict = self.payload.to_dict()

        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "seq": self.seq,
            "ts_utc": self.ts_utc,
            "level": self.level.value,
            "component": self.component.value,
            "step": self.step,
            "message": self.message,
            "payload_schema_id": payload_schema_id,
            "payload_schema_version": payload_schema_version,
            "payload": payload_dict,
            "tags": self.tags,
        }
    

@dataclass(frozen=True)
class ToolCallTrace:
    """
    Typed runtime artifact describing a single executed tool call.

    Notes:
    - This is NOT a DiagnosticPayload (not emitted to trace_events directly).
    - It is used to build RuntimeAnswer.tool_calls (API-facing).
    - Keep fields JSON-friendly and stable.
    """
    tool_name: str
    arguments: Dict[str, Any]
    output_preview: Optional[str]
    success: bool
    error_message: Optional[str]
    raw_trace: Dict[str, Any]
