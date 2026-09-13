# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Plane B diagnostic payload contract (Harness Observability Spine)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

DEFAULT_REDACTED_TEXT: str = "[REDACTED]"


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
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def redact(self) -> DiagnosticPayload:
        raise NotImplementedError
