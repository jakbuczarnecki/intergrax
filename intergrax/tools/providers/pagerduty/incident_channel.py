# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed incident surface for PagerDuty tool handlers."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, runtime_checkable


@runtime_checkable
class PagerDutyIncidentChannel(Protocol):
    def trigger_incident(
        self,
        *,
        summary: str,
        severity: str = "error",
        source: str = "intergrax",
        custom_details: Optional[Mapping[str, Any]] = None,
        dedup_key: Optional[str] = None,
    ) -> str: ...

    def acknowledge_incident(
        self,
        *,
        dedup_key: str,
        note: str | None = None,
    ) -> None: ...
