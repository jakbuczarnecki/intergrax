# © Artur Czarnecki. All rights reserved.

"""Typed incident investigation scope — application-owned admissibility boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
    COMPARISON_LINE_ID,
    LINE_ID,
    TimeWindowLabel,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_TELEMETRY_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_READ,
)


class IncidentScopeViolationError(ValueError):
    """Tool arguments fall outside the permitted incident investigation scope."""


@dataclass(frozen=True, slots=True)
class IncidentScope:
    """
    Immutable incident scope derived from the typed incident request / fixture.

    Application owns scope; the model may choose tools and bounded arguments within it.
    """

    line_id: str
    station_id: str
    incident_window: str
    comparison_window: str
    comparison_line_id: str
    reference_line_id: str
    shift_id: str

    @classmethod
    def from_operational_defaults(
        cls,
        *,
        station_id: str,
        shift_id: str = "shift_b",
    ) -> IncidentScope:
        return cls(
            line_id=LINE_ID,
            station_id=station_id,
            incident_window=TimeWindowLabel.INCIDENT,
            comparison_window=TimeWindowLabel.COMPARISON,
            comparison_line_id=COMPARISON_LINE_ID,
            reference_line_id=LINE_ID,
            shift_id=shift_id,
        )

    def validate_tool_input(self, tool_id: str, args: Mapping[str, Any]) -> None:
        """Reject model-selected arguments outside permitted incident scope."""
        if tool_id in {TOOL_WORKLOAD_READ, TOOL_THROUGHPUT_READ}:
            self._require_line_window(args, window=self.incident_window)
            return
        if tool_id in {TOOL_STAFFING_SCHEDULE_READ, TOOL_STAFFING_ATTENDANCE_READ}:
            self._require_staffing(args, window=self.incident_window)
            return
        if tool_id == TOOL_COMPARISON_READ:
            self._require_comparison(args)
            return
        if tool_id == TOOL_TELEMETRY_READ:
            self._require_telemetry(args)
            return
        raise IncidentScopeViolationError(f"tool_not_in_incident_scope:{tool_id}")

    def _require_line_window(self, args: Mapping[str, Any], *, window: str) -> None:
        line_id = str(args.get("line_id", ""))
        observed_window = str(args.get("window", ""))
        if line_id != self.line_id:
            raise IncidentScopeViolationError(
                f"line_id_out_of_scope:{line_id}!={self.line_id}"
            )
        if observed_window != window:
            raise IncidentScopeViolationError(
                f"window_out_of_scope:{observed_window}!={window}"
            )

    def _require_staffing(self, args: Mapping[str, Any], *, window: str) -> None:
        line_id = str(args.get("line_id", ""))
        shift_id = str(args.get("shift_id", ""))
        observed_window = str(args.get("window", ""))
        if line_id != self.line_id:
            raise IncidentScopeViolationError(
                f"line_id_out_of_scope:{line_id}!={self.line_id}"
            )
        if shift_id != self.shift_id:
            raise IncidentScopeViolationError(
                f"shift_id_out_of_scope:{shift_id}!={self.shift_id}"
            )
        if observed_window != window:
            raise IncidentScopeViolationError(
                f"window_out_of_scope:{observed_window}!={window}"
            )

    def _require_comparison(self, args: Mapping[str, Any]) -> None:
        reference_line_id = str(args.get("reference_line_id", ""))
        comparison_line_id = str(args.get("comparison_line_id", ""))
        observed_window = str(args.get("window", ""))
        if reference_line_id != self.reference_line_id:
            raise IncidentScopeViolationError(
                "reference_line_id_out_of_scope:"
                f"{reference_line_id}!={self.reference_line_id}"
            )
        if comparison_line_id != self.comparison_line_id:
            raise IncidentScopeViolationError(
                "comparison_line_id_out_of_scope:"
                f"{comparison_line_id}!={self.comparison_line_id}"
            )
        if observed_window != self.comparison_window:
            raise IncidentScopeViolationError(
                f"window_out_of_scope:{observed_window}!={self.comparison_window}"
            )

    def _require_telemetry(self, args: Mapping[str, Any]) -> None:
        station_id = str(args.get("station_id", ""))
        observed_window = str(args.get("window", ""))
        if station_id != self.station_id:
            raise IncidentScopeViolationError(
                f"station_id_out_of_scope:{station_id}!={self.station_id}"
            )
        if observed_window != self.incident_window:
            raise IncidentScopeViolationError(
                f"window_out_of_scope:{observed_window}!={self.incident_window}"
            )
