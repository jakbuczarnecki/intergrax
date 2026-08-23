# © Artur Czarnecki. All rights reserved.

"""Scenario production tools — registered on platform ToolRegistry; fixture behind provider."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler
from platform_proofs.scenarios.ai_incident_investigation.fixtures import (
    IncidentFixture,
    LINE_ID,
)
from testing_support.builder import tools_agent_make_contract

TOOL_WORKLOAD_READ = "production.workload.read"
TOOL_THROUGHPUT_READ = "production.throughput.read"
TOOL_TELEMETRY_READ = "production.telemetry.read"

SCENARIO_TOOL_IDS: tuple[str, ...] = (
    TOOL_WORKLOAD_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_TELEMETRY_READ,
)


class LineWindowInput(BaseModel):
    line_id: str = Field(min_length=1)
    window: str = Field(min_length=1)


class WorkloadOutput(BaseModel):
    line_id: str
    window: str
    order_volume_delta_pct: float
    admissible: bool


class ThroughputOutput(BaseModel):
    line_id: str
    window: str
    target_attainment_pct: float
    baseline_attainment_pct: float
    admissible: bool


class TelemetryInput(BaseModel):
    station_id: str = Field(min_length=1)
    window: str = Field(min_length=1)


class TelemetryOutput(BaseModel):
    station_id: str
    window: str
    signal_state: str
    complex_assembly_throughput_pct: float
    admissible: bool


class _WorkloadHandler(ToolHandler[LineWindowInput, WorkloadOutput]):
    def __init__(self, fixture: IncidentFixture) -> None:
        self._fixture = fixture

    def execute(self, request: ToolExecutionRequest[LineWindowInput]) -> WorkloadOutput:
        obs = self._fixture.workload
        if request.input.line_id != obs.line_id:
            return WorkloadOutput(
                line_id=request.input.line_id,
                window=request.input.window,
                order_volume_delta_pct=0.0,
                admissible=False,
            )
        return WorkloadOutput(
            line_id=obs.line_id,
            window=obs.window_label,
            order_volume_delta_pct=obs.order_volume_delta_pct,
            admissible=obs.admissible,
        )


class _ThroughputHandler(ToolHandler[LineWindowInput, ThroughputOutput]):
    def __init__(self, fixture: IncidentFixture) -> None:
        self._fixture = fixture

    def execute(self, request: ToolExecutionRequest[LineWindowInput]) -> ThroughputOutput:
        obs = self._fixture.throughput
        if request.input.line_id != obs.line_id:
            return ThroughputOutput(
                line_id=request.input.line_id,
                window=request.input.window,
                target_attainment_pct=0.0,
                baseline_attainment_pct=0.0,
                admissible=False,
            )
        return ThroughputOutput(
            line_id=obs.line_id,
            window=obs.window_label,
            target_attainment_pct=obs.target_attainment_pct,
            baseline_attainment_pct=obs.baseline_attainment_pct,
            admissible=obs.admissible,
        )


class _TelemetryHandler(ToolHandler[TelemetryInput, TelemetryOutput]):
    def __init__(self, fixture: IncidentFixture) -> None:
        self._fixture = fixture

    def execute(self, request: ToolExecutionRequest[TelemetryInput]) -> TelemetryOutput:
        obs = self._fixture.telemetry
        if request.input.station_id != obs.station_id:
            return TelemetryOutput(
                station_id=request.input.station_id,
                window=request.input.window,
                signal_state="no_data",
                complex_assembly_throughput_pct=0.0,
                admissible=False,
            )
        return TelemetryOutput(
            station_id=obs.station_id,
            window=obs.window_label,
            signal_state=obs.signal_state,
            complex_assembly_throughput_pct=obs.complex_assembly_throughput_pct,
            admissible=obs.admissible,
        )


def register_scenario_tools(registry: ToolRegistry, fixture: IncidentFixture) -> None:
    registry.register(
        tools_agent_make_contract(TOOL_WORKLOAD_READ, LineWindowInput, WorkloadOutput),
        _WorkloadHandler(fixture),
    )
    registry.register(
        tools_agent_make_contract(TOOL_THROUGHPUT_READ, LineWindowInput, ThroughputOutput),
        _ThroughputHandler(fixture),
    )
    registry.register(
        tools_agent_make_contract(TOOL_TELEMETRY_READ, TelemetryInput, TelemetryOutput),
        _TelemetryHandler(fixture),
    )


def default_line_window_input() -> dict[str, str]:
    return {"line_id": LINE_ID, "window": "incident_window"}


def default_telemetry_input(station_id: str) -> dict[str, str]:
    return {"station_id": station_id, "window": "incident_window"}
