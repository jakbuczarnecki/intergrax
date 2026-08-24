# © Artur Czarnecki. All rights reserved.

"""Test-only scripted native tool planner LLM for incident investigation (APP-2A)."""

from __future__ import annotations

import json
from collections.abc import Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from platform_proofs.scenarios.ai_incident_investigation.fixtures import LINE_ID, STATION_ID, TimeWindowLabel
from platform_proofs.scenarios.ai_incident_investigation.tools import (
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_TELEMETRY_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_READ,
)
from testing_support.builder import FakeLLMAdapter

_DEFAULT_INITIAL_SEQUENCE: tuple[str, ...] = (
    TOOL_WORKLOAD_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_STAFFING_SCHEDULE_READ,
)
_DEFAULT_REVISION_SEQUENCE: tuple[str, ...] = (
    TOOL_WORKLOAD_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_TELEMETRY_READ,
)


def _tool_args(tool_id: str, *, station_id: str = STATION_ID) -> dict[str, str]:
    if tool_id in {TOOL_WORKLOAD_READ, TOOL_THROUGHPUT_READ}:
        return {"line_id": LINE_ID, "window": TimeWindowLabel.INCIDENT}
    if tool_id in {TOOL_STAFFING_SCHEDULE_READ, TOOL_STAFFING_ATTENDANCE_READ}:
        return {"line_id": LINE_ID, "shift_id": "shift_b", "window": TimeWindowLabel.INCIDENT}
    if tool_id == TOOL_COMPARISON_READ:
        return {
            "reference_line_id": LINE_ID,
            "comparison_line_id": "line3",
            "window": TimeWindowLabel.COMPARISON,
        }
    if tool_id == TOOL_TELEMETRY_READ:
        return {"station_id": station_id, "window": TimeWindowLabel.INCIDENT}
    raise ValueError(f"unsupported tool id: {tool_id}")


def _decision_note(*basis_ids: str, purpose: str) -> str:
    basis = ",".join(basis_ids)
    return f"EVIDENCE_BASIS: {basis}\nPURPOSE: {purpose}"


class ScriptedIncidentInvestigationLLM(FakeLLMAdapter):
    """Native-tools LLM double that materializes planner-selected tool order for tests."""

    def __init__(
        self,
        *,
        initial_sequence: Sequence[str] | None = None,
        revision_sequence: Sequence[str] | None = None,
        station_id: str = STATION_ID,
    ) -> None:
        super().__init__(fixed_text="")
        self._initial_sequence = tuple(initial_sequence or _DEFAULT_INITIAL_SEQUENCE)
        self._revision_sequence = tuple(revision_sequence or _DEFAULT_REVISION_SEQUENCE)
        self._station_id = station_id
        self._round_by_phase: dict[str, int] = {"initial": 0, "revision": 0}
        self._prior_tool_call_ids: list[str] = []

    def supports_tools(self) -> bool:
        return True

    def _detect_phase(self, messages: Sequence[ChatMessage]) -> str:
        for message in messages:
            if message.role == "system" and "Investigation phase: revision" in (message.content or ""):
                return "revision"
        return "initial"

    def _active_sequence(self, phase: str) -> tuple[str, ...]:
        return self._revision_sequence if phase == "revision" else self._initial_sequence

    def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
        _ = tools_schema, kwargs
        phase = self._detect_phase(messages)
        has_tool_messages = any(message.role == "tool" for message in messages)
        if not has_tool_messages:
            self._prior_tool_call_ids = []
            self._round_by_phase[phase] = 0
        sequence = self._active_sequence(phase)
        round_index = self._round_by_phase[phase]
        if round_index >= len(sequence):
            return LLMAdapterResponse(content="evidence gathering complete", tool_calls=())

        tool_id = sequence[round_index]
        call_id = f"tc-{phase}-{round_index + 1}"
        self._round_by_phase[phase] = round_index + 1
        purpose = f"gather {tool_id.split('.')[-1]} evidence for incident investigation"
        if round_index == 0:
            content = _decision_note(purpose=purpose)
        else:
            basis = self._prior_tool_call_ids[-1]
            content = _decision_note(basis, purpose=purpose)
        self._prior_tool_call_ids.append(call_id)
        return LLMAdapterResponse(
            content=content,
            tool_calls=(
                LLMToolCall.from_openai_shape(
                    call_id=call_id,
                    name=tool_id,
                    arguments=_tool_args(tool_id, station_id=self._station_id),
                ),
            ),
        )
