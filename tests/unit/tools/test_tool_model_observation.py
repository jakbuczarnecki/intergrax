# © Artur Czarnecki. All rights reserved.

"""ToolModelObservation contract and model-visible formatting."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionResult, ToolModelObservation
from intergrax.tools.model_observation_format import (
    EVIDENCE_REF_PREFIX,
    format_tool_model_observation_content,
    parse_evidence_reference_from_tool_content,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _Out(BaseModel):
    value: int = 1


def test_from_execution_result_defaults_evidence_reference_to_none() -> None:
    observation = ToolModelObservation.from_execution_result(ToolExecutionResult.ok(_Out()))
    assert observation.evidence_reference is None
    assert json.loads(observation.content) == {"value": 1}


def test_from_execution_result_accepts_explicit_evidence_reference() -> None:
    observation = ToolModelObservation.from_execution_result(
        ToolExecutionResult.ok(_Out()),
        evidence_reference="evidence.workload.line4.incident_window",
    )
    assert observation.evidence_reference == "evidence.workload.line4.incident_window"
    assert "evidence_reference" not in json.loads(observation.content)


def test_format_tool_model_observation_content_envelope() -> None:
    observation = ToolModelObservation(
        content='{"value":1}',
        evidence_reference="evidence.workload.line4.incident_window",
    )
    rendered = format_tool_model_observation_content(observation)
    assert rendered.startswith(f"{EVIDENCE_REF_PREFIX}evidence.workload.line4.incident_window\n")
    assert rendered.endswith('{"value":1}')


def test_parse_evidence_reference_from_tool_content_round_trip() -> None:
    observation = ToolModelObservation(
        content='{"value":1}',
        evidence_reference="observation.probe.a.step-1",
    )
    rendered = format_tool_model_observation_content(observation)
    assert parse_evidence_reference_from_tool_content(rendered) == "observation.probe.a.step-1"
