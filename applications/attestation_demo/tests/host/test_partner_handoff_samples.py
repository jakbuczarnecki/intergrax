# © Artur Czarnecki. All rights reserved.

"""Committed partner handoff JSON samples must stay structurally valid."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attestation_demo.partner_handoff.contract_assertions import (
    assert_partner_failed_tool_dual_claims,
    assert_partner_poc_response_shape,
)

pytestmark = pytest.mark.unit

_HANDOFF_DIR = Path(__file__).resolve().parents[2] / "partner_handoff"


def test_partner_handoff_v2_success_sample_shape() -> None:
    payload = json.loads((_HANDOFF_DIR / "poc_run_response.v2.json").read_text(encoding="utf-8"))
    # Samples use placeholders — validate structure with synthetic ids for assertions
    payload["run_id"] = "run_sample"
    for event in payload["boundary_events"]:
        event["run_id"] = "run_sample"
        event["event_id"] = f"evt-{event['event_sequence']}"
        ref = event.get("lineage", {}).get("ref", "")
        event["lineage"]["ref"] = ref.replace("run-<dynamic>", "run_sample")
    assert_partner_poc_response_shape(payload)


def test_partner_handoff_v2_failed_sample_dual_claims() -> None:
    payload = json.loads(
        (_HANDOFF_DIR / "poc_run_response.failed.v2.json").read_text(encoding="utf-8"),
    )
    payload["run_id"] = "run_sample_fail"
    for event in payload["boundary_events"]:
        event["run_id"] = "run_sample_fail"
        event["event_id"] = f"evt-fail-{event['event_sequence']}"
        ref = event.get("lineage", {}).get("ref", "")
        event["lineage"]["ref"] = ref.replace("run-<dynamic>", "run_sample_fail")
    assert_partner_failed_tool_dual_claims(
        payload["boundary_events"],
        run_id="run_sample_fail",
        host_signed=False,
    )
