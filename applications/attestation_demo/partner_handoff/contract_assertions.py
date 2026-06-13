# © Artur Czarnecki. All rights reserved.

"""Partner PoC contract assertions (shared by tests and handoff verification)."""

from __future__ import annotations

import json
from typing import Any

from intergrax.runtime.attestation.execution_boundary_event import ExecutionBoundaryEventV1

PARTNER_REQUIRED_EVENT_FIELDS = frozenset(
    {
        "schema_id",
        "signed",
        "tool_id",
        "agent_id",
        "run_id",
        "step_id",
        "action_status",
        "input",
        "output",
        "input_hash",
        "output_hash",
        "occurred_at",
        "lineage",
        "boundary_type",
        "side_effects",
    }
)

PARTNER_REQUIRED_RESPONSE_FIELDS = frozenset(
    {
        "task_id",
        "run_id",
        "state",
        "agent_id",
        "boundary_events",
        "trust_model",
    }
)


def assert_partner_poc_response_shape(payload: dict[str, Any]) -> None:
    missing = PARTNER_REQUIRED_RESPONSE_FIELDS - set(payload.keys())
    assert not missing, f"missing response fields: {sorted(missing)}"
    trust = payload.get("trust_model") or {}
    assert trust.get("platform_signed") == "false"
    assert trust.get("recommended_receipt_role") == "client_observed"
    assert "server_attested" not in json.dumps(trust).lower()
    events = payload.get("boundary_events") or []
    assert events, "boundary_events must be non-empty"
    assert_partner_boundary_event(events[0], run_id=str(payload.get("run_id") or ""))


def assert_partner_boundary_event(event: dict[str, Any], *, run_id: str = "") -> ExecutionBoundaryEventV1:
    missing = PARTNER_REQUIRED_EVENT_FIELDS - set(event.keys())
    assert not missing, f"missing event fields: {sorted(missing)}"
    assert event.get("schema_id") == "execution_boundary_event.v1"
    assert event.get("signed") is False
    assert event.get("tool_id") == "records.put"
    assert event.get("agent_id") == "boundary_demo_agent"
    assert event.get("boundary_type") == "tool_execution"
    assert event.get("action_status") in {"executed", "failed"}
    assert str(event.get("input_hash", "")).startswith("sha256:")
    if event.get("action_status") == "failed":
        assert event.get("error_message")
    else:
        assert str(event.get("output_hash", "")).startswith("sha256:")
    if run_id:
        assert event.get("run_id") == run_id
        assert run_id in str((event.get("lineage") or {}).get("ref", ""))
    return ExecutionBoundaryEventV1.model_validate(event)
