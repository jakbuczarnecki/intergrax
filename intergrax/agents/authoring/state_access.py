# © Artur Czarnecki. All rights reserved.

"""Typed session state READ/UPDATE helpers (architecture §32.0 · ACP-DX-6)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from intergrax.contracts.acp_state import ACP_STATE_KEY, AcpSessionState
from intergrax.contracts.agent_step_context import AgentStepContext


def load_session_state(
    step_ctx: AgentStepContext,
    *,
    state_type: type[AcpSessionState] = AcpSessionState,
) -> AcpSessionState:
    """READ — deserialize typed session state from step context."""
    snapshot = step_ctx.state_snapshot or {}
    if ACP_STATE_KEY in snapshot and isinstance(snapshot[ACP_STATE_KEY], dict):
        snapshot = snapshot[ACP_STATE_KEY]
    return state_type.model_validate(snapshot)


def session_state_delta(
    model: BaseModel,
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    exclude_none: bool = True,
) -> dict[str, Any]:
    """
    UPDATE — build merge-patch keys from a typed Pydantic model dump.

    Harness-owned envelope fields are excluded by default.
    """
    default_exclude = {"schema_version", "state_version"}
    merged_exclude = default_exclude | (exclude or set())
    data = model.model_dump(mode="json", include=include, exclude=merged_exclude)
    if exclude_none:
        data = {key: value for key, value in data.items() if value is not None}
    return data
