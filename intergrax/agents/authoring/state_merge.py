# © Artur Czarnecki. All rights reserved.

"""RFC 7396 state_delta merge for acp.state.v1 (architecture §37.2 · ACP-CON-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.contracts.acp_state import ACP_STATE_KEY
from intergrax.contracts.agent_run_enums import AgentRunErrorCode

StateBlob = dict[str, Any]


@dataclass(frozen=True, slots=True)
class StateMergeResult:
    """Outcome of applying a merge patch to session state."""

    state: StateBlob
    error_code: AgentRunErrorCode | None = None
    error_message: str | None = None


def extract_acp_state_blob(state_root: StateBlob | None) -> StateBlob:
    """Return the acp.state.v1 object from a run state root."""
    if not state_root:
        return {"schema_version": "acp.state.v1", "_version": 0}
    nested = state_root.get(ACP_STATE_KEY)
    if isinstance(nested, dict):
        return dict(nested)
    if "schema_version" in state_root:
        return dict(state_root)
    return {"schema_version": "acp.state.v1", "_version": 0}


def wrap_acp_state_blob(acp_state: StateBlob) -> StateBlob:
    """Embed acp.state.v1 inside a run-level state root."""
    return {ACP_STATE_KEY: dict(acp_state)}


def apply_state_delta(current: StateBlob, delta: StateBlob) -> StateBlob:
    """Shallow JSON Merge Patch (RFC 7396) without version bump."""
    result = dict(current)
    for key, value in delta.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def bump_state_version(state: StateBlob) -> StateBlob:
    next_version = int(state.get("_version", 0)) + 1
    merged = dict(state)
    merged["_version"] = next_version
    return merged


def merge_session_state(
    current_root: StateBlob | None,
    delta: StateBlob,
    *,
    incoming_version: int | None = None,
    force_resume: bool = False,
) -> StateMergeResult:
    """
    Apply delta to acp.state.v1 and return updated run-level state root.

    When ``incoming_version`` is lower than the checkpoint version, returns
  ``VALIDATION_FAILED`` unless ``force_resume`` is set.
    """
    current = extract_acp_state_blob(current_root)
    checkpoint_version = int(current.get("_version", 0))
    if (
        incoming_version is not None
        and incoming_version < checkpoint_version
        and not force_resume
    ):
        return StateMergeResult(
            state=wrap_acp_state_blob(current),
            error_code=AgentRunErrorCode.VALIDATION_FAILED,
            error_message=(
                f"incoming state _version {incoming_version} < checkpoint {checkpoint_version}"
            ),
        )
    merged = bump_state_version(apply_state_delta(current, delta))
    return StateMergeResult(state=wrap_acp_state_blob(merged))
