# © Artur Czarnecki. All rights reserved.

"""Declarative tool action validation (ACP-PROD-2 · ACP-PROD-3)."""

from __future__ import annotations

from typing import Any

from intergrax.agents.persistence.idempotency_keys import build_default_idempotency_key
from intergrax.agents.persistence.idempotency_ledger_bridge import should_skip_side_effect_replay
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.contracts.agent_run_enums import SideEffectMode
from intergrax.contracts.side_effect import SideEffectKind
from intergrax.tools.tool_execution_profile import ToolExecutionProfile, ToolMutability


class ToolActionValidationError(Exception):
    def __init__(self, *, message: str, tool_id: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.tool_id = tool_id
        self.code = code


def validate_requested_actions(
    *,
    requested_actions: list[dict[str, Any]] | None,
    side_effect_mode: SideEffectMode,
    tool_profiles: dict[str, ToolExecutionProfile],
    run_id: str,
    step_index: int,
    ledger: SideEffectLedger | None,
    idempotency_store: IdempotencyStore | None = None,
    tenant_id: str = "default",
) -> list[dict[str, Any]]:
    """
    Validate declarative actions and return normalized actions.

    Raises ``ToolActionValidationError`` when mutating tools lack idempotency keys.
    """
    if not requested_actions:
        return []
    if side_effect_mode != SideEffectMode.DECLARATIVE:
        return list(requested_actions)

    normalized: list[dict[str, Any]] = []
    for action in requested_actions:
        tool_id = action.get("tool_id")
        if not isinstance(tool_id, str) or not tool_id:
            raise ToolActionValidationError(
                message="requested action missing tool_id",
                tool_id="",
                code="acp.tool.missing_id",
            )
        profile = tool_profiles.get(tool_id)
        if profile is None:
            profile = ToolExecutionProfile(tool_id=tool_id)
        idempotency_key = action.get("idempotency_key")
        if profile.mutability == ToolMutability.MUTATING and profile.requires_idempotency_key:
            if not isinstance(idempotency_key, str) or not idempotency_key:
                raise ToolActionValidationError(
                    message=f"mutating tool {tool_id!r} requires idempotency_key",
                    tool_id=tool_id,
                    code="acp.tool.idempotency_required",
                )
            if should_skip_side_effect_replay(
                idempotency_key=idempotency_key,
                ledger=ledger,
                idempotency_store=idempotency_store,
                tenant_id=tenant_id,
            ):
                normalized.append({**action, "replay_skipped": True})
                continue
        elif not idempotency_key and profile.mutability == ToolMutability.MUTATING:
            idempotency_key = build_default_idempotency_key(
                run_id=run_id,
                step_index=step_index,
                kind=SideEffectKind.TOOL,
                target=tool_id,
                args=action.get("args") if isinstance(action.get("args"), dict) else {},
            )
        item = dict(action)
        if idempotency_key:
            item["idempotency_key"] = idempotency_key
        normalized.append(item)
        if ledger is not None and isinstance(idempotency_key, str):
            ledger.register(
                idempotency_key=idempotency_key,
                run_id=run_id,
                step_index=step_index,
                target=tool_id,
            )
    return normalized
