# © Artur Czarnecki. All rights reserved.

"""Encryption enforcement middleware at memory and tool boundaries (Phase ENC-2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.base import RuntimeMiddleware
from intergrax.runtime.security.encryption_policy import (
    _classification_from_payload,
    evaluate_encryption_enforcement,
)
from intergrax.runtime.security.encryption_transform import RestrictedPayloadEncryptor
from intergrax.runtime.security.security_events import emit_encryption_denied

if TYPE_CHECKING:
    from intergrax.runtime.events.event_bus import RuntimeEventBus


class EncryptionEnforcementMiddleware(RuntimeMiddleware):
    """Enforce RESTRICTED classification at memory write and tool output boundaries."""

    priority = 48
    name = "EncryptionEnforcementMiddleware"

    def __init__(
        self,
        *,
        enforcement_enabled: bool,
        secrets_store_configured: bool,
        encryptor: RestrictedPayloadEncryptor | None = None,
        event_bus: RuntimeEventBus | None = None,
    ) -> None:
        self._enforcement_enabled = enforcement_enabled
        self._secrets_store_configured = secrets_store_configured
        self._encryptor = encryptor
        self._event_bus = event_bus

    async def before(self, point: HookPoint, ctx: HookContext) -> HookResult:
        if point not in {
            HookPoint.BEFORE_MEMORY_WRITE,
            HookPoint.AFTER_TOOL_CALL,
        }:
            return HookResult()
        payload = dict(ctx.runtime_state)
        decision = evaluate_encryption_enforcement(
            payload=payload,
            secrets_store_configured=self._secrets_store_configured,
            enforcement_enabled=self._enforcement_enabled,
        )
        if not decision.allowed:
            reason = "; ".join(decision.reasons) or "encryption enforcement blocked payload"
            classification = _classification_from_payload(payload)
            await emit_encryption_denied(
                self._event_bus,
                ctx=ctx,
                point=point,
                reason=reason,
                classification=classification.value if classification is not None else None,
            )
            return HookResult(action=HookAction.BLOCK, reason=reason)
        if (
            self._secrets_store_configured
            and self._encryptor is not None
            and self._enforcement_enabled
        ):
            classification = _classification_from_payload(payload)
            if classification is not None and classification.requires_encryption():
                encrypted = self._encryptor.encrypt_payload(payload, run_id=ctx.run_id)
                if encrypted != payload:
                    return HookResult(action=HookAction.MODIFY, modified_payload=encrypted)
        return HookResult()

    async def after(self, point: HookPoint, ctx: HookContext) -> HookResult:
        return HookResult()
