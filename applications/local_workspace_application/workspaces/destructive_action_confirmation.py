# © Artur Czarnecki. All rights reserved.

"""Generic HMAC destructive-action confirmation (ARCH-1 PRODUCT-4B)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeOperationV1,
)
from pydantic import BaseModel, ConfigDict, Field


class DestructiveActionKindV1:
    WORKSPACE_DELETE = "workspace.delete"
    KNOWLEDGE_DETACH = "knowledge.detach"


class DestructiveActionConfirmationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token: str
    tenant_id: str
    workspace_id: str
    action_kind: str = Field(min_length=1, max_length=128)
    target_id: str = Field(min_length=1, max_length=256)
    expected_state_version: int = Field(ge=0)
    expires_at: datetime


class DestructiveActionConfirmationError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _sign_payload(secret: bytes, payload: dict[str, object]) -> str:
    encoded_payload = _urlsafe_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(
        secret,
        encoded_payload.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{encoded_payload}.{_urlsafe_encode(signature)}"


def _verify_signed_payload(
    secret: bytes,
    token: str,
    *,
    expected_version: int,
) -> dict[str, object]:
    try:
        encoded_payload, encoded_signature = token.split(".", maxsplit=1)
        expected_signature = hmac.new(
            secret,
            encoded_payload.encode("ascii"),
            hashlib.sha256,
        ).digest()
        actual_signature = _urlsafe_decode(encoded_signature)
        if not hmac.compare_digest(actual_signature, expected_signature):
            raise DestructiveActionConfirmationError("destructive_confirmation_invalid")
        payload = json.loads(_urlsafe_decode(encoded_payload))
        if payload.get("v") != expected_version:
            raise DestructiveActionConfirmationError("destructive_confirmation_invalid")
        return payload
    except DestructiveActionConfirmationError:
        raise
    except (
        binascii.Error,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise DestructiveActionConfirmationError("destructive_confirmation_invalid") from exc


class HmacDestructiveActionConfirmationCodec:
    """Signed, stateless confirmation codec for destructive conversational actions."""

    _VERSION = 2

    def __init__(
        self,
        *,
        secret: bytes,
        ttl: timedelta = timedelta(minutes=5),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not secret:
            raise ValueError("confirmation secret must not be empty")
        if ttl <= timedelta(0):
            raise ValueError("confirmation ttl must be positive")
        self._secret = secret
        self._ttl = ttl
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def ttl(self) -> timedelta:
        return self._ttl

    def issue(self, confirmation: DestructiveActionConfirmationV1) -> str:
        payload = {
            "v": self._VERSION,
            "tenant_id": confirmation.tenant_id,
            "workspace_id": confirmation.workspace_id,
            "action_kind": confirmation.action_kind,
            "target_id": confirmation.target_id,
            "expected_state_version": confirmation.expected_state_version,
            "expires_at": int(confirmation.expires_at.timestamp()),
        }
        return _sign_payload(self._secret, payload)

    def verify(self, token: str) -> DestructiveActionConfirmationV1:
        payload = _verify_signed_payload(
            self._secret,
            token,
            expected_version=self._VERSION,
        )
        expires_at = datetime.fromtimestamp(int(payload["expires_at"]), tz=UTC)
        if expires_at <= self._clock():
            raise DestructiveActionConfirmationError("destructive_confirmation_expired")
        return DestructiveActionConfirmationV1(
            token=token,
            tenant_id=str(payload["tenant_id"]),
            workspace_id=str(payload["workspace_id"]),
            action_kind=str(payload["action_kind"]),
            target_id=str(payload["target_id"]),
            expected_state_version=int(payload["expected_state_version"]),
            expires_at=expires_at,
        )


def knowledge_operation_action_kind(operation: KnowledgeOperationV1) -> str:
    return f"knowledge.{operation.value}"


def knowledge_detach_action_kind() -> str:
    return knowledge_operation_action_kind(KnowledgeOperationV1.DETACH)


__all__ = [
    "DestructiveActionConfirmationError",
    "DestructiveActionConfirmationV1",
    "DestructiveActionKindV1",
    "HmacDestructiveActionConfirmationCodec",
    "knowledge_detach_action_kind",
    "knowledge_operation_action_kind",
]
