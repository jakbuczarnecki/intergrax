# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Teams outgoing-webhook HMAC verification (opt-in, §18 Phase H.5)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Mapping, Optional

from intergrax.runtime.interactions.verification.contract import InboundRequestVerifier

ENV_TEAMS_SECURITY_TOKEN = "INTERGRAX_TEAMS_SECURITY_TOKEN"
ENV_TEAMS_VERIFY_SIGNATURE = "INTERGRAX_TEAMS_VERIFY_SIGNATURE"


class TeamsSignatureVerifier(InboundRequestVerifier):
    """
    Validates Teams ``Authorization`` HMAC-SHA256 when explicitly enabled.

    Disabled by default — laboratory intake works without a security token.
    """

    def __init__(
        self,
        *,
        security_token: str = "",
        enabled: Optional[bool] = None,
    ) -> None:
        self._security_token = security_token.strip()
        if enabled is None:
            enabled = _env_truthy(ENV_TEAMS_VERIFY_SIGNATURE)
        self._enabled = enabled and bool(self._security_token)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def verify(self, *, headers: Mapping[str, str], body: bytes) -> None:
        if not self._enabled:
            return

        authorization = _header(headers, "Authorization")
        if not authorization:
            raise ValueError("missing Teams Authorization header")

        expected = _compute_teams_hmac(self._security_token, body)
        provided = _normalize_authorization(authorization)
        if not hmac.compare_digest(expected, provided):
            raise ValueError("invalid Teams signature")


def _compute_teams_hmac(security_token: str, body: bytes) -> str:
    digest = hmac.new(
        security_token.encode("utf-8"),
        body,
        hashlib.sha256,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def _normalize_authorization(value: str) -> str:
    normalized = value.strip()
    prefix = "HMACSHA256 "
    if normalized.upper().startswith(prefix):
        return normalized[len(prefix) :].strip()
    return normalized


def _header(headers: Mapping[str, str], name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return str(value)
    return ""


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_teams_security_token(explicit: Optional[str] = None) -> str:
    return (explicit or os.environ.get(ENV_TEAMS_SECURITY_TOKEN, "")).strip()
