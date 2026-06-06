# © Artur Czarnecki. All rights reserved.

"""Optional Tier-3 business outcome webhook contract (Phase W-ADAPT-7.2)."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from datetime import UTC, datetime
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field


class BusinessOutcomeWebhookConfig(BaseModel):
    """Adaptive harness business outcome webhook settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    signing_secret_env_var: str = "INTERGRAX_BUSINESS_OUTCOME_WEBHOOK_SECRET"
    signature_header: str = "X-Intergrax-Business-Outcome-Signature"
    timestamp_header: str = "X-Intergrax-Business-Outcome-Timestamp"
    max_skew_seconds: int = Field(default=300, ge=1)


class BusinessOutcomeWebhookPayload(BaseModel):
    """Signed payload for optional Tier-3 business outcome signal ingestion."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    tenant_id: str
    business_outcome: float = Field(ge=-1.0, le=1.0)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BusinessOutcomeWebhookVerifier:
    """HMAC-SHA256 verifier for business outcome webhook payloads."""

    def __init__(self, *, config: BusinessOutcomeWebhookConfig) -> None:
        self._config = config
        self._secret = os.environ.get(config.signing_secret_env_var, "").strip()

    @property
    def enabled(self) -> bool:
        return self._config.enabled and bool(self._secret)

    def verify(self, *, headers: Mapping[str, str], body: bytes) -> None:
        if not self.enabled:
            return
        signature = _header(headers, self._config.signature_header)
        timestamp = _header(headers, self._config.timestamp_header)
        if not signature or not timestamp:
            raise ValueError("missing business outcome webhook signature headers")
        try:
            request_ts = int(timestamp)
        except ValueError as exc:
            raise ValueError("invalid business outcome webhook timestamp") from exc
        now = int(time.time())
        if abs(now - request_ts) > self._config.max_skew_seconds:
            raise ValueError("business outcome webhook timestamp outside allowed skew")
        basestring = f"{timestamp}:{body.decode('utf-8')}"
        digest = hmac.new(
            self._secret.encode("utf-8"),
            basestring.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        expected = f"sha256={digest}"
        if not hmac.compare_digest(expected, signature):
            raise ValueError("invalid business outcome webhook signature")


def _header(headers: Mapping[str, str], name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return str(value)
    return ""
