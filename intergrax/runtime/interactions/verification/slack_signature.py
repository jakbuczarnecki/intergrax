# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack request signature verification (opt-in, §18 Phase H.3)."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Mapping, Optional

from intergrax.runtime.interactions.verification.contract import InboundRequestVerifier

ENV_SLACK_SIGNING_SECRET = "INTERGRAX_SLACK_SIGNING_SECRET"
ENV_SLACK_VERIFY_SIGNATURE = "INTERGRAX_SLACK_VERIFY_SIGNATURE"

DEFAULT_MAX_SKEW_SECONDS = 60 * 5


class SlackSignatureVerifier(InboundRequestVerifier):
    """
    Validates Slack ``X-Slack-Signature`` when explicitly enabled.

    Disabled by default — laboratory intake works without signing secret.
    """

    def __init__(
        self,
        *,
        signing_secret: str = "",
        enabled: Optional[bool] = None,
        max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS,
    ) -> None:
        self._signing_secret = signing_secret.strip()
        if enabled is None:
            enabled = _env_truthy(ENV_SLACK_VERIFY_SIGNATURE)
        self._enabled = enabled and bool(self._signing_secret)
        self._max_skew_seconds = max_skew_seconds

    @property
    def enabled(self) -> bool:
        return self._enabled

    def verify(self, *, headers: Mapping[str, str], body: bytes) -> None:
        if not self._enabled:
            return

        signature = _header(headers, "X-Slack-Signature")
        timestamp = _header(headers, "X-Slack-Request-Timestamp")
        if not signature or not timestamp:
            raise ValueError("missing Slack signature headers")

        try:
            request_ts = int(timestamp)
        except ValueError as exc:
            raise ValueError("invalid Slack request timestamp") from exc

        now = int(time.time())
        if abs(now - request_ts) > self._max_skew_seconds:
            raise ValueError("Slack request timestamp outside allowed skew")

        basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        digest = hmac.new(
            self._signing_secret.encode("utf-8"),
            basestring.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        expected = f"v0={digest}"
        if not hmac.compare_digest(expected, signature):
            raise ValueError("invalid Slack signature")


def _header(headers: Mapping[str, str], name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return str(value)
    return ""


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_slack_signing_secret(explicit: Optional[str] = None) -> str:
    return (explicit or os.environ.get(ENV_SLACK_SIGNING_SECRET, "")).strip()
