# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-7.2: business outcome webhook contract tests."""

from __future__ import annotations

import hashlib
import hmac
import time

import pytest

from intergrax.applications.contracts.business_outcome_webhook import (
    BusinessOutcomeWebhookConfig,
    BusinessOutcomeWebhookPayload,
    BusinessOutcomeWebhookVerifier,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_business_outcome_webhook_verifier_accepts_valid_signature(monkeypatch) -> None:
    secret = "test-secret"
    monkeypatch.setenv("INTERGRAX_BUSINESS_OUTCOME_WEBHOOK_SECRET", secret)
    config = BusinessOutcomeWebhookConfig(enabled=True)
    verifier = BusinessOutcomeWebhookVerifier(config=config)
    payload = BusinessOutcomeWebhookPayload(
        run_id="run-1",
        tenant_id="tenant-a",
        business_outcome=0.8,
    )
    body = payload.model_dump_json().encode("utf-8")
    timestamp = str(int(time.time()))
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}:{body.decode('utf-8')}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    headers = {
        "X-Intergrax-Business-Outcome-Timestamp": timestamp,
        "X-Intergrax-Business-Outcome-Signature": f"sha256={digest}",
    }
    verifier.verify(headers=headers, body=body)


def test_business_outcome_webhook_verifier_rejects_invalid_signature(monkeypatch) -> None:
    monkeypatch.setenv("INTERGRAX_BUSINESS_OUTCOME_WEBHOOK_SECRET", "test-secret")
    config = BusinessOutcomeWebhookConfig(enabled=True)
    verifier = BusinessOutcomeWebhookVerifier(config=config)
    payload = BusinessOutcomeWebhookPayload(run_id="run-1", tenant_id="tenant-a", business_outcome=0.5)
    with pytest.raises(ValueError, match="invalid business outcome webhook signature"):
        verifier.verify(
            headers={
                "X-Intergrax-Business-Outcome-Timestamp": str(int(time.time())),
                "X-Intergrax-Business-Outcome-Signature": "sha256=deadbeef",
            },
            body=payload.model_dump_json().encode("utf-8"),
        )
