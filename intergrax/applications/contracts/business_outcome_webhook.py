# © Artur Czarnecki. All rights reserved.

"""Optional Tier-3 business outcome webhook DTOs (Phase W-ADAPT-7.2)."""

from __future__ import annotations

from datetime import UTC, datetime

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
