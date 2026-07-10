# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor-neutral structured proof receipt contract (PROOF-RECEIPTS-1A)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PROOF_RECEIPT_SCHEMA_VERSION = "intergrax.proof_receipt.v1"
PROOF_RECEIPT_PARTITION_PREFIX = "proof_receipts"


class ProofReceiptResult(StrEnum):
    """Normalized proof outcome for structured receipt persistence."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProofReceipt(BaseModel):
    """Structured proof evidence persisted through the platform DocumentStore contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["intergrax.proof_receipt.v1"] = PROOF_RECEIPT_SCHEMA_VERSION
    proof_id: str = Field(..., min_length=1)
    proof_kind: str = Field(..., min_length=1)
    application_id: str = Field(..., min_length=1)
    result: ProofReceiptResult
    recorded_at: datetime = Field(default_factory=_utc_now)
    run_id: str = Field(..., min_length=1)
    correlation_id: str | None = None
    task_id: str | None = None
    provider_evidence: dict[str, Any] = Field(default_factory=dict)
    domain_evidence: dict[str, Any] = Field(default_factory=dict)
    guardrails: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
