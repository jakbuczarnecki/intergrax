# © Artur Czarnecki. All rights reserved.

"""Verification result for execution-evidence receipts — never authorizes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VerificationResult(BaseModel):
    """Offline verification outcome — no authorization side effects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    schema_valid: bool
    digest_valid: bool
    signature_valid: bool
    key_id: str = ""
    errors: tuple[str, ...] = Field(default_factory=tuple)
