# © Artur Czarnecki. All rights reserved.

"""Provider capability manifest for External Work (PC-9).

Provider-neutral feature flags — not a partner adapter and not a transport
discovery document.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_EXTERNAL_WORK_PROVIDER_CAPABILITIES_V1: Final = (
    "external_work_provider_capabilities.v1"
)
_NON_EMPTY = Field(min_length=1)


class ExternalWorkProviderCapabilities(BaseModel):
    """Declared provider features for orchestrator fail-closed / fallback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external_work_provider_capabilities.v1"] = (
        SCHEMA_EXTERNAL_WORK_PROVIDER_CAPABILITIES_V1
    )
    provider_id: str = _NON_EMPTY
    supports_create: bool = False
    supports_quote: bool = False
    supports_accept: bool = False
    supports_cancel: bool = False
    supports_status_polling: bool = False
    supports_idempotency: bool = False
    supports_native_provider_request_id: bool = False
    supports_tool_logs: bool = False
    supports_receipt_timeline: bool = False
    supports_payment_state: bool = False
    supports_human_wait_state: bool = False

    @field_validator("provider_id")
    @classmethod
    def _strip_provider_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("provider_id must be non-empty")
        return normalized


def quote_first_partner_capability_fixture(
    *,
    provider_id: str = "quote_first_partner_profile",
    supports_cancel: bool = True,
    supports_native_provider_request_id: bool = True,
) -> ExternalWorkProviderCapabilities:
    """Fixture matching a quote-first contractor profile (partner-shaped, unnamed)."""
    return ExternalWorkProviderCapabilities(
        provider_id=provider_id,
        supports_create=True,
        supports_quote=True,
        supports_accept=True,
        supports_cancel=supports_cancel,
        supports_status_polling=True,
        supports_idempotency=True,
        supports_native_provider_request_id=supports_native_provider_request_id,
        supports_tool_logs=True,
        supports_receipt_timeline=True,
        supports_payment_state=True,
        supports_human_wait_state=True,
    )
