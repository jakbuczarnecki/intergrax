from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.integrations.contracts.base import IntegrationCategory

LIVE_CONTRACT_VERSION = "1"

_SLUG = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*"
_OPERATION = r"(?:search|list|read|thread\.read|child\.read|content\.read)"
_CAPABILITY_RE = re.compile(
    rf"^vendor\.(?P<provider_id>{_SLUG})\.(?P<source_kind>{_SLUG})\.(?P<operation>{_OPERATION})$"
)


class LiveOperationV1(StrEnum):
    SEARCH = "search"
    LIST = "list"
    READ = "read"
    THREAD_READ = "thread.read"
    CHILD_READ = "child.read"
    CONTENT_READ = "content.read"


class CapabilityIdentityV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider_id: str = Field(..., min_length=1, max_length=64)
    integration_kind: IntegrationCategory
    source_kind: str = Field(..., min_length=1, max_length=64)
    capability_id: str = Field(..., min_length=1, max_length=128)
    contract_version: str = Field(..., min_length=1, max_length=32)
    operation: LiveOperationV1

    @field_validator("provider_id", "source_kind", "capability_id", "contract_version")
    @classmethod
    def _trimmed(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned != value or not cleaned:
            raise ValueError("live_identity_string_invalid")
        return cleaned

    @field_validator("contract_version")
    @classmethod
    def _supported_version(cls, value: str) -> str:
        if value != LIVE_CONTRACT_VERSION:
            raise ValueError("live_contract_version_unsupported")
        return value


def parse_capability_id(capability_id: str) -> tuple[str, str, LiveOperationV1]:
    if not isinstance(capability_id, str):
        raise TypeError("live_capability_id_invalid")
    match = _CAPABILITY_RE.fullmatch(capability_id)
    if match is None:
        raise ValueError("live_capability_id_invalid")
    try:
        operation = LiveOperationV1(match.group("operation"))
    except ValueError:
        raise ValueError("live_capability_operation_invalid") from None
    return match.group("provider_id"), match.group("source_kind"), operation


def validate_capability_identity(
    *,
    capability_id: str,
    provider_id: str,
    integration_kind: IntegrationCategory,
    source_kind: str,
    contract_version: str,
) -> CapabilityIdentityV1:
    id_provider, id_source, operation = parse_capability_id(capability_id)
    if provider_id != id_provider:
        raise ValueError("live_capability_provider_mismatch")
    if source_kind != id_source:
        raise ValueError("live_capability_source_kind_mismatch")
    if contract_version != LIVE_CONTRACT_VERSION:
        raise ValueError("live_contract_version_unsupported")
    return CapabilityIdentityV1(
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        capability_id=capability_id,
        contract_version=contract_version,
        operation=operation,
    )


def exact_capability_key(identity: CapabilityIdentityV1) -> tuple[str, IntegrationCategory, str, str]:
    return (
        identity.provider_id,
        identity.integration_kind,
        identity.capability_id,
        identity.contract_version,
    )
