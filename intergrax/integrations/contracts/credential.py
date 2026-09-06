# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Safe credential reference and late-resolution contracts (P1.7)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CredentialResolutionMode(StrEnum):
    """How a tenant-connection factory expects credential material at creation time."""

    RESOLVED_MATERIAL = "resolved_material"
    LATE_BOUND = "late_bound"


class CredentialResolutionError(Exception):
    """Base failure for credential resolution (message must not contain secret material)."""


class CredentialNotFoundError(CredentialResolutionError):
    """Configured credential reference could not be resolved."""


class CredentialScopeMismatchError(CredentialResolutionError):
    """Credential reference tenant/context does not match the active resolution scope."""


class CredentialProviderUnavailableError(CredentialResolutionError):
    """Credential backing provider is unavailable."""


class CredentialRef(BaseModel):
    """Immutable safe identity for locating secret material — never the secret itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1, max_length=128)
    secret_path: str = Field(min_length=1, max_length=512)
    version: str | None = Field(default=None, max_length=128)
    tenant_id: str | None = Field(default=None, max_length=128)

    @field_validator("provider_id", "secret_path")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must be a non-empty string")
        return cleaned

    @field_validator("version", "tenant_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @classmethod
    def from_secret_path(
        cls,
        *,
        provider_id: str,
        secret_path: str,
        tenant_id: str | None = None,
        version: str | None = None,
    ) -> CredentialRef:
        """Build a reference from durable configuration fields."""
        return cls(
            provider_id=provider_id,
            secret_path=secret_path,
            tenant_id=tenant_id,
            version=version,
        )

    def identity_fingerprint(self) -> str:
        """Stable semantic identity for profile/revision fingerprinting (not secret value)."""
        tenant = self.tenant_id or ""
        version = self.version or ""
        return f"{self.provider_id}|{tenant}|{self.secret_path}|{version}"

    def config_credential_ref(self) -> str:
        """Legacy configuration surface used by existing integration configs."""
        return self.secret_path


@dataclass(frozen=True, slots=True)
class CredentialResolutionContext:
    """Minimal identity context for scoped credential resolution."""

    tenant_id: str
    application_id: str | None = None
    execution_id: str | None = None
    operation: str | None = None


@dataclass(slots=True)
class ResolvedCredential:
    """Ephemeral resolved secret material — not safe to serialize."""

    ref: CredentialRef
    value: str
    resolved_version: str | None = None

    def __repr__(self) -> str:
        return (
            f"ResolvedCredential(ref={self.ref!r}, "
            f"resolved_version={self.resolved_version!r}, value=<redacted>)"
        )

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True, slots=True)
class CredentialUseEvidence:
    """Safe credential-use evidence — never contains secret material."""

    credential_ref: CredentialRef
    provider_id: str
    execution_id: str | None = None
    operation: str | None = None
    resolved_version: str | None = None


@runtime_checkable
class CredentialResolver(Protocol):
    """Provider-neutral late credential resolution seam."""

    def resolve(
        self,
        ref: CredentialRef,
        *,
        context: CredentialResolutionContext,
    ) -> ResolvedCredential:
        """Resolve secret material immediately before an operation needs it."""


__all__ = [
    "CredentialNotFoundError",
    "CredentialProviderUnavailableError",
    "CredentialRef",
    "CredentialResolutionContext",
    "CredentialResolutionError",
    "CredentialResolutionMode",
    "CredentialResolver",
    "CredentialScopeMismatchError",
    "CredentialUseEvidence",
    "ResolvedCredential",
]
