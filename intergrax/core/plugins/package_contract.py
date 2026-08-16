# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Package-level Platform Plugin coordination contract (PLATFORM-PLUGIN-3)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.core.distribution import DistributionPackageIdentity, PlatformCompatibility
from intergrax.core.plugins.errors import PlatformPluginManifestValidationError
from intergrax.core.security import (
    FORBIDDEN_KEY,
    SecretSafetyValidationError,
    SecretSafeValidationPolicy,
    validate_secret_safe_value,
)

MANIFEST_SCHEMA_VERSION: Literal[1] = 1

PLATFORM_PLUGIN_MANIFEST_SECRET_POLICY = SecretSafeValidationPolicy(
    forbidden_key_fragments=frozenset(
        {
            "api_key",
            "apikey",
            "password",
            "passwd",
            "secret",
            "token",
            "credential",
            "credentials",
            "connection_string",
            "private_key",
            "access_key",
            "client_secret",
        }
    ),
    scan_string_values=False,
    split_key_segments=False,
    traverse_sequences=True,
)


def _require_non_empty_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _normalize_capability_ids(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        normalized = _require_non_empty_text(value, field_name="capability_ids")
        return (normalized,)
    if isinstance(value, (list, tuple)):
        normalized_ids: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"capability_ids[{index}] must be a string")
            normalized_ids.append(_require_non_empty_text(item, field_name="capability_ids"))
        return tuple(normalized_ids)
    raise ValueError("capability_ids must be a string or list of strings")


def _capability_identity(
    *,
    domain: str,
    entry_point_group: str,
    entry_point_name: str,
) -> tuple[str, str, str]:
    return (domain, entry_point_group, entry_point_name)


def validate_platform_plugin_manifest_secrets(payload: object, *, path: str = "") -> None:
    """Reject secret-like Platform Plugin manifest keys using the domain policy."""
    try:
        validate_secret_safe_value(
            payload,
            policy=PLATFORM_PLUGIN_MANIFEST_SECRET_POLICY,
            path=path,
            context_label="manifest",
        )
    except SecretSafetyValidationError as exc:
        if exc.reason_code == FORBIDDEN_KEY:
            location = exc.path or "manifest"
            raise PlatformPluginManifestValidationError(
                f"secret-like manifest field is not allowed: {location}"
            ) from exc
        raise PlatformPluginManifestValidationError(str(exc)) from exc


class CapabilityDescriptor(BaseModel):
    """Package-level pointer to one domain capability (not a domain manifest)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: str
    entry_point_group: str
    entry_point_name: str
    capability_ids: tuple[str, ...] = ()

    @field_validator("domain")
    @classmethod
    def _validate_domain(cls, value: str) -> str:
        return _require_non_empty_text(value, field_name="capability domain")

    @field_validator("entry_point_group")
    @classmethod
    def _validate_entry_point_group(cls, value: str) -> str:
        return _require_non_empty_text(value, field_name="entry_point_group")

    @field_validator("entry_point_name")
    @classmethod
    def _validate_entry_point_name(cls, value: str) -> str:
        return _require_non_empty_text(value, field_name="entry_point_name")

    @field_validator("capability_ids", mode="before")
    @classmethod
    def _validate_capability_ids(cls, value: object) -> tuple[str, ...]:
        return _normalize_capability_ids(value)

    @property
    def identity_key(self) -> tuple[str, str, str]:
        return _capability_identity(
            domain=self.domain,
            entry_point_group=self.entry_point_group,
            entry_point_name=self.entry_point_name,
        )


class PlatformPluginManifest(BaseModel):
    """Optional package-level Platform Plugin coordination manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    package: DistributionPackageIdentity
    platform_compatibility: PlatformCompatibility
    capabilities: tuple[CapabilityDescriptor, ...] = ()
    author: str | None = None
    documentation_uri: str | None = None
    labels: tuple[str, ...] = ()

    @field_validator("author")
    @classmethod
    def _validate_author(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_empty_text(value, field_name="author")

    @field_validator("documentation_uri")
    @classmethod
    def _validate_documentation_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_empty_text(value, field_name="documentation_uri")

    @field_validator("labels", mode="before")
    @classmethod
    def _validate_labels(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (_require_non_empty_text(value, field_name="labels"),)
        if isinstance(value, (list, tuple)):
            normalized: list[str] = []
            for index, item in enumerate(value):
                if not isinstance(item, str):
                    raise ValueError(f"labels[{index}] must be a string")
                normalized.append(_require_non_empty_text(item, field_name="labels"))
            return tuple(normalized)
        raise ValueError("labels must be a string or list of strings")

    @model_validator(mode="after")
    def _reject_duplicate_capabilities(self) -> Self:
        seen: set[tuple[str, str, str]] = set()
        for descriptor in self.capabilities:
            identity = descriptor.identity_key
            if identity in seen:
                raise ValueError(f"duplicate capability descriptor: {identity!r}")
            seen.add(identity)
        return self


def build_platform_plugin_manifest(
    *,
    name: str,
    version: str,
    intergrax_version: str,
    capabilities: tuple[CapabilityDescriptor, ...] | list[CapabilityDescriptor] = (),
    author: str | None = None,
    documentation_uri: str | None = None,
    labels: tuple[str, ...] | list[str] = (),
    schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION,
) -> PlatformPluginManifest:
    """Construct a validated Platform Plugin manifest without side effects."""
    try:
        return PlatformPluginManifest(
            schema_version=schema_version,
            package=DistributionPackageIdentity(name=name, version=version),
            platform_compatibility=PlatformCompatibility(intergrax_version=intergrax_version),
            capabilities=tuple(capabilities),
            author=author,
            documentation_uri=documentation_uri,
            labels=tuple(labels),
        )
    except ValueError as exc:
        raise PlatformPluginManifestValidationError(str(exc)) from exc
