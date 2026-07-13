# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application profile core contracts (APP-HOST-1A.1)."""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema
from pydantic.types import SecretBytes, SecretStr

from intergrax.hosting.contracts.identity import (
    HostedApplicationIdentity,
    normalize_application_id,
    validate_application_factory_id,
)

HOSTED_APPLICATION_PROFILE_SPEC_VERSION = "1.0"


def _validate_json_value(value: object) -> JsonValue:
    """Validate and copy a JSON-safe metadata value.

    Metadata is public JSON-only data and must not contain secrets.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata must contain only finite JSON numbers")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        raise ValueError("metadata must contain only public JSON values")
    if isinstance(value, SecretStr | SecretBytes):
        raise ValueError("metadata must contain only public JSON values")
    if isinstance(value, list):
        return [_validate_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _validate_json_value(item) for key, item in value.items()}
    raise ValueError("metadata must contain only public JSON values")


def _normalize_metadata(metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: _validate_json_value(value) for key, value in metadata.items()}


def derive_stable_application_factory_id(factory: Callable[..., object]) -> str:
    """Derive a stable factory identifier from a module-level callable."""
    if isinstance(factory, functools.partial):
        raise ValueError(
            "application_factory identity is not reliably stable; "
            'provide an explicit application_factory_id="..."'
        )
    module = getattr(factory, "__module__", None)
    qualname = getattr(factory, "__qualname__", None)
    if not module or not qualname:
        raise ValueError(
            "application_factory identity is not reliably stable; "
            'provide an explicit application_factory_id="..."'
        )
    if qualname == "<lambda>" or "<locals>" in qualname:
        raise ValueError(
            "application_factory identity is not reliably stable; "
            'provide an explicit application_factory_id="..."'
        )
    return validate_application_factory_id(f"{module}.{qualname}")


class HostedApplicationProfilePublicView(BaseModel):
    """Explicit public projection of a hosted application profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_version: Literal["1.0"]
    identity: HostedApplicationIdentity
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _normalize_metadata(value)


def _canonical_public_view_bytes(public_view: HostedApplicationProfilePublicView) -> bytes:
    payload = public_view.model_dump(mode="json")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class HostedApplicationProfile(BaseModel):
    """Minimal hosted application profile core (APP-HOST-1A.1)."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    application_id: str
    application_factory: SkipJsonSchema[Callable[..., object]] = Field(exclude=True, repr=False)
    application_factory_id: str | None = None
    spec_version: Literal["1.0"] = HOSTED_APPLICATION_PROFILE_SPEC_VERSION
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("application_id")
    @classmethod
    def _validate_application_id(cls, value: str) -> str:
        return normalize_application_id(value)

    @field_validator("application_factory_id")
    @classmethod
    def _validate_explicit_application_factory_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_application_factory_id(value)

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _normalize_metadata(value)

    @model_validator(mode="after")
    def _resolve_application_factory_id(self) -> HostedApplicationProfile:
        if self.application_factory_id is not None:
            return self
        derived_factory_id = derive_stable_application_factory_id(self.application_factory)
        object.__setattr__(self, "application_factory_id", derived_factory_id)
        return self

    @property
    def identity(self) -> HostedApplicationIdentity:
        factory_id = self.application_factory_id
        if factory_id is None:
            raise RuntimeError("hosted application profile is missing application_factory_id")
        return HostedApplicationIdentity(
            application_id=self.application_id,
            application_factory_id=factory_id,
        )

    def public_view(self) -> HostedApplicationProfilePublicView:
        return HostedApplicationProfilePublicView(
            spec_version=self.spec_version,
            identity=self.identity,
            metadata=self.metadata,
        )

    def profile_digest(self) -> str:
        digest = hashlib.sha256(_canonical_public_view_bytes(self.public_view())).hexdigest()
        return f"sha256:{digest}"
