# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace collaboration-suite foundation contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, Protocol, runtime_checkable

from intergrax.integrations.contracts.base import IntegrationConfigurationError

_MAX_MEDIA_TYPE_LENGTH = 255
_MEDIA_TYPE_PATTERN = re.compile(
    r"^[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+$"
)


def normalize_google_workspace_media_type(value: object) -> str:
    """Validate and canonicalize one base media type without parameters."""
    if type(value) is not str:
        raise TypeError("media type must be a string")
    if not value or value != value.strip():
        raise ValueError("media type must be nonblank without surrounding whitespace")
    if len(value) > _MAX_MEDIA_TYPE_LENGTH:
        raise ValueError("media type exceeds maximum length")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError("media type contains control characters")
    if "," in value or "*" in value or ";" in value:
        raise ValueError("media type must not contain comma, wildcard, or parameters")
    normalized = value.lower()
    if not _MEDIA_TYPE_PATTERN.fullmatch(normalized):
        raise ValueError("media type syntax is invalid")
    return normalized

_INVALID_CREDENTIAL_MATERIAL_MESSAGE = "Google Workspace credential material is invalid"


def copy_google_workspace_credential_material(material: object) -> dict[str, str]:
    """Validate and defensively copy opaque Google Workspace credential material."""
    if not isinstance(material, Mapping):
        raise IntegrationConfigurationError(_INVALID_CREDENTIAL_MATERIAL_MESSAGE)
    try:
        items = tuple(material.items())
    except Exception:
        raise IntegrationConfigurationError(_INVALID_CREDENTIAL_MATERIAL_MESSAGE) from None
    if not items:
        raise IntegrationConfigurationError(_INVALID_CREDENTIAL_MATERIAL_MESSAGE)
    copied: dict[str, str] = {}
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise IntegrationConfigurationError(_INVALID_CREDENTIAL_MATERIAL_MESSAGE)
        key, value = item
        if not isinstance(key, str) or not key.strip():
            raise IntegrationConfigurationError(_INVALID_CREDENTIAL_MATERIAL_MESSAGE)
        if not isinstance(value, str):
            raise IntegrationConfigurationError(_INVALID_CREDENTIAL_MATERIAL_MESSAGE)
        copied[key] = value
    return copied


class GoogleWorkspaceSourceKind(StrEnum):
    """Supported Google Workspace knowledge source kinds for one provider integration."""

    DRIVE = "drive"
    DOCS = "docs"
    SHEETS = "sheets"
    SLIDES = "slides"
    CALENDAR = "calendar"
    MAIL = "mail"
    CHAT = "chat"


GOOGLE_WORKSPACE_SUPPORTED_SOURCE_KINDS: tuple[GoogleWorkspaceSourceKind, ...] = tuple(
    GoogleWorkspaceSourceKind
)


@runtime_checkable
class GoogleWorkspaceHttpResponse(Protocol):
    """Single HTTP response surface for one executor attempt."""

    status_code: int
    headers: Mapping[str, str]
    content: bytes

    def json(self) -> object:
        """Decode the response body as JSON."""


@runtime_checkable
class GoogleWorkspaceRequestExecutor(Protocol):
    """Perform exactly one HTTP GET attempt per call."""

    def get(
        self,
        *,
        url: str,
        params: Mapping[str, object] | None,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> GoogleWorkspaceHttpResponse:
        """Execute one HTTP GET request."""


@runtime_checkable
class GoogleWorkspaceRequestExecutorFactory(Protocol):
    """Authentication and deployment seam for request executors."""

    def create_request_executor(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> GoogleWorkspaceRequestExecutor:
        """Build a request executor from resolved credential material."""


@runtime_checkable
class GoogleWorkspaceTransport(Protocol):
    """Shared read-only Google Workspace request transport."""

    def get_json(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        """Issue a bounded GET request and return decoded JSON object payload."""


@dataclass(frozen=True, slots=True)
class GoogleWorkspaceBinaryPayload:
    data: bytes = field(repr=False)
    content_type: str

    def __post_init__(self) -> None:
        if type(self.data) is not bytes:
            raise TypeError("binary payload data must be bytes")
        canonical = normalize_google_workspace_media_type(self.content_type)
        if canonical != self.content_type:
            object.__setattr__(self, "content_type", canonical)


@runtime_checkable
class GoogleWorkspaceBinaryTransport(Protocol):
    """Bounded binary GET capability for Google Workspace provider APIs."""

    def get_bytes(
        self,
        *,
        source_kind: GoogleWorkspaceSourceKind,
        relative_path: str,
        params: Mapping[str, object] | None,
        expected_content_type: str,
        max_bytes: int,
        range_limited: bool,
    ) -> GoogleWorkspaceBinaryPayload:
        """Issue a bounded GET request and return validated binary payload."""


@runtime_checkable
class GoogleWorkspaceClientFamily(Protocol):
    """Shared Google Workspace API client family exposing one transport."""

    @property
    def transport(self) -> GoogleWorkspaceTransport:
        """Return the shared read-only transport for all Google Workspace APIs."""


@runtime_checkable
class GoogleWorkspaceCredentialResolver(Protocol):
    """Resolve opaque credential material from a safe credential reference."""

    def resolve_credential(self, credential_ref: str) -> Mapping[str, str]:
        """Return credential material for ``credential_ref`` without exposing secret storage."""


@runtime_checkable
class GoogleWorkspaceClientFactory(Protocol):
    """Create one shared Google Workspace client family from resolved credential material."""

    def create_client_family(
        self,
        *,
        credential_material: Mapping[str, str],
    ) -> GoogleWorkspaceClientFamily:
        """Build or return the shared client family for Google Workspace APIs."""
