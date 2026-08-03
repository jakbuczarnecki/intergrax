# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace collaboration-suite foundation contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping, Protocol, runtime_checkable


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
