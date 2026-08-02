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
class GoogleWorkspaceClientFamily(Protocol):
    """Opaque shared Google Workspace API client family (foundation port marker)."""


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
