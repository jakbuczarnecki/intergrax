# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace collaboration-suite typed configuration."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig


class GoogleWorkspaceCollaborationSuiteCompositionMode(StrEnum):
    """Explicit integration composition contract."""

    CREDENTIAL_REF = "credential_ref"
    INJECTED_CLIENT = "injected_client"


class GoogleWorkspaceCollaborationSuiteIntegrationConfig(CategoryIntegrationConfig):
    """Typed, secret-safe config for Google Workspace collaboration suite integration."""

    credential_ref: str = Field(default="", min_length=0)
    composition_mode: GoogleWorkspaceCollaborationSuiteCompositionMode = (
        GoogleWorkspaceCollaborationSuiteCompositionMode.CREDENTIAL_REF
    )

    @model_validator(mode="after")
    def _validate_enabled_composition(self) -> Self:
        if (
            self.enabled
            and self.composition_mode
            == GoogleWorkspaceCollaborationSuiteCompositionMode.CREDENTIAL_REF
            and not self.credential_ref.strip()
        ):
            raise ValueError(
                "credential_ref is required when Google Workspace integration is enabled "
                "in credential_ref composition mode",
            )
        return self
