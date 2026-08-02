# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace collaboration-suite typed configuration."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig


class GoogleWorkspaceCollaborationSuiteIntegrationConfig(CategoryIntegrationConfig):
    """Typed, secret-safe config for Google Workspace collaboration suite integration."""

    credential_ref: str = Field(default="", min_length=0)

    @model_validator(mode="after")
    def _require_credential_ref_when_enabled(self) -> Self:
        if self.enabled and not self.credential_ref.strip():
            raise ValueError(
                "credential_ref is required when Google Workspace integration is enabled",
            )
        return self
