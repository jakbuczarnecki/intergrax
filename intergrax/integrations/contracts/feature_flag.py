# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Feature flag integration contract (Phase M.6 P4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class FeatureFlagEvaluation(BaseModel):
    key: str
    enabled: bool = False
    variant: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class FeatureFlagBackend(Protocol):
    """Tenant-scoped feature flag evaluation (Unleash, LaunchDarkly, …)."""

    def is_enabled(self, flag_key: str, *, tenant_id: str, user_id: str = "") -> bool:
        """Return whether ``flag_key`` is enabled for the given tenant/user context."""

    def evaluate(
        self,
        flag_key: str,
        *,
        tenant_id: str,
        user_id: str = "",
    ) -> FeatureFlagEvaluation:
        """Return structured evaluation including optional variant."""
