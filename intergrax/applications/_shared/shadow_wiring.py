# © Artur Czarnecki. All rights reserved.

"""Shadow workspace manager wiring from environment profile (Phase H-APP.3.4)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ShadowWorkspaceProfile,
)
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager


def wire_shadow_workspace(
    env: ApplicationEnvironmentProfile,
) -> ShadowWorkspaceManager | None:
    """Configure ``ShadowWorkspaceManager`` when profile enables shadow workspace."""
    profile = env.shadow_workspace
    if profile is None:
        return None
    return ShadowWorkspaceManager(root=profile.root)
