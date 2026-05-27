# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shadow workspace runtime (Phase F.1, architecture §20)."""

from intergrax.runtime.workspace.manager import (
    DEFAULT_SHADOW_ROOT,
    ENV_SHADOW_ROOT,
    ShadowWorkspaceManager,
    resolve_shadow_root,
)
from intergrax.runtime.workspace.models import (
    ShadowArtifact,
    ShadowSnapshot,
    ShadowWorkspaceManifest,
)
from intergrax.runtime.workspace.shadow_workspace import (
    SHADOW_WORKSPACE_CLEANUP_KEY,
    SHADOW_WORKSPACE_FLAG,
    SHADOW_WORKSPACE_ID_KEY,
    ShadowWorkspace,
)

__all__ = [
    "DEFAULT_SHADOW_ROOT",
    "ENV_SHADOW_ROOT",
    "SHADOW_WORKSPACE_CLEANUP_KEY",
    "SHADOW_WORKSPACE_FLAG",
    "SHADOW_WORKSPACE_ID_KEY",
    "ShadowArtifact",
    "ShadowSnapshot",
    "ShadowWorkspace",
    "ShadowWorkspaceManager",
    "ShadowWorkspaceManifest",
    "resolve_shadow_root",
]
