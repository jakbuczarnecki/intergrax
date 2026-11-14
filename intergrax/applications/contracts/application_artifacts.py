# © Artur Czarnecki. All rights reserved.

"""Application artifact references — COMPAT-X shim (APP-CON-4, PDC-D8-R017).

The contracts are defined once, in :mod:`intergrax.contracts.task_artifacts`. This module only
re-exports them so existing import paths keep working (COMPAT-X R5 shim policy); it defines
no class of its own. New code should import from ``intergrax.contracts.task_artifacts``.
"""

from __future__ import annotations

from intergrax.contracts.task_artifacts import (
    APPLICATION_ARTIFACTS_STAGING_KEY,
    RUN_ARTIFACT_BUNDLE_METADATA_KEY,
    ApplicationArtifactRef,
    ArtifactRetentionPolicy,
    ArtifactSecurityClass,
    ArtifactVisibility,
    RunArtifactBundle,
    SandboxArtifactRef,
    WorkspaceArtifactRef,
)

__all__ = [
    "APPLICATION_ARTIFACTS_STAGING_KEY",
    "RUN_ARTIFACT_BUNDLE_METADATA_KEY",
    "ApplicationArtifactRef",
    "ArtifactRetentionPolicy",
    "ArtifactSecurityClass",
    "ArtifactVisibility",
    "RunArtifactBundle",
    "SandboxArtifactRef",
    "WorkspaceArtifactRef",
]
