# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""VENV_BUNDLE artifact path resolution and integrity checks (Agent Distribution)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Sequence

from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.errors import AgentDistributionError
from intergrax.agent_distribution.runtime_context_staging import (
    RUNTIME_LOCK_MANIFEST_FILENAME,
    directory_content_digest,
)
from intergrax.agent_distribution.runtime_revision import RuntimeRevision

_FLAT_SITE_PACKAGES: Final = "site-packages"


class VenvBundleDeploymentArtifactError(AgentDistributionError):
    """VENV_BUNDLE deployment artifact validation failed."""


def resolve_venv_bundle_artifact_directory(
    artifact_locator: str,
    *,
    trusted_parent_roots: Sequence[Path] | None = None,
    allow_qualification_locators: bool = False,
) -> Path:
    """Resolve an immutable bundle directory from a canonical locator (production: ``file://`` only)."""
    normalized = artifact_locator.strip()
    if not normalized:
        raise VenvBundleDeploymentArtifactError("artifact_locator must be non-empty")
    if normalized.startswith("reference://"):
        raise VenvBundleDeploymentArtifactError(
            "reference:// artifact locators are forbidden for production deployment"
        )
    if normalized.startswith("artifact://"):
        raise VenvBundleDeploymentArtifactError(
            "artifact:// locators are not authoritative deployment inputs"
        )
    if normalized.startswith("test://"):
        if not allow_qualification_locators:
            raise VenvBundleDeploymentArtifactError(
                "test:// artifact locators are forbidden in production deployment mode"
            )
        candidate = Path(normalized.removeprefix("test://"))
    elif normalized.startswith("file://"):
        candidate = Path(normalized.removeprefix("file://"))
        if not candidate.is_absolute():
            candidate = Path(normalized.removeprefix("file://").lstrip("/"))
    else:
        raise VenvBundleDeploymentArtifactError(
            f"unsupported artifact_locator scheme: {artifact_locator!r}"
        )

    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise VenvBundleDeploymentArtifactError(
            f"artifact locator does not resolve to a directory: {artifact_locator!r}"
        )
    if trusted_parent_roots is not None:
        if not any(
            _path_is_same_or_under(resolved, parent.resolve())
            for parent in trusted_parent_roots
        ):
            raise VenvBundleDeploymentArtifactError(
                "artifact directory is outside trusted deployment roots"
            )
    return resolved


def _path_is_same_or_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def discover_venv_bundle_site_packages(artifact_root: Path) -> Path:
    """Return site-packages directory for a VENV_BUNDLE layout."""
    venv_root = artifact_root / ".venv"
    if venv_root.is_dir():
        matches = sorted(venv_root.glob("lib/python*/site-packages"))
        for candidate in matches:
            if candidate.is_dir():
                return candidate.resolve()
    flat = artifact_root / _FLAT_SITE_PACKAGES
    if flat.is_dir():
        return flat.resolve()
    raise VenvBundleDeploymentArtifactError(
        f"venv bundle at {artifact_root!s} lacks importable site-packages layout"
    )


def load_venv_bundle_runtime_lock(artifact_root: Path) -> MaterializedRuntimeLock:
    manifest_path = artifact_root / RUNTIME_LOCK_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise VenvBundleDeploymentArtifactError(
            f"missing runtime lock manifest: {RUNTIME_LOCK_MANIFEST_FILENAME}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VenvBundleDeploymentArtifactError(
            f"invalid runtime lock manifest JSON in {RUNTIME_LOCK_MANIFEST_FILENAME}"
        ) from exc
    try:
        lock = MaterializedRuntimeLock.model_validate(payload)
    except ValueError as exc:
        raise VenvBundleDeploymentArtifactError(
            "runtime lock manifest failed validation"
        ) from exc
    return lock.with_content_identity()


def verify_venv_bundle_artifact_bytes(
    artifact_root: Path,
    expected_digest: str,
) -> str:
    """Compute and compare directory content digest (sanctioned authority)."""
    computed = directory_content_digest(artifact_root)
    if computed != expected_digest:
        raise VenvBundleDeploymentArtifactError(
            "materialization artifact digest mismatch with on-disk bundle contents"
        )
    return computed


def verify_venv_bundle_lock_matches_revision(
    revision: RuntimeRevision,
    lock: MaterializedRuntimeLock,
) -> None:
    if revision.materialized_runtime_lock_id is not None:
        if lock.lock_id != revision.materialized_runtime_lock_id:
            raise VenvBundleDeploymentArtifactError(
                "materialized runtime lock id mismatch with runtime revision"
            )
    if revision.materialized_runtime_lock_digest is not None:
        if lock.lock_digest != revision.materialized_runtime_lock_digest:
            raise VenvBundleDeploymentArtifactError(
                "materialized runtime lock digest mismatch with runtime revision"
            )
