# © Artur Czarnecki. All rights reserved.

"""Minimal VENV_BUNDLE directories for Agent Distribution deployment qualification."""

from __future__ import annotations

import json
from pathlib import Path

from intergrax.agent_distribution.dependency import (
    MaterializedAgentClosureEntry,
    MaterializedLockPackage,
    MaterializedRuntimeLock,
)
from intergrax.agent_distribution.runtime_context_staging import (
    RUNTIME_LOCK_MANIFEST_FILENAME,
    directory_content_digest,
)


def write_minimal_venv_bundle(
    artifact_root: Path,
    *,
    package_digest: str = "sha256:" + ("a" * 64),
) -> tuple[str, MaterializedRuntimeLock]:
    site_packages = artifact_root / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "bundle_marker.py").write_text(
        'MARKER = "qualification"\n',
        encoding="utf-8",
    )
    lock = MaterializedRuntimeLock(
        resolver_algorithm_id="intergrax.test",
        resolver_algorithm_version="1",
        inputs_digest="inputs-1",
        intergrax_version="0.1.0",
        python_version="3.12",
        packages=(
            MaterializedLockPackage(
                distribution_name="pkg-qual",
                version="1.0.0",
                package_digest=package_digest,
            ),
        ),
        agent_closure=(
            MaterializedAgentClosureEntry(
                distribution_package_id="pkg-qual",
                package_digest=package_digest,
                role="direct",
            ),
        ),
    ).with_content_identity()
    (artifact_root / RUNTIME_LOCK_MANIFEST_FILENAME).write_text(
        json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    digest = directory_content_digest(artifact_root)
    return digest, lock


def file_artifact_locator(artifact_root: Path) -> str:
    return f"file://{artifact_root.resolve().as_posix()}"
