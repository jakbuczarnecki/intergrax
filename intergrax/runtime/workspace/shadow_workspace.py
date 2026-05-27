# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Isolated temporary workspace for experiments (architecture §20)."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

from intergrax.runtime.workspace.models import (
    ShadowArtifact,
    ShadowSnapshot,
    ShadowWorkspaceManifest,
)
from intergrax.utils.time_provider import SystemTimeProvider

SHADOW_WORKSPACE_FLAG = "shadow_workspace"
SHADOW_WORKSPACE_ID_KEY = "shadow_workspace_id"
SHADOW_WORKSPACE_CLEANUP_KEY = "shadow_workspace_cleanup"


def _safe_relative_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {relative_path}")
    return path


class ShadowWorkspace:
    """
    Temporary isolated filesystem workspace.

    Provides write/read, artifact listing, snapshot/rollback, and cleanup.
    """

    def __init__(
        self,
        *,
        workspace_id: str,
        root: Path,
        tenant_id: str,
        task_id: str,
        created_at_utc: str,
    ) -> None:
        self.workspace_id = workspace_id
        self.root = root
        self.tenant_id = tenant_id
        self.task_id = task_id
        self.created_at_utc = created_at_utc
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        base_dir: Path,
        *,
        tenant_id: str,
        task_id: str,
        workspace_id: str | None = None,
    ) -> ShadowWorkspace:
        ws_id = workspace_id or f"shadow_{uuid4().hex[:16]}"
        root = base_dir / tenant_id / task_id / ws_id
        return cls(
            workspace_id=ws_id,
            root=root,
            tenant_id=tenant_id,
            task_id=task_id,
            created_at_utc=SystemTimeProvider.utc_now().isoformat(),
        )

    def write_text(self, relative_path: str, content: str, *, content_type: str = "text/plain") -> ShadowArtifact:
        rel = _safe_relative_path(relative_path)
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode("utf-8")
        target.write_bytes(encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        return ShadowArtifact(
            artifact_id=f"art_{uuid4().hex[:12]}",
            relative_path=rel.as_posix(),
            size_bytes=len(encoded),
            content_type=content_type,
            sha256=digest,
        )

    def read_text(self, relative_path: str) -> str:
        rel = _safe_relative_path(relative_path)
        target = self.root / rel
        return target.read_text(encoding="utf-8")

    def list_artifacts(self) -> list[ShadowArtifact]:
        artifacts: list[ShadowArtifact] = []
        if not self.root.exists():
            return artifacts
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            data = path.read_bytes()
            artifacts.append(
                ShadowArtifact(
                    artifact_id=f"art_{hashlib.sha256(rel.encode()).hexdigest()[:12]}",
                    relative_path=rel,
                    size_bytes=len(data),
                    content_type="application/octet-stream",
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
        return artifacts

    def manifest(self) -> ShadowWorkspaceManifest:
        artifacts = self.list_artifacts()
        return ShadowWorkspaceManifest(
            workspace_id=self.workspace_id,
            tenant_id=self.tenant_id,
            task_id=self.task_id,
            root_path=str(self.root),
            created_at_utc=self.created_at_utc,
            artifact_count=len(artifacts),
            artifacts=artifacts,
        )

    def snapshot(self) -> ShadowSnapshot:
        files: dict[str, str] = {}
        for artifact in self.list_artifacts():
            files[artifact.relative_path] = self.read_text(artifact.relative_path)
        return ShadowSnapshot(
            workspace_id=self.workspace_id,
            created_at_utc=SystemTimeProvider.utc_now().isoformat(),
            files=files,
        )

    def rollback(self, snapshot: ShadowSnapshot) -> None:
        if snapshot.workspace_id != self.workspace_id:
            raise ValueError("snapshot workspace_id mismatch")

        for path in list(self.root.rglob("*")):
            if path.is_file():
                path.unlink()

        for relative_path, content in snapshot.files.items():
            self.write_text(relative_path, content)

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def exists_on_disk(self) -> bool:
        return self.root.exists()
