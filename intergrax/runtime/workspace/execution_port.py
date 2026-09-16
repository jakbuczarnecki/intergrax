# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral workspace execution port (tool invocation boundary)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.runtime.workspace.models import ShadowArtifact, ShadowSnapshot


@runtime_checkable
class WorkspaceExecutionPort(Protocol):
    """Minimal workspace surface consumed by catalog workspace tools."""

    workspace_id: str
    task_id: str

    def write_text(
        self,
        relative_path: str,
        content: str,
        *,
        content_type: str = "text/plain",
    ) -> ShadowArtifact: ...

    def read_text(self, relative_path: str) -> str: ...

    def delete_file(self, relative_path: str) -> bool: ...

    def list_artifacts(self) -> list[ShadowArtifact]: ...

    def snapshot(self) -> ShadowSnapshot: ...

    def search_text(
        self,
        query: str,
        *,
        path_prefix: str = "",
        case_insensitive: bool = True,
        max_matches: int = 50,
    ) -> list[tuple[str, int, str]]: ...

    def read_artifact_bytes(self, relative_path: str) -> bytes | None: ...

    def write_artifact_bytes(
        self,
        relative_path: str,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> ShadowArtifact: ...
