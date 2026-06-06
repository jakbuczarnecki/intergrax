# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cloud sandbox host integration contract (Phase M.6 P6)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SandboxSession(BaseModel):
    """Active remote sandbox session handle."""

    session_id: str
    status: str = "running"
    metadata: dict[str, str] = Field(default_factory=dict)


class SandboxExecResult(BaseModel):
    """Command execution result inside a sandbox session."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


class SandboxArtifact(BaseModel):
    """Uploaded artifact descriptor from sandbox session."""

    artifact_id: str
    uri: str = ""
    size_bytes: int = 0


@runtime_checkable
class SandboxHostBackend(Protocol):
    """Remote isolation host for harness ``sandbox.exec`` tool bridge."""

    def create_session(self) -> SandboxSession:
        """Provision a new isolated sandbox session."""

    def exec(self, session_id: str, command: str) -> SandboxExecResult:
        """Run a shell command inside an existing session."""

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> SandboxArtifact:
        """Upload a local file into the sandbox session."""
