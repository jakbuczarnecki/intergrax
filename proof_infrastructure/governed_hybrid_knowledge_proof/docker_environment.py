# © Artur Czarnecki. All rights reserved.

"""Docker lifecycle abstraction for governed hybrid knowledge proof."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from proof_infrastructure.governed_hybrid_knowledge_proof.admin_port import (
    ControlledSecurityStatusAdminPort,
    HttpxControlledSecurityStatusAdminPort,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_COMPOSE_FILE = (
    _REPO_ROOT
    / "applications/local_workspace_application/docker/docker-compose.governed-hybrid-proof.yml"
)
_SECURITY_VENDOR_SERVICE = "security-status-vendor"


def _vendor_base_url() -> str:
    return os.environ.get(
        "GOVERNED_PROOF_SECURITY_VENDOR_URL",
        "http://127.0.0.1:8091",
    ).rstrip("/")


@dataclass(frozen=True, slots=True)
class GovernedHybridDockerEnvironmentV1:
    """Proof-only Docker environment for security-status vendor persistence scenarios."""

    compose_file: Path
    admin: ControlledSecurityStatusAdminPort
    vendor_base_url: str

    @classmethod
    def from_defaults(cls) -> GovernedHybridDockerEnvironmentV1:
        base_url = _vendor_base_url()
        return cls(
            compose_file=_DEFAULT_COMPOSE_FILE,
            admin=HttpxControlledSecurityStatusAdminPort(base_url=base_url),
            vendor_base_url=base_url,
        )

    def ensure_ready(self) -> None:
        self.admin.wait_until_ready()

    def restart_security_vendor(self) -> None:
        if not self.compose_file.is_file():
            raise RuntimeError(f"compose_file_missing: {self.compose_file}")
        completed = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(self.compose_file),
                "restart",
                _SECURITY_VENDOR_SERVICE,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "restart_failed"
            raise RuntimeError(f"security_vendor_restart_failed: {detail}")
        self.admin.wait_until_ready()
