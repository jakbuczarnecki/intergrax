# © Artur Czarnecki. All rights reserved.

"""Docker lifecycle abstraction for advanced flagship multi-vendor proof."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from proof_infrastructure.governed_hybrid_knowledge_proof.flagship_admin_ports import (
    FlagshipVendorAdminFacadeV1,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_COMPOSE_FILE = (
    _REPO_ROOT
    / "applications/local_workspace_application/docker/docker-compose.governed-hybrid-proof.yml"
)
_PROJECT_VENDOR_SERVICE = "project-status-vendor"
_SECURITY_VENDOR_SERVICE = "security-status-vendor"
_CHANGE_VENDOR_SERVICE = "change-approval-vendor"
_GOVERNANCE_VENDOR_SERVICE = "governance-approval-vendor"

_COMPOSE_UP_HINT = (
    "docker compose "
    f"-f {_DEFAULT_COMPOSE_FILE} "
    "up --build -d"
)


def _vendor_url(env_key: str, default: str) -> str:
    return os.environ.get(env_key, default).rstrip("/")


@dataclass(frozen=True, slots=True)
class AdvancedFlagshipDockerEnvironmentV1:
    """Proof-only Docker environment for four independent flagship vendors."""

    compose_file: Path
    admin: FlagshipVendorAdminFacadeV1
    project_vendor_base_url: str
    security_vendor_base_url: str
    change_vendor_base_url: str
    governance_vendor_base_url: str

    @classmethod
    def from_defaults(cls) -> AdvancedFlagshipDockerEnvironmentV1:
        project_url = _vendor_url(
            "GOVERNED_PROOF_PROJECT_VENDOR_URL",
            "http://127.0.0.1:8092",
        )
        security_url = _vendor_url(
            "GOVERNED_PROOF_SECURITY_VENDOR_URL",
            "http://127.0.0.1:8091",
        )
        change_url = _vendor_url(
            "GOVERNED_PROOF_CHANGE_VENDOR_URL",
            "http://127.0.0.1:8093",
        )
        governance_url = _vendor_url(
            "GOVERNED_PROOF_GOVERNANCE_VENDOR_URL",
            "http://127.0.0.1:8094",
        )
        return cls(
            compose_file=_DEFAULT_COMPOSE_FILE,
            admin=FlagshipVendorAdminFacadeV1.from_base_urls(
                project_base_url=project_url,
                security_base_url=security_url,
                change_base_url=change_url,
                governance_base_url=governance_url,
            ),
            project_vendor_base_url=project_url,
            security_vendor_base_url=security_url,
            change_vendor_base_url=change_url,
            governance_vendor_base_url=governance_url,
        )

    def ensure_ready(self) -> None:
        try:
            self.admin.wait_until_all_ready()
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}; required compose: {_COMPOSE_UP_HINT}"
            ) from exc

    def restart_vendor(self, service_name: str) -> None:
        if not self.compose_file.is_file():
            raise RuntimeError(f"compose_file_missing: {self.compose_file}")
        completed = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(self.compose_file),
                "restart",
                service_name,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "restart_failed"
            raise RuntimeError(f"vendor_restart_failed: {detail}")
        self.admin.wait_until_all_ready()

    def restart_project_vendor(self) -> None:
        self.restart_vendor(_PROJECT_VENDOR_SERVICE)

    def restart_security_vendor(self) -> None:
        self.restart_vendor(_SECURITY_VENDOR_SERVICE)

    def compose_up_hint(self) -> str:
        return _COMPOSE_UP_HINT
