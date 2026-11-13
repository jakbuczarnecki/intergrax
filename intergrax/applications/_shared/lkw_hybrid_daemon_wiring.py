# © Artur Czarnecki. All rights reserved.

"""LKW hybrid daemon wiring (AUDIT-IDEAL-28.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications._shared.lkw_hybrid_daemon import LkwHybridDaemonSpec, build_lkw_hybrid_daemon_spec
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    HostDeploymentProfile,
)


class LkwHybridDaemonDeployment(BaseModel):
    """LKW-owned hybrid daemon deployment keys (TL-FIX-C).

    The keys ride on the generic ``HostDeploymentProfile`` as product-owned flat scalar
    entries; this model is the only place that knows their names and validates them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lkw_hybrid_daemon_enabled: bool = False
    lkw_daemon_bind_host: str = "127.0.0.1"
    lkw_daemon_port: int = Field(default=8020, ge=1, le=65535)

    @classmethod
    def from_host_deployment(cls, deployment: HostDeploymentProfile) -> LkwHybridDaemonDeployment:
        """Read and validate the LKW keys from the generic host deployment profile.

        Unknown ``lkw_`` keys and out-of-range values fail closed.
        """
        extras = deployment.model_extra or {}
        return cls.model_validate({key: value for key, value in extras.items() if key.startswith("lkw_")})


@dataclass(frozen=True, slots=True)
class LkwHybridDaemonWiring:
    enabled: bool
    spec: LkwHybridDaemonSpec | None
    launcher_path: Path | None


def resolve_lkw_hybrid_daemon_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    repo_root: Path,
) -> LkwHybridDaemonWiring:
    """Resolve LKW hybrid daemon spec when host deployment profile enables CFG-14."""
    deployment = LkwHybridDaemonDeployment.from_host_deployment(env.host_deployment_profile)
    if not deployment.lkw_hybrid_daemon_enabled:
        return LkwHybridDaemonWiring(enabled=False, spec=None, launcher_path=None)

    launcher = repo_root / "scripts" / "lkw-host.py"
    if not launcher.is_file():
        return LkwHybridDaemonWiring(enabled=False, spec=None, launcher_path=None)

    spec = build_lkw_hybrid_daemon_spec(
        bind_host=deployment.lkw_daemon_bind_host,
        port=deployment.lkw_daemon_port,
    )
    return LkwHybridDaemonWiring(enabled=True, spec=spec, launcher_path=launcher)
