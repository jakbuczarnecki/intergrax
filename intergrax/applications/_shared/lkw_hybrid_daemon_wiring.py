# © Artur Czarnecki. All rights reserved.

"""LKW hybrid daemon wiring (AUDIT-IDEAL-28.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.applications._shared.lkw_hybrid_daemon import LkwHybridDaemonSpec, build_lkw_hybrid_daemon_spec
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


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
    deployment = env.host_deployment_profile
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
