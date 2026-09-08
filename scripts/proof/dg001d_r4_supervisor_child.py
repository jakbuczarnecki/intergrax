#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E402

"""DG-001D R4 qualification child: real supervisor with injected failing engine factory."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
_APPLICATIONS_BOOTSTRAP_ROOT = _REPO_BOOTSTRAP_ROOT / "applications"
_AGENTS_BOOTSTRAP_ROOT = _REPO_BOOTSTRAP_ROOT / "agents"
for _bootstrap_entry in (
    str(_APPLICATIONS_BOOTSTRAP_ROOT),
    str(_AGENTS_BOOTSTRAP_ROOT),
    str(_REPO_BOOTSTRAP_ROOT),
):
    if _bootstrap_entry not in sys.path:
        sys.path.insert(0, _bootstrap_entry)

import asyncio
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from intergrax.hosting import HostedApplicationExitKind
from intergrax.hosting.contracts.policies import RestartPolicy
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.engine.definition import resolve_hosted_application_definition
from intergrax.hosting.supervisor.supervisor import HostedApplicationSupervisor
from local_workspace_application.host.background_worker_main import (
    activate_local_workspace_reference_production_authority,
    build_local_workspace_worker_bootstrap_diagnostics,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.hosting.profile import build_local_workspace_hosted_profile
from local_workspace_application.workspaces.document_store_factory import (
    resolve_lkw_runtime_document_store,
)
from scripts.proof.dg001d_r4_qualification_contracts import (
    controlled_failing_hosted_application_engine_factory,
)

logger = logging.getLogger(__name__)


class _SystemWallClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def _resolve_instance_id_generator() -> Callable[[], str] | None:
    configured = os.environ.get("DG001D_R4_INSTANCE_ID", "").strip()
    if not configured:
        return None
    return lambda: configured


async def _run_supervisor() -> int:
    settings = LocalWorkspaceBackendSettings.from_env()
    environment_profile = build_local_workspace_environment_profile(settings)
    composition, registry_projection = activate_local_workspace_reference_production_authority(
        settings,
        environment_profile=environment_profile,
    )
    document_store = resolve_lkw_runtime_document_store(settings)
    bootstrap_diagnostics = build_local_workspace_worker_bootstrap_diagnostics(
        registry_projection=registry_projection,
        settings=settings,
        environment_profile=environment_profile,
        document_store=document_store,
    )
    profile = build_local_workspace_hosted_profile(
        process_composition=composition,
        settings=settings,
    )
    definition = resolve_hosted_application_definition(profile)
    definition = replace(definition, restart_policy=RestartPolicy.never())
    clock = _SystemWallClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    engine_factory = controlled_failing_hosted_application_engine_factory()
    instance_id_generator = _resolve_instance_id_generator()
    if instance_id_generator is not None:
        supervisor = HostedApplicationSupervisor(
            definition=definition,
            engine_factory=engine_factory,
            control=control,
            event_publisher=bootstrap_diagnostics.event_publisher,
            clock=clock,
            instance_id_generator=instance_id_generator,
        )
    else:
        supervisor = HostedApplicationSupervisor(
            definition=definition,
            engine_factory=engine_factory,
            control=control,
            event_publisher=bootstrap_diagnostics.event_publisher,
            clock=clock,
        )
    result = await supervisor.run()
    print(f"SUPERVISOR_EXIT_KIND={result.final_exit.exit_kind.value}")
    print(f"SUPERVISOR_REASON_CODE={result.final_exit.reason_code}")
    if result.attempts:
        print(f"SUPERVISOR_INSTANCE_ID={result.attempts[-1].instance_id}")
    if result.final_exit.exit_kind is HostedApplicationExitKind.SUPERVISOR_ERROR:
        return 1
    if result.final_exit.exit_kind is HostedApplicationExitKind.CLEAN_STOP:
        return 0
    return 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(_run_supervisor())


if __name__ == "__main__":
    sys.exit(main())
