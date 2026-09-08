#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E402

"""DG-001B R5-R1 qualification child: canonical LKW worker bootstrap with injected failure."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
_APPLICATIONS_BOOTSTRAP_ROOT = _REPO_BOOTSTRAP_ROOT / "applications"
for _bootstrap_entry in (str(_APPLICATIONS_BOOTSTRAP_ROOT), str(_REPO_BOOTSTRAP_ROOT)):
    if _bootstrap_entry not in sys.path:
        sys.path.insert(0, _bootstrap_entry)

import asyncio
import logging
import sys

from intergrax.hosting import HostedProcessBootstrapContext
from local_workspace_application.host.background_worker_main import (
    _resolve_settings,
    _run_guarded_worker_bootstrap,
    activate_local_workspace_reference_production_authority,
    build_local_workspace_worker_bootstrap_diagnostics,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.message_bus_wiring import local_workspace_message_bus_enabled
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.workspaces.document_store_factory import (
    resolve_lkw_runtime_document_store,
)
from scripts.proof.dg001b_r5_qualification_contracts import (
    controlled_failing_background_worker_constructor,
)

logger = logging.getLogger(__name__)
_BACKGROUND_WORKER_PROCESS_ROLE = "background_worker"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not local_workspace_message_bus_enabled():
        logger.error("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS must be true for the background worker")
        return 1

    settings = _resolve_settings(None)
    environment_profile = build_local_workspace_environment_profile(settings)
    _, registry_projection = activate_local_workspace_reference_production_authority(
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
    bootstrap_context = HostedProcessBootstrapContext.create(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role=_BACKGROUND_WORKER_PROCESS_ROLE,
    )
    asyncio.run(
        _run_guarded_worker_bootstrap(
            bootstrap_context=bootstrap_context,
            event_publisher=bootstrap_diagnostics.event_publisher,
            settings=settings,
            registry_projection=registry_projection,
            document_store=document_store,
            worker_constructor=controlled_failing_background_worker_constructor(),
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
