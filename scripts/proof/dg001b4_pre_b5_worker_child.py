#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E402

"""DG-001B4 qualification child: canonical worker pre-B5 guarded bootstrap with injected failure."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
_APPLICATIONS_BOOTSTRAP_ROOT = _REPO_BOOTSTRAP_ROOT / "applications"
for _bootstrap_entry in (str(_APPLICATIONS_BOOTSTRAP_ROOT), str(_REPO_BOOTSTRAP_ROOT)):
    if _bootstrap_entry not in sys.path:
        sys.path.insert(0, _bootstrap_entry)

import json
import logging

from local_workspace_application.host.background_worker_main import (
    _resolve_settings,
    activate_local_workspace_reference_production_authority,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.message_bus_wiring import local_workspace_message_bus_enabled
from local_workspace_application.workspaces.document_store_factory import (
    resolve_lkw_runtime_document_store,
)
from scripts.proof.dg001b4_pre_b5_qualification_contracts import (
    controlled_failing_worker_bootstrap_diagnostics_segment,
    qualification_secret_sentinel,
)
from scripts.proof.dg001b4_pre_b5_qualification_support import (
    RecordingBootstrapFailureReporter,
    run_canonical_worker_pre_b5_guarded_bootstrap,
)

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not local_workspace_message_bus_enabled():
        logger.error("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS must be true for the background worker")
        return 1

    reporter = RecordingBootstrapFailureReporter(records=[])
    failing_segment = controlled_failing_worker_bootstrap_diagnostics_segment()
    settings = _resolve_settings(None)
    environment_profile = build_local_workspace_environment_profile(settings)
    _, registry_projection = activate_local_workspace_reference_production_authority(
        settings,
        environment_profile=environment_profile,
    )
    document_store = resolve_lkw_runtime_document_store(settings)

    try:
        run_canonical_worker_pre_b5_guarded_bootstrap(
            diagnostics_segment=lambda: failing_segment(
                registry_projection=registry_projection,
                settings=settings,
                environment_profile=environment_profile,
                document_store=document_store,
            ),
            extra_reporters=[reporter],
        )
    except RuntimeError as exc:
        if qualification_secret_sentinel() not in str(exc):
            raise
        evidence_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
        if evidence_path is not None:
            record = reporter.records[0]
            payload = {
                "bootstrap_attempt_id": record.bootstrap_attempt_id,
                "readiness_at_failure": record.readiness_at_failure.value,
                "stage": record.stage.value,
                "surface_kind": record.surface_kind.value,
                "exception_type": record.failure_facts.exception_type,
                "instance_id": record.identity.instance_id,
                "diagnostic_tenant_id": record.identity.diagnostic_tenant_id,
                "primary_exception_type": type(exc).__name__,
            }
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
