# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E402

"""Shared helpers for DG-001B4 worker pre-B5 integration qualification."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
_APPLICATIONS_BOOTSTRAP_ROOT = _REPO_BOOTSTRAP_ROOT / "applications"
for _bootstrap_entry in (str(_APPLICATIONS_BOOTSTRAP_ROOT), str(_REPO_BOOTSTRAP_ROOT)):
    if _bootstrap_entry not in sys.path:
        sys.path.insert(0, _bootstrap_entry)

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from intergrax.hosting.bootstrap_failure import (
    BootstrapFailureReporter,
    BootstrapIdentitySnapshot,
    BootstrapReadinessLevel,
    BootstrapSurfaceKind,
    HostedBootstrapFailureProducer,
    HostedBootstrapFailureRecord,
    LoggingBootstrapFailureReporter,
    mint_bootstrap_attempt_id,
    run_guarded_hosted_bootstrap_segment,
)
from intergrax.hosting.process_bootstrap import HostedProcessBootstrapPhase
from local_workspace_application.host.background_worker_main import (
    LocalWorkspaceWorkerBootstrapDiagnostics,
    _resolve_settings,
    activate_local_workspace_reference_production_authority,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

_BACKGROUND_WORKER_PROCESS_ROLE = "background_worker"


@dataclass
class RecordingBootstrapFailureReporter:
    """Qualification reporter that captures emitted bootstrap failure records."""

    records: list[HostedBootstrapFailureRecord]

    def report(self, record: HostedBootstrapFailureRecord) -> None:
        self.records.append(record)


def build_production_equivalent_bootstrap_failure_producer(
    *,
    extra_reporters: Sequence[BootstrapFailureReporter] = (),
) -> HostedBootstrapFailureProducer:
    """Mirror ``background_worker_main._BOOTSTRAP_FAILURE_PRODUCER`` with optional observers."""
    return HostedBootstrapFailureProducer(
        reporters=[LoggingBootstrapFailureReporter(), *extra_reporters],
    )


def run_canonical_worker_pre_b5_guarded_bootstrap(
    *,
    diagnostics_segment: Callable[[], LocalWorkspaceWorkerBootstrapDiagnostics],
    extra_reporters: Sequence[BootstrapFailureReporter] = (),
) -> None:
    """Execute the canonical worker pre-B5 guarded bootstrap path from ``main()``."""
    bootstrap_attempt_id = mint_bootstrap_attempt_id()
    settings = _resolve_settings(None)
    environment_profile = build_local_workspace_environment_profile(settings)
    activate_local_workspace_reference_production_authority(
        settings,
        environment_profile=environment_profile,
    )
    bootstrap_identity = BootstrapIdentitySnapshot(
        bootstrap_attempt_id=bootstrap_attempt_id,
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role=_BACKGROUND_WORKER_PROCESS_ROLE,
    )
    producer = build_production_equivalent_bootstrap_failure_producer(
        extra_reporters=extra_reporters,
    )
    run_guarded_hosted_bootstrap_segment(
        producer=producer,
        readiness_at_failure=BootstrapReadinessLevel.B3_TENANT_BINDING,
        stage=HostedProcessBootstrapPhase.DEPENDENCY_RESOLUTION,
        identity=bootstrap_identity,
        surface_kind=BootstrapSurfaceKind.WORKER_BACKGROUND,
        segment=diagnostics_segment,
    )
