# © Artur Czarnecki. All rights reserved.

"""APP-HOST-8B — hosted LKW readiness bridge tests."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from intergrax.hosting.contracts.components import (
    HostedApplicationComponentHealth,
    HostedApplicationComponentState,
)
from intergrax.hosting.contracts.context import (
    HostedApplicationContext,
    HostedApplicationPaths,
    HostedApplicationProcessIdentity,
)
from intergrax.hosting.contracts.events import HostedApplicationEvent
from intergrax.hosting.contracts.lifecycle import (
    HostedApplicationLifecycleSnapshot,
    HostedApplicationLifecycleState,
    HostedApplicationShutdownCoordinator,
)
from intergrax.hosting.engine.health import (
    HostedApplicationHealthSnapshot,
    HostedApplicationReadinessService,
)
from intergrax.hosting.services import HostedApplicationServiceRegistry
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.hosting import build_local_workspace_hosted_profile
from local_workspace_application.hosting.readiness import _HostedLocalWorkspaceReadiness

pytestmark = pytest.mark.unit


class _Clock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class _Logger:
    def debug(self, message: str, **fields: object) -> None:
        del message, fields

    def info(self, message: str, **fields: object) -> None:
        del message, fields

    def warning(self, message: str, **fields: object) -> None:
        del message, fields

    def error(self, message: str, **fields: object) -> None:
        del message, fields


class _EventPublisher:
    async def publish(self, event: HostedApplicationEvent) -> None:
        del event


class _Shutdown:
    def request_shutdown(self, *, reason_code: str = "test") -> None:
        del reason_code

    def is_shutdown_requested(self) -> bool:
        return False

    async def wait_until_requested(self) -> None:
        return None


class _MutableLifecycleProvider:
    def __init__(self) -> None:
        self.snapshot_value = HostedApplicationLifecycleSnapshot(
            state=HostedApplicationLifecycleState.STARTING,
            accepting_new_work=False,
            shutdown_requested=False,
            last_transition_at=datetime.now(timezone.utc),
        )

    def snapshot(self) -> HostedApplicationLifecycleSnapshot:
        return self.snapshot_value


class _MutableReadinessService:
    def __init__(self) -> None:
        self.snapshot_value = HostedApplicationHealthSnapshot(
            live=True,
            ready=False,
            degraded=False,
            accepting_new_work=False,
            runtime_ready=False,
            instance_ownership_valid=True,
            shutdown_requested=False,
            last_evaluated_at=datetime.now(timezone.utc),
        )

    def snapshot(self) -> HostedApplicationHealthSnapshot:
        return self.snapshot_value

    def accepts_new_work(self) -> bool:
        return self.snapshot_value.accepting_new_work


def _make_context(
    lifecycle: _MutableLifecycleProvider,
    readiness: _MutableReadinessService,
) -> HostedApplicationContext:
    profile = build_local_workspace_hosted_profile(
        settings=LocalWorkspaceBackendSettings(),
    )
    services = HostedApplicationServiceRegistry()
    services.register(HostedApplicationReadinessService, readiness)
    return HostedApplicationContext(
        application_id=profile.application_id,
        instance_id="01TESTHOSTEDREADINESSINSTANCE001",
        profile=profile.public_view(),
        profile_digest=profile.profile_digest(),
        paths=HostedApplicationPaths(
            data_home=Path("build/test-lkw-hosting-8b-readiness"),
            run_directory=Path("build/test-lkw-hosting-8b-readiness/run"),
        ),
        process_identity=HostedApplicationProcessIdentity(
            process_id=1,
            started_at=datetime.now(timezone.utc),
        ),
        services=services,
        clock=_Clock(),
        logger=_Logger(),
        event_publisher=_EventPublisher(),
        shutdown=cast(HostedApplicationShutdownCoordinator, _Shutdown()),
        lifecycle=lifecycle,
    )


def _set_lifecycle(
    provider: _MutableLifecycleProvider,
    *,
    state: HostedApplicationLifecycleState,
    accepting_new_work: bool,
    shutdown_requested: bool = False,
) -> None:
    provider.snapshot_value = HostedApplicationLifecycleSnapshot(
        state=state,
        accepting_new_work=accepting_new_work,
        shutdown_requested=shutdown_requested,
        last_transition_at=datetime.now(timezone.utc),
    )


def _set_health(
    service: _MutableReadinessService,
    *,
    ready: bool = False,
    accepting_new_work: bool = False,
    runtime_ready: bool = True,
    instance_ownership_valid: bool = True,
    shutdown_requested: bool = False,
    health_evaluation_failed: bool = False,
    blocking_component_ids: tuple[str, ...] = (),
    component_snapshots: tuple[HostedApplicationComponentHealth, ...] = (),
) -> None:
    service.snapshot_value = HostedApplicationHealthSnapshot(
        live=True,
        ready=ready,
        degraded=False,
        accepting_new_work=accepting_new_work,
        runtime_ready=runtime_ready,
        instance_ownership_valid=instance_ownership_valid,
        shutdown_requested=shutdown_requested,
        blocking_component_ids=blocking_component_ids,
        component_snapshots=component_snapshots,
        health_evaluation_failed=health_evaluation_failed,
        last_evaluated_at=datetime.now(timezone.utc),
    )


@pytest.mark.parametrize(
    ("state", "expected_state", "expected_detail", "expected_error"),
    [
        (
            HostedApplicationLifecycleState.CREATED,
            "starting",
            "host_state=starting",
            "lkw_host_not_ready",
        ),
        (
            HostedApplicationLifecycleState.STARTING,
            "starting",
            "host_state=starting",
            "lkw_host_not_ready",
        ),
        (
            HostedApplicationLifecycleState.STOPPING,
            "stopping",
            "host_state=stopping",
            "lkw_host_stopping",
        ),
        (
            HostedApplicationLifecycleState.STOPPED,
            "stopped",
            "host_state=stopped",
            "lkw_host_not_ready",
        ),
        (
            HostedApplicationLifecycleState.FAILED,
            "failed",
            "host_state=failed",
            "lkw_host_not_ready",
        ),
    ],
)
def test_hosted_readiness_maps_lifecycle_states(
    state: HostedApplicationLifecycleState,
    expected_state: str,
    expected_detail: str,
    expected_error: str,
) -> None:
    lifecycle = _MutableLifecycleProvider()
    readiness = _MutableReadinessService()
    _set_lifecycle(lifecycle, state=state, accepting_new_work=False)
    _set_health(readiness, ready=False, accepting_new_work=False)
    bridge = _HostedLocalWorkspaceReadiness(_make_context(lifecycle, readiness))
    snapshot = bridge.readiness_snapshot()
    assert snapshot.state == expected_state
    assert snapshot.ready is False
    assert snapshot.accepts_new_work is False
    assert snapshot.detail == expected_detail
    assert snapshot.rejection_error_id == expected_error


def test_hosted_readiness_ready_accepts_work() -> None:
    lifecycle = _MutableLifecycleProvider()
    readiness = _MutableReadinessService()
    _set_lifecycle(
        lifecycle,
        state=HostedApplicationLifecycleState.READY,
        accepting_new_work=True,
    )
    _set_health(readiness, ready=True, accepting_new_work=True, runtime_ready=True)
    bridge = _HostedLocalWorkspaceReadiness(_make_context(lifecycle, readiness))
    snapshot = bridge.readiness_snapshot()
    assert snapshot.state == "ready"
    assert snapshot.ready is True
    assert snapshot.accepts_new_work is True
    assert snapshot.detail == "ready"
    assert snapshot.rejection_error_id == ""


@pytest.mark.parametrize(
    ("health_kwargs", "expected_detail"),
    [
        ({"runtime_ready": False}, "runtime_not_ready"),
        ({"instance_ownership_valid": False}, "instance_ownership_invalid"),
        ({"shutdown_requested": True}, "shutdown_requested"),
        ({"health_evaluation_failed": True}, "health_evaluation_failed"),
        ({"blocking_component_ids": ("comp_b",)}, "blocking_components"),
    ],
)
def test_hosted_readiness_rejects_on_health_factors(
    health_kwargs: dict[str, object],
    expected_detail: str,
) -> None:
    lifecycle = _MutableLifecycleProvider()
    readiness = _MutableReadinessService()
    _set_lifecycle(
        lifecycle,
        state=HostedApplicationLifecycleState.READY,
        accepting_new_work=True,
        shutdown_requested=bool(health_kwargs.get("shutdown_requested")),
    )
    kwargs: dict[str, object] = {
        "ready": False,
        "accepting_new_work": False,
        "runtime_ready": True,
        "instance_ownership_valid": True,
        "shutdown_requested": False,
        "health_evaluation_failed": False,
        "blocking_component_ids": (),
    }
    kwargs.update(health_kwargs)
    _set_health(readiness, **kwargs)  # type: ignore[arg-type]
    bridge = _HostedLocalWorkspaceReadiness(_make_context(lifecycle, readiness))
    snapshot = bridge.readiness_snapshot()
    assert snapshot.ready is False
    assert snapshot.accepts_new_work is False
    assert snapshot.detail == expected_detail
    if health_kwargs.get("shutdown_requested"):
        assert snapshot.rejection_error_id == "lkw_host_stopping"
    else:
        assert snapshot.rejection_error_id == "lkw_host_not_ready"


def test_hosted_readiness_projects_components_deterministically() -> None:
    lifecycle = _MutableLifecycleProvider()
    readiness = _MutableReadinessService()
    _set_lifecycle(
        lifecycle,
        state=HostedApplicationLifecycleState.READY,
        accepting_new_work=True,
    )
    now = datetime.now(timezone.utc)
    _set_health(
        readiness,
        ready=True,
        accepting_new_work=True,
        component_snapshots=(
            HostedApplicationComponentHealth(
                component_id="comp_z",
                enabled=True,
                required=False,
                state=HostedApplicationComponentState.READY,
                healthy=True,
                ready=True,
                detail_code="fallback_code",
                safe_message="",
                last_check_at=now,
            ),
            HostedApplicationComponentHealth(
                component_id="comp_a",
                enabled=True,
                required=True,
                state=HostedApplicationComponentState.READY,
                healthy=True,
                ready=True,
                detail_code="ignored_code",
                safe_message="safe detail",
                last_check_at=now,
            ),
        ),
    )
    bridge = _HostedLocalWorkspaceReadiness(_make_context(lifecycle, readiness))
    snapshot = bridge.readiness_snapshot()
    assert [component.name for component in snapshot.components] == ["comp_a", "comp_z"]
    assert snapshot.components[0].enabled is True
    assert snapshot.components[0].required is True
    assert snapshot.components[0].healthy is True
    assert snapshot.components[0].detail == "safe detail"
    assert snapshot.components[1].detail == "fallback_code"


def test_hosted_readiness_import_boundary() -> None:
    path = Path(__file__).resolve().parents[2] / "hosting" / "readiness.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "HostedApplicationEngine",
        "HostedApplicationSupervisor",
        "HostedApplicationHealthCoordinator",
        "HostedApplicationLifecycleController",
        "FileHostedApplicationInstanceGuard",
        "PortableForegroundSignalAdapter",
        "HostedApplicationControlCoordinator",
        "run_hosted_application",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[-1] not in forbidden
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name not in forbidden
            if node.module:
                for name in forbidden:
                    assert name not in node.module
