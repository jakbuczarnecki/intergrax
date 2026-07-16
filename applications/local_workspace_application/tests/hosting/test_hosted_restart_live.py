# © Artur Czarnecki. All rights reserved.

"""APP-HOST-8D — real supervisor restart and request-after-restart proof."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from intergrax.hosting.contracts.context import HostedApplicationPaths
from intergrax.hosting.contracts.events import (
    HostedApplicationEvent,
    HostedApplicationEventType,
)
from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.engine.definition import resolve_hosted_application_definition
from intergrax.hosting.instance.contracts import (
    HostedApplicationInstanceIdentity,
    InstanceAcquisitionClassification,
)
from intergrax.hosting.instance.file_guard import FileHostedApplicationInstanceGuard
from intergrax.hosting.runner import (
    _default_runner_factories,
    _run_resolved_hosted_application,
)
from intergrax.hosting.shutdown import (
    HostedApplicationShutdownPhase,
    HostedApplicationShutdownPhaseOutcome,
)
from intergrax.hosting.supervisor.classification import HostedApplicationExitKind
from local_workspace_application.hosting.profile import (
    build_local_workspace_hosted_profile,
)

pytestmark = [pytest.mark.unit]

_STARTUP_DEADLINE_SECONDS = 60.0
_TOTAL_DEADLINE_SECONDS = 90.0
_POLL_INTERVAL_SECONDS = 0.1
_BOUNDARY_NAME = "local_workspace_hosting_boundary"

_INSTANCE_ID_0 = "00000000-0000-4000-8000-000000000001"
_INSTANCE_ID_1 = "00000000-0000-4000-8000-000000000002"
_INSTANCE_ID_2 = "00000000-0000-4000-8000-000000000003"


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _configure_lkw_env(
    tmp_path: Path, port: int, monkeypatch: pytest.MonkeyPatch
) -> Path:
    home = tmp_path / "home"
    data_home = tmp_path / "lkw-data"
    sqlite = tmp_path / "sqlite"
    shadow = tmp_path / "shadow"
    workspace = tmp_path / "workspace"
    hosting_data = tmp_path / "hosting-data"
    hosting_run = tmp_path / "hosting-run"
    for path in (home, data_home, sqlite, shadow, workspace, hosting_data, hosting_run):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("LKW_DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_SQLITE_DATA_DIR", str(sqlite))
    monkeypatch.setenv("INTERGRAX_SHADOW_ROOT", str(shadow))
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(workspace))
    monkeypatch.setenv("LOCAL_WORKSPACE_BACKEND_HOST", "127.0.0.1")
    monkeypatch.setenv("LOCAL_WORKSPACE_BACKEND_PORT", str(port))
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_RAG", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_RAG_INGEST", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_MCP", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_SCHEDULER", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_INTERACTIONS", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_TASK_CONTROL", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED", "false")
    return workspace


def _http_json(
    method: str, url: str, payload: dict[str, Any] | None = None
) -> tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
        return int(response.status), json.loads(body) if body else None


async def _wait_http_ready(port: int, *, deadline: float) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/v1/local_workspace/readiness"
    last_error = ""
    while time.monotonic() < deadline:
        try:
            status, body = await asyncio.to_thread(_http_json, "GET", url)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = str(exc)
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            continue
        if (
            status == 200
            and isinstance(body, dict)
            and body.get("ready") is True
            and body.get("accepts_new_work") is True
            and body.get("state") == "ready"
        ):
            return body
        last_error = f"status={status} body={body!r}"
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
    raise AssertionError(f"readiness not reached before deadline: {last_error}")


def _assert_boundary_component(readiness: dict[str, Any]) -> None:
    components = readiness.get("components")
    assert isinstance(components, list)
    matches = [item for item in components if item.get("name") == _BOUNDARY_NAME]
    assert len(matches) == 1, components
    component = matches[0]
    assert component.get("enabled") is True
    assert component.get("required") is True
    assert component.get("healthy") is True
    assert component.get("detail") == "before_ready hook completed"


def _assert_ordered_subsequence(
    events: list[HostedApplicationEvent],
    expected: list[HostedApplicationEventType],
    *,
    instance_id: str | None = None,
) -> None:
    index = 0
    for event in events:
        if instance_id is not None and event.instance_id != instance_id:
            continue
        if event.event_type is expected[index]:
            index += 1
            if index == len(expected):
                return
    raise AssertionError(
        f"ordered subsequence not found for instance_id={instance_id!r}: "
        f"expected={expected!r} seen={[e.event_type for e in events]}"
    )


@dataclass
class _RecordingEventPublisher:
    events: list[HostedApplicationEvent] = field(default_factory=list)
    _condition: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)

    async def publish(self, event: HostedApplicationEvent) -> None:
        async with self._condition:
            self.events.append(event)
            self._condition.notify_all()

    async def wait_for(
        self,
        predicate: Any,
        *,
        deadline: float,
        description: str,
    ) -> HostedApplicationEvent:
        while True:
            async with self._condition:
                for event in self.events:
                    if predicate(event):
                        return event
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"timed out waiting for {description}; "
                        f"events={[e.event_type.value for e in self.events]}"
                    )
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except TimeoutError as exc:
                    raise AssertionError(
                        f"timed out waiting for {description}; "
                        f"events={[e.event_type.value for e in self.events]}"
                    ) from exc


@dataclass
class _ControlHolder:
    control: HostedApplicationControlCoordinator | None = None


class _CapturingSignalBridge:
    def __init__(
        self, holder: _ControlHolder, control: HostedApplicationControlCoordinator
    ) -> None:
        self._holder = holder
        self._control = control

    def install(self) -> None:
        self._holder.control = self._control

    def restore(self) -> None:
        return None


@dataclass
class _SequenceInstanceIds:
    values: list[str]
    _index: int = 0

    def __call__(self) -> str:
        if self._index >= len(self.values):
            raise AssertionError("instance_id_generator exhausted")
        value = self.values[self._index]
        self._index += 1
        return value


def _assert_critical_shutdown_phases(diagnostics: Any) -> None:
    execution = diagnostics.shutdown_execution
    assert execution is not None
    assert execution.timed_out is False
    assert execution.forced is False
    by_phase = {record.phase: record.outcome for record in execution.phase_records}
    for phase in (
        HostedApplicationShutdownPhase.COMPONENT_STOP,
        HostedApplicationShutdownPhase.RUNTIME_STOP,
        HostedApplicationShutdownPhase.LEASE_RELEASE,
    ):
        assert by_phase.get(phase) is HostedApplicationShutdownPhaseOutcome.COMPLETED, (
            by_phase
        )


@pytest.mark.asyncio
async def test_hosted_lkw_restart_creates_new_instance_and_accepts_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    port = _reserve_free_port()
    workspace = _configure_lkw_env(tmp_path, port, monkeypatch)
    fixture_path = workspace / "hosted-restart-proof.txt"
    fixture_path.write_text(
        "APP-HOST-8D hosted restart proof fixture for local.workspace.index\n",
        encoding="utf-8",
    )

    hosting_data = (tmp_path / "hosting-data").resolve()
    hosting_run = (tmp_path / "hosting-run").resolve()
    paths = HostedApplicationPaths(data_home=hosting_data, run_directory=hosting_run)

    profile = build_local_workspace_hosted_profile()
    definition = resolve_hosted_application_definition(profile)
    profile_digest = profile.profile_digest()

    defaults = _default_runner_factories()
    clock = defaults.create_clock()
    process_identity = defaults.create_process_identity(clock)
    publisher = _RecordingEventPublisher()
    control_holder = _ControlHolder()
    instance_ids = _SequenceInstanceIds(
        [_INSTANCE_ID_0, _INSTANCE_ID_1, _INSTANCE_ID_2]
    )

    factories = dataclasses.replace(
        defaults,
        create_paths=lambda _definition: paths,
        create_clock=lambda: clock,
        create_process_identity=lambda _clock: process_identity,
        create_event_publisher=lambda: publisher,
        create_signal_adapter=lambda control: cast(
            Any,
            _CapturingSignalBridge(control_holder, control),
        ),
        instance_id_generator=instance_ids,
    )

    deadline = time.monotonic() + _TOTAL_DEADLINE_SECONDS
    supervisor_task = asyncio.create_task(
        _run_resolved_hosted_application(definition, factories)
    )
    try:
        while control_holder.control is None:
            if time.monotonic() >= deadline:
                raise AssertionError("control coordinator was not captured")
            if supervisor_task.done():
                raise AssertionError(
                    f"supervisor exited before control capture: {supervisor_task.result()!r}"
                )
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        control = control_holder.control
        assert control is not None

        first_ready = await publisher.wait_for(
            lambda event: (
                event.event_type is HostedApplicationEventType.APPLICATION_READY
                and event.instance_id == _INSTANCE_ID_0
                and event.lifecycle_state is HostedApplicationLifecycleState.READY
            ),
            deadline=min(deadline, time.monotonic() + _STARTUP_DEADLINE_SECONDS),
            description="first APPLICATION_READY",
        )
        assert first_ready.instance_id == _INSTANCE_ID_0

        readiness = await _wait_http_ready(
            port,
            deadline=min(deadline, time.monotonic() + _STARTUP_DEADLINE_SECONDS),
        )
        _assert_boundary_component(readiness)
        assert not supervisor_task.done()

        control.request_restart(
            "lkw.hosting_proof.restart",
            source_id="lkw_hosting_proof",
        )
        restart_effective = control.current_effective_request()
        assert restart_effective is not None
        assert restart_effective.intent == "restart"
        assert restart_effective.reason_code == "lkw.hosting_proof.restart"
        assert restart_effective.source_id == "lkw_hosting_proof"

        second_ready = await publisher.wait_for(
            lambda event: (
                event.event_type is HostedApplicationEventType.APPLICATION_READY
                and event.instance_id == _INSTANCE_ID_1
                and event.lifecycle_state is HostedApplicationLifecycleState.READY
            ),
            deadline=deadline,
            description="second APPLICATION_READY",
        )
        assert second_ready.instance_id != first_ready.instance_id

        _assert_ordered_subsequence(
            publisher.events,
            [
                HostedApplicationEventType.APPLICATION_STOPPING,
                HostedApplicationEventType.COMPONENT_STOPPING,
                HostedApplicationEventType.COMPONENT_STOPPED,
                HostedApplicationEventType.INSTANCE_RELEASED,
                HostedApplicationEventType.APPLICATION_STOPPED,
            ],
            instance_id=_INSTANCE_ID_0,
        )
        _assert_ordered_subsequence(
            publisher.events,
            [
                HostedApplicationEventType.RESTART_REQUESTED,
                HostedApplicationEventType.RESTART_SCHEDULED,
                HostedApplicationEventType.RESTART_STARTED,
            ],
        )
        for event_type in (
            HostedApplicationEventType.RESTART_REQUESTED,
            HostedApplicationEventType.RESTART_SCHEDULED,
            HostedApplicationEventType.RESTART_STARTED,
        ):
            matches = [
                event for event in publisher.events if event.event_type is event_type
            ]
            assert matches, event_type
            payload = matches[0].payload
            assert "attempt_number" in payload
            assert "exit_kind" in payload
            assert "reason_code" in payload
            assert "profile_digest" in payload

        _assert_ordered_subsequence(
            publisher.events,
            [
                HostedApplicationEventType.INSTANCE_ACQUIRED,
                HostedApplicationEventType.APPLICATION_STARTING,
                HostedApplicationEventType.APPLICATION_STARTED,
                HostedApplicationEventType.APPLICATION_READY,
            ],
            instance_id=_INSTANCE_ID_1,
        )

        second_readiness = await _wait_http_ready(port, deadline=deadline)
        _assert_boundary_component(second_readiness)

        status, body = await asyncio.to_thread(
            _http_json,
            "POST",
            f"http://127.0.0.1:{port}/v1/local_workspace/run",
            {
                "tenant_id": "tenant-hosting-restart-proof",
                "workspace_id": "workspace-hosting-restart-proof",
                "message": "index hosted restart proof fixture",
                "capability": "local.workspace.index",
                "metadata": {
                    "source_paths": [str(fixture_path.resolve())],
                    "collection_id": "workspace-hosting-restart-proof",
                },
            },
        )
        assert status == 200
        assert isinstance(body, dict)
        assert body.get("state") == "completed"
        metadata = body.get("metadata")
        assert isinstance(metadata, dict)
        summary = metadata.get("application_run_summary.v1")
        assert isinstance(summary, dict)
        assert "terminal_status" in summary

        control.request_shutdown(
            "lkw.hosting_proof.complete",
            source_id="lkw_hosting_proof",
        )
        shutdown_effective = control.current_effective_request()
        assert shutdown_effective is not None
        assert shutdown_effective.intent == "stop"
        assert shutdown_effective.reason_code == "lkw.hosting_proof.complete"
        assert shutdown_effective.source_id == "lkw_hosting_proof"

        remaining = max(0.1, deadline - time.monotonic())
        result = await asyncio.wait_for(supervisor_task, timeout=remaining)

        assert result.application_id == "local_workspace"
        assert result.profile_digest == definition.profile_digest
        assert result.definition_digest == definition.definition_digest
        assert result.restart_exhausted is False
        assert len(result.attempts) == 2

        attempt0 = result.attempts[0]
        assert attempt0.attempt_number == 0
        assert attempt0.instance_id == _INSTANCE_ID_0
        assert attempt0.cleanup_verified is True
        assert attempt0.cleanup_issue == ""
        assert attempt0.exit_record is not None
        assert (
            attempt0.exit_record.exit_kind
            is HostedApplicationExitKind.RESTART_REQUESTED
        )
        assert attempt0.exit_record.retryable is True
        assert attempt0.exit_record.reason_code == "lkw.hosting_proof.restart"
        assert (
            attempt0.exit_record.terminal_lifecycle_state
            is HostedApplicationLifecycleState.STOPPED
        )
        assert attempt0.exit_record.profile_digest == result.profile_digest
        assert attempt0.terminal_result is not None
        assert (
            attempt0.terminal_result.terminal_state
            is HostedApplicationLifecycleState.STOPPED
        )
        diagnostics0 = attempt0.terminal_result.diagnostics
        assert diagnostics0.instance_lease_acquired is True
        assert diagnostics0.instance_lease_released is True
        assert diagnostics0.context_closed is True
        _assert_critical_shutdown_phases(diagnostics0)

        attempt1 = result.attempts[1]
        assert attempt1.attempt_number == 1
        assert attempt1.instance_id == _INSTANCE_ID_1
        assert attempt1.cleanup_verified is True
        assert attempt1.cleanup_issue == ""
        assert attempt1.exit_record is not None
        assert attempt1.exit_record.exit_kind is HostedApplicationExitKind.CLEAN_STOP
        assert attempt1.exit_record.retryable is False
        assert attempt1.exit_record.reason_code == "lkw.hosting_proof.complete"
        assert (
            attempt1.exit_record.terminal_lifecycle_state
            is HostedApplicationLifecycleState.STOPPED
        )
        assert attempt1.exit_record.profile_digest == result.profile_digest
        assert attempt1.terminal_result is not None
        assert (
            attempt1.terminal_result.terminal_state
            is HostedApplicationLifecycleState.STOPPED
        )
        diagnostics1 = attempt1.terminal_result.diagnostics
        assert diagnostics1.instance_lease_acquired is True
        assert diagnostics1.instance_lease_released is True
        assert diagnostics1.context_closed is True
        _assert_critical_shutdown_phases(diagnostics1)

        assert result.final_exit == attempt1.exit_record
        assert result.final_exit.exit_kind is HostedApplicationExitKind.CLEAN_STOP

        assert (
            attempt0.exit_record.profile_digest == attempt1.exit_record.profile_digest
        )
        assert attempt0.exit_record.profile_digest == profile_digest
        assert attempt0.exit_record.profile_digest == definition.profile_digest
        assert result.profile_digest == profile_digest
        assert result.definition_digest == definition.definition_digest

        guard = FileHostedApplicationInstanceGuard(
            run_directory=paths.run_directory,
            instance_policy=definition.instance_policy,
            process_identity=process_identity,
            clock=clock,
        )
        reacquire_identity = HostedApplicationInstanceIdentity(
            application_id="local_workspace",
            instance_id=_INSTANCE_ID_2,
            profile_digest=definition.profile_digest,
            process_identity=process_identity,
        )
        acquisition = await guard.acquire(reacquire_identity)
        assert (
            acquisition.classification
            is not InstanceAcquisitionClassification.ACTIVE_OWNER
        )
        assert acquisition.classification in {
            InstanceAcquisitionClassification.FRESH,
            InstanceAcquisitionClassification.STALE_OWNER,
            InstanceAcquisitionClassification.CORRUPTED_METADATA,
        }
        assert acquisition.lease.is_valid()
        await acquisition.lease.release()

        record_property("hosting.restart_requested", "true")
        record_property("hosting.first_instance_id", attempt0.instance_id)
        record_property("hosting.second_instance_id", attempt1.instance_id)
        record_property("hosting.instance_id_changed", "true")
        record_property(
            "hosting.first_attempt_exit_kind",
            attempt0.exit_record.exit_kind.value,
        )
        record_property("hosting.first_attempt_cleanup_verified", "true")
        record_property("hosting.first_lease_released", "true")
        record_property("hosting.first_context_closed", "true")
        record_property("hosting.stopped_events_verified", "true")
        record_property("hosting.restart_events_verified", "true")
        record_property("hosting.second_instance_ready", "true")
        record_property("hosting.real_index_after_restart", "true")
        record_property("hosting.profile_digest", result.profile_digest)
        record_property("hosting.definition_digest", result.definition_digest)
        record_property("hosting.profile_digest_preserved", "true")
        record_property("hosting.definition_digest_preserved", "true")
        record_property(
            "hosting.final_exit_kind",
            result.final_exit.exit_kind.value,
        )
        record_property("hosting.final_cleanup_verified", "true")
        record_property("hosting.final_lease_released", "true")
        record_property("hosting.final_context_closed", "true")
        record_property("hosting.final_lock_reacquired", "true")
    finally:
        if not supervisor_task.done():
            if control_holder.control is not None:
                control_holder.control.request_shutdown(
                    "lkw.hosting_proof.cleanup",
                    source_id="lkw_hosting_proof",
                )
                try:
                    await asyncio.wait_for(supervisor_task, timeout=15)
                except (TimeoutError, Exception):
                    supervisor_task.cancel()
                    await asyncio.gather(supervisor_task, return_exceptions=True)
            else:
                supervisor_task.cancel()
                await asyncio.gather(supervisor_task, return_exceptions=True)
