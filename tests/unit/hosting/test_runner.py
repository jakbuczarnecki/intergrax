# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from intergrax.hosting import (
    HostedApplicationProfile,
    HostedApplicationSupervisorResult,
    InstancePolicy,
    RestartPolicy,
    resolve_hosted_application_definition,
    run_hosted_application,
)
from intergrax.hosting.contracts.context import (
    HostedApplicationClock,
    HostedApplicationContext,
    HostedApplicationPaths,
    HostedApplicationProcessIdentity,
)
from intergrax.hosting.contracts.policies import InstanceExclusivityMode, RestartMode
from intergrax.hosting.engine.definition import HostedApplicationDefinition
from intergrax.hosting.engine.engine import HostedApplicationEngine
from intergrax.hosting.engine.ports import HostedApplicationInstanceGuardPort, HostedApplicationRuntime
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.errors import HostedApplicationConfigurationError, HostedApplicationDefinitionError
from intergrax.hosting.instance.file_guard import FileHostedApplicationInstanceGuard
from intergrax.hosting.runner import (
    _NonExclusiveInstanceGuard,
    _RunnerFactories,
    _run_resolved_hosted_application,
)
from intergrax.hosting.signals import HostedApplicationSignalBridge
from intergrax.hosting.supervisor.classification import HostedApplicationExitKind
from intergrax.hosting.shutdown import SystemMonotonicClock
from tests.unit.hosting.engine._fakes import (
    FakeInstanceGuard,
    FakeMonotonicClock,
    FixedClock,
    NoopLogger,
    RecordingPublisher,
    build_engine_paths,
)

pytestmark = pytest.mark.unit

_FORBIDDEN_RUNNER_FRAGMENTS = (
    "intergrax.runtime.task",
    "TaskResult",
    "NexusLoop",
    "intergrax.agents",
    "intergrax.tools",
    "intergrax.skills",
    "fastapi",
    "uvicorn",
    "local_workspace_application",
    "HarnessApplication",
)


class _RecordingSignalAdapter:
    def __init__(self) -> None:
        self.install_count = 0
        self.restore_count = 0
        self.raise_on_run: Exception | None = None

    def install(self) -> None:
        self.install_count += 1

    def restore(self) -> None:
        self.restore_count += 1


class _ShutdownOnStartRuntime:
    async def start(self, context: HostedApplicationContext) -> None:
        context.shutdown.request_shutdown("test.complete")

    async def stop(self, context: HostedApplicationContext) -> None:
        return None

    async def ready(self, context: HostedApplicationContext) -> bool:
        return True


def _fast_restart_policy(*, max_attempts: int = 2) -> RestartPolicy:
    return RestartPolicy(
        mode=RestartMode.ON_FAILURE,
        max_attempts=max_attempts,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.001,
        jitter_ratio=0.0,
    )


def _profile_with_runtime(
    runtime_factory: Callable[[], HostedApplicationRuntime],
    *,
    restart: RestartPolicy | None = None,
    instance_policy: InstancePolicy | None = None,
) -> HostedApplicationProfile:
    return HostedApplicationProfile(
        application_id="runner_test_app",
        application_factory=runtime_factory,
        application_factory_id="tests.unit.hosting.test_runner._profile_with_runtime",
        restart=restart or RestartPolicy.never(),
        instance=instance_policy or InstancePolicy(exclusivity_mode=InstanceExclusivityMode.MULTI_INSTANCE),
    )


def _recording_factories(
    tmp_path: Path,
    *,
    signal_adapter: _RecordingSignalAdapter | None = None,
    instance_id_generator: Callable[[], str] | None = None,
) -> tuple[_RunnerFactories, dict[str, Any]]:
    shared_clock = FixedClock()
    shared_monotonic = FakeMonotonicClock()
    shared_publisher = RecordingPublisher()
    shared_control_holder: dict[str, Any] = {}
    guards: list[HostedApplicationInstanceGuardPort] = []
    engines: list[HostedApplicationEngine] = []
    events: list[str] = []

    def create_paths(definition: HostedApplicationDefinition) -> HostedApplicationPaths:
        events.append("paths")
        return HostedApplicationPaths(
            data_home=(tmp_path / "data" / definition.application_id).resolve(),
            run_directory=(tmp_path / "run").resolve(),
        )

    def create_instance_guard(
        definition: HostedApplicationDefinition,
        paths: HostedApplicationPaths,
        process_identity: HostedApplicationProcessIdentity,
        clock: HostedApplicationClock,
    ) -> HostedApplicationInstanceGuardPort:
        events.append("guard")
        from intergrax.hosting.runner import _create_instance_guard

        guard = _create_instance_guard(definition, paths, process_identity, clock)
        guards.append(guard)
        return guard

    factories = _RunnerFactories(
        create_paths=create_paths,
        create_clock=lambda: shared_clock,
        create_monotonic_clock=lambda: shared_monotonic,
        create_logger=lambda _application_id: NoopLogger(),
        create_event_publisher=lambda: shared_publisher,
        create_process_identity=lambda clock: HostedApplicationProcessIdentity(
            process_id=4242,
            started_at=clock.now(),
        ),
        create_instance_guard=create_instance_guard,
        create_signal_adapter=lambda control: (
            signal_adapter
            if signal_adapter is not None
            else _RecordingSignalAdapter()
        ),
        instance_id_generator=instance_id_generator,
    )
    shared_control_holder.update(
        {
            "clock": shared_clock,
            "monotonic": shared_monotonic,
            "publisher": shared_publisher,
            "guards": guards,
            "engines": engines,
            "events": events,
        }
    )
    return factories, shared_control_holder


def test_public_api_import_and_signature() -> None:
    assert callable(run_hosted_application)
    signature = inspect.signature(run_hosted_application)
    assert "profile" in signature.parameters
    annotation = signature.return_annotation
    assert annotation in {HostedApplicationSupervisorResult, "HostedApplicationSupervisorResult"}
    assert not hasattr(__import__("intergrax.hosting", fromlist=["runner"]), "run_hosted_application_async")
    import intergrax.hosting as hosting_module

    assert "run_hosted_application" in hosting_module.__all__


def test_runner_source_import_boundaries() -> None:
    import intergrax.hosting.runner as runner_module

    source = inspect.getsource(runner_module)
    lowered = source.lower()
    for fragment in _FORBIDDEN_RUNNER_FRAGMENTS:
        assert fragment.lower() not in lowered, fragment


@pytest.mark.asyncio
async def test_definition_resolution_before_side_effects(tmp_path: Path) -> None:
    signal_adapter = _RecordingSignalAdapter()
    factories, holder = _recording_factories(tmp_path, signal_adapter=signal_adapter)
    profile = _profile_with_runtime(lambda: _ShutdownOnStartRuntime())  # type: ignore[return-value]
    definition = resolve_hosted_application_definition(profile)
    await _run_resolved_hosted_application(definition, factories)
    assert holder["events"][0] == "paths"
    assert holder["events"].count("guard") == 1
    assert signal_adapter.install_count == 1
    assert signal_adapter.restore_count == 1


def test_invalid_profile_creates_no_directories_or_signals() -> None:
    events: list[str] = []

    def _fail_paths(definition: HostedApplicationDefinition) -> HostedApplicationPaths:
        events.append("paths")
        return build_engine_paths()

    def _fail_signals(control: Any) -> HostedApplicationSignalBridge:
        events.append("signals")
        return _RecordingSignalAdapter()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "intergrax.hosting.runner._default_runner_factories",
        lambda: _RunnerFactories(
            create_paths=_fail_paths,
            create_clock=FixedClock,
            create_monotonic_clock=FakeMonotonicClock,
            create_logger=lambda _application_id: NoopLogger(),
            create_event_publisher=RecordingPublisher,
            create_process_identity=lambda clock: HostedApplicationProcessIdentity(
                process_id=1,
                started_at=clock.now(),
            ),
            create_instance_guard=lambda *_args, **_kwargs: FakeInstanceGuard(),
            create_signal_adapter=_fail_signals,
        ),
    )
    try:
        from intergrax.hosting import HostedApplicationComponentRegistration
        from tests.unit.hosting.engine._fakes import FakeComponent

        profile = HostedApplicationProfile(
            application_id="runner_test_app",
            application_factory=lambda: _ShutdownOnStartRuntime(),  # type: ignore[return-value]
            application_factory_id="tests.unit.hosting.test_runner.invalid",
            components=(
                HostedApplicationComponentRegistration(
                    component=FakeComponent("worker"),
                    dependencies=("missing_parent",),
                ),
            ),
        )

        with pytest.raises(HostedApplicationDefinitionError):
            run_hosted_application(profile)

        assert events == []
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_active_event_loop_rejects_without_signal_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_calls: list[str] = []

    def _recording_signal(control: Any) -> HostedApplicationSignalBridge:
        adapter = _RecordingSignalAdapter()
        original_install = adapter.install

        def _install() -> None:
            install_calls.append("install")
            original_install()

        adapter.install = _install  # type: ignore[method-assign]
        return adapter

    monkeypatch.setattr(
        "intergrax.hosting.runner._default_runner_factories",
        lambda: _RunnerFactories(
            create_paths=lambda definition: HostedApplicationPaths(
                data_home=(tmp_path / "data").resolve(),
                run_directory=(tmp_path / "run").resolve(),
            ),
            create_clock=FixedClock,
            create_monotonic_clock=FakeMonotonicClock,
            create_logger=lambda _application_id: NoopLogger(),
            create_event_publisher=RecordingPublisher,
            create_process_identity=lambda clock: HostedApplicationProcessIdentity(
                process_id=1,
                started_at=clock.now(),
            ),
            create_instance_guard=lambda *_args, **_kwargs: FakeInstanceGuard(),
            create_signal_adapter=_recording_signal,
        ),
    )

    profile = _profile_with_runtime(lambda: _ShutdownOnStartRuntime())  # type: ignore[return-value]

    with pytest.raises(HostedApplicationConfigurationError, match="active event loop"):
        run_hosted_application(profile)

    assert install_calls == []


def test_shared_dependencies_and_fresh_attempt_resources(tmp_path: Path) -> None:
    class _SequenceInstanceIds:
        def __init__(self, values: list[str]) -> None:
            self._values = iter(values)

        def __call__(self) -> str:
            return next(self._values)

    shared_clock = FixedClock()
    shared_monotonic = SystemMonotonicClock()
    shared_publisher = RecordingPublisher()
    controls: list[Any] = []
    guards: list[HostedApplicationInstanceGuardPort] = []
    engines: list[HostedApplicationEngine] = []
    original_engine_cls = HostedApplicationEngine
    attempt = {"n": 0}

    def _tracking_engine(*args: Any, **kwargs: Any) -> HostedApplicationEngine:
        engine = original_engine_cls(*args, **kwargs)
        engines.append(engine)
        return engine

    def create_signal_adapter(control: Any) -> HostedApplicationSignalBridge:
        controls.append(control)
        return _RecordingSignalAdapter()

    def create_instance_guard(
        definition: HostedApplicationDefinition,
        paths: HostedApplicationPaths,
        process_identity: HostedApplicationProcessIdentity,
        clock: HostedApplicationClock,
    ) -> HostedApplicationInstanceGuardPort:
        from intergrax.hosting.runner import _create_instance_guard

        guard = _create_instance_guard(definition, paths, process_identity, clock)
        guards.append(guard)
        return guard

    factories = _RunnerFactories(
        create_paths=lambda definition: HostedApplicationPaths(
            data_home=(tmp_path / "data" / definition.application_id).resolve(),
            run_directory=(tmp_path / "run").resolve(),
        ),
        create_clock=lambda: shared_clock,
        create_monotonic_clock=lambda: shared_monotonic,
        create_logger=lambda _application_id: NoopLogger(),
        create_event_publisher=lambda: shared_publisher,
        create_process_identity=lambda clock: HostedApplicationProcessIdentity(
            process_id=4242,
            started_at=clock.now(),
        ),
        create_instance_guard=create_instance_guard,
        create_signal_adapter=create_signal_adapter,
        instance_id_generator=_SequenceInstanceIds(["instance-001", "instance-002"]),
    )

    from tests.unit.hosting.engine._fakes import FakeRuntime

    def _runtime_factory() -> HostedApplicationRuntime:
        n = attempt["n"]
        attempt["n"] += 1
        if n == 0:
            return FakeRuntime(fail_start=True)  # type: ignore[return-value]
        return _ShutdownOnStartRuntime()  # type: ignore[return-value]

    profile = _profile_with_runtime(
        _runtime_factory,
        restart=_fast_restart_policy(max_attempts=2),
    )
    definition = resolve_hosted_application_definition(profile)

    import intergrax.hosting.runner as runner_module

    original_engine_cls_module = runner_module.HostedApplicationEngine
    runner_module.HostedApplicationEngine = _tracking_engine  # type: ignore[misc,assignment]
    try:
        result = asyncio.run(_run_resolved_hosted_application(definition, factories))
    finally:
        runner_module.HostedApplicationEngine = original_engine_cls_module  # type: ignore[misc,assignment]

    assert len(result.attempts) == 2
    assert len(controls) == 1
    assert len(guards) == 2
    assert guards[0] is not guards[1]
    assert len(engines) == 2
    assert engines[0] is not engines[1]
    assert engines[0].clock is shared_clock
    assert engines[1].clock is shared_clock
    assert engines[0].process_identity == engines[1].process_identity
    assert engines[0]._monotonic_clock is shared_monotonic
    assert engines[1]._monotonic_clock is shared_monotonic
    assert engines[0].event_publisher is shared_publisher
    assert engines[1].event_publisher is shared_publisher
    assert engines[0].shutdown is engines[1].shutdown is controls[0]


@pytest.mark.asyncio
async def test_signal_install_restore_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    signal_adapter = _RecordingSignalAdapter()
    factories, _holder = _recording_factories(tmp_path, signal_adapter=signal_adapter)
    profile = _profile_with_runtime(lambda: _ShutdownOnStartRuntime())  # type: ignore[return-value]
    definition = resolve_hosted_application_definition(profile)

    await _run_resolved_hosted_application(definition, factories)
    assert signal_adapter.install_count == 1
    assert signal_adapter.restore_count == 1

    from intergrax.hosting.supervisor.supervisor import HostedApplicationSupervisor

    async def _failing_run(self: HostedApplicationSupervisor) -> HostedApplicationSupervisorResult:
        raise RuntimeError("supervisor failed")

    monkeypatch.setattr(HostedApplicationSupervisor, "run", _failing_run)

    with pytest.raises(RuntimeError, match="supervisor failed"):
        await _run_resolved_hosted_application(definition, factories)

    assert signal_adapter.install_count == 2
    assert signal_adapter.restore_count == 2


def test_minimal_foreground_clean_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.hosting.runner._default_runner_factories",
        lambda: _RunnerFactories(
            create_paths=lambda definition: HostedApplicationPaths(
                data_home=(tmp_path / "data" / definition.application_id).resolve(),
                run_directory=(tmp_path / "run").resolve(),
            ),
            create_clock=FixedClock,
            create_monotonic_clock=FakeMonotonicClock,
            create_logger=lambda _application_id: NoopLogger(),
            create_event_publisher=RecordingPublisher,
            create_process_identity=lambda clock: HostedApplicationProcessIdentity(
                process_id=5150,
                started_at=clock.now(),
            ),
            create_instance_guard=lambda definition, paths, process_identity, clock: _NonExclusiveInstanceGuard(
                clock=clock
            ),
            create_signal_adapter=lambda control: _RecordingSignalAdapter(),
            instance_id_generator=(lambda: "instance-001"),
        ),
    )
    profile = _profile_with_runtime(lambda: _ShutdownOnStartRuntime())  # type: ignore[return-value]
    expected_digest = resolve_hosted_application_definition(profile).profile_digest

    result = run_hosted_application(profile)

    assert isinstance(result, HostedApplicationSupervisorResult)
    assert len(result.attempts) == 1
    assert result.final_exit.exit_kind is HostedApplicationExitKind.CLEAN_STOP
    assert result.profile_digest == expected_digest
    assert result.definition_digest
    terminal = result.attempts[0].terminal_result
    assert terminal is not None
    assert terminal.diagnostics.instance_lease_released is True
    assert terminal.diagnostics.context_closed is True


def test_restart_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import time

    from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState
    from tests.unit.hosting.engine._fakes import FakeRuntime

    class _SequenceInstanceIds:
        def __init__(self, values: list[str]) -> None:
            self._values = iter(values)

        def __call__(self) -> str:
            return next(self._values)

    controls: list[HostedApplicationControlCoordinator] = []
    engines: list[HostedApplicationEngine] = []
    original_engine_cls = HostedApplicationEngine
    original_control_cls = HostedApplicationControlCoordinator

    def _tracking_engine(*args: Any, **kwargs: Any) -> HostedApplicationEngine:
        engine = original_engine_cls(*args, **kwargs)
        engines.append(engine)
        return engine

    def _tracking_control(*args: Any, **kwargs: Any) -> Any:
        control = original_control_cls(*args, **kwargs)
        controls.append(control)
        return control

    monkeypatch.setattr("intergrax.hosting.runner.HostedApplicationEngine", _tracking_engine)
    monkeypatch.setattr("intergrax.hosting.runner.HostedApplicationControlCoordinator", _tracking_control)
    monkeypatch.setattr(
        "intergrax.hosting.runner._default_runner_factories",
        lambda: _RunnerFactories(
            create_paths=lambda definition: HostedApplicationPaths(
                data_home=(tmp_path / "data" / definition.application_id).resolve(),
                run_directory=(tmp_path / "run").resolve(),
            ),
            create_clock=FixedClock,
            create_monotonic_clock=SystemMonotonicClock,
            create_logger=lambda _application_id: NoopLogger(),
            create_event_publisher=RecordingPublisher,
            create_process_identity=lambda clock: HostedApplicationProcessIdentity(
                process_id=5150,
                started_at=clock.now(),
            ),
            create_instance_guard=lambda definition, paths, process_identity, clock: _NonExclusiveInstanceGuard(
                clock=clock
            ),
            create_signal_adapter=lambda control: _RecordingSignalAdapter(),
            instance_id_generator=_SequenceInstanceIds(["instance-001", "instance-002"]),
        ),
    )

    profile = _profile_with_runtime(
        lambda: FakeRuntime(),  # type: ignore[return-value]
        restart=_fast_restart_policy(max_attempts=2),
    )
    expected = resolve_hosted_application_definition(profile)
    result_holder: dict[str, HostedApplicationSupervisorResult] = {}

    def _run() -> None:
        result_holder["result"] = run_hosted_application(profile)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    deadline = time.time() + 5.0
    restarted = False
    while time.time() < deadline:
        if controls and engines:
            state = engines[-1].lifecycle_snapshot().state
            if not restarted and len(engines) == 1 and state is HostedApplicationLifecycleState.READY:
                controls[0].request_restart("test.restart")
                restarted = True
            elif restarted and len(engines) == 2 and state is HostedApplicationLifecycleState.READY:
                controls[0].request_shutdown("test.complete")
                break
        time.sleep(0.01)
    else:
        pytest.fail("restart lifecycle did not complete")

    thread.join(timeout=5.0)
    assert not thread.is_alive()
    result = result_holder["result"]

    assert len(result.attempts) == 2
    assert result.attempts[0].instance_id != result.attempts[1].instance_id
    assert result.attempts[0].exit_record is not None
    assert result.attempts[0].exit_record.exit_kind is HostedApplicationExitKind.RESTART_REQUESTED
    assert result.attempts[1].exit_record is not None
    assert result.attempts[1].exit_record.exit_kind is HostedApplicationExitKind.CLEAN_STOP
    assert result.profile_digest == expected.profile_digest
    assert result.definition_digest == expected.definition_digest


def test_instance_guard_selection_single_and_multi(tmp_path: Path) -> None:
    from intergrax.hosting.runner import _create_instance_guard

    clock = FixedClock()
    run_directory = tmp_path / "run"
    run_directory.mkdir(parents=True)
    paths = HostedApplicationPaths(
        data_home=(tmp_path / "data" / "app").resolve(),
        run_directory=run_directory.resolve(),
    )
    process_identity = HostedApplicationProcessIdentity(process_id=1, started_at=clock.now())
    single_definition = resolve_hosted_application_definition(
        _profile_with_runtime(
            lambda: _ShutdownOnStartRuntime(),  # type: ignore[return-value]
            instance_policy=InstancePolicy.standard(),
        )
    )
    multi_definition = resolve_hosted_application_definition(
        _profile_with_runtime(
            lambda: _ShutdownOnStartRuntime(),  # type: ignore[return-value]
            instance_policy=InstancePolicy(exclusivity_mode=InstanceExclusivityMode.MULTI_INSTANCE),
        )
    )

    single_guard = _create_instance_guard(single_definition, paths, process_identity, clock)
    multi_guard = _create_instance_guard(multi_definition, paths, process_identity, clock)

    assert isinstance(single_guard, FileHostedApplicationInstanceGuard)
    assert isinstance(multi_guard, _NonExclusiveInstanceGuard)
