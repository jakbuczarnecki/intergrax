# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.hosting import (
    BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE,
    HostedApplicationEventType,
    HostedApplicationLifecycleState,
    HostedProcessBootstrapContext,
    HostedProcessBootstrapPhase,
    run_guarded_hosted_process_bootstrap,
)
from intergrax.hosting.contracts.context import HostedApplicationEventPublisher
from intergrax.hosting.contracts.events import HostedApplicationEvent

pytestmark = pytest.mark.unit


class _RecordingPublisher(HostedApplicationEventPublisher):
    def __init__(self) -> None:
        self.events: list[HostedApplicationEvent] = []

    async def publish(self, event: HostedApplicationEvent) -> None:
        self.events.append(event)


class _FailingPublisher(HostedApplicationEventPublisher):
    async def publish(self, event: HostedApplicationEvent) -> None:
        raise RuntimeError("publisher failed")


def _context(
    *, process_role: str = "background_worker"
) -> HostedProcessBootstrapContext:
    return HostedProcessBootstrapContext.create(
        application_id="my_application",
        process_role=process_role,
    )


@pytest.mark.asyncio
async def test_success_returns_exact_callback_result() -> None:
    publisher = _RecordingPublisher()
    context = _context()
    result = await run_guarded_hosted_process_bootstrap(
        context=context,
        phase=HostedProcessBootstrapPhase.COMPOSITION,
        event_publisher=publisher,
        bootstrap=lambda: {"ready": True},
    )
    assert result == {"ready": True}


@pytest.mark.asyncio
async def test_success_emits_no_failure_event() -> None:
    publisher = _RecordingPublisher()
    context = _context()
    await run_guarded_hosted_process_bootstrap(
        context=context,
        phase=HostedProcessBootstrapPhase.STARTUP,
        event_publisher=publisher,
        bootstrap=lambda: None,
    )
    assert publisher.events == []


@pytest.mark.asyncio
async def test_failure_emits_exactly_one_application_failed_event() -> None:
    publisher = _RecordingPublisher()
    context = _context()
    with pytest.raises(ValueError):
        await run_guarded_hosted_process_bootstrap(
            context=context,
            phase=HostedProcessBootstrapPhase.WORKER_CONSTRUCTION,
            event_publisher=publisher,
            bootstrap=lambda: (_ for _ in ()).throw(
                ValueError("secret dependency token")
            ),
        )
    assert len(publisher.events) == 1
    assert (
        publisher.events[0].event_type is HostedApplicationEventType.APPLICATION_FAILED
    )


@pytest.mark.asyncio
async def test_failure_preserves_application_and_instance_identity() -> None:
    publisher = _RecordingPublisher()
    context = _context()
    with pytest.raises(RuntimeError):
        await run_guarded_hosted_process_bootstrap(
            context=context,
            phase=HostedProcessBootstrapPhase.COMPOSITION,
            event_publisher=publisher,
            bootstrap=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    event = publisher.events[0]
    assert event.application_id == context.application_id
    assert event.instance_id == context.instance_id


@pytest.mark.asyncio
async def test_failure_preserves_process_role_and_phase() -> None:
    publisher = _RecordingPublisher()
    context = _context(process_role="sidecar_daemon")
    with pytest.raises(TypeError):
        await run_guarded_hosted_process_bootstrap(
            context=context,
            phase=HostedProcessBootstrapPhase.DEPENDENCY_RESOLUTION,
            event_publisher=publisher,
            bootstrap=lambda: (_ for _ in ()).throw(TypeError("missing dependency")),
        )
    payload = publisher.events[0].payload
    assert payload["process_role"] == "sidecar_daemon"
    assert payload["phase"] == HostedProcessBootstrapPhase.DEPENDENCY_RESOLUTION.value


@pytest.mark.asyncio
async def test_failure_reason_code_and_exception_type_bounded() -> None:
    publisher = _RecordingPublisher()
    context = _context()
    with pytest.raises(KeyError):
        await run_guarded_hosted_process_bootstrap(
            context=context,
            phase=HostedProcessBootstrapPhase.CONFIGURATION,
            event_publisher=publisher,
            bootstrap=lambda: (_ for _ in ()).throw(KeyError("secret-key")),
        )
    payload = publisher.events[0].payload
    assert payload["reason_code"] == BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE
    assert payload["exception_type"] == "KeyError"


@pytest.mark.asyncio
async def test_failure_payload_excludes_raw_exception_message() -> None:
    publisher = _RecordingPublisher()
    context = _context()
    secret = "super-secret-bootstrap-message"
    with pytest.raises(ValueError):
        await run_guarded_hosted_process_bootstrap(
            context=context,
            phase=HostedProcessBootstrapPhase.STARTUP,
            event_publisher=publisher,
            bootstrap=lambda: (_ for _ in ()).throw(ValueError(secret)),
        )
    serialized = publisher.events[0].model_dump_json()
    assert secret not in serialized


@pytest.mark.asyncio
async def test_failure_re_raises_original_exception_object() -> None:
    publisher = _RecordingPublisher()
    context = _context()
    original = ValueError("classified failure")
    with pytest.raises(ValueError) as caught:
        await run_guarded_hosted_process_bootstrap(
            context=context,
            phase=HostedProcessBootstrapPhase.COMPOSITION,
            event_publisher=publisher,
            bootstrap=lambda: (_ for _ in ()).throw(original),
        )
    assert caught.value is original


@pytest.mark.asyncio
async def test_publisher_failure_does_not_replace_original_bootstrap_exception() -> (
    None
):
    publisher = _FailingPublisher()
    context = _context()
    original = RuntimeError("bootstrap failed")
    with pytest.raises(RuntimeError) as caught:
        await run_guarded_hosted_process_bootstrap(
            context=context,
            phase=HostedProcessBootstrapPhase.STARTUP,
            event_publisher=publisher,
            bootstrap=lambda: (_ for _ in ()).throw(original),
        )
    assert caught.value is original


@pytest.mark.asyncio
async def test_callback_invoked_exactly_once() -> None:
    publisher = _RecordingPublisher()
    context = _context()
    calls = 0

    def _bootstrap() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await run_guarded_hosted_process_bootstrap(
        context=context,
        phase=HostedProcessBootstrapPhase.COMPOSITION,
        event_publisher=publisher,
        bootstrap=_bootstrap,
    )
    assert result == "ok"
    assert calls == 1


@pytest.mark.asyncio
async def test_system_exit_not_converted_to_application_failed() -> None:
    publisher = _RecordingPublisher()
    context = _context()
    with pytest.raises(SystemExit):
        await run_guarded_hosted_process_bootstrap(
            context=context,
            phase=HostedProcessBootstrapPhase.STARTUP,
            event_publisher=publisher,
            bootstrap=lambda: (_ for _ in ()).throw(SystemExit(1)),
        )
    assert publisher.events == []


@pytest.mark.asyncio
async def test_keyboard_interrupt_not_converted_to_application_failed() -> None:
    publisher = _RecordingPublisher()
    context = _context()
    with pytest.raises(KeyboardInterrupt):
        await run_guarded_hosted_process_bootstrap(
            context=context,
            phase=HostedProcessBootstrapPhase.STARTUP,
            event_publisher=publisher,
            bootstrap=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    assert publisher.events == []


@pytest.mark.parametrize(
    ("application_id", "process_role"),
    [
        ("", "worker"),
        ("   ", "worker"),
        ("1invalid", "worker"),
        ("my_application", ""),
        ("my_application", "   "),
    ],
)
def test_invalid_context_validation(application_id: str, process_role: str) -> None:
    with pytest.raises(ValueError):
        HostedProcessBootstrapContext.create(
            application_id=application_id,
            process_role=process_role,
        )


def test_invalid_instance_id_rejected_on_direct_construction() -> None:
    with pytest.raises(ValueError, match="instance_id"):
        HostedProcessBootstrapContext(
            application_id="my_application",
            instance_id="   ",
            process_role="worker",
        )


@pytest.mark.asyncio
async def test_failure_event_lifecycle_state_failed() -> None:
    publisher = _RecordingPublisher()
    context = _context()
    with pytest.raises(OSError):
        await run_guarded_hosted_process_bootstrap(
            context=context,
            phase=HostedProcessBootstrapPhase.CONFIGURATION,
            event_publisher=publisher,
            bootstrap=lambda: (_ for _ in ()).throw(OSError("ignored")),
        )
    event = publisher.events[0]
    assert event.lifecycle_state is HostedApplicationLifecycleState.FAILED


def test_process_bootstrap_module_has_no_execution_identity_imports() -> None:
    module_path = (
        Path(__file__).resolve().parents[3]
        / "intergrax"
        / "hosting"
        / "process_bootstrap.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    forbidden = {"TaskId", "RunId", "AttemptId", "ExecutionId"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            violations.append(node.id)
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            violations.append(node.attr)
    assert violations == []


def test_process_bootstrap_module_has_no_diagnostics_or_queue_imports() -> None:
    module_path = (
        Path(__file__).resolve().parents[3]
        / "intergrax"
        / "hosting"
        / "process_bootstrap.py"
    )
    source = module_path.read_text(encoding="utf-8").lower()
    forbidden_fragments = (
        "intergrax.runtime.diagnostics",
        "diagnosticorchestrator",
        "kafka",
        "celery",
        "rabbitmq",
        "message_bus",
        "local_workspace_application",
        "execution_identity",
    )
    for fragment in forbidden_fragments:
        assert fragment not in source, fragment
