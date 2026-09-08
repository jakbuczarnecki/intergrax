# © Artur Czarnecki. All rights reserved.

"""DG-001B R5-R1 — generic background worker construction seam guards."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.applications._shared.host_queue_execution_wiring import HostQueueExecutionDependencies
from intergrax.hosting import HostedProcessBootstrapContext
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.observability.causal_evidence_persistence import CausalEvidencePersistence
from local_workspace_application.host.background_worker_constructor import (
    BackgroundWorkerConstructor,
    BlockingBackgroundWorker,
    create_default_background_worker,
)
from local_workspace_application.host.background_worker_factory import (
    build_local_workspace_background_worker_wiring,
)
from local_workspace_application.host.background_worker_main import (
    _run_guarded_worker_bootstrap,
    activate_local_workspace_reference_production_authority,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from local_workspace_application.workspaces.document_store_factory import (
    resolve_lkw_runtime_document_store,
)
from scripts.proof.dg001b_r5_qualification_contracts import (
    ControlledFailingBackgroundWorkerConstructor,
    qualification_secret_sentinel,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PRODUCTION_GUARD_PATHS = (
    _REPO_ROOT / "applications/local_workspace_application/host/background_worker_constructor.py",
    _REPO_ROOT / "applications/local_workspace_application/host/background_worker_factory.py",
    _REPO_ROOT / "applications/local_workspace_application/host/background_worker_main.py",
    _REPO_ROOT / "applications/local_workspace_application/host/settings.py",
)
_FORBIDDEN_PRODUCTION_TOKENS = (
    "DG001B",
    "R5-SECRET",
    "SECRET-SENTINEL",
    "typed_bootstrap_exception",
    "WORKER_CONSTRUCTION_FAULT",
    "worker_construction_fault",
)
_TYPE_IGNORE_GUARD_PATHS = (
    _REPO_ROOT / "applications/local_workspace_application/host/background_worker_constructor.py",
    _REPO_ROOT / "applications/local_workspace_application/host/background_worker_factory.py",
    _REPO_ROOT / "applications/local_workspace_application/host/background_worker_main.py",
)
_FORBIDDEN_PRODUCTION_DYNAMIC_SYMBOLS = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "inspect",
        "importlib",
    },
)


@dataclass(frozen=True, slots=True)
class _CapturedWorkerDependencies:
    kv_store: DistributedKVStore
    execution_registry: TaskExecutionRegistry
    idempotency_store: IdempotencyStore | None
    causal_evidence_persistence: CausalEvidencePersistence


class _RecordingWorkerConstructor:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.captured: _CapturedWorkerDependencies | None = None

    def __call__(
        self,
        *,
        kv_store: DistributedKVStore,
        execution_registry: TaskExecutionRegistry,
        idempotency_store: IdempotencyStore | None,
        causal_evidence_persistence: CausalEvidencePersistence,
    ) -> BlockingBackgroundWorker:
        self.captured = _CapturedWorkerDependencies(
            kv_store=kv_store,
            execution_registry=execution_registry,
            idempotency_store=idempotency_store,
            causal_evidence_persistence=causal_evidence_persistence,
        )
        if self.fail:
            raise TypeError("create_kafka_worker composition failure")
        worker = MagicMock(spec=BlockingBackgroundWorker)
        worker.start = MagicMock()
        return worker


def _settings(monkeypatch: pytest.MonkeyPatch) -> LocalWorkspaceBackendSettings:
    monkeypatch.setenv("INTERGRAX_HARNESS_API_KEY", "test-lkw-worker-constructor-seam-key")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_KAFKA_MESSAGE_BUS", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_ENABLE_REDIS", "true")
    monkeypatch.setenv(
        "INTERGRAX_PROBLEM_LIST_CURSOR_SECRET",
        "unit-test-lkw-worker-constructor-seam-cursor-secret",
    )
    return LocalWorkspaceBackendSettings.from_env()


def test_production_worker_seam_sources_have_no_qualification_tokens() -> None:
    violations: list[str] = []
    for path in _PRODUCTION_GUARD_PATHS:
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for token in _FORBIDDEN_PRODUCTION_TOKENS:
            if token in source:
                violations.append(f"{rel} references forbidden token {token}")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in _FORBIDDEN_PRODUCTION_DYNAMIC_SYMBOLS:
                violations.append(f"{rel}:{node.lineno} references {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_PRODUCTION_DYNAMIC_SYMBOLS:
                violations.append(f"{rel}:{node.lineno} references .{node.attr}")
    for path in _TYPE_IGNORE_GUARD_PATHS:
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if "type: ignore" in source:
            violations.append(f"{rel} references forbidden token type: ignore")
    assert violations == []


def _queue_dependencies() -> HostQueueExecutionDependencies:
    return HostQueueExecutionDependencies(
        kv_store=MagicMock(spec=DistributedKVStore),
        causal_evidence_persistence=MagicMock(spec=CausalEvidencePersistence),
    )


def test_settings_have_no_worker_construction_fault_field() -> None:
    fields = {field.name for field in LocalWorkspaceBackendSettings.__dataclass_fields__.values()}
    assert "worker_construction_fault" not in fields


def test_default_constructor_delegates_to_create_kafka_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    environment_profile = build_local_workspace_environment_profile(settings)
    _, projection = activate_local_workspace_reference_production_authority(
        settings,
        environment_profile=environment_profile,
    )
    kafka_worker = MagicMock(spec=BlockingBackgroundWorker)
    kafka_worker.start = MagicMock()

    with (
        patch(
            "local_workspace_application.host.background_worker_factory.resolve_host_queue_execution_dependencies",
            return_value=_queue_dependencies(),
        ),
        patch(
            "local_workspace_application.host.background_worker_constructor.create_kafka_worker",
            return_value=kafka_worker,
        ) as create_worker,
    ):
        wiring = build_local_workspace_background_worker_wiring(
            manifest=LOCAL_WORKSPACE_APPLICATION_MANIFEST,
            registry_projection=projection,
            settings=settings,
            document_store=resolve_lkw_runtime_document_store(settings),
        )

    create_worker.assert_called_once()
    assert wiring.worker is kafka_worker


def test_injected_constructor_receives_canonical_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    environment_profile = build_local_workspace_environment_profile(settings)
    _, projection = activate_local_workspace_reference_production_authority(
        settings,
        environment_profile=environment_profile,
    )
    recording = _RecordingWorkerConstructor()

    with patch(
        "local_workspace_application.host.background_worker_factory.resolve_host_queue_execution_dependencies",
        return_value=_queue_dependencies(),
    ):
        wiring = build_local_workspace_background_worker_wiring(
            manifest=LOCAL_WORKSPACE_APPLICATION_MANIFEST,
            registry_projection=projection,
            settings=settings,
            document_store=resolve_lkw_runtime_document_store(settings),
            worker_constructor=recording,
        )

    assert recording.captured is not None
    assert recording.captured.kv_store is wiring.kv_store
    assert recording.captured.execution_registry is wiring.registry
    assert recording.captured.idempotency_store is wiring.idempotency_store
    assert recording.captured.causal_evidence_persistence is not None


@pytest.mark.asyncio
async def test_injected_constructor_failure_occurs_inside_b6_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    environment_profile = build_local_workspace_environment_profile(settings)
    _, projection = activate_local_workspace_reference_production_authority(
        settings,
        environment_profile=environment_profile,
    )
    bootstrap_context = HostedProcessBootstrapContext.create(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        process_role="background_worker",
    )
    publisher = MagicMock()
    publisher.publish = MagicMock(return_value=None)

    with patch(
        "local_workspace_application.host.background_worker_factory.resolve_host_queue_execution_dependencies",
        return_value=_queue_dependencies(),
    ):
        with pytest.raises(TypeError, match="create_kafka_worker composition failure"):
            await _run_guarded_worker_bootstrap(
                bootstrap_context=bootstrap_context,
                event_publisher=publisher,
                settings=settings,
                registry_projection=projection,
                document_store=resolve_lkw_runtime_document_store(settings),
                worker_constructor=_RecordingWorkerConstructor(fail=True),
            )


def test_controlled_failing_constructor_preserves_original_type_error() -> None:
    constructor = ControlledFailingBackgroundWorkerConstructor()
    with pytest.raises(TypeError) as exc_info:
        constructor(
            kv_store=MagicMock(spec=DistributedKVStore),
            execution_registry=MagicMock(spec=TaskExecutionRegistry),
            idempotency_store=None,
            causal_evidence_persistence=MagicMock(spec=CausalEvidencePersistence),
        )
    assert qualification_secret_sentinel() in str(exc_info.value)
    assert type(exc_info.value) is TypeError


def test_create_default_background_worker_rejects_invalid_worker_type() -> None:
    with patch(
        "local_workspace_application.host.background_worker_constructor.create_kafka_worker",
        return_value=object(),
    ):
        with pytest.raises(TypeError, match="create_kafka_worker returned an invalid worker type"):
            create_default_background_worker(
                kv_store=MagicMock(spec=DistributedKVStore),
                execution_registry=MagicMock(spec=TaskExecutionRegistry),
                idempotency_store=None,
                causal_evidence_persistence=MagicMock(spec=CausalEvidencePersistence),
            )


def test_background_worker_constructor_protocol_is_type_safe() -> None:
    constructor: BackgroundWorkerConstructor = create_default_background_worker
    assert callable(constructor)
