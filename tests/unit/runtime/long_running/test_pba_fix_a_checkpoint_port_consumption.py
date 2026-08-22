# © Artur Czarnecki. All rights reserved.

"""PBA-FIX-A — generic runtime consumes TaskCheckpointPersistence ports, not SQLite."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
    ReliabilityProfile,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id
from intergrax.integrations.providers.relational_store.sqlite import (
    create_sqlite_task_checkpoint_store,
)
from intergrax.runtime.long_running.checkpoint_builder import build_task_checkpoint
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
    TaskCheckpointReader,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.orchestration.long_running_bridge import (
    maybe_checkpoint_long_running,
    maybe_restore_long_running,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.nexus_worker_execution import NexusWorkerRuntime
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GENERIC_RUNTIME_FILES = (
    _REPO_ROOT / "intergrax/runtime/nexus/nexus_loop.py",
    _REPO_ROOT / "intergrax/runtime/nexus/orchestration/long_running_bridge.py",
    _REPO_ROOT / "intergrax/runtime/long_running/coordinator.py",
    _REPO_ROOT / "intergrax/runtime/task/nexus_worker_execution.py",
    _REPO_ROOT / "intergrax/runtime/task/worker_bootstrap.py",
    _REPO_ROOT / "intergrax/applications/_shared/nexus_factory.py",
)


class _FakeCheckpointStore(TaskCheckpointPersistence):
    def __init__(self) -> None:
        self.saved: list[TaskCheckpoint] = []
        self._by_token: Dict[tuple[str, str, str], TaskCheckpoint] = {}
        self._latest: Dict[tuple[str, str], TaskCheckpoint] = {}

    def list_for_task(self, task_id: str, tenant_id: str) -> List[TaskCheckpoint]:
        return [
            checkpoint
            for checkpoint in self.saved
            if checkpoint.task_id == task_id and checkpoint.tenant_id == tenant_id
        ]

    def get_latest(self, task_id: str, tenant_id: str) -> Optional[TaskCheckpoint]:
        return self._latest.get((task_id, tenant_id))

    def get_by_token(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: str,
    ) -> Optional[TaskCheckpoint]:
        return self._by_token.get((task_id, tenant_id, resume_token))

    def list_paused(self) -> List[TaskCheckpoint]:
        return []

    def save(self, checkpoint: TaskCheckpoint) -> TaskCheckpoint:
        self.saved.append(checkpoint)
        self._latest[(checkpoint.task_id, checkpoint.tenant_id)] = checkpoint
        self._by_token[
            (checkpoint.task_id, checkpoint.tenant_id, checkpoint.resume_token)
        ] = checkpoint
        return checkpoint


def _long_running_task(*, resume_token: str | None = None) -> Task:
    return Task(
        tenant_id="t1",
        user_id="u1",
        message="long-running work",
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True,
                resume_token=resume_token,
            ),
        ),
    )


def test_a1_nexus_accepts_fake_checkpoint_port() -> None:
    fake = _FakeCheckpointStore()
    loop = NexusLoop(AgentRegistry(), checkpoint_store=fake)
    assert loop._checkpoint_store is fake  # noqa: SLF001


def test_a2_coordinator_persist_uses_fake_port(tmp_path: Path) -> None:
    fake = _FakeCheckpointStore()
    task = _long_running_task()
    checkpoint = LongRunningCoordinator.persist_checkpoint(
        task,
        fake,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        progress_message="step complete",
    )
    assert len(fake.saved) == 1
    assert fake.saved[0] is checkpoint
    assert checkpoint.progress_message == "step complete"
    assert "SQLiteTaskCheckpointStore" not in type(fake.saved[0]).__name__


def test_a3_coordinator_restore_with_fake_reader() -> None:
    fake = _FakeCheckpointStore()
    original = _long_running_task()
    checkpoint = LongRunningCoordinator.persist_checkpoint(
        original,
        fake,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        progress_message="awaiting human",
    )
    resume_task = Task(
        tenant_id="t1",
        user_id="u1",
        task_id=original.task_id,
        message="ignored until restore",
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True,
                resume_token=checkpoint.resume_token,
            ),
        ),
    )
    restored = LongRunningCoordinator.restore_if_resuming(resume_task, fake)
    assert restored is not None
    assert resume_task.message == "long-running work"
    assert isinstance(fake, TaskCheckpointReader)


def test_a4_build_task_checkpoint_is_provider_neutral() -> None:
    task = _long_running_task()
    checkpoint = build_task_checkpoint(task, progress_message="built")
    assert checkpoint.task_id == task.task_id
    assert checkpoint.progress_message == "built"
    source = inspect.getsource(build_task_checkpoint)
    assert "sqlite" not in source.lower()
    module_source = Path(
        inspect.getfile(build_task_checkpoint)  # type: ignore[arg-type]
    ).read_text(encoding="utf-8")
    assert "SQLiteTaskCheckpointStore" not in module_source


def test_a5_worker_runtime_accepts_fake_port() -> None:
    fake = _FakeCheckpointStore()
    runtime = NexusWorkerRuntime.from_registry(AgentRegistry(), checkpoint_store=fake)
    assert runtime.task_runner is not None


def test_a6_worker_bootstrap_contract_accepts_fake_port() -> None:
    bootstrap_path = _REPO_ROOT / "intergrax/runtime/task/worker_bootstrap.py"
    source = bootstrap_path.read_text(encoding="utf-8")
    assert "SQLiteTaskCheckpointStore" not in source
    assert "TaskCheckpointPersistence" in source

    celery = pytest.importorskip("celery", reason="celery optional for runtime bootstrap proof")
    del celery
    from intergrax.runtime.task.worker_bootstrap import (
        build_nexus_task_execution_registry,
        create_nexus_celery_worker_app,
    )

    fake = _FakeCheckpointStore()
    registry = build_nexus_task_execution_registry(
        AgentRegistry(),
        checkpoint_store=fake,
    )
    assert registry is not None
    app = create_nexus_celery_worker_app(
        app_name="pba-fix-a",
        broker_url="memory://",
        backend_url=None,
        agent_registry=AgentRegistry(),
        checkpoint_store=fake,
        task_always_eager=True,
    )
    assert app is not None


def test_a7_shared_nexus_factory_accepts_fake_port() -> None:
    fake = _FakeCheckpointStore()
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "reliability_profile": ReliabilityProfile(long_running_scheduler_enabled=True),
            "orchestration_profile": OrchestrationProfile(long_running_enabled=True),
        }
    )
    loop = build_nexus_loop_from_environment(
        AgentRegistry(),
        env=env,
        checkpoint_store=fake,
    )
    assert loop._checkpoint_store is fake  # noqa: SLF001


def test_r1_1_long_running_bridge_has_no_sqlite_checkpoint_import() -> None:
    bridge_path = _REPO_ROOT / "intergrax/runtime/nexus/orchestration/long_running_bridge.py"
    source = bridge_path.read_text(encoding="utf-8")
    assert "SQLiteTaskCheckpointStore" not in source
    assert "runtime.long_running.store" not in source
    assert "TaskCheckpointReader" in source
    assert "TaskCheckpointPersistence" in source


async def test_r1_2_bridge_restore_with_fake_reader() -> None:
    fake = _FakeCheckpointStore()
    original = _long_running_task()
    checkpoint = LongRunningCoordinator.persist_checkpoint(
        original,
        fake,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        progress_message="awaiting human",
    )
    resume_task = Task(
        tenant_id="t1",
        user_id="u1",
        task_id=original.task_id,
        message="ignored until restore",
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True,
                resume_token=checkpoint.resume_token,
            ),
        ),
    )
    published: list[RuntimeEvent] = []

    async def _publish(event: RuntimeEvent, *, task: Task | None = None) -> None:
        _ = task
        published.append(event)

    run_id = mint_run_id()
    stub_event = MagicMock()
    stub_event.model_copy.return_value = stub_event
    with patch(
        "intergrax.runtime.nexus.orchestration.long_running_bridge.runtime_event_from_task_state",
        return_value=stub_event,
    ):
        await maybe_restore_long_running(
            resume_task,
            checkpoint_store=fake,
            publish=_publish,
            notification_adapter=None,
            run_id=run_id,
        )
    assert resume_task.message == "long-running work"
    assert len(published) == 1
    assert isinstance(fake, TaskCheckpointReader)


async def test_r1_3_bridge_checkpoint_with_fake_persistence() -> None:
    fake = _FakeCheckpointStore()
    task = _long_running_task()
    published: list[RuntimeEvent] = []

    async def _publish(event: RuntimeEvent, *, task: Task | None = None) -> None:
        _ = task
        published.append(event)

    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    stub_event = MagicMock()
    stub_event.model_copy.return_value = stub_event
    with patch(
        "intergrax.runtime.nexus.orchestration.long_running_bridge.runtime_event_from_task_state",
        return_value=stub_event,
    ):
        await maybe_checkpoint_long_running(
            task,
            checkpoint_store=fake,
            publish=_publish,
            notification_adapter=None,
            progress_message="step complete",
            run_id=run_id,
            attempt_id=attempt_id,
        )
    assert len(fake.saved) == 1
    assert fake.saved[0].progress_message == "step complete"
    assert len(published) == 2
    assert isinstance(fake, TaskCheckpointPersistence)


def test_a8_generic_runtime_has_no_sqlite_checkpoint_import() -> None:
    banned = {"SQLiteTaskCheckpointStore", "open_task_checkpoint_store"}
    for path in _GENERIC_RUNTIME_FILES:
        source = path.read_text(encoding="utf-8")
        assert "SQLiteTaskCheckpointStore" not in source
        tree = ast.parse(source, filename=str(path))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
        leaked = banned & imported
        assert not leaked, f"{path}: leaked imports {leaked}"


def test_a9_sqlite_factory_satisfies_checkpoint_port(tmp_path: Path) -> None:
    store = create_sqlite_task_checkpoint_store(db_path=tmp_path / "ckpt.db")
    assert isinstance(store, TaskCheckpointPersistence)
