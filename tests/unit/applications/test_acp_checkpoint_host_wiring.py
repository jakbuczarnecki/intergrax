# © Artur Czarnecki. All rights reserved.

"""ACP-CLOSE-PROD-1/2: agent checkpoint store on harness product hosts."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.agents.persistence.checkpoint_store import (
    InMemoryAgentCheckpointStore,
    SQLiteAgentCheckpointStore,
)
from intergrax.applications._shared.acp_checkpoint_host_wiring import (
    resolve_agent_checkpoint_db_path,
    resolve_host_agent_checkpoint_store,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.task_control_wiring import build_reliability_task_enricher
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.runtime.task.task import Task
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resolve_agent_checkpoint_db_path_adjacent_to_task_checkpoints(tmp_path: Path) -> None:
    task_path = tmp_path / "checkpoints.db"
    assert resolve_agent_checkpoint_db_path(task_path) == tmp_path / "agent_checkpoints.db"


def test_resolve_host_agent_checkpoint_store_uses_explicit_store() -> None:
    store = InMemoryAgentCheckpointStore()
    resolved = resolve_host_agent_checkpoint_store(agent_checkpoint_store=store)
    assert resolved is store


def test_resolve_host_agent_checkpoint_store_opens_sqlite_when_path_known(tmp_path: Path) -> None:
    task_path = tmp_path / "checkpoints.db"
    store = resolve_host_agent_checkpoint_store(checkpoints_db_path=task_path)
    assert isinstance(store, SQLiteAgentCheckpointStore)


def test_build_harness_host_runtime_exposes_agent_checkpoint_store() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    assert env is not None
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    assert runtime.agent_checkpoint_store is not None


def test_build_reliability_task_enricher_injects_checkpoint_store() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = manifest.environment
    assert env is not None
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    enricher = build_reliability_task_enricher(
        env,
        agent_checkpoint_store=runtime.agent_checkpoint_store,
    )
    task = Task(
        task_id="task-acp-1",
        tenant_id="tenant-a",
        user_id="user-1",
        agent_id="echo",
        message="hello",
        metadata={},
    )
    enriched = enricher(task)
    assert enriched.metadata.get(AcpMetadataKey.CHECKPOINT_STORE) is runtime.agent_checkpoint_store
