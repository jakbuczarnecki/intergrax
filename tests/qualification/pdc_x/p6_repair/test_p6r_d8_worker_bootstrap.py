# © Artur Czarnecki. All rights reserved.

"""B-P6-R item 8 - the worker root registers the artifact retention sweep and builds the governed read port."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.nexus.artifacts.governed_access import GovernedArtifactReadPort
from intergrax.runtime.nexus.artifacts.retention import ARTIFACT_RETENTION_SWEEP_TASK_NAME
from intergrax.runtime.persistence.sqlite_opens import open_artifact_store_at
from tests.qualification.pdc_x.d8._d8_support import make_governance, reader_grants

pytest.importorskip("celery", reason="worker_bootstrap imports the optional celery extra")

from intergrax.runtime.task.worker_bootstrap import (  # noqa: E402
    WorkerArtifactServices,
    build_nexus_task_execution_registry,
    compose_worker_artifact_services,
)

pytestmark = pytest.mark.unit


def test_compose_registers_the_retention_sweep_on_the_existing_registry(tmp_path: Path) -> None:
    registry = TaskExecutionRegistry()
    services = compose_worker_artifact_services(registry, artifact_store=open_artifact_store_at(tmp_path / "a.db"))
    assert isinstance(services, WorkerArtifactServices)
    assert services.read_port is None  # no governance supplied: no read surface is invented
    assert callable(services.sweep_handler)
    assert callable(registry.get_handler(ARTIFACT_RETENTION_SWEEP_TASK_NAME))


def test_compose_builds_the_governed_read_port_when_governance_is_supplied(tmp_path: Path) -> None:
    harness = make_governance(granted=reader_grants())
    services = compose_worker_artifact_services(
        TaskExecutionRegistry(),
        artifact_store=open_artifact_store_at(tmp_path / "a.db"),
        governance=harness.boundary,
        audit=harness.recorder,
    )
    assert isinstance(services.read_port, GovernedArtifactReadPort)


def test_compose_refuses_half_a_governance_pair(tmp_path: Path) -> None:
    harness = make_governance(granted=reader_grants())
    with pytest.raises(ValueError):
        compose_worker_artifact_services(
            TaskExecutionRegistry(),
            artifact_store=open_artifact_store_at(tmp_path / "a.db"),
            governance=harness.boundary,
        )


def test_the_worker_registry_builder_wires_the_sweep_only_when_a_store_is_supplied(tmp_path: Path) -> None:
    seen: list[WorkerArtifactServices] = []
    with_store = build_nexus_task_execution_registry(
        MagicMock(),
        host_execution=MagicMock(),
        production_mode=False,
        artifact_store=open_artifact_store_at(tmp_path / "a.db"),
        on_artifact_services=seen.append,
    )
    assert callable(with_store.get_handler(ARTIFACT_RETENTION_SWEEP_TASK_NAME))
    assert len(seen) == 1

    without_store = build_nexus_task_execution_registry(MagicMock(), host_execution=MagicMock(), production_mode=False)
    with pytest.raises(ValueError):
        without_store.get_handler(ARTIFACT_RETENTION_SWEEP_TASK_NAME)
