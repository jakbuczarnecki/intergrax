# © Artur Czarnecki. All rights reserved.

"""P2-E01: OD-02 representative host trace (contract equivalence, no Kafka R replay)."""

from __future__ import annotations

import inspect

import pytest

from intergrax.queueing.providers.document_store.colocated_worker import DocumentStoreTaskWorker
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from local_workspace_application.host import background_worker_factory
from local_workspace_application.host import background_worker_main
from tests.qualification.dist_scale_x import p2_multiprocess_worker

pytestmark = pytest.mark.qualification


def test_p2_e01_lock_record_authorizes_document_store_worker_harness() -> None:
    from pathlib import Path

    lock = Path(__file__).resolve().parents[3] / (
        "docs/project/maintainers/qualification/DIST_SCALE_X_THRESHOLD_LOCK_RECORD.md"
    )
    text = lock.read_text(encoding="utf-8")
    assert "DocumentStoreTaskWorker" in text
    assert "Kafka" in text and "excluded" in text.lower()


def test_p2_e01_pg_worker_uses_platform_colocated_consumer_stack() -> None:
    source = inspect.getsource(p2_multiprocess_worker._build_worker)
    assert "DocumentStoreTaskWorker" in source
    assert "wire_background_execution_admission_dependencies" in source


def test_p2_e01_lkw_composition_root_exposes_host_task_execution_port() -> None:
    assert hasattr(HostTaskExecutionPort, "execute")
    assert hasattr(background_worker_main, "main")
    factory_source = inspect.getsource(background_worker_factory.build_local_workspace_background_worker_wiring)
    assert "host_execution" in factory_source
    assert "_BackgroundIngestHostExecutionRunner" in factory_source


def test_p2_e01_od02_broker_split_documented() -> None:
    """P2 mandatory M uses Rabbit qual worker; LKW binary entry is Kafka R (excluded @ P2)."""
    from tests.qualification.dist_scale_x import p2_rabbit_multiprocess_worker

    assert "DocumentStoreTaskWorker" in inspect.getsource(p2_multiprocess_worker._build_worker)
    assert "create_rabbitmq_worker" in inspect.getsource(p2_rabbit_multiprocess_worker.poll_loop)
