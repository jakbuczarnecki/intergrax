# © Artur Czarnecki. All rights reserved.

"""HARDEN-4C — product-host clean execution / no false-positive Problem E2E."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.observability.operator_wiring import ObservabilityExportOperatorConfig
from tests.integration.runtime.diag_final_otel_support import (
    assert_runtime_event_truth,
    build_diag_final_product_host,
    build_read_service,
    execute_host_run,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_TENANT = "harden-4c-clean"


@pytest.fixture
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def _assert_clean_runtime_event_sequence(
    composition,
    *,
    tenant_id: str,
    run_id: str,
) -> None:
    store = composition.runtime.observability.runtime_event_store
    assert store is not None
    events = store.list_for_run(run_id, tenant_id=tenant_id)
    assert events, f"expected RuntimeEvents for run_id={run_id!r}"

    completed_indexes = [
        index
        for index, event in enumerate(events)
        if event.event_type is RuntimeEventType.TASK_COMPLETED
    ]
    assert completed_indexes, "expected TASK_COMPLETED in run event sequence"

    violation_types = {RuntimeEventType.RETRY_SCHEDULED}
    for event in events:
        assert event.event_type not in violation_types, (
            f"clean run must not contain violation event {event.event_type.value!r}"
        )

    first_completed_index = completed_indexes[0]
    for event in events[first_completed_index + 1 :]:
        assert event.event_type not in violation_types, (
            "known diag-final violation pattern: event after TASK_COMPLETED"
        )


def test_harden_4c_clean_product_host_execution_creates_no_problem(
    tmp_path: Path,
    _stub_host_llm: None,
) -> None:
    document_store = InMemoryDocumentStore()
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)
    export_config = ObservabilityExportOperatorConfig(enabled=False)

    composition = build_diag_final_product_host(
        tmp_path=storage_root,
        document_store=document_store,
        observability_export=export_config,
        tenant_id=_TENANT,
        inject_violation=False,
    )
    assert composition.runtime.diagnostic_wiring.attached is True

    run = execute_host_run(
        composition,
        tenant_id=_TENANT,
        message="harden-4c clean execution",
    )
    run_id = str(run["run_id"])
    task_id = str(run["task_id"])
    assert task_id

    terminal_event = assert_runtime_event_truth(
        composition,
        tenant_id=_TENANT,
        run_id=run_id,
        task_id=task_id,
    )
    _assert_clean_runtime_event_sequence(
        composition,
        tenant_id=_TENANT,
        run_id=run_id,
    )

    read_service = build_read_service(composition)
    listed = read_service.list_problems(tenant_id=_TENANT)
    assert listed.total_count == 0
    assert listed.problems == ()
    assert listed.returned_count == 0
    assert listed.is_truncated is False

    assert str(terminal_event.run_id) == run_id
    assert str(terminal_event.task_id) == task_id
    assert str(terminal_event.attempt_id)
    assert str(terminal_event.execution_id)
