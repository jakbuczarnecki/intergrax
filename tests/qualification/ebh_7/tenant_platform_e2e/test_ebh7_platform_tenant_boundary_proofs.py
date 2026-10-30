# © Artur Czarnecki. All rights reserved.

"""EBH-7 platform tenant boundary proofs (production contracts, controlled fixtures)."""

from __future__ import annotations

import pytest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from tests.qualification.compat_x.test_compat_x_plugin_provider_r4_gates import (
    test_r4_probe_g_cross_tenant_evidence,
)
from tests.qualification.dist_scale_x.p2_task_load import assert_cross_tenant_queue_fail_closed
from tests.qualification.trace_x.test_trace_x_p4_model_context_attribution import (
    test_txp4r1_q20_cross_tenant_attribution_rejected,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_ten_e2e_10_distributed_queue_cross_tenant_fail_closed() -> None:
    store = InMemoryDocumentStore()
    queue = DocumentStoreTaskQueue(store)
    owner = "tenant-a"
    victim = "tenant-b"
    handle = queue.enqueue(
        TaskRequest(
            tenant_id=owner,
            run_id="run-ebh7-10",
            task_name="ebh7.platform.tenant.queue",
            payload=b"{}",
            idempotency_key="ebh7-ten-e2e-10",
        )
    )
    claimed = queue.claim_pending(limit=1)
    assert len(claimed) == 1
    queue.mark_succeeded(handle, output=b"{}")
    assert queue.get_status(handle) is TaskStatus.SUCCEEDED
    assert_cross_tenant_queue_fail_closed(queue, owner_tenant=owner, victim_tenant=victim)


def test_ebh7_ten_e2e_11_trace_attribution_cross_tenant_rejected(governed_execution) -> None:
    test_txp4r1_q20_cross_tenant_attribution_rejected(governed_execution)


def test_ebh7_ten_e2e_13_plugin_compat_cross_tenant_rejected() -> None:
    test_r4_probe_g_cross_tenant_evidence()


def test_ebh7_ten_e2e_20_compat_assess_cross_tenant_rejected() -> None:
    test_r4_probe_g_cross_tenant_evidence()
