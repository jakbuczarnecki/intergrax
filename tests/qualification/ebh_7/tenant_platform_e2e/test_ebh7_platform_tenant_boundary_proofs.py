# © Artur Czarnecki. All rights reserved.

"""EBH-7 platform tenant boundary proofs (production contracts, controlled fixtures)."""

from __future__ import annotations

from pathlib import Path

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
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_plugin_host_compat_cross_tenant_activation_denied,
    )

    assert_plugin_host_compat_cross_tenant_activation_denied()


def test_ebh7_ten_e2e_20_compat_assess_cross_tenant_rejected() -> None:
    test_r4_probe_g_cross_tenant_evidence()


def test_ebh7_ten_e2e_03_gov_exec_cross_tenant_denied() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_gov_exec_cross_tenant_projection_denied,
    )

    assert_gov_exec_cross_tenant_projection_denied()


def test_ebh7_ten_e2e_04_exec_tool_tenant_boundary() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_exec_tool_cross_tenant_invocation_denied,
    )

    assert_exec_tool_cross_tenant_invocation_denied()


@pytest.mark.asyncio
async def test_ebh7_ten_e2e_04_exec_tool_tenant_boundary_tool_runtime() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_exec_tool_cross_tenant_via_tool_runtime,
    )

    await assert_exec_tool_cross_tenant_via_tool_runtime()


def test_ebh7_ten_e2e_05_config_provider_cross_tenant() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_config_provider_cross_tenant_rejected,
    )

    assert_config_provider_cross_tenant_rejected()


def test_ebh7_ten_e2e_12_skill_compose_cross_tenant() -> None:
    from tests.unit.skills.test_skill_x_batch_e_integrated_authority_tenant_qualification import (
        _pinning_backends,
        test_ec8_cross_tenant_read_returns_none,
    )

    for _name, store in _pinning_backends():
        test_ec8_cross_tenant_read_returns_none(_name, store)


def test_ebh7_ten_e2e_14_backend_switch_preserves_tenant(tmp_path: Path) -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_backend_switch_preserves_tenant_binding,
    )

    assert_backend_switch_preserves_tenant_binding(tmp_path)


def test_ebh7_ten_e2e_17_obs_tenant_isolate() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_obs_tenant_configuration_isolated,
    )

    assert_obs_tenant_configuration_isolated()


def test_ebh7_ten_e2e_18_hitl_resume_tenant() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_hitl_resume_tenant_widen_denied,
    )

    assert_hitl_resume_tenant_widen_denied()


def test_ebh7_ten_e2e_19_child_exec_cross_tenant() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_parent_child_exec_cross_tenant_denied,
    )

    assert_parent_child_exec_cross_tenant_denied()


def test_ebh7_ten_e2e_21_control_plane_cross_tenant() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_control_plane_cross_tenant_mutation_denied,
    )

    assert_control_plane_cross_tenant_mutation_denied()


def test_ebh7_ten_e2e_22_scheduler_tenant_drift() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_scheduler_worker_tenant_drift_denied,
    )

    assert_scheduler_worker_tenant_drift_denied()


def test_ebh7_ten_e2e_23_trace_replay_cross_tenant() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
        assert_trace_replay_cross_tenant_denied,
    )

    assert_trace_replay_cross_tenant_denied()
