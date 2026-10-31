# © Artur Czarnecki. All rights reserved.

"""Cross-domain qualification helpers — execution via sanctioned proof runner (no direct test_* calls)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from intergrax.integrations.contracts.base import IntegrationCategory
from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import catalog_by_concern_id
from tests.qualification.ebh_5._package_c_v2_primitives import assert_unregistered_fail_closed
from tests.qualification.ebh_5._package_c_v2_proof_helpers import run_package_c_v2_dual_slug_for_entry
from tests.qualification.ebh_7._proof_execution import execute_proof
from tests.qualification.tenant_x._r3_adversarial import (
    assert_checkpoint_recovery_cross_tenant_rejects,
)
from tests.qualification.tenant_x._r4_adversarial import (
    assert_vendor_worker_payload_job_tenant_mismatch_rejects,
)


def _run(module: str, function: str, **kwargs: object) -> None:
    execute_proof(module, function, **kwargs)


def _run_async(module: str, function: str) -> None:
    from tests.qualification.ebh_7._proof_execution import load_proof_callable

    fn = load_proof_callable(module, function)
    asyncio.run(fn())


def qualify_tool_skill_plugin_shadow_denied() -> None:
    _run(
        "tests.qualification.tenant_x.test_tenant_x_r1_fast",
        "test_tx_r1_q12_adversarial_tool_scope_same_tenant",
    )
    _run(
        "tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        "test_ebh7_ten_e2e_12_skill_compose_cross_tenant",
    )
    _run(
        "tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary",
        "test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone",
    )


def qualify_trace_replay_tenant_lineage_fail_closed() -> None:
    _run(
        "tests.unit.applications.test_uca6c_marketplace_qualified_execution_composition",
        "test_tenant_mismatch_blocks_materialization_pin_and_io",
    )


def qualify_compat_historical_evidence_fail_closed() -> None:
    _run(
        "tests.qualification.tenant_x.test_tenant_x_p0_fast",
        "test_tx_p0_r1_q02_ctrl_x_historical_evidence_sha",
    )
    _run(
        "tests.qualification.tenant_x.test_tenant_x_p0_fast",
        "test_tx_p0_r1_q09_historical_evidence_requires_replay",
    )
    _run(
        "tests.qualification.compat_x.test_compat_x_plugin_provider_r4_gates",
        "test_r4_probe_g_cross_tenant_evidence",
    )


def qualify_failure_diagnostics_tenant_leak_denied() -> None:
    _run(
        "tests.qualification.trace_x.test_trace_x_p5_p0_qualification_gates",
        "test_txp5p0_q10_configuration_tenant_mismatch_fail_closed",
    )
    _run(
        "tests.qualification.trace_x.test_trace_x_p5_p0_qualification_gates",
        "test_txp5p0_q11_profile_binding_tenant_mismatch_fail_closed",
    )


def qualify_recovery_gov_restore_without_grant_denied() -> None:
    assert_checkpoint_recovery_cross_tenant_rejects()
    _run_async(
        "tests.qualification.governance.test_governance_e2e_decision",
        "test_scenario_l_stale_proposal_fail_closed",
    )


def qualify_llm_tenant_gov_sdk_block() -> None:
    _run(
        "tests.qualification.tenant_x.test_tenant_x_r4_fast",
        "test_tx_r4_q05_adversarial_context_llm_cross_tenant",
    )
    _run_async(
        "tests.qualification.governance.test_governance_e2e_decision",
        "test_scenario_z_cross_tenant_fail_closed",
    )


def qualify_worker_lease_tenant_denied() -> None:
    assert_vendor_worker_payload_job_tenant_mismatch_rejects()
    _run(
        "tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        "test_ebh7_ten_e2e_10_distributed_queue_cross_tenant_fail_closed",
    )


def qualify_hitl_trace_tenant_scope_denied() -> None:
    _run_async(
        "tests.qualification.governance.test_governance_e2e_decision",
        "test_scenario_m_decision_accepted_governance_deny_blocks_operation",
    )
    _run(
        "tests.qualification.trace_x.test_trace_x_p5_p0_qualification_gates",
        "test_txp5p0_q11_profile_binding_tenant_mismatch_fail_closed",
    )


def qualify_config_compat_tenant_fail_closed() -> None:
    _run(
        "tests.qualification.config_x.test_config_x_adversarial_cx_gates",
        "test_cx_g_cross_tenant_provider_config_rejected_by_realization_guards",
    )
    _run(
        "tests.qualification.compat_x.test_compat_x_plugin_provider_r4_gates",
        "test_r4_probe_g_cross_tenant_evidence",
    )


def qualify_agent_distribution_rogue_register_denied() -> None:
    _run(
        "tests.qualification.dup_x.test_dup_x_mechanical_exit_gates",
        "test_dup_x_exit_closed_world_replay",
    )
    _run(
        "tests.qualification.dup_x.test_dup_x_mechanical_exit_gates",
        "test_dup_x_section_13_measure_open_bypass",
    )


def qualify_nexus_agent_private_loop_denied() -> None:
    _run(
        "tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary",
        "test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone",
    )
    _run(
        "tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary",
        "test_b4_materialize_host_orchestration_backend_callers_allowlisted",
    )


def qualify_integration_external_di_boundary(tmp_path: Path) -> None:
    _run(
        "tests.qualification.dup_x.test_dup_x_npsc_boundary_gates",
        "test_dup_x_npsc_exclusion_record_covers_ten_modules",
    )
    _run(
        "tests.qualification.ebh_5.test_ebh_5_behavioral_proofs",
        "test_ebh5_behavioral_relational_store_negative_unregistered_provider_fail_closed",
        tmp_path=tmp_path,
    )


def qualify_scale_backpressure_tenant_isolate() -> None:
    _run(
        "tests.qualification.tenant_x.test_tenant_x_r4_fast",
        "test_tx_r4_q08_adversarial_vendor_async_queue",
    )
    _run(
        "tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        "test_ebh7_ten_e2e_10_distributed_queue_cross_tenant_fail_closed",
    )


def qualify_relational_store_sqlite_behavioral(tmp_path: Path) -> None:
    _run(
        "tests.qualification.ebh_5.test_ebh_5_behavioral_proofs",
        "test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile",
        tmp_path=tmp_path,
    )
    _run(
        "tests.qualification.ebh_5.test_ebh_5_behavioral_proofs",
        "test_ebh5_behavioral_relational_store_negative_unregistered_provider_fail_closed",
        tmp_path=tmp_path,
    )


def qualify_vbn_composition_factory_dual_slug(tmp_path: Path) -> None:
    entry = catalog_by_concern_id()["integration.relational_store"]
    run_package_c_v2_dual_slug_for_entry(entry, tmp_path=tmp_path)
    assert_unregistered_fail_closed(category=IntegrationCategory.RELATIONAL_STORE)
