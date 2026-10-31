# © Artur Czarnecki. All rights reserved.

"""Cross-domain qualification helpers — delegated proof execution (not composed test imports in integrated layer)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from intergrax.integrations.contracts.base import IntegrationCategory
from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import catalog_by_concern_id
from tests.qualification.ebh_5._package_c_v2_primitives import assert_unregistered_fail_closed
from tests.qualification.ebh_5._package_c_v2_proof_helpers import run_package_c_v2_dual_slug_for_entry
from tests.qualification.tenant_x._r3_adversarial import (
    assert_checkpoint_recovery_cross_tenant_rejects,
)
from tests.qualification.tenant_x._r4_adversarial import (
    assert_vendor_worker_payload_job_tenant_mismatch_rejects,
)


def qualify_tool_skill_plugin_shadow_denied() -> None:
    from tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary import (
        test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone,
    )
    from tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs import (
        test_ebh7_ten_e2e_12_skill_compose_cross_tenant,
    )
    from tests.qualification.tenant_x.test_tenant_x_r1_fast import (
        test_tx_r1_q12_adversarial_tool_scope_same_tenant,
    )

    test_tx_r1_q12_adversarial_tool_scope_same_tenant()
    test_ebh7_ten_e2e_12_skill_compose_cross_tenant()
    test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone()


def qualify_trace_replay_tenant_lineage_fail_closed() -> None:
    from tests.unit.applications.test_uca6c_marketplace_qualified_execution_composition import (
        test_tenant_mismatch_blocks_materialization_pin_and_io,
    )

    test_tenant_mismatch_blocks_materialization_pin_and_io()


def qualify_compat_historical_evidence_fail_closed() -> None:
    from tests.qualification.compat_x.test_compat_x_plugin_provider_r4_gates import (
        test_r4_probe_g_cross_tenant_evidence,
    )
    from tests.qualification.tenant_x.test_tenant_x_p0_fast import (
        test_tx_p0_r1_q02_ctrl_x_historical_evidence_sha,
        test_tx_p0_r1_q09_historical_evidence_requires_replay,
    )

    test_tx_p0_r1_q02_ctrl_x_historical_evidence_sha()
    test_tx_p0_r1_q09_historical_evidence_requires_replay()
    test_r4_probe_g_cross_tenant_evidence()


def qualify_failure_diagnostics_tenant_leak_denied() -> None:
    from tests.qualification.trace_x.test_trace_x_p5_p0_qualification_gates import (
        test_txp5p0_q10_configuration_tenant_mismatch_fail_closed,
        test_txp5p0_q11_profile_binding_tenant_mismatch_fail_closed,
    )

    test_txp5p0_q10_configuration_tenant_mismatch_fail_closed()
    test_txp5p0_q11_profile_binding_tenant_mismatch_fail_closed()


def qualify_recovery_gov_restore_without_grant_denied() -> None:
    from tests.qualification.governance.test_governance_e2e_decision import (
        test_scenario_l_stale_proposal_fail_closed,
    )

    assert_checkpoint_recovery_cross_tenant_rejects()
    asyncio.run(test_scenario_l_stale_proposal_fail_closed())


def qualify_llm_tenant_gov_sdk_block() -> None:
    from tests.qualification.governance.test_governance_e2e_decision import (
        test_scenario_z_cross_tenant_fail_closed,
    )
    from tests.qualification.tenant_x.test_tenant_x_r4_fast import (
        test_tx_r4_q05_adversarial_context_llm_cross_tenant,
    )

    test_tx_r4_q05_adversarial_context_llm_cross_tenant()
    asyncio.run(test_scenario_z_cross_tenant_fail_closed())


def qualify_worker_lease_tenant_denied() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs import (
        test_ebh7_ten_e2e_10_distributed_queue_cross_tenant_fail_closed,
    )

    assert_vendor_worker_payload_job_tenant_mismatch_rejects()
    test_ebh7_ten_e2e_10_distributed_queue_cross_tenant_fail_closed()


def qualify_hitl_trace_tenant_scope_denied() -> None:
    from tests.qualification.governance.test_governance_e2e_decision import (
        test_scenario_m_decision_accepted_governance_deny_blocks_operation,
    )
    from tests.qualification.trace_x.test_trace_x_p5_p0_qualification_gates import (
        test_txp5p0_q11_profile_binding_tenant_mismatch_fail_closed,
    )

    asyncio.run(test_scenario_m_decision_accepted_governance_deny_blocks_operation())
    test_txp5p0_q11_profile_binding_tenant_mismatch_fail_closed()


def qualify_config_compat_tenant_fail_closed() -> None:
    from tests.qualification.compat_x.test_compat_x_plugin_provider_r4_gates import (
        test_r4_probe_g_cross_tenant_evidence,
    )
    from tests.qualification.config_x.test_config_x_adversarial_cx_gates import (
        test_cx_g_cross_tenant_provider_config_rejected_by_realization_guards,
    )

    test_cx_g_cross_tenant_provider_config_rejected_by_realization_guards()
    test_r4_probe_g_cross_tenant_evidence()


def qualify_agent_distribution_rogue_register_denied() -> None:
    from tests.qualification.dup_x.test_dup_x_mechanical_exit_gates import (
        test_dup_x_exit_closed_world_replay,
        test_dup_x_section_13_measure_open_bypass,
    )

    test_dup_x_exit_closed_world_replay()
    test_dup_x_section_13_measure_open_bypass()


def qualify_nexus_agent_private_loop_denied() -> None:
    from tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary import (
        test_b4_materialize_host_orchestration_backend_callers_allowlisted,
        test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone,
    )

    test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone()
    test_b4_materialize_host_orchestration_backend_callers_allowlisted()


def qualify_integration_external_di_boundary(tmp_path: Path) -> None:
    from tests.qualification.dup_x.test_dup_x_npsc_boundary_gates import (
        test_dup_x_npsc_exclusion_record_covers_ten_modules,
    )
    from tests.qualification.ebh_5.test_ebh_5_behavioral_proofs import (
        test_ebh5_behavioral_relational_store_negative_unregistered_provider_fail_closed,
    )

    test_dup_x_npsc_exclusion_record_covers_ten_modules()
    test_ebh5_behavioral_relational_store_negative_unregistered_provider_fail_closed(tmp_path)


def qualify_scale_backpressure_tenant_isolate() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs import (
        test_ebh7_ten_e2e_10_distributed_queue_cross_tenant_fail_closed,
    )
    from tests.qualification.tenant_x.test_tenant_x_r4_fast import (
        test_tx_r4_q08_adversarial_vendor_async_queue,
    )

    test_tx_r4_q08_adversarial_vendor_async_queue()
    test_ebh7_ten_e2e_10_distributed_queue_cross_tenant_fail_closed()


def qualify_relational_store_sqlite_behavioral(tmp_path: Path) -> None:
    from tests.qualification.ebh_5.test_ebh_5_behavioral_proofs import (
        test_ebh5_behavioral_relational_store_negative_unregistered_provider_fail_closed,
        test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile,
    )

    test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile(tmp_path)
    test_ebh5_behavioral_relational_store_negative_unregistered_provider_fail_closed(tmp_path)


def qualify_vbn_composition_factory_dual_slug(tmp_path: Path) -> None:
    entry = catalog_by_concern_id()["integration.relational_store"]
    run_package_c_v2_dual_slug_for_entry(entry, tmp_path=tmp_path)
    assert_unregistered_fail_closed(category=IntegrationCategory.RELATIONAL_STORE)
