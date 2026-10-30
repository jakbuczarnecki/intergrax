# © Artur Czarnecki. All rights reserved.

"""EBH-7 cross-domain platform proofs (integrated delegation @ HEAD)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_xdm_010_backend_persistence_sqlite_behavioral(tmp_path: Path) -> None:
    from tests.qualification.ebh_5.test_ebh_5_behavioral_proofs import (
        test_ebh5_behavioral_relational_store_negative_unregistered_provider_fail_closed,
        test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile,
    )

    test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile(tmp_path)
    test_ebh5_behavioral_relational_store_negative_unregistered_provider_fail_closed(tmp_path)


def test_ebh7_xdm_007_tool_skill_plugin_shadow_denied() -> None:
    from tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary import (
        test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone,
    )
    from tests.qualification.tenant_x.test_tenant_x_r1_fast import (
        test_tx_r1_q12_adversarial_tool_scope_same_tenant,
    )

    test_tx_r1_q12_adversarial_tool_scope_same_tenant()
    test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone()


def test_ebh7_xdm_012_trace_replay_tenant_boundary() -> None:
    from tests.qualification.tenant_x.test_tenant_x_r3_fast import (
        test_tx_r3_q09_retry_resume_preserves_tenant,
    )

    test_tx_r3_q09_retry_resume_preserves_tenant()


def test_ebh7_xdm_014_cancel_external_effect_tenant() -> None:
    from tests.qualification.tenant_x.test_tenant_x_r3_fast import (
        test_tx_r3_q09_retry_resume_preserves_tenant,
    )

    test_tx_r3_q09_retry_resume_preserves_tenant()


def test_ebh7_xdm_015_compat_historical_evidence_reject() -> None:
    from tests.qualification.compat_x.test_compat_x_plugin_provider_r4_gates import (
        test_r4_probe_g_cross_tenant_evidence,
    )

    test_r4_probe_g_cross_tenant_evidence()


def test_ebh7_xdm_016_vbn_composition_factory_behavioral(tmp_path: Path) -> None:
    test_ebh7_xdm_010_backend_persistence_sqlite_behavioral(tmp_path)


def test_ebh7_xdm_017_failure_diagnostics_tenant_leak() -> None:
    from tests.qualification.tenant_x.test_tenant_x_r2_fast import (
        test_tx_r2_q06_adversarial_missing_tenant_credential_scope,
    )

    test_tx_r2_q06_adversarial_missing_tenant_credential_scope()


def test_ebh7_xdm_018_plugin_tenant_activate_denied() -> None:
    from tests.qualification.compat_x.test_compat_x_plugin_provider_r4_gates import (
        test_r4_probe_g_cross_tenant_evidence,
    )

    test_r4_probe_g_cross_tenant_evidence()


def test_ebh7_xdm_019_recovery_gov_restore_denied() -> None:
    from tests.qualification.tenant_x.test_tenant_x_r3_fast import (
        test_tx_r3_q05_adversarial_checkpoint_recovery_cross_tenant,
    )

    test_tx_r3_q05_adversarial_checkpoint_recovery_cross_tenant()


def test_ebh7_xdm_022_llm_tenant_gov_block() -> None:
    from tests.qualification.tenant_x.test_tenant_x_r1_fast import (
        test_tx_r1_q08_adversarial_runtime_request_tenant_mismatch,
    )

    test_tx_r1_q08_adversarial_runtime_request_tenant_mismatch()


def test_ebh7_xdm_023_worker_lease_tenant() -> None:
    from tests.qualification.tenant_x.test_tenant_x_r4_fast import (
        test_tx_r4_q08_adversarial_vendor_async_queue,
    )

    test_tx_r4_q08_adversarial_vendor_async_queue()


def test_ebh7_xdm_024_retry_side_effect_tenant() -> None:
    from tests.qualification.tenant_x.test_tenant_x_r3_fast import (
        test_tx_r3_q09_retry_resume_preserves_tenant,
    )

    test_tx_r3_q09_retry_resume_preserves_tenant()


def test_ebh7_xdm_025_hitl_trace_tenant() -> None:
    from tests.qualification.tenant_x.test_tenant_x_r4_fast import (
        test_tx_r4_q05_adversarial_context_llm_cross_tenant,
    )

    test_tx_r4_q05_adversarial_context_llm_cross_tenant()


def test_ebh7_xdm_026_config_compat_tenant() -> None:
    from tests.qualification.compat_x.test_compat_x_plugin_provider_r4_gates import (
        test_r4_probe_g_cross_tenant_evidence,
    )

    test_r4_probe_g_cross_tenant_evidence()


def test_ebh7_xdm_027_agent_distribution_tenant() -> None:
    from tests.qualification.dup_x.test_dup_x_mechanical_exit_gates import (
        test_dup_x_exit_closed_world_replay,
    )

    test_dup_x_exit_closed_world_replay()


def test_ebh7_xdm_028_nexus_agent_tenant() -> None:
    from tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary import (
        test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone,
    )

    test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone()


def test_ebh7_xdm_029_integration_external_di_boundary() -> None:
    from tests.qualification.dup_x.test_dup_x_npsc_boundary_gates import (
        test_dup_x_npsc_exclusion_record_covers_ten_modules,
    )

    test_dup_x_npsc_exclusion_record_covers_ten_modules()


def test_ebh7_xdm_030_obs_evidence_tenant_isolate() -> None:
    from tests.qualification.ebh_5.test_ebh_5_behavioral_proofs import (
        test_ebh5_behavioral_otlp_transport_negative_invalid_distributed_configuration,
    )

    test_ebh5_behavioral_otlp_transport_negative_invalid_distributed_configuration()


def test_ebh7_xdm_031_scale_backpressure_tenant() -> None:
    from tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs import (
        test_ebh7_ten_e2e_10_distributed_queue_cross_tenant_fail_closed,
    )

    test_ebh7_ten_e2e_10_distributed_queue_cross_tenant_fail_closed()
