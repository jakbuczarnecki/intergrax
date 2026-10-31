# © Artur Czarnecki. All rights reserved.

"""EBH-7 cross-domain platform proofs (integrated delegation @ HEAD)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_7.cross_domain_e2e._ebh7_cross_domain_integrated_assertions import (
    assert_xdm_agent_distribution_rogue_register_denied,
    assert_xdm_cancel_external_effect_tenant_boundary,
    assert_xdm_compat_historical_evidence_fail_closed,
    assert_xdm_config_compat_tenant_fail_closed,
    assert_xdm_failure_diagnostics_tenant_leak_denied,
    assert_xdm_hitl_trace_tenant_scope_denied,
    assert_xdm_integration_external_di_boundary,
    assert_xdm_llm_tenant_gov_sdk_block,
    assert_xdm_nexus_agent_private_loop_denied,
    assert_xdm_obs_evidence_tenant_isolate,
    assert_xdm_recovery_gov_restore_without_grant_denied,
    assert_xdm_retry_side_effect_cross_tenant_denied,
    assert_xdm_scale_backpressure_tenant_isolate,
    assert_xdm_tool_skill_plugin_shadow_denied,
    assert_xdm_trace_replay_tenant_lineage_fail_closed,
    assert_xdm_vbn_composition_factory_dual_slug,
    assert_xdm_worker_lease_tenant_denied,
)
from tests.qualification.ebh_7.tenant_platform_e2e._ebh7_integrated_tenant_boundary_assertions import (
    assert_plugin_host_compat_cross_tenant_activation_denied,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_xdm_010_backend_persistence_sqlite_behavioral(tmp_path: Path) -> None:
    from tests.qualification.ebh_7.cross_domain_e2e._ebh7_xdm_qualification_helpers import (
        qualify_relational_store_sqlite_behavioral,
    )

    qualify_relational_store_sqlite_behavioral(tmp_path)


def test_ebh7_xdm_007_tool_skill_plugin_shadow_denied() -> None:
    assert_xdm_tool_skill_plugin_shadow_denied()


def test_ebh7_xdm_012_trace_replay_tenant_boundary() -> None:
    assert_xdm_trace_replay_tenant_lineage_fail_closed()


def test_ebh7_xdm_014_cancel_external_effect_tenant() -> None:
    assert_xdm_cancel_external_effect_tenant_boundary()


def test_ebh7_xdm_015_compat_historical_evidence_reject() -> None:
    assert_xdm_compat_historical_evidence_fail_closed()


def test_ebh7_xdm_016_vbn_composition_factory_behavioral(tmp_path: Path) -> None:
    assert_xdm_vbn_composition_factory_dual_slug(tmp_path)


def test_ebh7_xdm_017_failure_diagnostics_tenant_leak() -> None:
    assert_xdm_failure_diagnostics_tenant_leak_denied()


def test_ebh7_xdm_018_plugin_tenant_activate_denied() -> None:
    assert_plugin_host_compat_cross_tenant_activation_denied()


def test_ebh7_xdm_019_recovery_gov_restore_denied() -> None:
    assert_xdm_recovery_gov_restore_without_grant_denied()


def test_ebh7_xdm_022_llm_tenant_gov_block() -> None:
    assert_xdm_llm_tenant_gov_sdk_block()


def test_ebh7_xdm_023_worker_lease_tenant() -> None:
    assert_xdm_worker_lease_tenant_denied()


def test_ebh7_xdm_024_retry_side_effect_tenant() -> None:
    assert_xdm_retry_side_effect_cross_tenant_denied()


def test_ebh7_xdm_025_hitl_trace_tenant() -> None:
    assert_xdm_hitl_trace_tenant_scope_denied()


def test_ebh7_xdm_026_config_compat_tenant() -> None:
    assert_xdm_config_compat_tenant_fail_closed()


def test_ebh7_xdm_027_agent_distribution_tenant() -> None:
    assert_xdm_agent_distribution_rogue_register_denied()


def test_ebh7_xdm_028_nexus_agent_tenant() -> None:
    assert_xdm_nexus_agent_private_loop_denied()


def test_ebh7_xdm_029_integration_external_di_boundary(tmp_path: Path) -> None:
    assert_xdm_integration_external_di_boundary(tmp_path)


def test_ebh7_xdm_030_obs_evidence_tenant_isolate() -> None:
    assert_xdm_obs_evidence_tenant_isolate()


def test_ebh7_xdm_031_scale_backpressure_tenant() -> None:
    assert_xdm_scale_backpressure_tenant_isolate()
