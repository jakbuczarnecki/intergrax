# © Artur Czarnecki. All rights reserved.

"""EBH-7 integrated cross-domain assertions — orchestration entrypoints (no direct test_* calls)."""

from __future__ import annotations

from pathlib import Path

from tests.qualification.ebh_7.cross_domain_e2e._ebh7_xdm_qualification_helpers import (
    qualify_agent_distribution_rogue_register_denied,
    qualify_compat_historical_evidence_fail_closed,
    qualify_config_compat_tenant_fail_closed,
    qualify_failure_diagnostics_tenant_leak_denied,
    qualify_hitl_trace_tenant_scope_denied,
    qualify_integration_external_di_boundary,
    qualify_llm_tenant_gov_sdk_block,
    qualify_nexus_agent_private_loop_denied,
    qualify_recovery_gov_restore_without_grant_denied,
    qualify_scale_backpressure_tenant_isolate,
    qualify_tool_skill_plugin_shadow_denied,
    qualify_trace_replay_tenant_lineage_fail_closed,
    qualify_vbn_composition_factory_dual_slug,
    qualify_worker_lease_tenant_denied,
)
from tests.qualification.tenant_x._r3_adversarial import assert_retry_resume_preserves_tenant_a
from tests.qualification.tenant_x._r4_adversarial import (
    assert_background_blank_transport_rejects,
    assert_background_transport_task_tenant_mismatch_rejects,
    assert_vendor_queue_request_tenant_mismatch_rejects,
)


def assert_xdm_tool_skill_plugin_shadow_denied() -> None:
    qualify_tool_skill_plugin_shadow_denied()


def assert_xdm_trace_replay_tenant_lineage_fail_closed() -> None:
    qualify_trace_replay_tenant_lineage_fail_closed()


def assert_xdm_cancel_external_effect_tenant_boundary() -> None:
    assert_background_transport_task_tenant_mismatch_rejects()
    assert_background_blank_transport_rejects()


def assert_xdm_compat_historical_evidence_fail_closed() -> None:
    qualify_compat_historical_evidence_fail_closed()


def assert_xdm_vbn_composition_factory_dual_slug(tmp_path: Path) -> None:
    qualify_vbn_composition_factory_dual_slug(tmp_path)


def assert_xdm_failure_diagnostics_tenant_leak_denied() -> None:
    qualify_failure_diagnostics_tenant_leak_denied()


def assert_xdm_recovery_gov_restore_without_grant_denied() -> None:
    qualify_recovery_gov_restore_without_grant_denied()


def assert_xdm_llm_tenant_gov_sdk_block() -> None:
    qualify_llm_tenant_gov_sdk_block()


def assert_xdm_worker_lease_tenant_denied() -> None:
    qualify_worker_lease_tenant_denied()


def assert_xdm_retry_side_effect_cross_tenant_denied() -> None:
    assert_vendor_queue_request_tenant_mismatch_rejects()
    assert_retry_resume_preserves_tenant_a()


def assert_xdm_hitl_trace_tenant_scope_denied() -> None:
    qualify_hitl_trace_tenant_scope_denied()


def assert_xdm_config_compat_tenant_fail_closed() -> None:
    qualify_config_compat_tenant_fail_closed()


def assert_xdm_agent_distribution_rogue_register_denied() -> None:
    qualify_agent_distribution_rogue_register_denied()


def assert_xdm_nexus_agent_private_loop_denied() -> None:
    qualify_nexus_agent_private_loop_denied()


def assert_xdm_integration_external_di_boundary(tmp_path: Path) -> None:
    qualify_integration_external_di_boundary(tmp_path)


def assert_xdm_obs_evidence_tenant_isolate() -> None:
    assert_xdm_failure_diagnostics_tenant_leak_denied()


def assert_xdm_scale_backpressure_tenant_isolate() -> None:
    qualify_scale_backpressure_tenant_isolate()
