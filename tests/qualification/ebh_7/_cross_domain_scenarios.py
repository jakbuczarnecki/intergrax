# © Artur Czarnecki. All rights reserved.

"""Annex F — cross-domain scenarios with explicit qualification state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.ebh_7._certification_status import CertificationStatus
from tests.qualification.ebh_7._constants import EBH7_CROSS_DOMAIN_COUNT
from tests.qualification.ebh_7._gate_strength import GateStrength


@dataclass(frozen=True, slots=True)
class CrossDomainScenario:
    scenario_id: str
    requirement_id: str
    combo: str
    failure_model: str
    components: str
    attack: str
    expected: str
    reuse_program: str
    proof_module: str | None
    proof_function: str | None
    proof_needs_tmp_path: bool
    materialization_status: CertificationStatus
    qualification_status: CertificationStatus
    gate_strength: GateStrength
    evidence_id: str


def _xdm(
    n: int,
    *,
    combo: str,
    failure_model: str,
    components: str,
    attack: str,
    expected: str,
    reuse: str,
    proof_module: str | None = None,
    proof_function: str | None = None,
    proof_needs_tmp_path: bool = False,
) -> CrossDomainScenario:
    sid = f"EBH7-XDM-{n:03d}"
    has_proof = proof_module is not None and proof_function is not None
    return CrossDomainScenario(
        scenario_id=sid,
        requirement_id=sid,
        combo=combo,
        failure_model=failure_model,
        components=components,
        attack=attack,
        expected=expected,
        reuse_program=reuse,
        proof_module=proof_module,
        proof_function=proof_function,
        proof_needs_tmp_path=proof_needs_tmp_path,
        materialization_status=CertificationStatus.MATERIALIZED,
        qualification_status=(
            CertificationStatus.QUALIFIED if has_proof else CertificationStatus.QUALIFICATION_PENDING
        ),
        gate_strength=GateStrength.ADVERSARIAL if has_proof else GateStrength.E2E,
        evidence_id=f"EVID-EBH7-XDM-{n:03d}",
    )


EBH7_CROSS_DOMAIN_SCENARIOS: Final[tuple[CrossDomainScenario, ...]] = (
    _xdm(
        1,
        combo="Tenant x Gov",
        failure_model="Gov admit",
        components="Tenant resolver",
        attack="wrong tenant admit",
        expected="DENY",
        reuse="tenant_x R1",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r1_fast",
        proof_function="test_tx_r1_q08_adversarial_runtime_request_tenant_mismatch",
    ),
    _xdm(
        2,
        combo="Tenant x Exec",
        failure_model="Dispatch",
        components="UnifiedTaskRunner",
        attack="cross dispatch",
        expected="DENY",
        reuse="tenant_x R1",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r1_fast",
        proof_function="test_tx_r1_q08_adversarial_runtime_request_tenant_mismatch",
    ),
    _xdm(
        3,
        combo="Tenant x State",
        failure_model="Checkpoint",
        components="STATE-X",
        attack="cross resume",
        expected="REJECT",
        reuse="tenant_x R3",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r3_fast",
        proof_function="test_tx_r3_q05_adversarial_checkpoint_recovery_cross_tenant",
    ),
    _xdm(
        4,
        combo="Tenant x Recovery",
        failure_model="Replay",
        components="Recovery svc",
        attack="replay bleed",
        expected="DENY",
        reuse="tenant_x R3",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r3_fast",
        proof_function="test_tx_r3_q06_adversarial_missing_tenant_no_global_recovery",
    ),
    _xdm(
        5,
        combo="Tenant x Evidence",
        failure_model="Export",
        components="Trace",
        attack="attribution swap",
        expected="REJECT",
        reuse="tenant_x R4",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r4_fast",
        proof_function="test_tx_r4_q09_adversarial_functional_evidence_tenant",
    ),
    _xdm(
        6,
        combo="Tenant x Distributed",
        failure_model="Queue",
        components="Worker",
        attack="queue cross",
        expected="DENY",
        reuse="tenant_x R4",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r4_fast",
        proof_function="test_tx_r4_q08_adversarial_vendor_async_queue",
    ),
    _xdm(
        7,
        combo="Tool x Skill x Plugin",
        failure_model="Compose",
        components="Skill,Tool,Plugin",
        attack="shadow tool",
        expected="DENY",
        reuse="skill_x",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_007_tool_skill_plugin_shadow_denied",
    ),
    _xdm(
        8,
        combo="Provider x Config",
        failure_model="Profile",
        components="CONFIG-X",
        attack="shadow env",
        expected="REJECT",
        reuse="ebh_6 config_x",
        proof_module="tests.qualification.ebh_6.test_ebh_6_config_x_head_reconciliation",
        proof_function="test_ebh6_config_x_all_concerns_mapped_at_head",
    ),
    _xdm(
        9,
        combo="Provider x Gov",
        failure_model="Provider call",
        components="Gov",
        attack="unauth provider",
        expected="DENY",
        reuse="tenant_x R2",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r2_fast",
        proof_function="test_tx_r2_q06_adversarial_missing_tenant_credential_scope",
    ),
    _xdm(
        10,
        combo="Backend x Persistence",
        failure_model="SQLite",
        components="EBH-5",
        attack="sqlite bleed",
        expected="REJECT",
        reuse="ebh_5 behavioral",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_010_backend_persistence_sqlite_behavioral",
        proof_needs_tmp_path=True,
    ),
    _xdm(
        11,
        combo="Backend x Recovery",
        failure_model="Restore",
        components="STATE-X",
        attack="wrong tenant restore",
        expected="REJECT",
        reuse="tenant_x R3",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r3_fast",
        proof_function="test_tx_r3_q08_adversarial_partial_recovery_tenant",
    ),
    _xdm(
        12,
        combo="Trace x Replay",
        failure_model="Lineage",
        components="TRACE-X",
        attack="break lineage",
        expected="FAIL",
        reuse="trace_x",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_012_trace_replay_tenant_boundary",
    ),
    _xdm(
        13,
        combo="Distributed x Idempotency",
        failure_model="Retry",
        components="DIST",
        attack="dup effect",
        expected="REJECT",
        reuse="tenant_x R3",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r3_fast",
        proof_function="test_tx_r3_q09_retry_resume_preserves_tenant",
    ),
    _xdm(
        14,
        combo="Cancel x External",
        failure_model="Cancel",
        components="TOOL-LIFE",
        attack="late effect",
        expected="REJECT",
        reuse="tool_life",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_014_cancel_external_effect_tenant",
    ),
    _xdm(
        15,
        combo="Compat x Historical",
        failure_model="SHA drift",
        components="COMPAT-X",
        attack="wrong evidence",
        expected="REJECT",
        reuse="compat_x",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_015_compat_historical_evidence_reject",
    ),
    _xdm(
        16,
        combo="VBN x Composition",
        failure_model="Factory",
        components="EBH-6",
        attack="hardcoded slug",
        expected="REJECT",
        reuse="ebh_6 findings",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_016_vbn_composition_factory_behavioral",
        proof_needs_tmp_path=True,
    ),
    _xdm(
        17,
        combo="Failure x Diagnostics",
        failure_model="Error leak",
        components="CTRL-X",
        attack="tenant leak",
        expected="REJECT",
        reuse="ctrl_x",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_017_failure_diagnostics_tenant_leak",
        proof_needs_tmp_path=True,
    ),
    _xdm(
        18,
        combo="Plugin x Tenant",
        failure_model="Activate",
        components="Plugin host",
        attack="B in A",
        expected="DENY",
        reuse="tenant_x R2",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_018_plugin_tenant_activate_denied",
    ),
    _xdm(
        19,
        combo="Recovery x Gov",
        failure_model="Restore auth",
        components="Gov+STATE",
        attack="restore w/o grant",
        expected="DENY",
        reuse="gov_x2",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_019_recovery_gov_restore_denied",
    ),
    _xdm(
        20,
        combo="Gov x Exec x Tool",
        failure_model="Triple",
        components="Gov,Exec,Tool",
        attack="bypass",
        expected="DENY",
        reuse="dup_x B4",
        proof_module="tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary",
        proof_function="test_b4_orchestration_backend_access_import_confined_to_execution_owner_zone",
    ),
    _xdm(
        21,
        combo="Memory x RAG x Tenant",
        failure_model="Retrieve",
        components="RAG",
        attack="index cross",
        expected="DENY",
        reuse="tenant_x R2",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r2_fast",
        proof_function="test_tx_r2_q07_adversarial_vector_scope_cross_tenant",
    ),
    _xdm(
        22,
        combo="LLM x Tenant x Gov",
        failure_model="SDK block",
        components="LLM adapters",
        attack="direct SDK",
        expected="DENY",
        reuse="sys_inv",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_022_llm_tenant_gov_block",
    ),
    _xdm(
        23,
        combo="Worker x Lease x Tenant",
        failure_model="Lease",
        components="DIST",
        attack="lease steal",
        expected="DENY",
        reuse="dist_scale",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_023_worker_lease_tenant",
    ),
    _xdm(
        24,
        combo="Retry x SideEffect x Tenant",
        failure_model="Retry",
        components="TOOL-LIFE",
        attack="cross retry",
        expected="DENY",
        reuse="tool_life",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_024_retry_side_effect_tenant",
    ),
    _xdm(
        25,
        combo="HITL x Trace x Tenant",
        failure_model="HITL chain",
        components="Gov,Trace",
        attack="scope break",
        expected="DENY",
        reuse="gov_x2",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_025_hitl_trace_tenant",
    ),
    _xdm(
        26,
        combo="Config x Compat x Tenant",
        failure_model="Profile",
        components="COMPAT",
        attack="mismatch",
        expected="FAIL-CLOSED",
        reuse="compat_x",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_026_config_compat_tenant",
    ),
    _xdm(
        27,
        combo="Agent x Distribution x Tenant",
        failure_model="AD path",
        components="Agent distribution",
        attack="rogue register",
        expected="DENY",
        reuse="sys_inv",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_027_agent_distribution_tenant",
    ),
    _xdm(
        28,
        combo="Nexus x Agent x Tenant",
        failure_model="Nexus loop",
        components="Nexus,Agent",
        attack="private loop",
        expected="DENY",
        reuse="sys_inv",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_028_nexus_agent_tenant",
    ),
    _xdm(
        29,
        combo="Integration x External",
        failure_model="DI boundary",
        components="INT-EXTCOMP",
        attack="DI bypass",
        expected="DENY",
        reuse="int_extcomp",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_029_integration_external_di_boundary",
    ),
    _xdm(
        30,
        combo="Obs x Evidence x Tenant",
        failure_model="Diagnostics",
        components="Obs",
        attack="diag authority",
        expected="ISOLATE",
        reuse="trace_x",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_030_obs_evidence_tenant_isolate",
    ),
    _xdm(
        31,
        combo="Scale x Backpressure x Tenant",
        failure_model="Queue fair",
        components="DIST",
        attack="starvation",
        expected="BOUNDED",
        reuse="dist_scale",
        proof_module="tests.qualification.ebh_7.cross_domain_e2e.test_ebh7_cross_domain_platform_proofs",
        proof_function="test_ebh7_xdm_031_scale_backpressure_tenant",
    ),
    _xdm(
        32,
        combo="Skill x Tool auth x Tenant",
        failure_model="Skill proposal",
        components="Skill",
        attack="direct side effect",
        expected="DENY",
        reuse="tenant_x R1",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r1_fast",
        proof_function="test_tx_r1_q12_adversarial_tool_scope_same_tenant",
    ),
    _xdm(
        33,
        combo="Host x Composition",
        failure_model="Second root",
        components="DUP-X RF-CMP",
        attack="duplicate root",
        expected="REJECT",
        reuse="dup_x exit",
        proof_module="tests.qualification.dup_x.test_dup_x_mechanical_exit_gates",
        proof_function="test_dup_x_section_13_measure_open_bypass",
    ),
    _xdm(
        34,
        combo="NPSC x Orchestration x Tenant",
        failure_model="Backend escape",
        components="NPSC-4.2",
        attack="backend escape",
        expected="REJECT",
        reuse="npsc gate",
        proof_module="tests.qualification.dup_x.test_dup_x_b4_npsc_orchestration_backend_negative_boundary",
        proof_function="test_b4_materialize_host_orchestration_backend_callers_allowlisted",
    ),
)


def assert_ebh7_cross_domain_registry_integrity() -> None:
    assert len(EBH7_CROSS_DOMAIN_SCENARIOS) == EBH7_CROSS_DOMAIN_COUNT
    ids = [s.scenario_id for s in EBH7_CROSS_DOMAIN_SCENARIOS]
    assert ids == [f"EBH7-XDM-{n:03d}" for n in range(1, EBH7_CROSS_DOMAIN_COUNT + 1)]


def cross_domain_pending_scenario_ids() -> tuple[str, ...]:
    from tests.qualification.ebh_7._semantic_proof_binding_audit import cross_domain_open_ids_from_audit

    return cross_domain_open_ids_from_audit()


def cross_domain_qualified_scenarios() -> tuple[CrossDomainScenario, ...]:
    from tests.qualification.ebh_7._semantic_proof_binding_audit import cross_domain_semantically_qualified_ids

    allowed = cross_domain_semantically_qualified_ids()
    return tuple(s for s in EBH7_CROSS_DOMAIN_SCENARIOS if s.scenario_id in allowed)
