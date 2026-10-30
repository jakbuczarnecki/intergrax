# © Artur Czarnecki. All rights reserved.

"""Annex E — platform tenant E2E scenarios with explicit qualification state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.ebh_7._certification_status import CertificationStatus
from tests.qualification.ebh_7._constants import EBH7_TENANT_E2E_COUNT
from tests.qualification.ebh_7._gate_strength import GateStrength


@dataclass(frozen=True, slots=True)
class TenantE2EScenario:
    scenario_id: str
    requirement_id: str
    entrypoint: str
    tenant_a: str
    tenant_b: str
    production_path: str
    attack_failure: str
    expected_behavior: str
    assertions: str
    evidence_id: str
    reuse_program: str
    proof_module: str | None
    proof_function: str | None
    proof_needs_tmp_path: bool
    materialization_status: CertificationStatus
    qualification_status: CertificationStatus
    gate_strength: GateStrength


def _scenario(
    n: int,
    *,
    entrypoint: str,
    production_path: str,
    attack: str,
    expected: str,
    assertions: str,
    proof_module: str | None = None,
    proof_function: str | None = None,
    proof_needs_tmp_path: bool = False,
    qualification_status: CertificationStatus | None = None,
) -> TenantE2EScenario:
    sid = f"EBH7-TEN-E2E-{n:02d}"
    has_proof = proof_module is not None and proof_function is not None
    q_status = qualification_status
    if q_status is None:
        q_status = CertificationStatus.QUALIFIED if has_proof else CertificationStatus.QUALIFICATION_PENDING
    return TenantE2EScenario(
        scenario_id=sid,
        requirement_id="EBH7-RQ-008",
        entrypoint=entrypoint,
        tenant_a="tenant-A",
        tenant_b="tenant-B",
        production_path=production_path,
        attack_failure=attack,
        expected_behavior=expected,
        assertions=assertions,
        evidence_id=f"EVID-EBH7-TEN-E2E-{n:02d}",
        reuse_program="TENANT-X adversarial @ HEAD (explicit proof node)",
        proof_module=proof_module,
        proof_function=proof_function,
        proof_needs_tmp_path=proof_needs_tmp_path,
        materialization_status=CertificationStatus.MATERIALIZED,
        qualification_status=q_status,
        gate_strength=GateStrength.ADVERSARIAL if has_proof else GateStrength.E2E,
    )


EBH7_TENANT_E2E_SCENARIOS: Final[tuple[TenantE2EScenario, ...]] = (
    _scenario(
        1,
        entrypoint="RootExecutionAuthorityAdmissionPort",
        production_path="Identity→Gov admission",
        attack="missing tenant",
        expected="FAIL-CLOSED",
        assertions="admission rejects missing tenant",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r1_fast",
        proof_function="test_tx_r1_q09_adversarial_root_admission_missing_tenant",
    ),
    _scenario(
        2,
        entrypoint="UnifiedTaskRunner.run_task",
        production_path="Identity→Context pipeline",
        attack="switched tenant in runtime request",
        expected="DENY",
        assertions="runtime tenant mismatch rejected",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r1_fast",
        proof_function="test_tx_r1_q08_adversarial_runtime_request_tenant_mismatch",
    ),
    _scenario(
        3,
        entrypoint="Governance admission",
        production_path="Gov→Exec",
        attack="tenant A auth with tenant B execution context",
        expected="DENY",
        assertions="cross-tenant gov/exec denied",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_integrated_tenant_boundary_proofs",
        proof_function="test_ebh7_integrated_ten_e2e_03_gov_exec_cross_tenant_denied",
    ),
    _scenario(
        4,
        entrypoint="ToolRuntime.invoke",
        production_path="Exec→Tool",
        attack="cross-tenant tool scope",
        expected="DENY",
        assertions="tool scope cannot cross tenants",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_integrated_tenant_boundary_proofs",
        proof_function="test_ebh7_integrated_ten_e2e_04_exec_tool_cross_tenant_denied",
    ),
    _scenario(
        5,
        entrypoint="Integration factory",
        production_path="Config→Provider",
        attack="tenant A profile on tenant B backend",
        expected="REJECT",
        assertions="credential broker rejects cross-tenant refs",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_integrated_tenant_boundary_proofs",
        proof_function="test_ebh7_integrated_ten_e2e_05_config_provider_cross_tenant",
    ),
    _scenario(
        6,
        entrypoint="State recovery",
        production_path="State→Recovery",
        attack="resume tenant B from tenant A checkpoint",
        expected="REJECT",
        assertions="checkpoint recovery cross-tenant rejected",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r3_fast",
        proof_function="test_tx_r3_q05_adversarial_checkpoint_recovery_cross_tenant",
    ),
    _scenario(
        7,
        entrypoint="Cache layer",
        production_path="Persistence→Cache",
        attack="cross-tenant durable load",
        expected="DENY",
        assertions="durable load cannot bleed tenants",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r3_fast",
        proof_function="test_tx_r3_q07_adversarial_durable_load_cross_tenant",
        proof_needs_tmp_path=True,
    ),
    _scenario(
        8,
        entrypoint="RAG retrieve",
        production_path="Memory→RAG",
        attack="cross-tenant vector scope",
        expected="DENY",
        assertions="vector scope cross-tenant rejected",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r2_fast",
        proof_function="test_tx_r2_q07_adversarial_vector_scope_cross_tenant",
    ),
    _scenario(
        9,
        entrypoint="Async worker",
        production_path="Queue→Worker",
        attack="stale tenant on async dequeue",
        expected="DENY",
        assertions="vendor async queue tenant attribution rejects bleed",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r4_fast",
        proof_function="test_tx_r4_q08_adversarial_vendor_async_queue",
    ),
    _scenario(
        10,
        entrypoint="Distributed redrive",
        production_path="DIST→Exec",
        attack="cross redrive",
        expected="DENY",
        assertions="cross-tenant queue status/result fail-closed",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        proof_function="test_ebh7_ten_e2e_10_distributed_queue_cross_tenant_fail_closed",
    ),
    _scenario(
        11,
        entrypoint="Trace export",
        production_path="Trace→Evidence",
        attack="cross export",
        expected="REJECT",
        assertions="trace attribution rejects tenant mismatch",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        proof_function="test_ebh7_ten_e2e_11_trace_attribution_cross_tenant_rejected",
        proof_needs_tmp_path=False,
    ),
    _scenario(
        12,
        entrypoint="Skill compose",
        production_path="Skill→Gov→Exec",
        attack="cross skill",
        expected="DENY",
        assertions="TBD",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        proof_function="test_ebh7_ten_e2e_12_skill_compose_cross_tenant",
    ),
    _scenario(
        13,
        entrypoint="Plugin activate",
        production_path="Plugin→Host",
        attack="B plugin in A host",
        expected="DENY",
        assertions="compat assess rejects cross-tenant evidence",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        proof_function="test_ebh7_ten_e2e_13_plugin_compat_cross_tenant_rejected",
    ),
    _scenario(
        14,
        entrypoint="Backend switch",
        production_path="Provider→Persistence",
        attack="switch drops tenant",
        expected="PRESERVE",
        assertions="TBD",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        proof_function="test_ebh7_ten_e2e_14_backend_switch_preserves_tenant",
        proof_needs_tmp_path=True,
    ),
    _scenario(
        15,
        entrypoint="Idempotency",
        production_path="State→Retry",
        attack="cross-tenant retry",
        expected="DENY",
        assertions="retry/resume preserves tenant boundary",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r3_fast",
        proof_function="test_tx_r3_q09_retry_resume_preserves_tenant",
    ),
    _scenario(
        16,
        entrypoint="Credential ref",
        production_path="Secrets→Provider",
        attack="tenant A cred on tenant B provider",
        expected="REJECT",
        assertions="missing tenant credential scope rejected",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r2_fast",
        proof_function="test_tx_r2_q06_adversarial_missing_tenant_credential_scope",
    ),
    _scenario(
        17,
        entrypoint="Observability",
        production_path="Obs→Tenant",
        attack="metrics bleed",
        expected="ISOLATE",
        assertions="TBD",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        proof_function="test_ebh7_ten_e2e_17_obs_tenant_isolate",
    ),
    _scenario(
        18,
        entrypoint="HITL resume",
        production_path="Gov→Resume",
        attack="tenant widen",
        expected="DENY",
        assertions="TBD",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        proof_function="test_ebh7_ten_e2e_18_hitl_resume_tenant",
    ),
    _scenario(
        19,
        entrypoint="Child exec",
        production_path="Parent→Child",
        attack="cross child",
        expected="DENY",
        assertions="TBD",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        proof_function="test_ebh7_ten_e2e_19_child_exec_cross_tenant",
    ),
    _scenario(
        20,
        entrypoint="Compat assess",
        production_path="COMPAT→Tenant",
        attack="cross assess",
        expected="REJECT",
        assertions="compat service rejects cross-tenant evidence",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        proof_function="test_ebh7_ten_e2e_20_compat_assess_cross_tenant_rejected",
    ),
    _scenario(
        21,
        entrypoint="Control plane",
        production_path="CTRL→Gov",
        attack="CP cross-tenant",
        expected="DENY",
        assertions="TBD",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        proof_function="test_ebh7_ten_e2e_21_control_plane_cross_tenant",
    ),
    _scenario(
        22,
        entrypoint="Scheduler",
        production_path="SCHED→Worker",
        attack="delayed drift",
        expected="DENY",
        assertions="TBD",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        proof_function="test_ebh7_ten_e2e_22_scheduler_tenant_drift",
    ),
    _scenario(
        23,
        entrypoint="Evidence replay",
        production_path="Trace→Replay",
        attack="cross replay",
        expected="DENY",
        assertions="TBD",
        proof_module="tests.qualification.ebh_7.tenant_platform_e2e.test_ebh7_platform_tenant_boundary_proofs",
        proof_function="test_ebh7_ten_e2e_23_trace_replay_cross_tenant",
    ),
    _scenario(
        24,
        entrypoint="Context widen",
        production_path="Context",
        attack="LLM context cross-tenant attribution",
        expected="DENY",
        assertions="context LLM cross-tenant rejected",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r4_fast",
        proof_function="test_tx_r4_q05_adversarial_context_llm_cross_tenant",
    ),
    _scenario(
        25,
        entrypoint="Request intake",
        production_path="Request→Identity",
        attack="forged tenant / missing evidence",
        expected="DENY",
        assertions="functional evidence missing tenant rejected",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r1_fast",
        proof_function="test_tx_r1_q10_adversarial_functional_evidence_missing_tenant",
    ),
    _scenario(
        26,
        entrypoint="Full path",
        production_path="Runner→External effect",
        attack="platform aggregate cross-tenant allowance",
        expected="DENY",
        assertions="final R4 certification reports zero cross-tenant allowed",
        proof_module="tests.qualification.tenant_x.test_tenant_x_r4_fast",
        proof_function="test_tx_r4_q10_closure_derived_security_metrics",
    ),
)


def assert_ebh7_tenant_e2e_registry_integrity() -> None:
    assert len(EBH7_TENANT_E2E_SCENARIOS) == EBH7_TENANT_E2E_COUNT
    ids = [s.scenario_id for s in EBH7_TENANT_E2E_SCENARIOS]
    assert ids == [f"EBH7-TEN-E2E-{n:02d}" for n in range(1, EBH7_TENANT_E2E_COUNT + 1)]


def tenant_e2e_pending_scenario_ids() -> tuple[str, ...]:
    from tests.qualification.ebh_7._semantic_proof_binding_audit import tenant_e2e_open_ids_from_audit

    return tenant_e2e_open_ids_from_audit()


def tenant_e2e_qualified_scenarios() -> tuple[TenantE2EScenario, ...]:
    from tests.qualification.ebh_7._semantic_proof_binding_audit import tenant_e2e_semantically_qualified_ids

    allowed = tenant_e2e_semantically_qualified_ids()
    return tuple(s for s in EBH7_TENANT_E2E_SCENARIOS if s.scenario_id in allowed)
