# © Artur Czarnecki. All rights reserved.

"""INSPECT-01-A Q1..Q15 evidence catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Inspect01AQEvidence:
    q_id: str
    title: str
    pytest_node_ids: tuple[str, ...]


def _nid(test_file: str, test_name: str) -> str:
    return f"tests/qualification/inspect_01/{test_file}::{test_name}"


@dataclass(frozen=True, slots=True)
class Inspect01BQEvidence:
    q_id: str
    title: str
    pytest_node_ids: tuple[str, ...]


INSPECT_01_A_Q_CATALOG: tuple[Inspect01AQEvidence, ...] = (
    Inspect01AQEvidence("A-Q1", "Canonical ExecutionId snapshot", (_nid("test_inspect_01a_federation.py", "test_a_q1_canonical_execution_snapshot"),)),
    Inspect01AQEvidence("A-Q2", "Deterministic bounded timeline", (_nid("test_inspect_01a_federation.py", "test_a_q2_timeline_bounded_deterministic"),)),
    Inspect01AQEvidence("A-Q3", "Diagnostics reuse DiagnosticReadService", (_nid("test_inspect_01a_federation.py", "test_a_q3_diagnostics_delegate_to_read_service"),)),
    Inspect01AQEvidence("A-Q4", "Evidence references not copied store", (_nid("test_inspect_01a_federation.py", "test_a_q4_evidence_refs_only"),)),
    Inspect01AQEvidence("A-Q5", "Completeness semantics", (_nid("test_inspect_01a_federation.py", "test_a_q5_completeness_complete_vs_partial"),)),
    Inspect01AQEvidence("A-Q6", "Partial source failure", (_nid("test_inspect_01a_federation.py", "test_a_q6_partial_source_failure"),)),
    Inspect01AQEvidence("A-Q7", "Typed NOT_FOUND", (_nid("test_inspect_01a_federation.py", "test_a_q7_typed_not_found"),)),
    Inspect01AQEvidence("A-Q8", "Tenant isolation", (_nid("test_inspect_01a_federation.py", "test_a_q8_tenant_isolation"),)),
    Inspect01AQEvidence("A-Q9", "Redaction / no secrets", (_nid("test_inspect_01a_safety.py", "test_a_q9_secrets_not_leaked"),)),
    Inspect01AQEvidence("A-Q10", "No side effects", (_nid("test_inspect_01a_safety.py", "test_a_q10_zero_side_effects"),)),
    Inspect01AQEvidence("A-Q11", "Replaceable read sources", (_nid("test_inspect_01a_federation.py", "test_a_q11_replaceable_adapters"),)),
    Inspect01AQEvidence("A-Q12", "No opaque ABI on canonical path", (_nid("test_inspect_01a_contract_integrity.py", "test_a_q12_canonical_contracts_no_opaque_abi"),)),
    Inspect01AQEvidence("A-Q13", "No registry/locator/reflection", (_nid("test_inspect_01a_contract_integrity.py", "test_a_q13_federation_static_gates"),)),
    Inspect01AQEvidence("A-Q14", "Tier-3 backwards compatibility", (_nid("test_inspect_01a_qualification_batch.py", "test_a_q14_p14_runtime_inspection_regression"),)),
    Inspect01AQEvidence("A-Q15", "Frozen subsystem regressions", (_nid("test_inspect_01a_qualification_batch.py", "test_a_q15_frozen_subsystem_regression_paths"),)),
)

INSPECT_01_B_Q_CATALOG: tuple[Inspect01BQEvidence, ...] = (
    Inspect01BQEvidence("B-Q1", "Tool success read", (_nid("test_inspect_01b_core_domains.py", "test_b_q1_tool_success_read"),)),
    Inspect01BQEvidence("B-Q2", "Tool failure read", (_nid("test_inspect_01b_core_domains.py", "test_b_q2_tool_failure_read"),)),
    Inspect01BQEvidence("B-Q3", "Governance ALLOW", (_nid("test_inspect_01b_core_domains.py", "test_b_q3_governance_allow"),)),
    Inspect01BQEvidence("B-Q4", "Governance DENY", (_nid("test_inspect_01b_core_domains.py", "test_b_q4_governance_deny"),)),
    Inspect01BQEvidence("B-Q5", "Governance HITL", (_nid("test_inspect_01b_core_domains.py", "test_b_q5_governance_hitl"),)),
    Inspect01BQEvidence("B-Q6", "Continuation WAITING", (_nid("test_inspect_01b_core_domains.py", "test_b_q6_continuation_waiting"),)),
    Inspect01BQEvidence("B-Q7", "Continuation RESUMED", (_nid("test_inspect_01b_core_domains.py", "test_b_q7_continuation_resumed"),)),
    Inspect01BQEvidence("B-Q8", "Continuation recovery read", (_nid("test_inspect_01b_core_domains.py", "test_b_q8_continuation_recovery_read"),)),
    Inspect01BQEvidence("B-Q9", "Partial tool source", (_nid("test_inspect_01b_core_domains.py", "test_b_q9_partial_tool_source"),)),
    Inspect01BQEvidence("B-Q10", "Partial governance source", (_nid("test_inspect_01b_core_domains.py", "test_b_q10_partial_governance_source"),)),
    Inspect01BQEvidence("B-Q11", "Partial continuation source", (_nid("test_inspect_01b_core_domains.py", "test_b_q11_partial_continuation_source"),)),
    Inspect01BQEvidence("B-Q12", "Empty vs unavailable", (_nid("test_inspect_01b_core_domains.py", "test_b_q12_empty_vs_unavailable"),)),
    Inspect01BQEvidence("B-Q13", "Cross-tenant integrity fail closed", (_nid("test_inspect_01b_core_domains.py", "test_b_q13_cross_tenant_integrity_fail_closed"),)),
    Inspect01BQEvidence("B-Q14", "Secret redaction", (_nid("test_inspect_01b_core_domains.py", "test_b_q14_secret_redaction"),)),
    Inspect01BQEvidence("B-Q15", "Zero side effects", (_nid("test_inspect_01b_core_domains.py", "test_b_q15_zero_side_effects"),)),
    Inspect01BQEvidence("B-Q16", "Replaceable read sources", (_nid("test_inspect_01b_core_domains.py", "test_b_q16_replaceable_read_sources"),)),
    Inspect01BQEvidence("B-Q17", "No opaque ABI on new sections", (_nid("test_inspect_01b_core_domains.py", "test_b_q17_no_opaque_abi_on_new_sections"),)),
    Inspect01BQEvidence("B-Q18", "No registry in federation", (_nid("test_inspect_01b_core_domains.py", "test_b_q18_federation_still_no_registry"),)),
    Inspect01BQEvidence("B-Q19", "INSPECT-01-A regression", (_nid("test_inspect_01b_qualification_batch.py", "test_b_q19_inspect_01_a_regression_paths"),)),
    Inspect01BQEvidence("B-Q20", "Frozen TR/GV/SESSION regression", (_nid("test_inspect_01b_qualification_batch.py", "test_b_q20_frozen_subsystem_regression_paths"),)),
)
