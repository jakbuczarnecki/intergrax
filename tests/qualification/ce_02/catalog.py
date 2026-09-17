# © Artur Czarnecki. All rights reserved.

"""CE-02 CE2-Q1..CE2-Q18 evidence catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Ce02QEvidence:
    q_id: str
    title: str
    pytest_node_ids: tuple[str, ...]


def _nid(test_file: str, test_name: str) -> str:
    return f"tests/qualification/ce_02/{test_file}::{test_name}"


def _ext(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


CE_02_Q_CATALOG: tuple[Ce02QEvidence, ...] = (
    Ce02QEvidence(
        "CE2-Q1",
        "One authoritative global context budget owner",
        (_nid("test_ce_02_gates.py", "test_ce2_q1_single_budget_resolution_entry"),),
    ),
    Ce02QEvidence(
        "CE2-Q2",
        "Typed budget contract (ResolvedModelContextBudget)",
        (_nid("test_ce_02_gates.py", "test_ce2_q2_typed_resolved_budget_contract"),),
    ),
    Ce02QEvidence(
        "CE2-Q3",
        "Provider-neutral model capability snapshot",
        (_nid("test_ce_02_gates.py", "test_ce2_q3_capability_snapshot_no_vendor_types_in_tier0"),),
    ),
    Ce02QEvidence(
        "CE2-Q4",
        "Replaceable token counting",
        (_nid("test_ce_02_gates.py", "test_ce2_q4_custom_token_counter_injection"),),
    ),
    Ce02QEvidence(
        "CE2-Q5",
        "Replaceable budget allocation / model budget policy",
        (_nid("test_ce_02_gates.py", "test_ce2_q5_custom_model_budget_policy"),),
    ),
    Ce02QEvidence(
        "CE2-Q6",
        "Replaceable compaction strategy",
        (_nid("test_ce_02_gates.py", "test_ce2_q6_custom_compaction_strategy"),),
    ),
    Ce02QEvidence(
        "CE2-Q7",
        "Replaceable degradation policy",
        (_nid("test_ce_02_gates.py", "test_ce2_q7_custom_degradation_policy"),),
    ),
    Ce02QEvidence(
        "CE2-Q8",
        "Mandatory content preservation in allocator",
        (_nid("test_ce_02_gates.py", "test_ce2_q8_mandatory_fragment_preserved"),),
    ),
    Ce02QEvidence(
        "CE2-Q9",
        "Unsatisfiable mandatory budget fails closed",
        (_nid("test_ce_02_gates.py", "test_ce2_q9_unsatisfiable_mandatory_budget"),),
    ),
    Ce02QEvidence(
        "CE2-Q10",
        "Tenant/authority/sensitivity invariants on compaction",
        (_nid("test_ce_02_gates.py", "test_ce2_q10_compaction_preserves_governance_fields"),),
    ),
    Ce02QEvidence(
        "CE2-Q11",
        "Compaction provenance lineage",
        (_nid("test_ce_02_gates.py", "test_ce2_q11_compaction_provenance"),),
    ),
    Ce02QEvidence(
        "CE2-Q12",
        "Final compiled context respects model window",
        (_nid("test_ce_02_gates.py", "test_ce2_q12_model_window_after_compile"),),
    ),
    Ce02QEvidence(
        "CE2-Q13",
        "Planner/compiler semantic consistency (FINAL_COMPILE_MUTATED_PLAN)",
        (_nid("test_ce_02_gates.py", "test_ce2_q13_compile_plan_invariant_guard_present"),),
    ),
    Ce02QEvidence(
        "CE2-Q14",
        "Deterministic replay of budget resolution",
        (_nid("test_ce_02_gates.py", "test_ce2_q14_deterministic_budget_replay"),),
    ),
    Ce02QEvidence(
        "CE2-Q15",
        "No hidden truncation in CE tier-0 / canonical engine scan",
        (_nid("test_ce_02_gates.py", "test_ce2_q15_hidden_truncation_gate"),),
    ),
    Ce02QEvidence(
        "CE2-Q16",
        "No vendor SDK imports in CE tier-0 budget module",
        (_nid("test_ce_02_gates.py", "test_ce2_q16_tier0_budget_vendor_import_gate"),),
    ),
    Ce02QEvidence(
        "CE2-Q17",
        "No dynamic ABI anti-patterns in budget contracts",
        (_nid("test_ce_02_gates.py", "test_ce2_q17_budget_contract_static_abi"),),
    ),
    Ce02QEvidence(
        "CE2-Q18",
        "Executable CE-02 qualification batch",
        (_nid("test_ce_02_qualification_batch.py", "test_ce_02_full_qualification_evidence_batch"),),
    ),
)
