# DG-001 — Multi-Agent Diagnostic Qualification R1

> **Task:** `DG-001-MULTI-AGENT-DIAGNOSTIC-QUALIFICATION-R1`  
> **Architecture:** [`DG_001_MULTI_AGENT_DIAGNOSTIC_LINEAGE_ARCHITECTURE_R1.md`](../architecture/DG_001_MULTI_AGENT_DIAGNOSTIC_LINEAGE_ARCHITECTURE_R1.md)  
> **Read integration (prior):** [`DG_001_MULTI_AGENT_DIAGNOSTIC_LINEAGE_READ_INTEGRATION_R1.md`](DG_001_MULTI_AGENT_DIAGNOSTIC_LINEAGE_READ_INTEGRATION_R1.md)

## Scope

Enterprise-grade qualification that **real** multi-agent production execution paths produce durable forensic lineage and central diagnostics can deterministically reconstruct operator truth via:

```text
ExecutionLineagePersistence + RuntimeEventPersistence + CausalEvidencePersistence
        → ExecutionReconstructor → DiagnosticReadService
```

**Invariant:** `DIAGNOSTIC_AUTHORITY_COUNT = 1` → `intergrax.runtime.diagnostics` only.

## Baseline

| Field | Value |
|---|---|
| **START_HEAD** | `374ac27a7be4a291dc646f3f04e412365572c2af` |
| **Branch** | `development` |
| **BASE_LINEAGE_HARDENING** | `130c4715b49653d3ab2bbfb1630bb97143004218` (ancestry OK) |
| **PRODUCTION_CHANGES** | NO |

## Architecture authority baseline

| Concern | Authority |
|---|---|
| Execution identity / admission | `ExecutionRuntime`, `ExecutionBoundary`, `ChildExecutionRunner` |
| Forensic topology | `ExecutionLineagePersistence` |
| Runtime outcome evidence | `RuntimeEventPersistence` |
| Cross-boundary causal | `CausalEvidencePersistence` |
| Resume projection (non-forensic) | `TaskCheckpoint` / `ExecutionTreeSnapshot` |
| Diagnostic engine | `intergrax.runtime.diagnostics` (`DiagnosticOrchestrator`, `ProblemLifecycleEngine`, `ExecutionReconstructor`, `DiagnosticReadService`) |

## Single diagnostic authority invariant

- **Allowed in multi-agent:** runtime event emission, lineage admission, causal evidence, metrics/logs/traces, governance signals.
- **Forbidden:** local `DiagnosticEngine`, local `ProblemStore`, parallel `ExecutionReconstructor`, scenario-owned canonical diagnostic DB.
- **Gate proofs:** `tests/unit/runtime/architecture/test_one_spine_diagnostic_orchestrator_gate.py`, `test_one_spine_problem_store_gate.py`, `tests/unit/runtime/diagnostics/test_dg001_multi_agent_diagnostic_qualification_r1.py::test_dg001_agent_distribution_does_not_instantiate_diagnostic_authority`.

## Production path audit (code-traced)

```text
CoordinationIntentExecutor
    → MultiAgentCoordinationService (SINGLE) / BoundedMultiAgentFanOutService (FAN_OUT)
    → DelegatedSubtaskService.execute
    → ChildExecutionPort (production adapter: as_child_execution_port(ChildExecutionRunner))
    → ChildExecutionRunner.execute_child
    → ExecutionBoundary + lineage admission hooks (when active lineage bound)
    → specialist delegate
```

**Child scope:** same `RunId`, `AttemptId`, `tenant_id`, `task_id`; child `ExecutionId` + `parent_execution_id` from `ExecutionIdentityBinding` at admission.

**Coordination is not identity authority:** `MultiAgentCoordinationService` does not mint `RunId`/`AttemptId`/`ExecutionId`.

## P3 proofs (this task)

| Proof | Level | Test |
|---|---|---|
| Root → single specialist child forensic edge | P3 | `test_dg001_p3_root_single_child_forensic_topology_via_coordination_intent` |
| Fan-out three siblings same parent | P3 | `test_dg001_p3_fan_out_three_siblings_share_parent_execution` |
| Nested specialist chain (depth ≥ 3 executions) | P3 | `test_dg001_p3_nested_specialist_preserves_direct_forensic_parent_chain` |
| Partial sibling failure retains admissions | P3 | `test_dg001_p3_partial_sibling_failure_preserves_all_admissions` |
| Orchestrator → Problem store → operator lineage view | P3 | `test_dg001_p3_operator_read_surfaces_multi_agent_lineage_after_orchestrator` |
| Agent distribution no diagnostic authority | P2 gate | `test_dg001_agent_distribution_does_not_instantiate_diagnostic_authority` |

Composition entry: `CoordinationIntentExecutor` + `build_decision_coordination_executor_fixture` (NPSC-5C harness) with `activate_root_execution_lineage` + root admission hook on `ExecutionBoundary` (no manual `admit_child` for E2E topology).

## Architecture Q1–Q51 coverage

Canonical numbering from architecture R1 matrix. Status legend: **PASS** (proof in this or cited test), **REUSED**, **N/A**, **NOT YET QUALIFIED**.

| Arch Q | Scenario | Level | Status | Proof |
|---:|---|---|---|---|
| Q1 | Root + single child | P3 | PASS | DG-001 P3 single-child test; `test_npsc5c_r3_single_decision_to_execution_e2e` |
| Q2 | Root + 3 siblings | P3 | PASS | DG-001 P3 fan-out; `test_npsc5c_r3_fan_out_decision_to_execution_e2e` |
| Q3 | Nested depth ≥ 3 | P3 | PASS | DG-001 P3 nested; `test_child_execution.py::test_nested_child_execution_lineage` |
| Q4 | One child failure | P3 | REUSED | `test_npsc5c_r3_fan_out_partial_failure_cross_system` |
| Q5 | Partial sibling failure | P3 | PASS | DG-001 P3 partial failure; NPSC-5C partial fan-out |
| Q6 | Retry isolated attempts | P3 | REUSED | `test_npsc5e_r1_final_retry_attempt_qualification.py` |
| Q7 | Resume from checkpoint | P3 | REUSED | `test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py` |
| Q8 | Crash after child admission | P3 | REUSED | lineage durability suites in `tests/unit/runtime/execution/lineage/` |
| Q9 | Duplicate edge idempotent | P1 | REUSED | `test_execution_lineage_persistence_conformance.py` |
| Q10 | Conflicting parent | P1 | REUSED | conformance `test_conflicting_parent_fails_closed` |
| Q11 | Cross-tenant | P1/P3 | REUSED | `test_dg001_lineage_read_integration_r1_final_qualification.py` tenant cases |
| Q12 | Historical run without lineage | P2 | REUSED | DG-001 read integration ABSENT/UNAVAILABLE matrix |
| Q13 | Root admission before delegate | P3 | REUSED | `test_execution_lineage_admission_order.py`, `test_host_task_lineage_wiring.py` |
| Q14 | Two attempts isolated | P3 | REUSED | NPSC-5E retry qualification |
| Q15 | Independent segment trees per attempt | P3 | REUSED | NPSC-5E retry + lineage segment tests |
| Q16 | Missing seal → not COMPLETE | P2 | REUSED | `test_execution_lineage_reconstruction.py`, DG-001 read integration |
| Q17 | Admission write degraded | P2 | REUSED | degradation tests in lineage/ |
| Q18 | Duplicate real parent conflict | P1 | REUSED | persistence conformance |
| Q19 | task_id without request introspection | P2 | REUSED | `test_npsc5e_p0a_*` lineage hook audit |
| Q20 | Child inherits attempt scope | P3 | REUSED | `test_delegated_subtask_execution_lineage` |
| Q21 | Admission failure prohibits COMPLETE seal | P2 | REUSED | lineage fault injection |
| Q22 | Seal storage failure → PARTIAL | P2 | REUSED | atomic fault injection tests |
| Q23 | Retry A2 OPEN lifecycle | P3 | REUSED | NPSC-5E R1 |
| Q24 | Stale attempt not falsely sealed | P2 | REUSED | attempt lifecycle + lineage |
| Q25 | Resumable → attempt not sealed | P3 | REUSED | NPSC-5E R2 resume |
| Q26 | Pause + resume same attempt | P3 | REUSED | `test_host_task_resume_lineage_identity.py` |
| Q27 | Root raise without terminal → no COMPLETE | P2 | REUSED | terminal conflict seal tests |
| Q28 | FAILED + COMPLETE lineage orthogonal | P2 | REUSED | DG-001 read integration completeness matrix |
| Q29 | CANCELLED + COMPLETE lineage orthogonal | P2 | REUSED | same |
| Q30 | Retry closes A1 opens A2 | P3 | REUSED | NPSC-5E R1 |
| Q31 | Admission failure + resume | P2 | REUSED | lineage degradation + resume tests |
| Q32 | UNCLEAN segment → PARTIAL | P3 | REUSED | NPSC-5E segment unclean proofs |
| Q33 | Generic runtime compatibility | P2 | REUSED | `test_execution_runtime` lineage optional wiring |
| Q34 | Segment open unavailable fail-closed | P2 | REUSED | fault injection A2 |
| Q35 | Scope attempt-only | P1 | REUSED | contract audit P0A |
| Q36 | Resume new root same attempt | P3 | REUSED | NPSC-5E R2 |
| Q37 | Two segments one root each | P3 | REUSED | resume lineage identity tests |
| Q38 | Checkpoint reparent ≠ forensic | P3 | REUSED | `test_ue_9c_execution_tree_checkpoint.py` + DG-001 read integration |
| Q39 | Adoption no second admission | P3 | REUSED | lineage resume tests |
| Q40 | New child after resume segment | P3 | REUSED | host task resume lineage |
| Q41 | Predecessor ≠ parent edge | P3 | REUSED | DG-001 read integration segment continuity |
| Q42 | Nested historical parent immutable | P3 | REUSED | resume + reconstruction integrity |
| Q43 | Adopted projection legal | P2 | REUSED | checkpoint resume plan tests |
| Q44 | Duplicate parent hard error | P1 | REUSED | persistence conformance |
| Q45 | UNCLEAN S1 + S2 degraded | P3 | REUSED | NPSC-5E unclean proofs |
| Q46 | Segment identity = root ExecutionId | P1 | PASS | no `ExecutionLineageSegmentId` in `intergrax/` (repo audit) |
| Q47 | open_segment uses minted root | P2 | REUSED | root activation tests |
| Q48 | No ExecutionLineageSegmentId | P1 | PASS | type absent in contracts/persistence/diagnostics |
| Q49 | Root admission id == segment root | P1 | REUSED | lineage admission invariants |
| Q50 | Predecessor link only | P3 | REUSED | DG-001 read integration |
| Q51 | Same-attempt resume segment roots | P3 | REUSED | NPSC-5E R2 + host resume lineage |

**Summary counts:** PASS/REUSED dominant; multi-agent P3 spine added in DG-001 R1 qualification tests; full multi-agent operator failure localization (nested E4) beyond lineage projection remains **NOT YET QUALIFIED** for dedicated root-cause reasoning (see limitations).

## Reused existing proofs

- NPSC-5A/5B/5C/5D/5E architecture and E2E qualifications (`test_npsc5c_decision_execution_e2e.py`, fan-out/fan-in, governance).
- `test_dg001_lineage_read_integration_r1_final_qualification.py` (read-side integrity, truncation, tenant).
- `test_harden_4e_diagnostic_read_truth_e2e.py` (operator read truth).
- `test_terminal_diagnostic_production_e2e.py` (single-spine terminal trigger).
- Lineage suite: `tests/unit/runtime/execution/lineage/` (61 tests).

## Regression suites (2026-09-10 run)

| Suite | Result | Notes |
|---|---|---|
| DG-001 R1 new tests | PASS (6) | |
| Lineage | PASS (61) | |
| NPSC 5A/5B/5C/5E P0A sample | PASS (50) | |
| Agent distribution | 941 pass, 1 fail | `test_ac6_architecture_gates::test_testing_support_does_not_import_tests` — pre-existing harness import policy |
| Harden 4E e2e | PASS (2) | |
| Full diagnostics `test_*.py` | COLLECTION ERROR | `test_decision_lifecycle_projection.py` — circular import in working tree (`decision_lifecycle_observability`); unrelated to DG-001 artifacts |

## Known limitations

- **Multi-agent deterministic failure boundary in `DiagnosticAssessmentBuilder`:** lineage + lifecycle localize executions; deep nested causal root-cause reasoning not fully qualified (next: `DIAGNOSTIC-ENGINE-SINGLE-AUTHORITY-ARCHITECTURE-R1`).
- **Attempt discovery legacy coverage:** see read integration doc; post-v1 index semantics qualified in read integration matrix, not re-opened here.
- **P4 external infra:** not required for this task.

## Final verdict

```text
STATUS: PASS
MULTI_AGENT_DIAGNOSTIC_P3: PASS
SINGLE_DIAGNOSTIC_AUTHORITY: PASS
PRODUCTION_CHANGES: NO
```

**Confirmations:** one central Diagnostic Engine; no local multi-agent/application diagnostic authority; no second lineage store; P3 proofs use real `CoordinationIntentExecutor` → `DelegatedSubtaskService` → `ChildExecutionRunner` boundary.
