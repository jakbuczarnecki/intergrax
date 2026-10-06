# TRACE-X-P3 — Tool, Provider & Side-Effect Authorization Attribution

**Parent:** TRACE-X — End-to-End Traceability & Evidence Certification

**START_HEAD:** `3572e6ed1c894859ac419770126e02d79e07208e` (P3 work baseline — ancestry anchor, not final evidence SHA)

**P3 qualification evidence commit (initial):** `d92e78432d58bc9f77f70292c2c59bb2e2003534`

**P3-Q1 remediation baseline:** `d92e78432d58bc9f77f70292c2c59bb2e2003534` (`TRACE_X_P3_Q1_START_HEAD`)

**Qualification replayability defect:** `P3-Q-BLK-01` — TXP3-Q01 required `HEAD == START_HEAD` instead of `START_HEAD` ancestor of `HEAD` (fixed in TRACE-X-P3-Q1).

**Mechanical SSOT:** `tests/qualification/trace_x/_trace_x_p3_support.py`

**Applicable FRZ (P3 only):** `FRZ-TRC-03`, `FRZ-TRC-04`, `FRZ-TRC-06`

**Production delta @ START_HEAD:** qualification-first — **no production code changes**

**Status:** **TRACE-X-P3 = BLOCKED** (partial closure; see FRZ dispositions) · **TRACE-X-P3-Q1** — replayability + authorization-suite reconciliation (see §8)

**Remediation track:** `TRACE-X-P3-R1` — architecture decision for execution-level identity on governed provider/boundary evidence

---

## 1. Scope

Prove whether canonical evidence reconstructs, without heuristic joins:

```text
ExecutionId → tool invocation → provider invocation → external effect → Governance authorization evidence
```

Preserved ownership (unchanged from TRACE-X-P0..P2):

| Concern | Owner |
|---|---|
| Runtime execution facts | `RuntimeEvent` |
| Execution topology | `ExecutionLineage` |
| Factual reconstruction | `ExecutionReconstructor` (exactly-one) |
| Governance decisions (evidence) | `GovernanceDecisionEvidenceFact` |
| Permission semantics | `Governance` |
| Execution semantics | `Execution` |
| Harness export | `ExecutionBoundaryEventV1` (**non-authoritative**) |

**Critical:** `intergrax/contracts/execution_evidence/boundary_event.py` (`ExecutionBoundaryEvent`) ≠ `intergrax/runtime/attestation/execution_boundary_event.py` (`ExecutionBoundaryEventV1`).

---

## 2. Attribution chains (@ START_HEAD)

### Tool (FRZ-TRC-03)

```text
Active execution identity
  → RuntimeToolInvoker.state.trace_event(tool_invocation_*)
  → trace_event_to_runtime_event (TOOL_REQUESTED / COMPLETED / DENIED / FAILED)
  → RuntimeEvent (tenant, task, run, attempt, execution)
  → ExecutionReconstructor
```

Tool diagnostic payloads (`ToolInvocationStartDiagV1`) carry `tool_id` / `step_id`; **execution identity lives on the RuntimeEvent envelope**, not in the diag payload alone.

### Provider (FRZ-TRC-04) — **BLOCKED**

```text
Governed ExecutionBoundaryEvent: task_id + run_id + ProviderInvocationSection(invocation_id)
  — no AttemptId / ExecutionId fields
```

`GovernedProofProfile.execution_ref` is optional opaque string — **not** canonical `ExecutionId`.

**STOP — ARCHITECTURE DECISION REQUIRED** (`TRACE-X-P3-R1`) before public contract expansion.

### Side-effect authorization (FRZ-TRC-06) — **BLOCKED**

`GovernanceDecisionEvidenceFact` **can** carry full execution correlation (`has_full_execution_correlation`), and `MeaningfulSideEffectAuthorizationPort` enforces fresh scope-bound authorization (see `test_fresh_side_effect_authorization.py`).

However, linking **this exact external effect** to **this exact authorization decision** for **this exact execution** through governed boundary/provider evidence cannot be completed at execution granularity without boundary contract identity (`P3-B06-01`).

---

## 3. Closed-world inventory (summary)

| Path | Producer | Persistence | Execution identity | Auth identity |
|---|---|---|---|---|
| Tool trace → RuntimeEvent | `RuntimeToolInvoker` | `EvidencePersistencePort` | Full typed ids on `RuntimeEvent` | Pre-invoke governance gates |
| Governed boundary | `compose_execution_boundary_event` | execution evidence ports | task + run only | `PolicyDecisionSection` + evidence pointer |
| Attestation export | `ExecutionBoundaryEmitter` | `BoundaryEventBuffer` | task + run + step_id | export verdicts only |
| Governance evidence | `GovernanceEvidenceRecorder` | `GovernanceEvidencePersistence` | optional full correlation | `evidence_id` |

---

## 4. FRZ dispositions (Cursor recommendation — not PASS)

| Criterion | Disposition |
|---|---|
| FRZ-TRC-03 | **READY FOR INDEPENDENT CLOSURE REVIEW** |
| FRZ-TRC-04 | **BLOCKED** — `P3-B04-01` |
| FRZ-TRC-06 | **BLOCKED** — `P3-B06-01` |

---

## 5. Findings

| ID | Classification | Summary |
|---|---|---|
| P3-B04-01 | IN-SCOPE BLOCKER | Provider/boundary evidence lacks canonical AttemptId/ExecutionId |
| P3-B06-01 | IN-SCOPE BLOCKER | Effect↔authorization exact chain blocked at boundary join |

---

## 6. Roadmap

```text
TRACE-X-P3 = BLOCKED
TRACE-X = CURRENT / BLOCKED
TRACE-X-P4 = NOT ENTERED
```

Unblock via **TRACE-X-P3-R1** (contract evolution decision), then re-run qualification.

---

## 7. Tests

1. `tests/qualification/trace_x/test_trace_x_p3_tool_provider_effect_attribution.py`
2. `tests/unit/runtime/tools/test_fresh_side_effect_authorization.py` (+ tool correlation / trace bridge unit tests)
3. Boundary / governance evidence unit tests (`ExecutionBoundaryEvent`, `GovernanceDecisionEvidenceFact`, attestation emitter)

---

## 8. TRACE-X-P3-Q1 (replayability)

| Item | State |
|---|---|
| P3-Q-BLK-01 | Remediated — provenance gate uses `merge-base --is-ancestor` semantics |
| P3-B04-01 | **Open** — provider/boundary execution identity gap unchanged |
| P3-B06-01 | **Open** — effect↔authorization exact chain still blocked |
| Authorization suite | Stale fixture — tests bind `canonical_governed_execution_scope` + governance identity for declarative policy paths (`record_governance_policy_decision_evidence_for_active_identity`) |

**Q1 final evidence commit:** recorded in repository `development` HEAD after push (not embedded here to avoid commit-loop).

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
