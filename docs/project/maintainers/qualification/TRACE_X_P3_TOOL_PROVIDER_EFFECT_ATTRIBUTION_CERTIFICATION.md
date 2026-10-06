# TRACE-X-P3 — Tool, Provider & Side-Effect Authorization Attribution

**Parent:** TRACE-X — End-to-End Traceability & Evidence Certification

**START_HEAD:** `3572e6ed1c894859ac419770126e02d79e07208e` (P3 work baseline — ancestry anchor, not final evidence SHA)

**P3 qualification evidence commit (initial):** `d92e78432d58bc9f77f70292c2c59bb2e2003534`

**P3-Q1 accepted final evidence:** `1780e2efebb6160b262e49bb3f8e8b4c0cf957c3`

**Qualification replayability defect:** `P3-Q-BLK-01` — **RESOLVED** in TRACE-X-P3-Q1 (provenance gate uses `merge-base --is-ancestor` semantics).

**Mechanical SSOT:** `tests/qualification/trace_x/_trace_x_p3_support.py`

**Applicable FRZ (P3 only):** `FRZ-TRC-03`, `FRZ-TRC-04`, `FRZ-TRC-06`

**Status:**

| Stage | State |
|---|---|
| **TRACE-X-P3** | **BLOCKED** (partial closure) |
| **TRACE-X-P3-Q1** | **CLOSED / INDEPENDENTLY ACCEPTED** @ `1780e2efebb6160b262e49bb3f8e8b4c0cf957c3` |
| **TRACE-X-P3-R1** | **REQUIRED / NEXT / NOT ENTERED** |

**Next engineering task:** **TRACE-X-P3-R1** — Versioned Governed Provider/Effect Execution Identity Binding — preserve canonical Task/Run/Attempt/Execution identity already known by Execution through the governed provider/effect boundary so a provider invocation and external side effect can be attributed to the exact execution and authorization without heuristics or a second evidence authority.

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
| Execution Evidence boundary composition | exactly-one boundary evidence composition owner (**unchanged**) |

**Critical:** `intergrax/contracts/execution_evidence/boundary_event.py` (`ExecutionBoundaryEvent`) ≠ `intergrax/runtime/attestation/execution_boundary_event.py` (`ExecutionBoundaryEventV1`).

**Forbidden (P3-R1 direction):** new `ProviderAttributionService`, `ProviderEffectReconstructor`, or side-effect truth store. Governance remains permission owner; boundary evidence records the already-made decision only. Execution remains Task/Run/Attempt/Execution identity authority; evidence propagates only.

---

## 2. Attribution chains

### Tool (FRZ-TRC-03) — **PASS / independently accepted**

```text
Active execution identity
  → RuntimeToolInvoker.state.trace_event(tool_invocation_*)
  → trace_event_to_runtime_event (TOOL_REQUESTED / COMPLETED / DENIED / FAILED)
  → RuntimeEvent (tenant, task, run, attempt, execution)
  → ExecutionReconstructor
```

Accepted invariant: tool invocation → `RuntimeEvent` → exact `AttemptId` + `ExecutionId` via canonical active execution identity and sanctioned trace→runtime-event bridge.

Tool diagnostic payloads carry `tool_id` / `step_id`; execution identity lives on the `RuntimeEvent` envelope.

### Provider (FRZ-TRC-04) — **BLOCKED** (`P3-B04-01`)

```text
Governed ExecutionBoundaryEvent v1: task_id + run_id + ProviderInvocationSection(invocation_id)
  — no AttemptId / ExecutionId fields
```

`GovernedProofProfile.execution_ref` ≠ canonical `ExecutionId` authority (may default to `run_id` for compatibility; no semantic aliasing in P3-R1).

### Side-effect authorization (FRZ-TRC-06) — **BLOCKED** (`P3-B06-01`)

`GovernanceDecisionEvidenceFact` can carry full execution correlation; fresh scope-bound authorization is enforced in production paths.

Linking **this exact external effect** to **this exact authorization decision** for **this exact execution** through governed boundary/provider evidence cannot be completed at execution granularity without boundary contract identity (`P3-B06-01`).

---

## 3. GovernedExecutionResult (current code)

Before boundary-event composition, the system already knows:

```text
GovernedExecutionResult.execution_id
+ GovernedExecutionResult.provider_invocation
+ GovernedExecutionResult.evaluated_policy_decision
+ GovernedExecutionResult.provider_outcome
+ GovernedExecutionResult.proof
```

Remaining gap includes canonical **`AttemptId`** propagation through governed boundary evidence — **not** claimed solved.

---

## 4. Reliability & compatibility decisions

| Item | Decision |
|---|---|
| `ProviderInvocationReliabilityFact` | **Must NOT** become canonical TRACE-X execution truth (projection only; observer optional; observer failure non-blocking). Not a substitute for FRZ-TRC-04/06 attribution. |
| `governed_execution_boundary_event.v1` | Compatibility / legacy schema — **do not** silently extend in place |
| `governed_execution_boundary_event.v2` | Future canonical writer for fully execution-attributable governed boundary evidence (P3-R1 implementation; migration mechanics → COMPAT-X later) |
| v1 deprecation | **Not** in P3-Q1 scope |

---

## 5. FRZ dispositions (independent acceptance)

| Criterion | Disposition |
|---|---|
| FRZ-TRC-03 | **PASS** / independently accepted @ `1780e2efebb6160b262e49bb3f8e8b4c0cf957c3` |
| FRZ-TRC-04 | **BLOCKED** — `P3-B04-01` |
| FRZ-TRC-06 | **BLOCKED** — `P3-B06-01` |

---

## 6. Findings

| ID | Classification | Summary |
|---|---|---|
| P3-B04-01 | **OPEN / CONFIRMED** | Provider/boundary evidence lacks canonical AttemptId/ExecutionId |
| P3-B06-01 | **OPEN / CONFIRMED** | Effect↔authorization exact chain blocked at boundary join |

---

## 7. Approved P3-R1 architecture direction

```text
governed_execution_boundary_event.v2
  owned by the SAME existing Execution Evidence semantic/composition owner
  carries full canonical execution identity: TaskId, RunId, AttemptId, ExecutionId
  propagated from Execution only
```

Identity must **never** be minted by the evidence layer, inferred from `correlation_id`, inferred from `step_id`, or reconstructed from timestamps.

---

## 8. TRACE-X-P3-Q1 closure record

| Item | State |
|---|---|
| TRACE-X-P3-Q1 | **CLOSED / INDEPENDENTLY ACCEPTED** @ `1780e2efebb6160b262e49bb3f8e8b4c0cf957c3` |
| P3-Q-BLK-01 | **RESOLVED** |
| FRZ-TRC-03 | **PASS** / independently accepted |
| Authorization suite | **Stale test fixture** — production defect = **NO** |
| P3-B04-01 | **OPEN / CONFIRMED** |
| P3-B06-01 | **OPEN / CONFIRMED** |

---

## 9. Roadmap (current)

```text
TRACE-X     = CURRENT / BLOCKED
TRACE-X-P3  = BLOCKED
TRACE-X-P3-Q1 = CLOSED / independently accepted
TRACE-X-P3-R1 = NEXT / REQUIRED / NOT ENTERED
TRACE-X-P4  = NOT ENTERED
TRACE-X-P5  = NOT ENTERED
TRACE-X-P6  = NOT ENTERED
TRACE-X-CERT = NOT ENTERED
```

---

## 10. Tests (evidence baseline)

Evidence accepted @ `1780e2efebb6160b262e49bb3f8e8b4c0cf957c3` (not re-run for P3-Q1 bookkeeping closure):

1. `tests/qualification/trace_x/test_trace_x_p3_tool_provider_effect_attribution.py`
2. `tests/unit/runtime/tools/test_fresh_side_effect_authorization.py`
3. Boundary / governance evidence unit tests

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
