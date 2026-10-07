# TRACE-X-P3 — Tool, Provider & Side-Effect Authorization Attribution

**Parent:** TRACE-X — End-to-End Traceability & Evidence Certification

**START_HEAD:** `3572e6ed1c894859ac419770126e02d79e07208e` (P3 work baseline — ancestry anchor, not final evidence SHA)

**P3 qualification evidence commit (initial):** `d92e78432d58bc9f77f70292c2c59bb2e2003534`

**P3-Q1 accepted final evidence:** `1780e2efebb6160b262e49bb3f8e8b4c0cf957c3`

**P3 accepted evidence/code baseline (final):** `3799b2d974369e6002ac5326e62c8c8b381e7944`

**Qualification replayability defect:** `P3-Q-BLK-01` — **RESOLVED** in TRACE-X-P3-Q1 (provenance gate uses `merge-base --is-ancestor` semantics).

**Mechanical SSOT:** `tests/qualification/trace_x/_trace_x_p3_support.py`

**P3-R1 mechanical SSOT:** `tests/qualification/trace_x/_trace_x_p3_r1_support.py`

**Applicable FRZ (P3):** `FRZ-TRC-03`, `FRZ-TRC-04`, `FRZ-TRC-06`

**Status:**

| Stage | State |
|---|---|
| **TRACE-X-P3** | **CLOSED / INDEPENDENTLY ACCEPTED** @ `3799b2d974369e6002ac5326e62c8c8b381e7944` |
| **TRACE-X-P3-Q1** | **CLOSED / INDEPENDENTLY ACCEPTED** @ `1780e2efebb6160b262e49bb3f8e8b4c0cf957c3` |
| **TRACE-X-P3-R1** | **CLOSED / INDEPENDENTLY ACCEPTED** @ `3799b2d974369e6002ac5326e62c8c8b381e7944` |
| **TRACE-X-P4** | **NEXT / REQUIRED / NOT ENTERED** |

**R1 qualification record:** [`TRACE_X_P3_R1_GOVERNED_BOUNDARY_V2_CERTIFICATION.md`](TRACE_X_P3_R1_GOVERNED_BOUNDARY_V2_CERTIFICATION.md)

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

**Authority planes (unchanged):** **Governance** = permission authority; **Execution** = effect/execution authority; **Evidence Plane** = descriptive factual recording only.

**Critical:** `intergrax/contracts/execution_evidence/boundary_event.py` (`ExecutionBoundaryEvent`) ≠ `intergrax/runtime/attestation/execution_boundary_event.py` (`ExecutionBoundaryEventV1`).

**Forbidden:** new `ProviderAttributionService`, `ProviderEffectReconstructor`, or side-effect truth store. Governance remains permission owner; boundary evidence records the already-made decision only. Execution remains Task/Run/Attempt/Execution identity authority; evidence propagates only.

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

### Provider (FRZ-TRC-04) — **PASS / independently accepted** @ `3799b2d974369e6002ac5326e62c8c8b381e7944`

```text
governed_execution_boundary_event.v2
  TaskId + RunId + AttemptId + ExecutionId (from Execution only)
  + ProviderInvocationSection(invocation_id)
  atomically in versioned execution evidence
```

**Execution** = sole execution identity authority. No heuristic join; no host-minted canonical `ExecutionId`. **`P3-B04-01`** = **RESOLVED**.

### Side-effect authorization (FRZ-TRC-06) — **PASS / independently accepted** @ `3799b2d974369e6002ac5326e62c8c8b381e7944`

Certified chain:

```text
authorization
  → Task/Run/Attempt/Execution
  → provider invocation
  → provider outcome
  → governed side-effect evidence
  → signed ProofReceiptV2
```

Adversarial evidence: cross-execution mismatch rejected; cross-attempt mismatch rejected; cross-tenant mismatch rejected; DENY cannot produce successful receipt.

**GovernanceEvidenceRef** may be emitted only when `GovernanceEvidencePersistenceOutcome.persisted == True` and `evidence_id` matches exactly. `persisted=False` → no **GovernanceEvidenceRef** → no dangling signed evidence reference.

**GR-8** evidence persistence failure does **not** convert ALLOW into DENY — it only means no durable **GovernanceEvidenceRef** may be claimed for the unpersisted fact. Evidence persistence is **not** authorization authority.

**`P3-B06-01`** = **RESOLVED**.

---

## 3. GovernedExecutionResult

At boundary composition, canonical identity and attribution are bound through **v2** boundary evidence and **ProofReceiptV2** pairing (cross-version pairings fail closed).

---

## 4. Reliability & compatibility decisions

| Item | Decision |
|---|---|
| `ProviderInvocationReliabilityFact` | **Must NOT** become canonical TRACE-X execution truth (projection only; observer optional; observer failure non-blocking). Not a substitute for FRZ-TRC-04/06 attribution. |
| `governed_execution_boundary_event.v1` | Compatibility / legacy schema — retained; **do not** silently extend in place |
| `governed_execution_boundary_event.v2` | Canonical current execution-attributable writer |
| ProofReceipt v1 ↔ boundary v1; ProofReceipt v2 ↔ boundary v2 | Cross-version pairings fail closed |
| v1 deprecation | **Not** required for P3 closure |

---

## 5. FRZ dispositions (independent acceptance)

| Criterion | Disposition |
|---|---|
| FRZ-TRC-03 | **PASS** / independently accepted @ `1780e2efebb6160b262e49bb3f8e8b4c0cf957c3` |
| FRZ-TRC-04 | **PASS** / independently accepted @ `3799b2d974369e6002ac5326e62c8c8b381e7944` |
| FRZ-TRC-06 | **PASS** / independently accepted @ `3799b2d974369e6002ac5326e62c8c8b381e7944` |

---

## 6. Findings

| ID | Classification | Summary |
|---|---|---|
| P3-B04-01 | **RESOLVED** | Provider boundary preserves canonical Task/Run/Attempt/Execution + `ProviderInvocation.invocation_id` via v2 |
| P3-B06-01 | **RESOLVED** | Exact authorization→effect chain via v2 + ProofReceiptV2 |
| P3-Q-BLK-01 | **RESOLVED** | Qualification replayability / provenance gate |
| P3-R1-BLK-TASK-IDENTITY-01 | **RESOLVED** | Task identity propagation through governed boundary |
| P3-R1-BLK-TENANT-01 | **RESOLVED** | Tenant isolation on governed attribution path |

---

## 7. P3-R1 architecture (accepted)

```text
governed_execution_boundary_event.v2
  owned by the SAME existing Execution Evidence semantic/composition owner
  carries full canonical execution identity: TaskId, RunId, AttemptId, ExecutionId
  propagated from Execution only
```

Identity must **never** be minted by the evidence layer, inferred from `correlation_id`, inferred from `step_id`, or reconstructed from timestamps.

---

## 8. TRACE-X-P3 closure record

| Item | State |
|---|---|
| TRACE-X-P3 | **CLOSED / INDEPENDENTLY ACCEPTED** @ `3799b2d974369e6002ac5326e62c8c8b381e7944` |
| TRACE-X-P3-Q1 | **CLOSED / INDEPENDENTLY ACCEPTED** @ `1780e2efebb6160b262e49bb3f8e8b4c0cf957c3` |
| TRACE-X-P3-R1 | **CLOSED / INDEPENDENTLY ACCEPTED** @ `3799b2d974369e6002ac5326e62c8c8b381e7944` |
| P3-Q-BLK-01 | **RESOLVED** |
| P3-B04-01 / P3-B06-01 | **RESOLVED** |
| FRZ-TRC-03 / FRZ-TRC-04 / FRZ-TRC-06 | **PASS** / independently accepted |

---

## 9. Roadmap (current)

```text
TRACE-X     = CURRENT
TRACE-X-P3  = CLOSED / independently accepted @ 3799b2d9…
TRACE-X-P3-Q1 = CLOSED / independently accepted @ 1780e2e…
TRACE-X-P3-R1 = CLOSED / independently accepted @ 3799b2d9…
TRACE-X-P4  = NEXT / REQUIRED / NOT ENTERED
TRACE-X-P5  = NOT ENTERED
TRACE-X-P6  = NOT ENTERED
TRACE-X-CERT = NOT ENTERED
```

---

## 10. Tracked freeze debt (does not block P3)

| Item | Classification |
|---|---|
| `applications/governed_contractor_application/tests/host/test_gr6_wire_production_decision_governance.py::test_strict_host_composition_wires_agent_boundary_and_integration` → `ToolDependencyAttemptBoundaryMaterializationError` | **TRACKED FREEZE DEBT** — Reliability / Composition / PROD-Q / QUAL-X |

---

## 11. Tests (evidence baseline)

Evidence accepted @ `3799b2d974369e6002ac5326e62c8c8b381e7944` (independent audit baseline — not re-run for docs-only closure bookkeeping):

1. `tests/qualification/trace_x/test_trace_x_p3_tool_provider_effect_attribution.py`
2. `tests/qualification/trace_x/test_trace_x_p3_r1_governed_boundary_v2.py`
3. `tests/unit/runtime/tools/test_fresh_side_effect_authorization.py`
4. Boundary / governance evidence unit tests (see P3-R1 certification)

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
