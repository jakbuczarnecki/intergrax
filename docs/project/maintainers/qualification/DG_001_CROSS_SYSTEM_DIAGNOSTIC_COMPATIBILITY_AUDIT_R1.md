# DG-001 — Cross-system diagnostic compatibility audit (R1)

> **Task:** `DG-001-CROSS-SYSTEM-DIAGNOSTIC-COMPATIBILITY-AUDIT-R1`  
> **Correction:** `DG-001-CROSS-SYSTEM-DIAGNOSTIC-COMPATIBILITY-AUDIT-R1-CORRECTION` (classification only)  
> **Mode:** architecture + contract + regression impact audit — **no implementation**  
> **Branch:** `development`  
> **Audit date:** 2026-09-09

---

## 1. Baseline

| Item | Value |
| ---- | ----- |
| **START HEAD** | `1b8f42143ed59a8d70fc7c5db9ae1e7283b40a3f` |
| **Correction ancestor** | `bbf21a7b1338d0c60183501a0854a00216ade37a` → `git merge-base --is-ancestor` exit 0 |
| **DG-001B3 ancestor** | `e0e99e907e5f7e1b02e452bab0e1eefc5822a07b` → `git merge-base --is-ancestor` exit 0 |
| **Reference instruction point** | `1af51065070dae2f5b777abc2e0bc9fc1821bb67` (included in ancestry) |
| **Reference commits audited** | `0936a614` (identity authority) · `30750b42` (runtime convergence) · `d3225587` (decision→execution) · `1af510650` (decision failure diagnostics) |
| **Production changes** | none |
| **Diagnostics core changed** | no |

---

## 2. Changed systems (post DG-001B3, diagnostic-relevant slice)

| System | Commits / artifacts reviewed | Diagnostic-relevant change |
| ------ | ---------------------------- | -------------------------- |
| **Execution Engine** | `0936a614` — `identity_authority.py` | Single minting authority for root/child/retry/background identity |
| **Application/runtime convergence** | `30750b42` — `HostTaskExecutionPort`, queue adapters | HTTP/queue/Celery paths converge through canonical `ExecutionRuntime` |
| **Decision System** | `1af510650`, `cc65d684`, `bf0c6dbf` | Typed coordination semantic + reconciliation failure diagnostics (scenario-owned) |
| **Agent Distribution** | `bf0c6dbf`, `d3225587` | `AuthoritativeAcceptedDecision` → `CoordinationIntent` → child executions |
| **Diagnostic read composition** | `diagnostic_read_wiring.py` (current) | Shared harness host persistence; no legacy Nexus read fallback |

---

## 3. Execution identity impact

**Verdict: PASS**

`intergrax/runtime/execution/identity_authority.py` is the sole ExecutionRuntime-owned minting surface:

| Identity | Mint owner | Consumer (Diagnostics) |
| -------- | ---------- | ------------------------ |
| Root `ExecutionId` / `RunId` / `AttemptId` | `mint_root_execution_identity()` via `ExecutionRuntime.resolve_root_execution_context` | Consumed via `TerminalExecutionDiagnosticTrigger` (`task_id`, `run_id`) and `ExecutionReconstructor` |
| Child `ExecutionId` | `mint_child_execution_id()` in `ChildExecutionRunner` | Per-event `execution_id` in `RuntimeEvent` within reconstruction scope |
| Retry `AttemptId` | `mint_retry_attempt_id()` in `attempt_lifecycle/service` | Grouped in `ExecutionReconstructor._build_attempts` |
| Background transport `TaskId`/`RunId`/`AttemptId` | `mint_background_transport_identity()` + durable `identity_persistence` | Causal evidence `RuntimeExecutionRef` at transport→execution boundary |
| `TaskId` (root host task) | `Task` model / `task_run_bridge` at task creation — not Diagnostics | Reconstruction scope key |

Diagnostics **does not mint** `ExecutionId`/`RunId`/`AttemptId` in production paths. Minting in `intergrax/runtime/diagnostics/persistence_conformance.py` and `functional_evidence_persistence_conformance.py` is conformance-only.

---

## 4. Runtime / event impact

**Verdict: PASS**

Canonical flow confirmed:

```text
HostTaskExecutionPort (host_task.py)
  → ExecutionRuntime (runtime.py)
  → StrategyExecutionRouter + terminal delegate
  → HostTaskTerminalPublisher.publish_terminal(run_id, attempt_id, execution_id)
  → RuntimeEventPersistence + causal evidence admission
  → TerminalExecutionDiagnosticTrigger → DiagnosticOrchestrator
```

Post `30750b42`, queue/HTTP paths use `HostTaskExecutionRunAdapter` / `QueuedHostTaskExecutionAdapter` — single convergence path to `ExecutionRuntime`. Background queue identity preserved via `identity_persistence` and `CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION`.

Diagnostics no longer depends on Nexus as an execution root for read composition. `diagnostic_read_wiring.py` resolves `RuntimeEventPersistence` and `CausalEvidencePersistence` from `HarnessHostRuntime` only.

---

## 5. Diagnostic reconstruction impact

**Verdict: PASS (run / attempt / event reconstruction) · GAP (multi-execution parent/child tree)**

`ExecutionReconstructor` (`execution_reconstruction.py`) reconstructs by `(tenant_id, task_id, run_id)` from `RuntimeEventPersistence` + `CausalEvidencePersistence`. Attempt grouping uses `attempt_id` from events and causal evidence.

| Check | Result |
| ----- | ------ |
| `RunId` / `AttemptId` / `TaskId` align with authority model | Yes — validated at reconstruction scope |
| Run / attempt / event reconstruction within scope | **PASS** — existing DIAG-2 paths remain valid for qualified use cases |
| `ExecutionId` in read model | Present on individual `RuntimeEvent` rows inside `positioned_events`; tags execution membership within one run — **available** |
| Exact direct parent→child execution edge | **Not available** in DIAG-2 canonical read sources — `RuntimeEvent` carries `execution_id` but not `parent_execution_id`; no parent→child `CausalRelationKind`; `DELEGATION_GRANTED` does not carry direct parent execution identity |
| Parent/child tree projection | **GAP** — DIAG-2 cannot deterministically reconstruct an arbitrary parent/child execution tree from its canonical sources alone |
| Retry attempts | Supported via attempt grouping |
| Background execution | Supported via causal evidence + shared run scope |

Do not degrade correctly working DIAG-2 for prior qualified cases. The gap is confined to multi-execution parent/child lineage read architecture (see §6, GAP-R1-01).

---

## 6. Multi-agent impact

**Verdict: GAP**

Multi-agent coordination (`CoordinationIntent` → `CoordinationIntentExecutor` → `ChildExecutionRunner`) preserves execution lineage at the **runtime** boundary:

- Children share parent `run_id` / `attempt_id`, mint distinct `execution_id`.
- `ExecutionIdentityBinding.parent_execution_id` is set in `child.py`.
- `RuntimeEvent.execution_id` tags per-event execution context — execution **membership within one run** is available in DIAG-2 canonical sources.

**Read-side architecture gap (not Execution Engine defect):** `ExecutionReconstructor` consumes `RuntimeEventPersistence` + `CausalEvidencePersistence`. `RuntimeEvent` has `execution_id` but not `parent_execution_id`. `CausalRelationKind` has no parent-execution → child-execution relation. `DELEGATION_GRANTED` does not carry direct parent execution identity. Therefore DIAG-2 **cannot deterministically reconstruct** an arbitrary parent/child execution tree from its canonical read sources. The exact direct parent→child edge is **not available** in those sources today.

**Existing canonical Execution Tree (reuse first — do not propose a parallel model):** the platform already owns:

- `ExecutionTreeSnapshot`
- `ExecutionCheckpointEntry.parent_execution_id`
- `ExecutionTreeRecorder`

The next architecture task must assess whether the correct remediation is:

- **A.** safe reuse of existing canonical Execution Tree / checkpoint truth on the Diagnostics read side,
- **B.** extension of canonical causal/runtime evidence with a parent→child relation,
- **C.** another existing public contract discovered during that future audit.

**Do not choose A/B/C in this audit.** No new parallel Execution Tree, private API, dynamic contracts, or diagnostics pipeline bypass.

---

## 7. Decision System impact

**Verdict: PASS (failure boundary) · QUALIFICATION_REQUIRED (lineage read path)**

### Decision-owned failure facts

`CompletionReconciliationDiagnostic` / `CompletionReconciliationFailureReason` / `CompletionReconciliationError` (`platform_proofs/.../completion_reconciliation.py`, commit `1af510650`) are **typed scenario reconciliation facts** attached to a raised exception. They:

- do **not** create `Problem`,
- do **not** write to Problem store,
- do **not** invoke `ProblemLifecycleEngine` or `DiagnosticOrchestrator`.

Core Decision contracts (`intergrax/contracts/decision_*.py`, `agent_distribution/decision_coordination_projection.py`) contain no `Problem` lifecycle.

### Decision → Execution identity boundary

- `DecisionIdentity` owns `decision_id` + `DecisionExecutionLineage` (`task_id`, `run_id`, `attempt_id`, optional `execution_id`).
- Decision does **not** mint `ExecutionId`; Execution authority remains in `identity_authority.py`.
- `DecisionCoordinationProjection` preserves `source_decision_identity` when projecting to `CoordinationIntent`.

Cross-run operator view `Decision → Execution → child failures` requires joining decision observability (`decision_lifecycle_observability.py`) with DIAG-2 reconstruction. Decision Lifecycle `RuntimeEvent` payload already carries `decision_id`, `task_id`, `run_id`, `attempt_id`, and `execution_id` — existing public read contracts may suffice; end-to-end diagnostic read qualification **not yet proven**. Do not introduce a new Decision diagnostics lifecycle.

---

## 8. DG-001B3 / B4 impact

**DG-001B3: VALID**

`HostedBootstrapFailureRecord` (`intergrax/hosting/bootstrap_failure.py`) operates **before B5 / before canonical execution**. Post-B3/B4/B5 parallel changes to ExecutionRuntime, Decision, and Agent Distribution do not alter:

- pre-B5 bootstrap phase semantics,
- `bootstrap_attempt_id` minting,
- explicit docstring: *"not a Problem"*.

**DG-001B4: B4_READY (architecture / proof design valid)**

Existing B4 qualification was executed at `1af51065070dae2f5b777abc2e0bc9fc1821bb67` (`DG_001B4_WORKER_PRE_B5_INTEGRATION_QUALIFICATION.md`). Worker bootstrap path is orthogonal to canonical execution identity changes; B4 architecture and proof design remain valid. Focused bootstrap test `tests/unit/hosting/test_bootstrap_failure_record.py` passes at correction HEAD.

Before formal B4 closure, a **routine focused B4 proof re-run on current HEAD** is required. The multi-agent diagnostic lineage gap (GAP-R1-01) **does not block B4** — pre-B5 bootstrap occurs before canonical task execution.

---

## 9. Identified gaps

| ID | Component | Classification | Impact | Minimal recommended action | Proposed task |
| -- | --------- | -------------- | ------ | ------------------------ | ------------- |
| GAP-R1-01 | `ExecutionReconstructor` / DIAG-2 read sources | **ARCHITECTURAL READ-MODEL GAP** (`GAP`) | `ExecutionReconstructor` consumes `RuntimeEventPersistence` + `CausalEvidencePersistence`. `RuntimeEvent` has `execution_id` but not `parent_execution_id`. `CausalRelationKind` has no parent→child execution relation. `DELEGATION_GRANTED` does not carry direct parent execution identity. Execution membership within one run is available; the exact direct parent→child edge is not. DIAG-2 cannot deterministically reconstruct an arbitrary parent/child execution tree. Platform already has canonical `ExecutionTreeSnapshot`, `ExecutionCheckpointEntry.parent_execution_id`, and `ExecutionTreeRecorder` — reuse first; no parallel tree. | Architecture decision (options A/B/C in §6); then read-side integration — no new diagnostics pipeline, private API, or bypass | `DG-001-CROSS-SYSTEM-MULTI-AGENT-DIAGNOSTIC-LINEAGE-ARCHITECTURE-R1` |
| GAP-R1-02 | Decision→Execution diagnostic read | **QUALIFICATION_REQUIRED** | `DecisionExecutionLineage` contract exists; operator read join not qualified. Decision Lifecycle `RuntimeEvent` payload carries `decision_id`, `task_id`, `run_id`, `attempt_id`, `execution_id` — existing public read contracts may suffice; proof still required. | Qualification harness joining `DecisionIdentity` observability with DIAG-2 scope — not a new Decision diagnostics lifecycle | `DG-001-DECISION-EXECUTION-DIAGNOSTIC-LINEAGE-QUALIFICATION-R1` |
| GAP-R1-03 | `CausalRelationKind` | **ARCHITECTURE DECISION DEFERRED TO GAP-R1-01 REMEDIATION** | Only `TRANSPORT_TASK_TRIGGERED_EXECUTION` today; no parent→child execution relation. `execution_id` identifies an execution but does not encode its direct parent — not sufficient to dismiss this gap. | Defer causal relation design until GAP-R1-01 architecture remediation chooses reuse vs evidence extension vs other public contract | *(subsumed by GAP-R1-01 architecture task)* |

No STOP-condition blockers. No production code change required to unblock B4/B5 track.

---

## 10. Recommended roadmap

1. **DG-001B4 current-HEAD revalidation** — independent bootstrap track; routine focused B4 proof re-run on current HEAD (no harness redesign).
2. **`DG-001-CROSS-SYSTEM-MULTI-AGENT-DIAGNOSTIC-LINEAGE-ARCHITECTURE-R1`** — read-side architecture decision (reuse Execution Tree / extend evidence / other public contract); enterprise rules: reuse existing contracts first, no parallel Execution Tree, no private API, no `getattr`/`setattr`, no dynamic contracts, no bypass, no new diagnostics pipeline, no Problem lifecycle in Decision System.
3. **Multi-agent diagnostic qualification** — after architecture decision from step 2, not before.
4. **`DG-001-DECISION-EXECUTION-DIAGNOSTIC-LINEAGE-QUALIFICATION-R1`** — read-composition proof for Decision→Execution operator linkage.
5. **Further DG-001 closure** — only after results from steps 1–4.

---

## 11. Final verdict

```text
DG-001 CROSS-SYSTEM DIAGNOSTIC COMPATIBILITY AUDIT R1 = GAPS_FOUND
```

Cross-system compatibility for existing **DG-001B3/B4 boundaries remains valid**, but this audit discovered **one real read-side architecture gap** for multi-agent parent/child execution lineage (GAP-R1-01), plus **separate qualification debt** for Decision→Execution operator linkage (GAP-R1-02). Large parallel changes after DG-001B3 do not invalidate Diagnostic Engine identity consumption, canonical runtime event flow, Decision failure-fact ownership, or B3/B4 bootstrap contracts. The multi-agent gap belongs to **Diagnostics read-side integration**, not Execution Engine correctness. No architectural bypass or competing diagnostics lifecycle is required.

---

## Boundary table

| Boundary | Previous assumption | Current truth | Diagnostics impact | Verdict |
| -------- | ------------------- | ------------- | ------------------ | ------- |
| Execution identity | Single authority after NPSC-3C | `identity_authority.py` owns root/child/retry/background minting | Diagnostics consumes, does not mint | **PASS** |
| Root execution | `ExecutionRuntime` mints root triple | `resolve_root_execution_context` → `mint_root_execution_identity` | Terminal trigger + reconstruction use same IDs | **PASS** |
| Child execution | Child mints under active parent | `ChildExecutionRunner` → `mint_child_execution_id`, shared `run_id`/`attempt_id` | Membership via `execution_id` in run scope; direct parent→child edge not in DIAG-2 canonical sources | **GAP** |
| Retries | Retry mints new `AttemptId` | `mint_retry_attempt_id` in attempt lifecycle | Attempt grouping in reconstruction | **PASS** |
| Queue/background execution | Transport identity persisted before runtime | `mint_background_transport_identity` + `identity_persistence` + causal evidence | Reconstruction sees transport→execution link | **PASS** |
| Terminal events | Host task publishes terminal with active identity | `_HostTaskTerminalPublishingDelegate` publishes `run_id`/`attempt_id`/`execution_id` | `TerminalExecutionDiagnosticTrigger` fires on terminal scope | **PASS** |
| Causal evidence | Transport→execution link | `TRANSPORT_TASK_TRIGGERED_EXECUTION` only | Sufficient for background transport boundary; no parent→child execution relation — decision deferred to GAP-R1-01 | **GAP (lineage subset)** |
| Decision failure facts | Decision owns meaning, not Problem lifecycle | `CompletionReconciliationDiagnostic` is typed exception payload; no Problem store | No competing diagnostics lifecycle | **PASS** |
| Decision→Execution lineage | Separate identity domains with binding | `DecisionExecutionLineage` on `DecisionIdentity`; projection preserves `source_decision_identity` | Join not yet qualified in DiagnosticReadService | **QUALIFICATION_REQUIRED** |
| Multi-agent fan-out | Parent + N children under coordination | `CoordinationIntentExecutor` + `ChildExecutionRunner`; runtime `ExecutionTreeRecorder` / checkpoint truth exists | DIAG-2 canonical read sources lack direct parent→child lineage; architecture decision before qualification | **GAP** |
| Diagnostic read composition | One spine, no Nexus fallback | `diagnostic_read_wiring.py` → harness host `RuntimeEventPersistence` + shared causal/Problem stores | Legacy Nexus read fallback absent — positive | **PASS** |
| DG-001B pre-B5 | Bootstrap failure before canonical execution | `HostedBootstrapFailureRecord` unchanged in role; orthogonal to ExecutionRuntime | No regression from parallel system changes | **VALID** |

---

## Tests executed

Focused regression (157 passed, 7.52s):

```text
tests/unit/contracts/test_execution_identity.py
tests/unit/runtime/architecture/test_execution_identity_single_authority_gate.py
tests/unit/runtime/architecture/test_ue_10r2_single_canonical_root_execution_id_gate.py
tests/unit/runtime/execution/test_execution_runtime.py
tests/unit/runtime/execution/test_host_task_terminal_publisher.py
tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py
tests/unit/runtime/diagnostics/test_execution_reconstruction.py
tests/unit/runtime/diagnostics/test_diagnostic_read_service.py
tests/unit/hosting/test_bootstrap_failure_record.py
tests/unit/runtime/architecture/test_npsc5c_decision_execution_e2e.py
tests/unit/applications/architecture/test_npsc3g_application_runtime_convergence_gate.py
tests/unit/runtime/architecture/test_one_spine_legacy_causal_diagnostics_gate.py
```

Log: `.tmp/session/dg001-cross-audit-r1/pytest.log`

**Unrelated failures:** none in focused suite.

**New audit tests:** none (static audit + existing regression only).
