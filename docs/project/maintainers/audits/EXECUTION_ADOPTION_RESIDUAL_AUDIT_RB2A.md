# RB-2A — Current Execution Adoption Residual Audit

**Status:** READ-ONLY qualification (no production code changed in RB-2A)  
**Parent ledger:** [`CROSS_LAYER_ARCHITECTURE_REBASE_RB0_LEDGER.md`](CROSS_LAYER_ARCHITECTURE_REBASE_RB0_LEDGER.md) (subordinate — does not replace index/detail register)  
**Parent report:** [`CROSS_LAYER_ARCHITECTURE_REBASE_RB0.md`](CROSS_LAYER_ARCHITECTURE_REBASE_RB0.md)  
**Task:** RB-2A — Current Execution Adoption Residual Audit  

| Gate | Value |
|------|-------|
| **RB2A_BASELINE_HEAD** | `4bcc0255dd21f082d74749fd5e0c2fc6003c83c7` |
| **origin/development @ audit** | `4bcc0255dd21f082d74749fd5e0c2fc6003c83c7` |
| **HEAD == origin/development** | **YES** |
| **Delta inspected since** | `385d89fe34e4e75d85f72c97716764dabe78cf4c` → `4bcc0255d` (execution touch: `4bcc0255d` delegated correlation **query read model** only) |
| **Frozen Execution Engine** | Not modified in RB-2A |

## Executive answer

On current `development` @ `4bcc0255d`, **no production-capable path was proven** that performs canonical platform work while avoiding frozen **ExecutionRuntime** authority for root lifecycle, identity minting, admission, strategy routing, retry/recovery semantics, or terminal convergence.

Remaining RB-2 work is **adoption debt**: historical audit **LEGACY** flags, dead compatibility surfaces, UER-FIX consumer re-proof, documentation drift, and U5 re-qualification at HEAD — not a second execution architecture.

**Proven production execution bypass count:** **0** (consistent with U5 P0 inventory metrics; RB-2A re-verifies call paths @ HEAD).

---

## Phase 1 — Production execution ingress inventory (22 EP + extensions)

Source baseline: [`PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md`](../qualification/PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md) (22 rows). RB-2A re-traced representative production modules @ HEAD.

| Surface | Entry module | Prod? | Input contract | Normalization | Execution contract | Implementation | Mint identity? | Own retry/resume? | Strategy select? | Direct Nexus root? | Legacy adapter? | Canonical verdict |
|---------|--------------|-------|----------------|---------------|-------------------|----------------|----------------|-------------------|------------------|-------------------|-----------------|-------------------|
| Interaction HTTP intake | `InteractionIntakeService` | YES | Vendor/lab payload | `intake_payload_to_task` + optional enricher | `HostTaskExecutionPort.execute` | `HostTaskExecution` → `ExecutionRuntime` | EE | EE | EE router | NO | `HostTaskExecutionExecutor` (L2) | **CANONICAL** |
| Harness task HTTP | `mount_canonical_harness_task_routes` | YES | Harness REST bodies | Reliability/ACP enrichers | `HostTaskExecutionExecutor` / `HostTaskExecutionPort` | `HostTaskExecution` | EE | EE | EE | NO | L2 executor | **CANONICAL** |
| Tier-3 app hosts | App factories | YES | App task contracts | Host wiring | `HostTaskExecution` / env host execution | `ExecutionRuntime` | EE | EE | EE | NO | None in app tree (gates) | **CANONICAL** |
| Scenario / proof | `execute_scenario_task` | YES (proof) | Scenario task | Scenario baseline | Host task port | Same as EP-02 | EE | EE | EE | NO | Same production path | **SAME PRODUCTION PATH** |
| Queue / background | `execute_logical_task` | YES | Logical task envelope | Worker bootstrap admission | Handler → canonical intake | Registry handlers | Admitted IDs | Handler via EE | Via handler | NO | Delivery ≠ execution identity | **CANONICAL** |
| AW worker dispatch | `WorkerExecutionDispatchService` | YES | `WorkerExecutionDispatchRequest` | AW admission + budget | `RootExecutionLaunchPort.launch` | Canonical launcher → intake | EE | EE | EE | NO | Correlation fields non-authoritative | **CANONICAL** |
| CLI `intergrax run` | `intergrax/cli/run.py` | YES | App module | Delegated | Via Tier-3 host | App composition | Via app | Via app | Via app | NO | — | **CANONICAL** |
| Nexus agentic step | `NexusLoop.handle_task` | YES (in-run) | Active execution context | Runtime context | Inside ORCHESTRATION/AGENTIC strategy | Tool/agent engines | Child under EE | EE | Strategy-owned | Internal only | — | **CANONICAL** |
| Graph / child work | `ChildExecutionRunner` | YES | Child ports | Admission hooks | Frozen child plane | Runner | EE child mint | EE | EE | NO | Delegated adapters | **CANONICAL** |
| Compensation drain | `CompensationSideEffectExecutionPort` | YES | Job payload | Queue worker | `ExecutionRuntime` | Tool invoke session | EE | EE | EE | NO | — | **CANONICAL** |
| HITL / task control resume | `governed_resume_checkpoint_task_with_host_execution` | YES | Checkpoint + governance | Task control | `HostTaskExecutionPort.execute` | `HostTaskExecution` + recovery handoff | EE | Recovery plane | EE | NO | Scheduler timing only | **CANONICAL** |
| Long-running scheduler | `wire_long_running_scheduler_with_host_execution` | YES (when enabled) | Checkpoint schedule | `HostTaskResumeExecutor` | `host_execution.execute` | `HostTaskExecution` | EE | EE resume semantics | EE | NO | Legacy `wire_long_running_scheduler` unused (L3) | **CANONICAL** |
| Debug API | `create_debug_app` | NO (lab) | Debug routes | Debug services | `host_execution=runtime.execution` | Same host port when configured | EE when run | EE | EE | NO | Reachable only in debug composition | **TEST/DEBUG** |
| Experiments workflow | `intergrax/experiments/workflow.py` | NO | Experiment task | Local | `UnifiedTaskRunner.run_task` | `execute_root_task` → EE | Partial harness bridge | EE internally | ORCHESTRATION-only path | Internal | L3 | **TEST-ONLY PATH** |
| Nexus eval | `eval/nexus_eval_runner.py` | NO | Eval | Eval harness | `UnifiedTaskRunner` | `execute_root_task` → EE | Eval scope | EE internally | Eval | Internal | L3 | **TEST-ONLY PATH** |
| AW work-stage loop | `WorkStageCapabilityLoop` | NO prod imports | Stage catalog tests | Test wiring | Tool port in tests | Not composable from apps | Test correlation | N/A | N/A | NO | EP-17 | **TEST-ONLY PATH** |
| MCP (harness) | Task control / host routes | YES when mounted | MCP via host HTTP stack | Same as harness | `HostTaskExecution` when canonical routes | Same as EP-04 | EE | EE | EE | NO | — | **CANONICAL** (same EP-04 chain) |

**Production execution ingress count (inventory):** **22** (P0 table); **19** CANONICAL production + **3** LEGACY NON-PRODUCTION (EP-17/20/21).

---

## Phase 2–3 — Bypass definition & L1–L7 (RB-2 LEGACY flags @ ledger)

| Finding ID | RB-2A L-class | Rationale |
|------------|---------------|-----------|
| AUDIT-20260818-EXECUTION_RUNTIME-01 … 06 | **L1** | Historical UER-FIX register items; EE frozen and canonical. **LEGACY** = consumer proof debt, not production bypass. |
| AUDIT-20260818-INTERFACE_TASK_INTAKE-01 … 06 | **L2** | `HostTaskExecutionExecutor` 1:1 delegates to `HostTaskExecution` → `ExecutionRuntime`. No `NexusLoopTaskExecutor`; no production `nexus_loop=` intake fallback. |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-01 … 10 | **L4** | Production harness uses `HarnessHostRuntime.execution: HostTaskExecution`. Residual platform symbols (`UnifiedTaskRunner`, `mount_harness_task_routes`) are gated out of factories — adoption/cleanup, not alternate lifecycle. |
| AUDIT-20260818-ORCHESTRATION-01 … 05 | **L1** | Nexus reachable only under ORCHESTRATION strategy inside EE; RB-2 monitor only. |

**L1–L7 counts (22 LEGACY ledger rows):** L1=11, L2=6, L3=0 (ledger), L4=10, L5=0, L6=0, L7=0.  
*(ORCHESTRATION 5× L1 + EXECUTION_RUNTIME 6× L1 = 11; SHM 10× L4; ITI 6× L2.)*

**Additional non-ledger surfaces (P0):** EP-17/20/21 → **L3** (3).

---

## Phase 4 — UnifiedTaskRunner

| Responsibility | Owner @ HEAD |
|----------------|--------------|
| Identity minting | `resolve_root_task_identity` + **ExecutionRuntime** (not runner) |
| ActiveTaskRegistry | Registration wrapper only around canonical call |
| Retry / resume / checkpoint | **ExecutionRuntime** / recovery plane via `execute_root_task` |
| Strategy routing | **ExecutionRuntime** + router inside `execute_root_task` |
| Production Tier-3 entry | **Forbidden** by class docstring; factories use `HostTaskExecution` |

**Verdict:** **KEEP TEMPORARILY — compatibility adapter** (scheduler/test/eval/internal `execute_root_task` bridge). **Not** a duplicate execution owner.

Evidence: `intergrax/runtime/task/unified_task_runner.py` → `execute_root_task` → `ExecutionRuntime.execute`.

---

## Phase 5 — Host task execution (public host surface)

| Field | Value |
|-------|-------|
| Public port | `HostTaskExecutionPort` (`intergrax/runtime/execution/host_task.py`) |
| Canonical impl | `HostTaskExecution` (dataclass composition root) |
| Delegates to frozen EE | **YES** — builds per-task `ExecutionRuntime`, `DefaultRootExecutionLauncher`, governance admission |
| Production consumers | `HarnessHostRuntime.execution`, `HostTaskExecutionExecutor`, scenario baseline, app host factories, debug optional |
| Leaks Nexus as root API | **NO** — orchestration via injected `OrchestrationExecutor` under strategy router |

**Verdict:** **KEEP — canonical responsibility**

---

## Phase 6 — Interaction intake

1. Production wiring: `wire_harness_host_interaction_intake` → `HostTaskExecutionExecutor` (`harness_host_auxiliary_wiring.py`).
2. `nexus_loop=` fallback on intake: **absent** (`NexusLoopTaskExecutor` token absent from runtime tree; gate tests).
3. Task preparation: optional enricher / `TaskPreparationExecutor.prepare` — no runtime reflection; no duplicate EE admission.
4. **Verdict:** **CANONICAL** production path; **L2** adapter only.

---

## Phase 7 — Background / worker

- `WorkerExecutionDispatchService` documents delivery vs execution identity (`worker_execution_dispatch.py`).
- Dispatch → `RootExecutionLaunchPort` → canonical intake; worker does not mint trusted authority.
- Queue worker → `execute_logical_task` + admitted identity bootstrap (EP-06/07 unchanged @ HEAD).

**Verdict:** **CANONICAL** — worker owns **WHEN/delivery**; EE owns execution semantics.

---

## Phase 8 — Scheduler / resume / HITL

- Production harness: `wire_long_running_scheduler_with_host_execution` → `HostTaskResumeExecutor` → `HostTaskExecution.execute`.
- Legacy `wire_long_running_scheduler(UnifiedTaskRunner)`: **no production callers** (only definition + convergence gate tests) — **L3**.
- Task control resume: `governed_resume_checkpoint_task_with_host_execution` — governance + host execution; no independent checkpoint semantics.

**Verdict:** **CANONICAL** on supported composition; legacy scheduler wiring **L3** retired from factories.

---

## Phase 9 — Delegated / child execution @ HEAD

- Commit `4bcc0255d`: delegated **correlation query** read model (`correlation_query_persistence.py`, `query_service.py`) — observability/read path; **does not** mint Run/Attempt/Execution or resume.
- Child work remains `ChildExecutionRunner` + U4 closed wiring.

**Verdict:** **CANONICAL** child re-entry; new read model **not** execution authority.

---

## Phase 10 — Proof / scenario / eval

| Class | Examples |
|-------|----------|
| SAME PRODUCTION PATH | `platform_proofs` → `execute_scenario_task` |
| TEST-ONLY PATH | EP-20 experiments, EP-21 eval runner |
| DUPLICATE EXECUTION PATH | **None proven** at production composition |

---

## Phase 11 — Debug / maintenance

- `create_debug_app` composable in lab harness only; uses `HostTaskExecution` when executing — **no bypass** when used; scope is debug composition.
- Maintenance scripts outside audited production ingress: treat as operator tooling unless imported by app factories (none proven).

---

## Phase 12 — Pluginability / contract check

Production ingress consumers depend on **`HostTaskExecutionPort`**, **`RootExecutionLaunchPort`**, **`TaskExecutor`**, **`ExecutionRuntime`** (internal to host), not concrete Nexus types for root admission. Swapping agent engine / orchestration executor remains composition-root concern; extensions cannot mint root identity without EE admission.

**Pluginable implementation ≠ pluginable authority:** **PASS** on audited paths.

---

## Phase 13 — Duplication table (suspects vs Execution)

| Component | Lifecycle | Identity | Retry | Resume | Routing | Admission | Verdict |
|-----------|-----------|----------|-------|--------|---------|-----------|---------|
| `HostTaskExecution` | EE | EE | EE | EE + recovery | Strategy router | Launcher + ports | Canonical delegate |
| `UnifiedTaskRunner` | EE | via resolve_* | EE | via execute_root_task | EE | via EE | Adapter only |
| `ActiveTaskRegistry` | track active | no mint | no | no | no | no | Not duplicate owner |
| `WorkerExecutionDispatchService` | no | no mint | no | no | no | AW + root launch | Admission only |
| `LongRunningScheduler` | schedule | no | no | invokes host port | no | no | **WHEN** only |
| Delegated correlation query | no | no | no | no | no | no | Read model |

**Duplicate execution owner count:** **0**

---

## Phase 14 — U5 zero-bypass reconciliation

| Question | Answer @ `4bcc0255d` |
|----------|----------------------|
| **Q1** U5 matches HEAD? | **Partially stale artifact HEAD** (`151f3d71…` in U5 doc); **inventory semantics still hold** — no new BYPASS row; AST gates in `test_platform_execution_unification_u5_final_zero_bypass.py` still enforce metrics. |
| **Q2** Covers all production ingress types? | **YES** for composable Tier-3/proof/worker paths; MCP collapses to harness HTTP (EP-04). |
| **Q3** RB-1 LEGACY = production bypass? | **NO** — compatibility/adoption/proof debt (this audit). |
| **Q4** New entry after U5? | **NO** new execution ingress; `4bcc0255d` adds delegated correlation **query** only. |
| **Q5** Truthful `0 proven production execution bypasses`? | **YES** |

**Current U5 zero-bypass verdict:** **HOLDS** (re-qualification at HEAD recommended — RB-2B slice — not a bypass regression).

---

## Phase 15 — RB-2 finding refresh summary

All **27** RB-2 ledger IDs accounted for (22 LEGACY + 5 ORCHESTRATION monitor).  
**Bypass Risk** remains **LEGACY** on adoption rows until RB-2B closes proof/doc debt — **not** upgraded to PRODUCTION bypass.

**Evidence Freshness SHA (RB-2A):** `4bcc0255dd21f082d74749fd5e0c2fc6003c83c7`

Detail register patches: ledger § Phase RB-2A delta + per-finding fields updated programmatically from this artifact.

---

## Phase 16 — Proposed RB-2B remediation slices (not implemented)

| Slice | Scope | Owner boundary |
|-------|-------|----------------|
| **RB-2B1** | Retire dead harness execution surfaces: `mount_harness_task_routes`, `build_task_runner_with_enricher`, unused `wire_long_running_scheduler` | `intergrax/applications/_shared`, `runtime/long_running/wiring.py` |
| **RB-2B2** | UER-FIX consumer re-proof @ HEAD (agents/kernel) without EE core edits | Consumers only |
| **RB-2B3** | Tier-3 doc drift (`UnifiedTaskRunner` references in application docs) | `applications/*/docs` |
| **RB-2B4** | U5 + P0 inventory re-qualification session @ `4bcc0255d` (incl. delegated query non-authority) | Qualification docs + gate HEAD stamp |
| **RB-2B5** | SHM/ITI adoption closure record + convergence gate expansion if needed | RB-8 qualification companion |

**Removed from slices (already canonical @ HEAD):** interaction Nexus fallback removal (done); production scheduler host wiring (done); production intake HostTaskExecution (done).

---

## Phase 17 — Priority & architecture stops

No **STOP / ARCHITECTURE DECISION REQUIRED** for RB-2A findings — remediation is consumer/adopt/retire only.

**Workstreams blocked by parallel sessions:** RB-4 (functional_evidence WIP on working tree — out of RB-2A scope; do not conflate with execution bypass).

---

## Validation checklist (RB-2A PASS)

- [x] All RB-2 finding IDs accounted for (27)
- [x] Production ingress accounted for (22 EP + MCP note)
- [x] LEGACY execution flags classified L1–L7
- [x] No duplicate ingress records
- [x] L5/L6 empty (no unproven critical claims)
- [x] Remediation slices bounded; no frozen EE mutation
- [x] No production code changes in RB-2A

---

## RB-2B1 — Dead compatibility surface retirement (@ `ed780d47e7bc60e0ac019fb9bee8961dac9493c5`)

| Candidate | Verdict | Production callers before → after | Notes |
|-----------|---------|-------------------------------------|-------|
| `mount_harness_task_routes` | **DELETED** | 0 → 0 | Canonical: `mount_canonical_harness_task_routes` + `wire_harness_task_control`. Unit tests migrated via `mount_canonical_harness_task_routes_for_tests`. |
| `build_task_runner_with_enricher` | **DELETED** | 0 → 0 | Enrichment remains `build_reliability_task_enricher` / composition; `UnifiedTaskRunner` constructed directly in eval/test-only paths. |
| `wire_long_running_scheduler(UnifiedTaskRunner)` | **DELETED** | 0 → 0 | Canonical: `wire_long_running_scheduler_with_host_execution` / `wire_harness_host_long_running_scheduler`. |
| `UnifiedTaskRunner` | **KEEP TEMPORARILY** | unchanged | RB-2A adapter; not in scope for deletion. |

**Zero production execution bypass count:** **0** (unchanged).  
**Frozen Execution Engine:** not modified.  
**Pluginability:** public surfaces remain `HostTaskExecutionPort`, `TaskExecutor`, scheduler host wiring contracts.

**RB2B1_BASELINE_HEAD:** `ed780d47e7bc60e0ac019fb9bee8961dac9493c5`
