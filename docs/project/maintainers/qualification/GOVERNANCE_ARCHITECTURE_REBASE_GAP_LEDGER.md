# Governance Architecture Rebase — Gap Ledger (GR-0)

**GOV-FINAL-1 reconciliation HEAD:** `497abad1f4a2520b3ab8ab0d453247e41d751904` on `development` (documentation truth sync; independent code audit still required)

**GOV-FINAL-2 runtime blockers (GR-3 allowlist + GR-4 policy-core Nexus decouple):** session on `development` — `test_gr3_inner_enforcement_architecture_gates.py` + `test_gr4_policy_core_architecture_gates.py` target PASS; independent GitHub audit still required before closure.

**Rebase audit HEAD (session):** `fe2edc8077234437b13345daaf46633867fe8f31` on `development`  
**H9.2C maintainer/doc reconciliation HEAD:** `1de3b7fca6ca76e015c284ce21b8d543487ef677` on `development`  
**H9.2C GR-3-R1 doc/code consistency audit HEAD:** `0c810fdeebd6edc85106b88cea6008ce50682c09` on `development` (GR-3-R1 production: `a7fbfb7e6`)  
**GR-3-R2 explicit active task scope composition audit HEAD:** session on `development` (composition-only `ActiveTaskRegistryTaskScopeResolver`)  
**Canonical architecture baseline:** Unified Execution Runtime, five-ID execution identity, Decision System, Evidence Plane, Central Diagnostics (verified against listed arch docs + production paths below).

**Purpose:** Evidence-based gap inventory after platform-wide execution-centric rebase. Supersedes stale maintainer PG-FIX status rows where code truth differs; does not erase historical audit references.

**GOVERNANCE-FINAL certification audit HEAD:** `a4aa388f672a41ccd32be58241db8c05cfe3cfe0` on `development` — record: [`GOVERNANCE_FINAL_ENTERPRISE_CERTIFICATION.md`](GOVERNANCE_FINAL_ENTERPRISE_CERTIFICATION.md). Verdict: **`NOT CERTIFIED — ENTERPRISE BLOCKERS REMAIN`**. Mandatory suite re-run: GR-5-R4 architecture gate **FAIL** (`intake_runner.py` / `HumanPauseCoordinator.is_resumed`); GR-7 host qualification **61/61 PASS** on same SHA. Independent GitHub audit still required.

**GR-5-ADR1 HEAD:** `9336beff5ec72c747b440e63f3fb2dddc0b4bf8d` on `development` — canonical HITL continuation ownership: [ADR-GR-5-001](../../technical/adr/entries/2026-09-15/ADR-GR-5-001.md). Verdict: **`INTRODUCE_CANONICAL_EXECUTION_CONTINUATION_CONTRACT`** (`ExecutionContinuationPort`). Nexus = **internal** Execution Engine orchestration; not external peer layer.

**GR-5-R1 HEAD:** `9979e3b3af64b436708d0e95fac669dc9ef6e3d1` base → commit on `development` — contract module `intergrax/contracts/execution_continuation.py` (two-phase `apply_resolution` / `resume`, CAS `revision`, `RESUME_AUTHORIZED` state). Runtime integration **not** in R1.

---

## Executive summary

On current `development`, GR-1…GR-7 **implementation slices are present** (identity, root admission, inner guard, policy core, continuation port, Decision requirement at meaningful effects, ProviderInvocation reliability boundary). **Enterprise certification of the full Governance Plane is not claimed.** Remaining work is qualification and coverage: **GR-8** governance evidence, **GR-10** strategy matrix, **GR-12** control-plane mutation, **GR-11/GR-13/GR-16** enterprise proof and claims discipline.

**Enterprise verdict for implementation:** `QUALIFICATION_AND_COVERAGE_OPEN` — no blocking fork for GOV-FINAL-2 runtime gaps documented in § Open gaps; architecture SSOT: [`GOVERNED_EXECUTION.md`](../../architecture/GOVERNED_EXECUTION.md) § Governance implementation truth.

---

## Candidate hypothesis verdicts (GOV-REBASE-01 … 10)

| ID | Verdict | Severity | Evidence (summary) |
| --- | --- | --- | --- |
| GOV-REBASE-01 | **CLOSED** | P0 | GR-1: `GovernedContinuationApprovalGrant`, correlation, and `matches_current_requirement` bind `task_id` + `run_id` + `attempt_id` + `execution_id`. |
| GOV-REBASE-02 | **CLOSED** | P0 | GR-1: `MeaningfulSideEffectRequest` requires canonical four-ID execution identity; governed continuation chain propagates without loss. |
| GOV-REBASE-03 | **CONFIRMED** | P1 | `authorize_and_execute` calls `lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)` and `HumanPauseCoordinator` on `Task` governance state (`meaningful_side_effect_authorization.py`, `governed_continuation_bridge.py`). Canonical arch assigns pause/resume to UER — implementation still Task/Nexus-centric. |
| GOV-REBASE-04 | **PARTIAL** | P1 | Production HITL bridge avoids direct `nexus.*` import except `declarative_hitl_grant` → `RuntimeRequest`. Pause/resume semantics and tests remain Nexus-orchestration-shaped; inference/agentic HITL not qualified. Arch target (GR-5-ADR1): HITL via **`ExecutionContinuationPort`**; Nexus orchestrates **internally** for ORCHESTRATION — not a second execution truth. |
| GOV-REBASE-05 | **PARTIAL** | P1 | `authorize_and_execute` runs caller `execute` callback after auth — not a second `ExecutionBoundary`, but can execute **outside** active canonical Execution context when caller omits UER binding (`agents/external_contractor_adapter/external_work_adapter.py` call site). Inner-op pattern (A) is intended; enforcement of (A) is **gap**. |
| GOV-REBASE-06 | **PARTIAL** | P1 | GR-6: `DecisionRequirementPolicy` + material ref on MSE boundary for wired hosts; not platform-universal. `ContinuationEvidenceRefs.hitl_decision_id` remains HITL store id, not Decision authority. Version pinning qual open (GOV-GAP-006). |
| GOV-REBASE-07 | **CONFIRMED** | P1 | Policy decisions use `audit_payload` on `PolicyDecision`; no systematic RuntimeEvent emission with five-ID correlation from governance spine. `governance_audit_event` is separate agent-governance channel — risk of parallel audit semantics. |
| GOV-REBASE-08 | **CONFIRMED** | P1 | G3B table (`GOVERNED_EXECUTION.md`) wires most points via Nexus/UAEP; strategy matrix §9 encodes GR-10-R1 INFERENCE applicability (PRE_MODEL **BLOCKED**; MSE/HITL/Continuation/Reliability **N/A** on inference path); AGENTIC/ORCHESTRATION MSE + HITL qualification remains open. |
| GOV-REBASE-09 | **CLOSED** | P2 | H9.2C: maintainer plan + arch G3B / Protocol v2.2 pointers reconciled with PG-FIX mechanism tests (`test_pg_fix_*`, `test_g5c2b*`); enterprise CLOSED still not claimed. |
| GOV-REBASE-10 | **CONFIRMED** | P1 | G3B + plan: **CONTROL_PLANE_MUTATION** remains **GAP**; domain tests (e.g. ECP) are partial slices, not platform-wide shared boundary. |

---

## Gap ledger (canonical rows)

| ID | Severity | Area | Current code truth | Target architecture | Mismatch | Risk | Required remediation | Dependencies | Roadmap slice | Evidence paths | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GOV-GAP-001 | P0 | Identity | GR-1-R1: atomic Attempt+Execution resolution; no hybrid caller/active assembly | Authorization bound to AttemptId + ExecutionId where security-sensitive | Pre-R1 partial override could mix caller Attempt with active Execution | Cross-attempt side-effect authorization | GR-1-R1 resolver + pause atomic bind | GR-0 audit | GR-1-R1 | `meaningful_side_effect.py`, `pause.py`, `test_meaningful_side_effect_execution_identity_resolution.py` | **CLOSED** |
| GOV-GAP-002 | P0 | Identity | GR-1: `HumanApprovalResolution` carries attempt/execution; governed pause validates correlation | Resolution correlated to active Execution | Residual UER pause ownership (GR-5) | HITL replay if resolution forged | GR-1 pause validation + grant derivation | GR-1 | GR-5 | `task_contract.py`, `pause.py` | **VERIFIED** |
| GOV-GAP-003 | P1 | HITL ownership | TaskLifecycle `WAITING_FOR_HUMAN`; Task governance blob | UER owns PAUSE/WAIT/RESUME same Execution via `ExecutionContinuationPort` (ADR-GR-5-001) | Doc/code split on pause owner | Nexus-shaped pause on non-orchestration strategies | GR-5-R1 contract + GR-5-R2+ integration; Task projection only | GR-5-ADR1 | GR-5-R1 | `meaningful_side_effect_authorization.py`, `governed_continuation_bridge.py`, ADR-GR-5-001 | OPEN (design **DONE**) |
| GOV-GAP-004 | P1 | Nexus coupling | Governed continuation docstrings + AgentExecutionResult bridge | HITL without mandatory Nexus | Orchestration-only proof for HITL path | Agentic/inference hosts lack qualified HITL | Strategy-neutral HITL entry; qualify non-Nexus paths | GR-5 | GR-5, GR-10 | `governed_continuation.py` (module doc), `governed_continuation_bridge.py` | OPEN |
| GOV-GAP-005 | P1 | Execution boundary | GR-3 + GR-3-R1 + **GR-3-R2**: contract-only `MeaningfulSideEffectAuthorizationBoundary`; `DefaultCanonicalInnerExecutionGuard` requires `ActiveExecutionTaskScopePort`; `ActiveTaskRegistryTaskScopeResolver` only in `meaningful_side_effect_authorization_composition.py`; four-ID inner enforcement + `authorize_and_execute` allowlist gate | Side effects only inside admitted Execution | Residual Reliability delivery semantics (GR-7) | Hidden concrete in reusable consumers | GR-3-R1/R2 explicit composition + architecture import gates | GR-1, GR-2 | GR-7 | `canonical_inner_execution_guard.py`, `meaningful_side_effect_authorization_composition.py`, `test_gr3_*`, `test_gr3_r1_*`, `test_gr3_r2_*` | **CANDIDATE CLOSED** (gap: GR-7; **GR-3-R1 DONE**; **GR-3-R2 DONE** — independent GitHub audit pending) |
| GOV-GAP-006 | P1 | Decision integration | `DecisionRequirementPolicy` + material ref at MSE boundary (GR-6) | Provenance for classified consequential effects on wired hosts | Platform-wide adoption + version qual | V2 decision reuses V1 approval without binding | Extend hosts + GR-10/GR-13 qual | Decision System + Governance | GR-6 | `test_gr6_*`, governed contractor GR-6 suites | **PARTIAL** |
| GOV-GAP-007 | P1 | Evidence | Legacy `audit_payload` remains diagnostic-only | Typed `GovernanceDecisionEvidenceFact` + `GovernanceEvidencePersistencePort` on root admission + MSE spine | Residual inner evaluation points / full strategy matrix | Duplicate emission guarded by idempotency key | Extend emission to remaining GEPs under GR-10/13 | Governance + Evidence | GR-8 | `governed_execution_governance_evidence.py`, ADR-GR-8-001 | **CANDIDATE CLOSED — spine complete + public contract frozen** ([ADR-GR-8-001](../../technical/adr/entries/2026-09-17/ADR-GR-8-001.md)); residual GEP → GR-10/13; independent final audit required |
| GOV-GAP-008 | P1 | Strategy coverage | G3B Nexus/UAEP-heavy COVERED rows | Matrix for INFERENCE/AGENTIC/ORCHESTRATION | “Nexus works” ≠ platform proof | **PRE_MODEL public contract** revised in **GR-10-R2-C1** (`principal_id` required); **runtime conformance** still **OPEN** (GR-10-R2-R1) | `tests/qualification/governance/strategy/` + matrix §9 | GR-10 | GR-10-R2-R1 | `GOVERNED_EXECUTION.md` §G3B | **PARTIAL** |
| GOV-GAP-009 | P1 | Control plane | CONTROL_PLANE_MUTATION GAP | Shared authority context per domain executor | No unified taxonomy enforcement | Unsafe platform mutations | Per-domain adoption of shared boundary; no god executor | Domain plans | GR-12 | plan CLA block; ECP tests partial | OPEN |
| GOV-GAP-010 | P2 | Maintainer truth | Stale PG-FIX / Protocol v2.2 status in docs | IMPLEMENTED / VERIFIED / CLOSED distinguished | Operators mis-plan | False closure claims | Reconcile plan + arch pointers (GR-0) | GR-0 | GR-0 | `plans/GOVERNED_EXECUTION.md` | **CLOSED** (H9.2C) |
| GOV-GAP-011 | P2 | Policy plugins | Catalog + handler slices; Nexus types in `policy_bundle.py` | Vendor-neutral core; platform plugin admission | Residual Nexus coupling in **documented adapter modules only** (`policy_bundle.py`, `tool_policy_resolution.py`, …) | Tier violation / test burden | **GR-4-R1:** neutral bundle assembly types; GR-11 owns certification | Platform plugins | GR-4-R1, GR-11 | `policy_bundle.py`, `tool_policy_resolution.py`, `test_gr4_policy_core_architecture_gates.py` | **PARTIAL** (neutral evaluator/catalog/handler core qualified; assembly adapters remain) |
| GOV-GAP-012 | P2 | Admission vs inner | GR-2 root admission (`RuntimeExecutionPolicyAdmissionPort`) vs GR-3 inner guard (`CanonicalInnerExecutionGuardPort`) vs `ExecutionAuthorityPolicy` (child narrowing) | Admission = may start Execution; inner = may proceed; child authority ≠ governance evaluator | — | Misuse of admission as inner policy | Ports composed explicitly (`execution_admission_composition.py`, `meaningful_side_effect_authorization_composition.py`) | GR-2, GR-3 | GR-4 | `runtime_execution_policy_admission.py`, `meaningful_side_effect_authorization.py` | **CANDIDATE CLOSED** (independent audit pending) |
| GOV-GAP-013 | P0 | Root admission coverage | **GR-2-R3-R1:** AST invocation/import gates (`gr2_r3_model_c1_ast.py`); `execute_root_task` **INTERNAL CERTIFIED HARNESS ENTRY**; legacy caller allowlists in `gr2_r3_model_c1_gate_policy.py` | One platform-wide Governance gate before every root Execution (INFERENCE / AGENTIC / ORCHESTRATION) | Legacy harness path remains launcher-external but gated (not Tier-3 production entry) | Independent GitHub audit pending | GR-2-R3-R1 + GR-2-R3 qualification tests | GR-2-R3-R1 | GR-2, GR-10 | `test_gr2_r3_model_c1_architecture_gates.py`, `orchestration.py` | **CANDIDATE CLOSED** (pending independent audit) |

---

## GR-2-R1 — platform-wide root admission audit (code truth)

**Audit HEAD:** `0123a11a6f337e264e503041ba1c84e750b6fe00` on `development` (note: differs from instruction pin `a409cce…` — no reset performed).

**Verdict:** `ARCHITECTURAL_DECISION_REQUIRED` — `FROZEN_EXECUTION_SEAM_INSUFFICIENT` (composition contracts exist; frozen canonical host does not consume them; `ExecutionRuntime` cannot enforce Governance without reopening).

| Root type | Production entry | Governance admission | Canonical intake | ExecutionRuntime | Bypass possible |
| --- | --- | --- | --- | --- | --- |
| INFERENCE | `HostTaskExecution.execute` → `build_host_task_strategy_router` → `StrategyExecutionRouter` → `InferenceExecutor` | None on host path | Skipped (`resolve_root_parent_execution_authority(task.execution_authority)`) | `Execution` → `ExecutionRuntime.execute` | **YES** |
| AGENTIC | Same host path → `AgentExecutor` | None on host path | Skipped | Same | **YES** |
| ORCHESTRATION | Host path → orchestration delegate; alternate AW `WorkerExecutionDispatchService` → `RootExecutionAuthorityAdmissionPort` → `CanonicalExecutionIntakePort` | AW dispatch only (not used by `HostTaskExecution` / harness / lab) | AW dispatch only | Both reach `ExecutionRuntime` | **YES** (host + direct `Execution`/`ExecutionRuntime` composition) |

**Root execution definition (frozen):** `ExecutionRuntime.execute(request, root_context: RootExecutionContext)` with `resolve_root_execution_context` / `mint_root_execution_identity` (`runtime.py`, `identity_authority.py`); lineage root invariant `execution_id == segment_root_execution_id` and `parent_execution_id is None` (`contracts/execution_lineage.py`).

**§13 seam:** `NO — MISSING_PLATFORM_GENERIC_GOVERNANCE_ADMISSION_CONTRACT` at the mandatory frozen host boundary. Optional AW-5A pair (`RootExecutionAuthorityAdmissionPort` → `CanonicalExecutionIntakePort`) is strategy-neutral but not enforced for `HostTaskExecution`.

**WORKER_ROOT_EXECUTION_OPERATION (§12):** **A** — `RuntimeExecutionPolicyAdmissionRequest.execution_operation` is generic; `RootExecutionAuthorityAdmissionService` hardcodes `WORKER_ROOT_EXECUTION_OPERATION` when calling policy (worker-named default, not worker-only port type).

**Admission contract roles:** `RuntimeExecutionPolicyAdmissionPort` / `RootExecutionAuthorityAdmissionPort` = GOVERNANCE AUTHORIZATION; `ExecutionAdmissionHook` = EXECUTION VALIDATION; `ExecutionCapacityAdmissionPort` = CAPACITY; `ExecutionAuthorityPolicy` = CHILD AUTHORITY; `CanonicalExecutionIntakePort` = INTAKE (trusted authority → runtime).

---

## GR-2-R2 — canonical root admission trust boundary (architecture)

**Design HEAD:** `722143bc37ff4459e126e9118c2e9596fdbbe9db` (`origin/development` baseline for GR-2-R2-R1).

**Verdict:** `ARCHITECTURE_APPROVAL_RECOMMENDED` — **Option C** (mandatory external intake + unified `RootExecutionLaunchPort`).

**Legal root entry (target):** `RootExecutionLaunchPort` → `RootExecutionAuthorityAdmissionPort` → `CanonicalExecutionIntakePort` → `ExecutionRuntime`.

**Artifact:** [`docs/project/maintainers/architecture/GR_2_R2_CANONICAL_ROOT_ADMISSION_TRUST_BOUNDARY.md`](../architecture/GR_2_R2_CANONICAL_ROOT_ADMISSION_TRUST_BOUNDARY.md)

**Status:** GR-2-R2 design **DONE**. GR-2 **OPEN**. GR-2-R3 **NEXT after independent audit**. GR-3 **BLOCKED** until GR-2 root bypass closed.

---

## GR-2-R2-R1 — root admission anti-bypass architecture correction

**Scope:** Documentation only (no runtime/contracts/tests).

**Corrections:** Option B bypass analysis (mandatory runtime hook = **HIGH** runtime no-bypass); Option C **MODEL C1** structural repository enforcement with **mandatory** CI architecture gates; anti-forgery = **provenance-by-path** + admission-provenanced authority (no false “mint monopoly”); API classifications; gate allowlists; removal of optional P0 hardening.

**Enforcement model:** `ARCHITECTURE_GATE_ENFORCED` (MODEL C1) for Option C.

**Status:** **DONE**. GR-2-R2 → **ARCHITECTURE APPROVED CANDIDATE** pending independent audit.

---

## Audit question snapshots

### A — Canonical execution identity (contracts)

| Contract / artifact | TaskId | RunId | AttemptId | ExecutionId |
| --- | --- | --- | --- | --- |
| MeaningfulSideEffectRequest | yes | yes | yes | yes |
| GovernedContinuationRequest / Correlation | yes | yes | yes | yes |
| GovernedContinuationApprovalGrant | yes | yes | yes | yes |
| HumanApprovalResolution | yes | optional run | optional | optional |
| Pause record | yes | — | partial | partial |
| Policy evidence (PolicyDecision) | via payload/context only | partial | no | no |

**Cross-execution reuse (post–GR-1):** Grants and `matches_current_requirement` bind `attempt_id` + `execution_id`; mismatched Attempt/Execution fails closed — historical P0 closed (GOV-GAP-001).

### B — Attempt semantics (code truth, not invented)

| Scenario | Approval validity (current) |
| --- | --- |
| Same Attempt / same Execution | Grant matches if task/run/scope/policy + attempt/execution unchanged |
| Same Run / new Attempt | **Invalid** when grant attempt/execution differ — **conformant** (GR-1) |
| Same Attempt / different Execution | **Invalid** when execution_id differs — **conformant** (GR-1) |
| Resumed same Execution | Bound when resolution/grant carry matching execution_id |
| New child Execution | Not authorized by parent Execution grant — **conformant** |

### C — HITL lifecycle

| Component | Owner today | Target |
| --- | --- | --- |
| Human decision | Governance / HITL (`HumanPauseCoordinator`, store) | Governance |
| Pause state | `Task` governance + `TaskLifecycle.WAITING_FOR_HUMAN` | Execution Runtime |
| Resume | Nexus/UAEP + task governance (tests) | UER same Execution |
| Classification | Legacy ownership leakage on pause; valid inner bridge for interrupt composition | Migration required (GR-5) |

### D — Nexus dependency (production `intergrax/runtime/human`)

- **Legitimate:** None required in core bridge modules (except declarative HITL grant schema import).
- **Gap:** Semantic coupling via `AgentExecutionResult`, Nexus-shaped tests, orchestration resume runners — HITL not proven without Nexus.

### E — `authorize_and_execute`

- **Design:** Evaluation + optional `execute` callback; doc states caller owns execution.
- **Risk:** Callers may invoke without `ExecutionBoundary` / active identity → **parallel execution path (B)** unless GR-3 enforces inner-op contract.

### F — Admission vs inner

- Root admission: `RuntimeExecutionPolicyAdmissionEvaluator` → `evaluate_root_execution_admission`.
- Child authority: `ExecutionAuthorityPolicy` (narrowing only) — **conformant** if not used as policy replacement.
- Inner: collaborative gate + `RuntimePolicyEngine.evaluate_meaningful_side_effect` — separate layer; duplication risk low if admission stays at Execution start only.

### G — Pluginability

- **Partial / enterprise-not-ready:** Policy catalog contracts (G2B), plugin contribution tests (G4B-1), handler registration exist; Nexus-typed bundle fragments and orchestration config resolution remain.
- **Fail-closed:** Unknown handler paths tested in declarative enforcer slices.

### H — PG-FIX-B (policy resolution)

| Status | Assessment |
| --- | --- |
| IMPLEMENTED | `runtime_policy_engine.py` specificity + conservative precedence + MODIFY normalization |
| VERIFIED | Targeted `test_pg_fix_b_side_effect_policy_precedence.py` (not full enterprise qualification) |
| PARTIAL | Other rule families share pattern; not all evaluation points |
| OPEN | Independent maintainer CLOSED bit |

### I — PG-FIX-C (scoped approval)

| Dimension | Mechanism sound? | Platform identity conformance? |
| --- | --- | --- |
| Task/run/scope/policy/pause binding | **MECHANISM SOUND** (tests G5C-2B*) | **PASS** (GR-1 attempt/execution on grant + matcher) |
| consume-before-effect | **MECHANISM SOUND** | Reliability still owns retry/idempotency |

### J — Decision System

- Governance does not adjudicate decision correctness — **conformant** at separation level.
- Decision-derived side effects: GR-6 mechanism on wired hosts; platform-wide binding/version qual — **partial** (GOV-GAP-006).

### K — Observability

- Governance facts: **partial** — `PolicyDecision.audit_payload`, task governance state; limited RuntimeEvent five-ID correlation.

### L — Diagnostics

- No evidence that Diagnostics authorizes execution (`policy_trace_diagnostics` is trace-only). **Conformant** today; future DIAG should consume governance evidence (GR-9).

### M — Reliability

- `consume-before-effect` documented as at-most-once **authorization**, not exactly-once effect — **conformant** with reliability boundary.

### N — Control plane

- **GAP** per G3B; partial domain tests only.

### O — Strategy matrix (governance capability)

| Capability | Inference | Agentic | Orchestration |
| --- | --- | --- | --- |
| Execution admission | UNVERIFIED | UNVERIFIED | PARTIAL |
| Meaningful side effect | GAP | PARTIAL (adapter) | PARTIAL |
| HITL | UNVERIFIED | UNVERIFIED | COVERED (Nexus path) |
| Resume | UNVERIFIED | UNVERIFIED | PARTIAL |
| Evidence | PARTIAL | PARTIAL | PARTIAL |
| Post-run governance | PARTIAL | PARTIAL | COVERED (when service injected) |

### P — Old roadmap disposition (summary)

| Old stage | Disposition | Reason |
| --- | --- | --- |
| G1 / G1A / G1B | KEEP | Evaluation-point architecture + contract hardening still valid |
| G2 / G2A / G2B / G2C | REVALIDATE | Catalog/handler identity done; enterprise plugin qual open → GR-11 |
| G3 / G3B | REVALIDATE | Coverage table needs post-UER + strategy matrix |
| G4A / G4B | REVALIDATE | Admission tests exist; merge into GR-11 |
| G5A–G5D / G5C-* | REOPEN + MERGE → GR-5 | Mechanism implemented; identity + UER pause rebase open |
| G6–G8 | SUPERSEDE → GR-6–GR-8 | Decision, reliability boundary, evidence integration |
| G9–G12 | SUPERSEDE → GR-9–GR-13 | Qualification / control plane / proof matrix |

---

## PG-FIX current truth (GOV-FINAL-1)

| Block | IMPLEMENTED? | VERIFIED? | Status | Notes |
| --- | --- | --- | --- | --- |
| PG-FIX-A | **Yes** (core spine) | **Partial** | **SUPERSEDED_BY_GR_3** — **PARTIAL** | Inner guard + MSE boundary; consumer coverage not universal |
| PG-FIX-B | **Yes** | **Yes** | **SUPERSEDED_BY_GR_4** — not enterprise **CLOSED** | `test_pg_fix_b_*`; GR-13 sign-off open |
| PG-FIX-C | **Yes** (mechanism) | **Partial** | Identity **CLOSED** (GR-1); continuation **PARTIAL** (GR-5) | Scoped grant semantics; UER port qual open |
| PG-FIX-D | **Yes** (typed matching) | **Yes** | **SUPERSEDED_BY_GR_4** — not enterprise **CLOSED** | `test_pg_fix_d_*`; GR-13 sign-off open |

Historical AUDIT-5 findings remain valid context; **enterprise CLOSED** for PG-FIX requires GR-13 / GR-16 — not claimed.

---

## GR enterprise roadmap (frozen at GR-0)

| ID | Task | Status | Simple goal | Depends on | Replaces (old) | Implementation expected? | Proof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GR-0 | Architecture Rebase & Gap Ledger | **DONE** | Establish code-truth gaps and roadmap | — | — | No (docs) | This ledger |
| GR-1 | Execution Identity Rebinding | **CLOSED** | Bind grants, side effects, HITL resolution to Attempt+Execution (GR-1-R1 atomic correction) | GR-0 | G5C identity follow-on | Yes | Contract + matcher + GR-1-R1 security tests |
| GR-1-R1 | Frozen boundary + atomic identity | **DONE** | Fail-closed partial identity; Governance consumes frozen active context | GR-1 | Audit defect A/B | Yes | `test_meaningful_side_effect_execution_identity_resolution.py` |
| GR-1-R2 | Frozen Nexus baseline restoration | **DONE** | Remove Governance-specific Attempt/Execution forwarding from Nexus; identity via active context | GR-1-R1 | GR-1 Nexus coupling | Yes | `test_gr1_execution_identity_rebinding.py` (Nexus path) |
| GR-2-R1 | Platform-wide root admission & frozen seam decision | **DONE** (`ARCHITECTURAL_DECISION_REQUIRED`) | Prove one Governance admission for INFERENCE/AGENTIC/ORCHESTRATION without layer violations | GR-1-R2 | — | No (audit) | This ledger § GR-2-R1; GOV-GAP-013 |
| GR-2-R2 | Canonical root admission trust boundary architecture | **DONE** (`ARCHITECTURE_APPROVAL_RECOMMENDED`) | One mandatory contract-first root trust boundary (design) | GR-2-R1 | — | No (docs) | `GR_2_R2_CANONICAL_ROOT_ADMISSION_TRUST_BOUNDARY.md` |
| GR-2-R2-R1 | Root admission anti-bypass architecture correction | **DONE** | MODEL C1 gates; correct Option B; anti-forgery nomenclature | GR-2-R2 | — | No (docs) | Same artifact + this ledger § GR-2-R2-R1 |
| GR-2-R3 | Root admission implementation | **CANDIDATE DONE** (audit pending) | Launcher + MODEL C1 gates + host/AW migration | GR-2-R2-R1 | — | Yes | `test_gr2_r3_root_execution_launcher.py`, `test_gr2_r3_model_c1_architecture_gates.py` |
| GR-2 | Execution Admission Governance | **CANDIDATE CLOSED** (audit pending) | Single admission at **every** root Execution start | GR-2-R3 | G3 admission rows | Yes | GR-2-R3 qualification suite |
| GR-3 | Canonical Inner Enforcement | **CANDIDATE CLOSED** (GR-3-R1 **DONE**, GR-3-R2 **DONE**); PG-FIX-A harness drift remains non-blocking for composition | Active Execution four-ID binding; contract-first inner guard + task scope | GR-1, GR-2 | PG-FIX-A membership proofs | Yes | `test_gr3_*`, `test_gr3_r1_*`, `test_gr3_r2_*`, architecture AST gates |
| GR-4 | Policy & Plugin Requalification | **CANDIDATE CLOSED** (independent GitHub audit pending); **GR-4-R1** = Nexus-neutral `RuntimePolicyBundle` assembly | Deterministic fail-closed policy resolution; contract-first handlers/catalog; no new framework | GR-3 | G2C, PG-FIX-B/D | Yes (qualification + gate) | `test_pg_fix_b_*`, `test_pg_fix_d_*`, `test_policy_registry.py`, `test_policy_plugin_contribution.py`, `test_gr4_policy_core_architecture_gates.py` |
| GR-4-R1 | Policy bundle Nexus decouple | **PLANNED** | Replace Nexus types in bundle assembly with neutral contracts + adapters | GR-4 | GOV-GAP-011 residual | Yes | Neutral bundle types; migrate `policy_bundle.py` / `tool_policy_resolution.py` |
| GR-5-ADR1 | Canonical Execution HITL Continuation Ownership | **DONE** | One lifecycle authority; Nexus internal; `ExecutionContinuationPort` direction | GR-1 | — | No (docs) | ADR-GR-5-001 |
| GR-5-R1 | Canonical Execution Continuation Contract | **DONE** | Typed `ExecutionContinuationPort` + DTOs (`execution_continuation.py`) | GR-5-ADR1 | — | Yes | Contract + boundary tests |
| GR-5-R2 | Canonical Pause/Resume Integration | **CANDIDATE CLOSED** (await audit) | UER owns transitions via `ExecutionContinuationService`; not root admission on resume; **TRANSITIONAL — NOT GR-5 COMPLETE** (R3/R4/R5) | GR-5-R1 | — | Yes | Lifecycle integration tests |
| GR-5-R3 | Task / HumanPauseCoordinator projection alignment | CANDIDATE CLOSED | Canonical continuation first; Task/Human replayable materialized view | GR-5-R2, GR-5-R3-R1 | — | Yes | Projection parity + canonical-first tests |
| GR-5-R3-R1 | Canonical-first resolution & atomic projection commit | DONE | `apply_resolution` before accepted Task projection; projection prepare/commit | GR-5-R3 | — | Yes | `test_gr5_r3_r1_canonical_first_atomic_projection.py` |
| GR-5-R4 | Execution Engine Internal HITL Orchestration Alignment | **CANDIDATE CLOSED — awaiting independent GitHub audit** (GR-5-R4-R1: Category C runners use `canonical_execution_is_resumed` only; GOVERNANCE-FINAL failure on `a4aa388f` remains historical) | Internal orchestration via port; Category C runners must not use Task projection lifecycle authority | GR-5-R2 | — | Yes | Orchestration HITL qual + architecture gate |
| GR-5-R5 | Checkpoint restart + exact identity qualification | **CANDIDATE CLOSED** (await audit) | Durable restart → same four IDs + current episode | GR-5-R3, GR-5-R4 | — | Yes | `test_gr5_r5_restart_exact_identity.py` |
| GR-5 | HITL / Governed Continuation Rebase | **CANDIDATE CLOSED** (await independent R5 audit) | End-to-end same-Execution pause/resume | GR-5-R1…R5 | G5*, PG-FIX-C | Yes | HITL E2E per strategy |
| GR-6 | Decision → Governance Integration | **IMPLEMENTED** — qualification **OPEN** | `DecisionRequirementPolicy`, canonical action/resource binding, production host composition | GR-1 | GOV-GAP-006 partial | Yes | `test_gr6_*`, architecture gates, governed contractor GR-6 suites |
| GR-7 | Reliability / External Effect Boundary | **IMPLEMENTED** — qualification **PARTIAL** (host 61/61 + GR-7-A8 unit pass on GOVERNANCE-FINAL SHA; not strategy-wide) | Durable ProviderInvocation; SUCCESS/FAILED/UNKNOWN; repeat/recovery/reconcile; reliability evidence (≠ GR-8) | GR-3, GR-5 | — | Yes | `test_gr7_*`, ERL contracts, governed contractor GR-7 suites |
| GR-8 | Governance Evidence Integration | **CANDIDATE CLOSED — PUBLIC CONTRACT FROZEN** — ADR-GR-8-001; independent final audit pending | Typed facts + pluginable persistence; root + MSE wired | GR-1 | — | Yes | `test_gr8_*`, ADR-GR-8-001 gates |
| GR-9 | Diagnostic Consumption Proof | PLANNED | DIAG reads governance evidence only | GR-8 | — | Mostly tests/docs | DIAG fixtures |
| GR-10 | Execution Strategy Coverage | **PARTIAL** | Inference/agentic/orchestration matrix + gates | GR-3, GR-5 | G3B + `tests/qualification/governance/strategy/` | Yes (INFERENCE PRE_MODEL qualified only after GR-10-R2-C1/R1) | Strategy qualification suite |
| GR-10-R2-ADR1 | PRE_MODEL identity + intake boundary | **CANDIDATE CLOSED** — SSOT reconciled; independent GitHub audit required | [ADR-GR-10-001](../../technical/adr/entries/2026-09-18/ADR-GR-10-001.md); intake **KEEP_REQUIRED**; carrier **ActiveExecutionGovernanceIdentity**; architecture decision **PASS**; runtime conformance **PENDING GR-10-R2-R1** | GR-10-R1 | GR-10-R2-ADR1-R1 | No (ADR + docs) | `test_gr10_adr1_pre_model_identity_boundary.py` |
| GR-10-R2-ADR1-R1 | PRE_MODEL identity SSOT reconciliation | **CANDIDATE CLOSED** — independent GitHub audit required | Current maintainer/arch/qual/gap-ledger status aligned to `GR10_INFERENCE_CAPABILITY_SEMANTICS` | GR-10-R2-ADR1 | — | No (docs + gates) | `test_gr10_gates.py` doc SSOT gates |
| GR-10-R2 | INFERENCE PRE_MODEL wiring | **BLOCKED** | Runtime present; not ADR-compliant until C1+R1 | GR-10-R2-ADR1 | — | Yes | `test_inference_executor.py` PRE_MODEL tests |
| GR-11 | Plugin Enterprise Certification | PLANNED | Admission, provenance, fail-closed | GR-4 | G4B | Yes | Plugin qual bundle |
| GR-12 | Control-Plane Governance | PLANNED | Shared boundary; domain executors | GR-2 | CLA control-plane | Yes | Per-domain conformance |
| GR-13 | Full Governance Proof Matrix | PLANNED | E2E positive/negative scenarios | GR-1–GR-12 | G9–G12 | Yes | Proof matrix |
| GR-14 | LKW Integration | PLANNED | Real application validation | GR-13 | — | Yes | LKW scenarios |
| GR-15 | Governance UX / App Contract | PLANNED | Reusable approval contract | GR-5 | — | Yes | API contract tests |
| GR-16 | Enterprise Qualification & Claims | PLANNED | Limit claims to evidence | GR-13 | — | Docs/process | Qualification sign-off |

---

## Platform ownership conformance (summary)

| Concern | Canonical owner | Implementation owner (today) | Conformant? |
| --- | --- | --- | --- |
| Execution lifecycle | Execution Runtime | UER + Nexus/UAEP paths | **Partial** |
| Five-ID identity | Execution Runtime / contracts | ContextVar + partial contract adoption | **Partial** |
| Execution authority (child) | ExecutionAuthorityPolicy | `execution/authority/policy.py` | **Yes** (narrowing) |
| Policy / inner evaluation | Governance plane | RuntimePolicyEngine, collaborative gate | **Partial** |
| HITL authorization | Governance | Human pause + grants on Task | **Partial** (grant **VERIFIED**; lifecycle port **OPEN**) |
| Decision correctness | Decision System | Separate; weak Governance link | **Partial** |
| Reliability / retry | Reliability | Not owned by governance spine | **Yes** |
| Evidence facts | Observability | Partial / parallel audit events | **Partial** |
| Diagnostics | Central Diagnostics | Policy trace diagnostics only | **Yes** (no auth coupling) |
| Nexus | Internal Execution orchestration | Primary **internal** HITL resume machinery (orchestration) | **Partial** — must route through `ExecutionContinuationPort` (GR-5-R4) |

---

*GR-0 artifact — independent re-audit required before implementation closure claims.*
