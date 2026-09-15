# Governance Architecture Rebase — Gap Ledger (GR-0)

**Rebase audit HEAD (session):** `fe2edc8077234437b13345daaf46633867fe8f31` on `development`  
**H9.2C maintainer/doc reconciliation HEAD:** `1de3b7fca6ca76e015c284ce21b8d543487ef677` on `development`  
**Canonical architecture baseline:** Unified Execution Runtime, five-ID execution identity, Decision System, Evidence Plane, Central Diagnostics (verified against listed arch docs + production paths below).

**Purpose:** Evidence-based gap inventory after platform-wide execution-centric rebase. Supersedes stale maintainer PG-FIX status rows where code truth differs; does not erase historical audit references.

---

## Executive summary

Governance **mechanisms** (collaborative-work enforcement gate, `MeaningfulSideEffectAuthorizationBoundary`, scoped `GovernedContinuationApprovalGrant`, `RuntimePolicyEngine` precedence) are materially present on `development`, but **platform identity conformance** (AttemptId / ExecutionId binding), **HITL pause ownership** (Task/Nexus lifecycle vs UER), **evidence plane correlation**, **Decision provenance**, and **strategy-wide qualification** remain open. Enterprise readiness is **not** claimed.

**Enterprise verdict for implementation:** `READY_FOR_GR-1` (identity rebind is the critical path; no blocking architectural fork requiring operator decision before GR-1).

---

## Candidate hypothesis verdicts (GOV-REBASE-01 … 10)

| ID | Verdict | Severity | Evidence (summary) |
| --- | --- | --- | --- |
| GOV-REBASE-01 | **CLOSED** | P0 | GR-1: `GovernedContinuationApprovalGrant`, correlation, and `matches_current_requirement` bind `task_id` + `run_id` + `attempt_id` + `execution_id`. |
| GOV-REBASE-02 | **CLOSED** | P0 | GR-1: `MeaningfulSideEffectRequest` requires canonical four-ID execution identity; governed continuation chain propagates without loss. |
| GOV-REBASE-03 | **CONFIRMED** | P1 | `authorize_and_execute` calls `lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)` and `HumanPauseCoordinator` on `Task` governance state (`meaningful_side_effect_authorization.py`, `governed_continuation_bridge.py`). Canonical arch assigns pause/resume to UER — implementation still Task/Nexus-centric. |
| GOV-REBASE-04 | **PARTIAL** | P1 | Production HITL bridge avoids direct `nexus.*` import except `declarative_hitl_grant` → `RuntimeRequest`. Pause/resume semantics and tests remain Nexus-orchestration-shaped; inference/agentic HITL not qualified. Arch target: HITL independent of Nexus. |
| GOV-REBASE-05 | **PARTIAL** | P1 | `authorize_and_execute` runs caller `execute` callback after auth — not a second `ExecutionBoundary`, but can execute **outside** active canonical Execution context when caller omits UER binding (`agents/external_contractor_adapter/external_work_adapter.py` call site). Inner-op pattern (A) is intended; enforcement of (A) is **gap**. |
| GOV-REBASE-06 | **PARTIAL** | P1 | No `DecisionId`/version on meaningful-side-effect or governed-continuation contracts. `ContinuationEvidenceRefs.hitl_decision_id` is HITL store id, not Decision System authority. Decision-derived effects lack typed Decision provenance where required. |
| GOV-REBASE-07 | **CONFIRMED** | P1 | Policy decisions use `audit_payload` on `PolicyDecision`; no systematic RuntimeEvent emission with five-ID correlation from governance spine. `governance_audit_event` is separate agent-governance channel — risk of parallel audit semantics. |
| GOV-REBASE-08 | **CONFIRMED** | P1 | G3B table (`GOVERNED_EXECUTION.md`) wires most points via Nexus/UAEP; no qualified matrix for INFERENCE/AGENTIC meaningful-side-effect + HITL (see strategy matrix below). |
| GOV-REBASE-09 | **CLOSED** | P2 | H9.2C: maintainer plan + arch G3B / Protocol v2.2 pointers reconciled with PG-FIX mechanism tests (`test_pg_fix_*`, `test_g5c2b*`); enterprise CLOSED still not claimed. |
| GOV-REBASE-10 | **CONFIRMED** | P1 | G3B + plan: **CONTROL_PLANE_MUTATION** remains **GAP**; domain tests (e.g. ECP) are partial slices, not platform-wide shared boundary. |

---

## Gap ledger (canonical rows)

| ID | Severity | Area | Current code truth | Target architecture | Mismatch | Risk | Required remediation | Dependencies | Roadmap slice | Evidence paths | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GOV-GAP-001 | P0 | Identity | GR-1-R1: atomic Attempt+Execution resolution; no hybrid caller/active assembly | Authorization bound to AttemptId + ExecutionId where security-sensitive | Pre-R1 partial override could mix caller Attempt with active Execution | Cross-attempt side-effect authorization | GR-1-R1 resolver + pause atomic bind | GR-0 audit | GR-1-R1 | `meaningful_side_effect.py`, `pause.py`, `test_meaningful_side_effect_execution_identity_resolution.py` | **CLOSED** |
| GOV-GAP-002 | P0 | Identity | GR-1: `HumanApprovalResolution` carries attempt/execution; governed pause validates correlation | Resolution correlated to active Execution | Residual UER pause ownership (GR-5) | HITL replay if resolution forged | GR-1 pause validation + grant derivation | GR-1 | GR-5 | `task_contract.py`, `pause.py` | **VERIFIED** |
| GOV-GAP-003 | P1 | HITL ownership | TaskLifecycle `WAITING_FOR_HUMAN`; Task governance blob | UER owns PAUSE/WAIT/RESUME same Execution | Doc/code split on pause owner | Nexus-shaped pause on non-orchestration strategies | Rebase pause onto Execution lifecycle; Task bridge only | UER APIs | GR-5 | `meaningful_side_effect_authorization.py`, `governed_continuation_bridge.py` | OPEN |
| GOV-GAP-004 | P1 | Nexus coupling | Governed continuation docstrings + AgentExecutionResult bridge | HITL without mandatory Nexus | Orchestration-only proof for HITL path | Agentic/inference hosts lack qualified HITL | Strategy-neutral HITL entry; qualify non-Nexus paths | GR-5 | GR-5, GR-10 | `governed_continuation.py` (module doc), `governed_continuation_bridge.py` | OPEN |
| GOV-GAP-005 | P1 | Execution boundary | GR-3 + GR-3-R1: contract-only `MeaningfulSideEffectAuthorizationBoundary`; default guard wired only in `meaningful_side_effect_authorization_composition.py`; four-ID inner enforcement + `authorize_and_execute` allowlist gate | Side effects only inside admitted Execution | Residual Reliability delivery semantics (GR-7) | Hidden concrete guard in policy consumer | GR-3-R1 explicit composition + architecture import gate | GR-1, GR-2 | GR-7 | `canonical_inner_governance.py`, `meaningful_side_effect_authorization.py`, `meaningful_side_effect_authorization_composition.py`, `test_gr3_*`, `test_gr3_r1_*` | **CANDIDATE CLOSED** (GR-3-R1 DONE; pending independent audit) |
| GOV-GAP-006 | P1 | Decision integration | No DecisionId/version on governance contracts | Provenance only for decision-derived consequential effects | Cannot prove authorization matches decision version | V2 decision reuses V1 approval | Neutral provenance ref + Decision binding where material | Decision System contracts | GR-6 | contracts grep; `DECISION_SYSTEM.md` | OPEN |
| GOV-GAP-007 | P1 | Evidence | Policy/HITL facts mostly in-memory Task state + `audit_payload` | Canonical Evidence Plane / RuntimeEvent with five IDs | Incomplete forensic reconstruction | Duplicate or missing governance facts | Emit correlated governance facts via observability ports | Observability | GR-8 | `runtime_policy_engine.py`, `agent_runtime_governance.py` | OPEN |
| GOV-GAP-008 | P1 | Strategy coverage | G3B Nexus/UAEP-heavy COVERED rows | Matrix for INFERENCE/AGENTIC/ORCHESTRATION | “Nexus works” ≠ platform proof | Ungoverned paths on non-orchestration strategies | Qualification matrix + close gaps | GR-10 | GR-10 | `GOVERNED_EXECUTION.md` G3B table | OPEN |
| GOV-GAP-009 | P1 | Control plane | CONTROL_PLANE_MUTATION GAP | Shared authority context per domain executor | No unified taxonomy enforcement | Unsafe platform mutations | Per-domain adoption of shared boundary; no god executor | Domain plans | GR-12 | plan CLA block; ECP tests partial | OPEN |
| GOV-GAP-010 | P2 | Maintainer truth | Stale PG-FIX / Protocol v2.2 status in docs | IMPLEMENTED / VERIFIED / CLOSED distinguished | Operators mis-plan | False closure claims | Reconcile plan + arch pointers (GR-0) | GR-0 | GR-0 | `plans/GOVERNED_EXECUTION.md` | **CLOSED** (H9.2C) |
| GOV-GAP-011 | P2 | Policy plugins | Catalog + handler slices; Nexus types in `policy_bundle.py` | Vendor-neutral core; platform plugin admission | Residual Nexus coupling in policy assembly | Tier violation / test burden | Gradual decouple bundle assembly from Nexus models | Platform plugins | GR-4, GR-11 | `policy_bundle.py`, `tool_policy_resolution.py` | OPEN |
| GOV-GAP-012 | P2 | Admission vs inner | GR-2 root admission (`RootExecutionAuthorityAdmissionPort`) vs GR-3 inner guard (`CanonicalInnerExecutionGuardPort`) — distinct contracts | Admission = may start Execution; inner = may proceed | Policy requalification still GR-4 | Misuse of admission as inner policy | Document ports; GR-4 closes policy plugin gaps | GR-2, GR-3 | GR-4 | `execution_admission_composition.py`, `canonical_inner_governance.py` | **PARTIAL** (inner vs root split documented; policy plugin gaps remain GR-4) |
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
- Missing DecisionId/version binding for decision-derived side effects — **gap** (GOV-GAP-006).

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

## PG-FIX current truth (GR-0 code audit)

| Block | IMPLEMENTED? | VERIFIED? | CLOSED? | Notes |
| --- | --- | --- | --- | --- |
| PG-FIX-A | **Yes** (core spine) | **Partial** (adapter + unit tests) | **No** | `CollaborativeWorkEnforcementGate` + `MeaningfulSideEffectAuthorizationBoundary`; universal consumer coverage not proven |
| PG-FIX-B | **Yes** | **Partial** (`test_pg_fix_b_*`) | **No** | Deterministic specificity in `RuntimePolicyEngine` |
| PG-FIX-C | **Yes** (mechanism) | **Partial** (G5C / PG-FIX-C tests) | **No** | Identity conformance **CLOSED** (GR-1); UER HITL ownership open (GR-5) |
| PG-FIX-D | **Yes** (typed matching) | **Partial** (`test_pg_fix_d_*`) | **No** | `rule_id` suffix dispatch rejected in tests |

Historical AUDIT-5 findings remain valid context; closure requires identity rebind + enterprise qualification (GR-13).

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
| GR-3 | Canonical Inner Enforcement | **CANDIDATE DONE** (audit pending) | Active Execution four-ID binding before meaningful side effects; no second executor | GR-1, GR-2 | PG-FIX-A requalified | Yes | `test_gr3_canonical_inner_enforcement.py`, `test_gr3_inner_enforcement_architecture_gates.py` |
| GR-4 | Policy Resolution & Catalog Requalification | **BLOCKED until independent GR-3 audit** | Close PG-FIX-B/D qualification gaps | GR-3 | G2C, PG-FIX-B/D | Yes | Precedence + catalog tests |
| GR-5 | HITL / Governed Continuation Rebase | PLANNED | UER pause/resume; scoped approval preserved | GR-1 | G5*, PG-FIX-C | Yes | HITL E2E per strategy |
| GR-6 | Decision → Governance Integration | PLANNED | Decision provenance where material | GR-1 | — | Yes | Decision-version binding tests |
| GR-7 | Reliability / External Effect Boundary | PLANNED | Authorization vs retry/idempotency | GR-3, GR-5 | — | Yes | Reliability boundary tests |
| GR-8 | Governance Evidence Integration | PLANNED | Five-ID correlated facts in Evidence Plane | GR-1 | — | Yes | RuntimeEvent correlation tests |
| GR-9 | Diagnostic Consumption Proof | PLANNED | DIAG reads governance evidence only | GR-8 | — | Mostly tests/docs | DIAG fixtures |
| GR-10 | Execution Strategy Coverage | PLANNED | Inference/agentic/orchestration matrix | GR-3, GR-5 | G3B | Yes | Strategy qualification suite |
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
| HITL authorization | Governance | Human pause + grants on Task | **Partial** |
| Decision correctness | Decision System | Separate; weak Governance link | **Partial** |
| Reliability / retry | Reliability | Not owned by governance spine | **Yes** |
| Evidence facts | Observability | Partial / parallel audit events | **Partial** |
| Diagnostics | Central Diagnostics | Policy trace diagnostics only | **Yes** (no auth coupling) |
| Nexus | Orchestration strategy | Still primary HITL resume path | **Partial** vs target |

---

*GR-0 artifact — independent re-audit required before implementation closure claims.*
