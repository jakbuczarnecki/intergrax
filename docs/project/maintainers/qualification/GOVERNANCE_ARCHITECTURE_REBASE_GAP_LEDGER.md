# Governance Architecture Rebase — Gap Ledger (GR-0)

**Rebase audit HEAD (session):** `fe2edc8077234437b13345daaf46633867fe8f31` on `development`  
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
| GOV-REBASE-09 | **CONFIRMED** | P2 | Maintainer plan still lists PG-FIX-B/C/D as ACCEPTED/PLANNED and arch Protocol v2.2 as “not implemented” while targeted tests exist (`test_pg_fix_*`, `test_g5c2b*`). |
| GOV-REBASE-10 | **CONFIRMED** | P1 | G3B + plan: **CONTROL_PLANE_MUTATION** remains **GAP**; domain tests (e.g. ECP) are partial slices, not platform-wide shared boundary. |

---

## Gap ledger (canonical rows)

| ID | Severity | Area | Current code truth | Target architecture | Mismatch | Risk | Required remediation | Dependencies | Roadmap slice | Evidence paths | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GOV-GAP-001 | P0 | Identity | GR-1-R1: atomic Attempt+Execution resolution; no hybrid caller/active assembly | Authorization bound to AttemptId + ExecutionId where security-sensitive | Pre-R1 partial override could mix caller Attempt with active Execution | Cross-attempt side-effect authorization | GR-1-R1 resolver + pause atomic bind | GR-0 audit | GR-1-R1 | `meaningful_side_effect.py`, `pause.py`, `test_meaningful_side_effect_execution_identity_resolution.py` | **CLOSED** |
| GOV-GAP-002 | P0 | Identity | GR-1: `HumanApprovalResolution` carries attempt/execution; governed pause validates correlation | Resolution correlated to active Execution | Residual UER pause ownership (GR-5) | HITL replay if resolution forged | GR-1 pause validation + grant derivation | GR-1 | GR-5 | `task_contract.py`, `pause.py` | **VERIFIED** |
| GOV-GAP-003 | P1 | HITL ownership | TaskLifecycle `WAITING_FOR_HUMAN`; Task governance blob | UER owns PAUSE/WAIT/RESUME same Execution | Doc/code split on pause owner | Nexus-shaped pause on non-orchestration strategies | Rebase pause onto Execution lifecycle; Task bridge only | UER APIs | GR-5 | `meaningful_side_effect_authorization.py`, `governed_continuation_bridge.py` | OPEN |
| GOV-GAP-004 | P1 | Nexus coupling | Governed continuation docstrings + AgentExecutionResult bridge | HITL without mandatory Nexus | Orchestration-only proof for HITL path | Agentic/inference hosts lack qualified HITL | Strategy-neutral HITL entry; qualify non-Nexus paths | GR-5 | GR-5, GR-10 | `governed_continuation.py` (module doc), `governed_continuation_bridge.py` | OPEN |
| GOV-GAP-005 | P1 | Execution boundary | `authorize_and_execute` invokes arbitrary callable | Side effects only inside admitted Execution | Parallel execution path vs `ExecutionBoundary` | Bypass of execution admission/identity context | Require active execution identity or explicit inner-op port; document contract | GR-1, GR-2 | GR-3, GR-7 | `meaningful_side_effect_authorization.py`, `external_work_adapter.py` | OPEN |
| GOV-GAP-006 | P1 | Decision integration | No DecisionId/version on governance contracts | Provenance only for decision-derived consequential effects | Cannot prove authorization matches decision version | V2 decision reuses V1 approval | Neutral provenance ref + Decision binding where material | Decision System contracts | GR-6 | contracts grep; `DECISION_SYSTEM.md` | OPEN |
| GOV-GAP-007 | P1 | Evidence | Policy/HITL facts mostly in-memory Task state + `audit_payload` | Canonical Evidence Plane / RuntimeEvent with five IDs | Incomplete forensic reconstruction | Duplicate or missing governance facts | Emit correlated governance facts via observability ports | Observability | GR-8 | `runtime_policy_engine.py`, `agent_runtime_governance.py` | OPEN |
| GOV-GAP-008 | P1 | Strategy coverage | G3B Nexus/UAEP-heavy COVERED rows | Matrix for INFERENCE/AGENTIC/ORCHESTRATION | “Nexus works” ≠ platform proof | Ungoverned paths on non-orchestration strategies | Qualification matrix + close gaps | GR-10 | GR-10 | `GOVERNED_EXECUTION.md` G3B table | OPEN |
| GOV-GAP-009 | P1 | Control plane | CONTROL_PLANE_MUTATION GAP | Shared authority context per domain executor | No unified taxonomy enforcement | Unsafe platform mutations | Per-domain adoption of shared boundary; no god executor | Domain plans | GR-12 | plan CLA block; ECP tests partial | OPEN |
| GOV-GAP-010 | P2 | Maintainer truth | Stale PG-FIX / Protocol v2.2 status in docs | IMPLEMENTED / VERIFIED / CLOSED distinguished | Operators mis-plan | False closure claims | Reconcile plan + arch pointers (GR-0) | GR-0 | GR-0 | `plans/GOVERNED_EXECUTION.md` | IN_PROGRESS |
| GOV-GAP-011 | P2 | Policy plugins | Catalog + handler slices; Nexus types in `policy_bundle.py` | Vendor-neutral core; platform plugin admission | Residual Nexus coupling in policy assembly | Tier violation / test burden | Gradual decouple bundle assembly from Nexus models | Platform plugins | GR-4, GR-11 | `policy_bundle.py`, `tool_policy_resolution.py` | OPEN |
| GOV-GAP-012 | P2 | Admission vs inner | `evaluate_root_execution_admission` + inner meaningful-side-effect | Admission = may start Execution; inner = may proceed | Potential semantic overlap if misused | Admission replaces policy | Keep `ExecutionAuthorityPolicy` as child narrowing only; document ports | GR-2 | GR-2, GR-3 | `execution/authority/policy.py`, `runtime_execution_policy_admission.py` | OPEN |

---

## Audit question snapshots

### A — Canonical execution identity (contracts)

| Contract / artifact | TaskId | RunId | AttemptId | ExecutionId |
| --- | --- | --- | --- | --- |
| MeaningfulSideEffectRequest | yes | yes | no | no |
| GovernedContinuationRequest / Correlation | yes | yes | no | no |
| GovernedContinuationApprovalGrant | yes | yes | no | no |
| HumanApprovalResolution | yes | optional run | no | no |
| Pause record | yes | — | no | no |
| Policy evidence (PolicyDecision) | via payload/context only | partial | no | no |

**Cross-execution reuse:** Within same Task+Run, matching scope + policy bundle → **grant can authorize semantically identical side effect on another Execution/Attempt** → **P0**.

### B — Attempt semantics (code truth, not invented)

| Scenario | Approval validity (current) |
| --- | --- |
| Same Attempt / same Execution | Grant matches if task/run/scope/policy unchanged |
| Same Run / new Attempt | **Still valid** (no attempt dimension) — **gap** |
| Same Attempt / different Execution | **Still valid** if task/run/scope match — **gap** |
| Resumed same Execution | Not bound to execution_id — **unclear / likely gap** |
| New child Execution | Not bound to parent/child execution — **gap** |

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
| Task/run/scope/policy/pause binding | **MECHANISM SOUND** (tests G5C-2B*) | **FAIL** (no attempt/execution) |
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
| PG-FIX-C | **Yes** (mechanism) | **Partial** (G5C tests) | **No** | Identity conformance fails GOV-REBASE-01 |
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
| GR-2 | Execution Admission Governance | **NEXT** | Single admission story at Execution start | GR-1-R2 | G3 admission rows | Yes | Admission integration tests |
| GR-3 | Inner Evaluation Spine | PLANNED | One inner enforcement path; safe `authorize_and_execute` | GR-1, GR-2 | PG-FIX-A completion | Yes | Bypass gate tests |
| GR-4 | Policy Resolution & Catalog Requalification | PLANNED | Close PG-FIX-B/D qualification gaps | GR-3 | G2C, PG-FIX-B/D | Yes | Precedence + catalog tests |
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
