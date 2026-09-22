# UCA-6C-ARCH-R2 — Canonical HITL Re-entry Integration Proof

**Audited commit:** `58e3056dede3f5c2292e7ca340bb35f79dd1abaa` (`development`, clean worktree at audit start)  
**Reconciliation reference (prior audit):** `f6a538f4c2291d83d5c5423ac50bf3e9da45a033`  
**Verdict:** **ARCHITECTURAL DECISION REQUIRED**

---

## A. Verdict

**ARCHITECTURAL DECISION REQUIRED**

The platform implements a complete canonical declarative-tool HITL loop (Nexus orchestration / tool loop / catalog dispatch), but the **UCA-6C CodeCraft `code.exec` path does not participate** in that loop. `REQUIRE_HITL` from `RuntimeToolInvoker` on the execution-bound path is **not** translated to `DeclarativePolicyHitlPauseRequired`, is **not** persisted as `declarative_hitl_pending`, and does **not** reach `ExecutionContinuationPort`. It escapes as `DeclarativePolicyHitlRequiredError` and is wrapped into `CanonicalExecutionInvocationFailed` at `CanonicalExecutionRuntimeAdapter`.

R6 cannot reuse the existing canonical HITL re-entry **as-is** without new Execution-owned continuation/HITL integration at the qualified-capability seam and without reconciling `invocation_scope_id` semantics (bridge-minted `dhr_*` vs UCA-derived `uca6c-scope:*`).

---

## B. Repository state

| Field | Value |
|-------|--------|
| START_HEAD | `58e3056dede3f5c2292e7ca340bb35f79dd1abaa` |
| BRANCH | `development` |
| WORKTREE_STATE | clean |
| FINAL_COMMIT | (this document commit) |

---

## C. Canonical HITL entry (`REQUIRE_HITL` → signal)

1. `RuntimeToolInvoker._require_current_attempt_authorization` → `DeclarativePolicyEnforcer.evaluate_tool_invocation` (`intergrax/runtime/nexus/tools/invoker.py`, ~622–661).
2. On enforced `PolicyRuleAction.REQUIRE_HITL` → **`DeclarativePolicyHitlRequiredError`** (same file, ~654–661).
3. **Bridge (mandatory for pause):** callers catch that error and call **`raise_hitl_pause_from_tool_invocation`** (`intergrax/runtime/nexus/tools/declarative_policy_hitl_bridge.py`), which mints `invocation_scope_id` via **`generate_invocation_scope_id()`** (`dhr_{uuid}`), builds `DeclarativePolicyHitlSignal` / `DeclarativeHitlPendingApproval`, and raises **`DeclarativePolicyHitlPauseRequired`**.

Canonical callers with bridge: `tool_loop._finish_canonical_tool_invocation`, `catalog_dispatch.invoke_catalog_tool_request` / batch dispatch — **not** `NexusExecutionBoundCatalogToolInvoker.invoke`.

---

## D. Invocation scope

| Stage | Owner / location |
|-------|------------------|
| Mint (first REQUIRE_HITL) | `signal_from_error` → `generate_invocation_scope_id()` in `declarative_policy_hitl_bridge.py` |
| Pending persistence | `HumanPauseCoordinator` / task `TaskGovernanceState.declarative_hitl_pending` (via `ExecutionInterruptHandler` + graph runner) |
| Grant mint | `DeclarativeHitlGrantCoordinator.create_grant_from_pending` — copies `pending.invocation_scope_id` |
| Resumed assignment | `maybe_assign_declarative_hitl_scope` → `ToolExecutionRequest.declarative_hitl_invocation_scope_id` (tool loop / catalog dispatch only) |

**UCA path:** on resume, scope may be supplied only via `ToolInvocationGovernanceApprovalEvidence.invocation_scope_id`, validated against **`derive_qualified_capability_governance_invocation_scope_id(execution_request_id)`** (`uca6c-scope:uca6c.bound:{id}`) — **not** the bridge-minted `dhr_*` scope from a real policy pause.

---

## E. Continuation

| Step | Symbol / owner |
|------|----------------|
| Pause owner | Task orchestration + `HumanPauseCoordinator`; EE **`ExecutionContinuationPort`** (`intergrax/contracts/execution_continuation.py`) for governed continuation lifecycle |
| Grant at resume (orchestration) | `DeclarativeHitlGrantCoordinator.transfer_persisted_grant_for_resume` → `RuntimeRequest.declarative_hitl_grant` (`graph_executor.py` execute_fn) |
| Resume execution | Graph / Nexus loop resumes task; grant consumed at orchestration boundary |

**Qualified capability composition** (`build_qualified_capability_execution_dispatch_service`): **no** `ExecutionContinuationPort` membership.

---

## F. Grant transport (canonical orchestration)

```text
TaskGovernanceState.declarative_hitl_grant
  → DeclarativeHitlGrantCoordinator.transfer_persisted_grant_for_resume
  → RuntimeRequest.declarative_hitl_grant (response_schema.RuntimeRequest)
  → RuntimeState.declarative_hitl_grant (field on state; set in execution-bound path directly)
  → PolicyEvaluationContext.approval_grant (RuntimeToolInvoker, invoker.py ~633)
  → ToolExecutionRequest.declarative_hitl_invocation_scope_id (after maybe_assign_declarative_hitl_scope)
```

Execution-bound alternate (UCA only):

```text
ToolInvocationGovernanceApprovalEvidence
  → declarative_hitl_grant_from_invocation_evidence (Nexus adapter)
  → RuntimeState.declarative_hitl_grant
  → declarative_hitl_invocation_scope_id on ToolExecutionRequest (from evidence only)
```

---

## G. Call graph A — Canonical existing HITL E2E

```text
ToolExecutionRequest
  → RuntimeToolInvoker.invoke
  → DeclarativePolicyHitlRequiredError
  → raise_hitl_pause_from_tool_invocation
  → DeclarativePolicyHitlPauseRequired
  → orchestration / NexusLoop / graph_runner (pending on task)
  → human APPROVE → DeclarativeHitlGrantCoordinator.create_grant_from_pending
  → transfer_persisted_grant_for_resume → RuntimeRequest.declarative_hitl_grant
  → maybe_assign_declarative_hitl_scope
  → RuntimeToolInvoker.invoke (same logical invocation dimensions)
  → Governance re-eval → MSE → ToolExecutor
```

**E2E test:** `tests/integration/runtime/test_declarative_policy_hitl_nexus_e2e.py` (Nexus lab task → pause → `resume_lab_nexus_hitl` → same tool).

---

## H. Call graph B — Current UCA CodeCraft

```text
WorkerQualifiedCapabilityResumeCoordinator.resume
  → WorkerQualifiedCapabilityExecutionEngineAdapter
  → QualifiedCapabilityExecutionDispatchService.dispatch
  → DefaultRootExecutionLauncher.launch
  → CanonicalExecutionRuntimeAdapter.dispatch
  → ExecutionRuntime → QualifiedCapabilityExecutionRuntimeDelegate.execute
  → CodeCraftQualifiedCapabilityExecutionHandler.dispatch_once
  → WiringCodeCraftBoundCapabilityExecution.execute
       → resolve_codecraft_exec_authorization (CodeCraft-local HITL store)
       → ExecutionBoundCatalogToolInvokeRequest (optional governance_approval_evidence)
  → NexusExecutionBoundCatalogToolInvoker.invoke
       → fresh RuntimeState (local projection)
       → RuntimeToolInvoker.invoke (NO HITL bridge catch)
```

---

## I. First divergence

**`NexusExecutionBoundCatalogToolInvoker.invoke`** (`intergrax/runtime/nexus/tools/nexus_execution_bound_catalog_tool_invoker.py`, ~111–116): calls `tool_invoker.invoke` without catching `DeclarativePolicyHitlRequiredError` and without `raise_hitl_pause_from_tool_invocation`, `maybe_assign_declarative_hitl_scope`, or orchestration pause handling.

Secondary: **`CanonicalExecutionRuntimeAdapter`** (~55–65) wraps any uncaught exception (including `DeclarativePolicyHitlRequiredError`) as **`CanonicalExecutionInvocationFailed`**.

---

## J. REQUIRE_HUMAN behavior on UCA `code.exec`

**Does not pause canonically.** Policy `REQUIRE_HITL` raises through execution-bound invoker; intake adapter converts to **`CanonicalExecutionInvocationFailed`** (see `test_uca6c_r5_r5_end_to_end_approval_evidence_propagation.py::test_worker_resume_strict_without_evidence_fails_before_mse`). No `DeclarativePolicyHitlPauseRequired`, no task pending, no `ExecutionContinuationPort` transition.

If invoker returned a failed `ToolExecutionResult` instead, `_map_tool_execution_result` would map `policy_error` to **REJECTED** — still not HITL lifecycle.

---

## K. Public re-entry surface

**Canonical (orchestration / EE task host):** `ExecutionContinuationPort` + task governance fields (`DeclarativeHitlPendingApproval` / `DeclarativeHitlApprovalGrant`) — wired in Nexus graph/task execution, **not** on qualified-capability root launch.

**UCA carrier:** `WorkerQualifiedCapabilityResumeRequest.governance_approval_evidence` (`ToolInvocationGovernanceApprovalEvidence`) — **parallel transport**, not produced by canonical pause on this path.

**For UCA CodeCraft HITL re-entry without Nexus public API:** **NONE** that connects `REQUIRE_HITL` → pause → grant → same invocation automatically.

---

## L. ToolInvocationGovernanceApprovalEvidence

**Verdict: PARTIAL — adapter justified only at execution-bound boundary; otherwise duplicate transport.**

- Neutral contract: `intergrax/contracts/tool_invocation_governance_approval_evidence.py`
- Nexus-only reconstruction: `declarative_hitl_grant_from_invocation_evidence`
- Reverse adapter: `tool_invocation_governance_approval_evidence_from_declarative_hitl` — **tests/fixtures only**, no production path from canonical pause to worker resume
- UCA validation forces **pre-derived** `invocation_scope_id`, incompatible with bridge-minted scopes from real policy HITL

**Duplicate transport analysis:** **YES** (same responsibility as `DeclarativeHitlApprovalGrant` → `RuntimeState.declarative_hitl_grant`, plus UCA-specific scope derivation).

---

## M. Binding fallback

`CatalogDeclarativeRunBinding.declarative_hitl_grant` + `_declarative_hitl_grant_for_request` fallback: **COMPATIBILITY / BLOCKING HIDDEN STATE** — not used on canonical graph resume (task governance transfer is canonical); **not required** for orchestration E2E, but present as hidden mutable binding. **R6 must not depend on binding mutation.**

`CatalogDeclarativeRunBinding.declarative_hitl_grant` required for canonical production re-entry? **NO** (orchestration uses task grant transfer). Required for execution-bound without evidence? Only via hidden binding — **R6 NOT READY** if that were the design.

---

## N–O. Freeze / Nexus

- Identity core modified: **NO**
- HITL / ExecutionContinuationPort semantics modified: **NO**
- Nexus on public contracts / UCA / AW: **NO** (execution-bound default impl is Nexus-internal)
- EE internals: Nexus **INTERNAL ONLY**

---

## P. Two-HITL analysis

| Mechanism | Trigger | Owner | Pause lifecycle | Approval artifact | Resume owner | Canonical? |
|-----------|---------|-------|-----------------|-------------------|--------------|------------|
| `resolve_codecraft_exec_authorization` | supervised / `require_hitl_before_exec` profile | CodeCraft wiring + `HumanDecisionStore` | None (local pending/denied) | HumanDecisionRecord (craft-scoped notes) | Worker re-call with store approval | **No** (local pre-exec gate) |
| Declarative tool HITL | `PolicyRuleAction.REQUIRE_HITL` | Governance + bridge + task HITL | Task pause / pending | `DeclarativeHitlApprovalGrant` | Orchestration resume + grant transfer | **Yes** |
| Governed continuation | MSE `ToolGovernanceApprovalRequiredError` + continuation request | MSE + governed continuation bridge | `ExecutionContinuationPort` | `GovernedContinuationApprovalGrant` | Continuation coordinator | **Yes** (distinct concern) |
| `ExecutionContinuationPort` | Governed pause commands | Execution Engine | PAUSED → WAITING_FOR_HUMAN → RESUME_AUTHORIZED → RESUMED | Continuation snapshots | EE port implementor | **Yes** |

UCA `code.exec` can hit **CodeCraft local exec authorization** before ToolRuntime, and **declarative/MSE HITL** inside `RuntimeToolInvoker` — **two gates**, only the latter is canonical platform HITL; the former is a **separate concern** (pre-exec craft approval), not a substitute for policy HITL.

---

## Q. Same execution on resume?

**Canonical orchestration HITL:** **YES** — same task/run, grant tied to invocation dimensions; E2E integration test demonstrates handler recall after resume.

**UCA qualified-capability path on policy HITL:** **NO** — no pause; failure/exception ends the intake invocation; no bridge persistence. Separate worker resume with new root launch is a **new execution episode** (distinct `execution_id` across recovery resumes in R5 tests).

---

## R. R6 readiness matrix

| Gate | Result |
|------|--------|
| Existing HITL path reusable for UCA `code.exec` | **NO** |
| Public re-entry surface exists for this path | **NO** |
| Exact invocation scope reusable (bridge ↔ UCA) | **NO** |
| Canonical grant transport reusable without parallel carrier | **NO** |
| No identity changes | YES |
| No continuation contract changes | YES |
| No Governance changes | YES |
| No ToolRuntime core changes | YES |
| No Nexus public dependency | YES (but impl is Nexus-internal) |
| No binding mutation | YES (must stay forbidden) |
| No new approval carrier needed | **NO** (gap: need EE-owned seam or scope unification) |

**R6 BLOCKED** until architectural decision.

---

## S. Exact R6 scope

*Not applicable (verdict ≠ PASS).*

---

## T. R6 forbidden scope (always)

Frozen families per UCA-6C-ARCH-R2 charter: `execution_identity*`, `execution_continuation*`, declarative/governed HITL contracts and coordinators, `DeclarativePolicyEnforcer` semantics, `RuntimeToolInvoker` / ToolExecutor core, public Nexus dependency, `CatalogDeclarativeRunBinding` grant mutation as consumer contract.

---

## U. Session goal check

| Question | Answer |
|----------|--------|
| One canonical capability fulfillment flow? | **NO** (UCA tool HITL not on canonical pause loop) |
| Second HITL mechanism required? | **NO** (but second *transport* and scope model exist today) |
| Second approval transport required? | **YES** (TIGAE vs task grant — not unified) |
| Second identity mechanism required? | **NO** |
| Public Nexus dependency required? | **NO** |

---

## GITHUB AUDIT REQUIRED

Wnioski UCA-6C-ARCH-R2 muszą zostać niezależnie zaudytowane na podstawie dokładnego kodu i dokumentacji z commitu na GitHubie.

**COMMIT_SHA:** (set at commit)

---

## Roadmap

| Etap | Status | Cel |
|------|--------|-----|
| UCA-6C-ARCH-R1 | CLOSED | Canonical HITL Boundary & Surgical Reconciliation |
| UCA-6C-ARCH-R2 | CURRENT | Canonical HITL Re-entry Integration Proof |
| UCA-6C-R6 | BLOCKED UNTIL PASS | Surgical Boundary Correction |
| UCA-6C-DOC | WAIT | Governed Capability Fulfillment ownership freeze + gates |
| UCA-6C Final Closure Audit R2 | WAIT | final enterprise audit |
| UCA-7A | WAIT | second consumer |
| UCA-7B | WAIT | cross-consumer E2E |
| UCA-7C | WAIT | adversarial qualification |
| UCA-8 | WAIT | durable/distributed enterprise closure |

---

> Udowodnić, czy istniejący canonical HITL i ExecutionContinuationPort potrafią wznowić dokładną `code.exec` invocation w UCA-6C bez nowego approval transportu, zmian identity i publicznej zależności od Nexus.

**UCA-6C-ARCH-R2 — Canonical HITL Re-entry Integration Proof**
