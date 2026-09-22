# UCA-6C-ARCH-R1 — Canonical HITL Boundary & Surgical Reconciliation Plan

**Classification:** META_ARCHITECTURE / RECONCILIATION (not a new domain pair)  
**Task:** UCA-6C-ARCH-R1  
**Production code in this artifact:** none (plan only)

---

## 1. Context

UCA-6C R5-R4/R5-R5 introduced **approval evidence propagation** from Autonomous Work (worker capability recovery) through Execution Engine dispatch into CodeCraft bound execution and catalog `code.exec` invocation. Independent audit at `2ed6d181628deffe3ff921d0bdeaad53255c36dd` classified the UCA architecture core as sound but flagged:

- **Approval propagation direction** (AW → pre-authorized future tool step) as architecture drift.
- **Consumer-supplied root `ExecutionId`** as not acceptable.

Post-audit commit `62c533a3c90ba36c1a7d7abe7fb00894e9ed9add` (**HEAD at ARCH-R1 authoring**) partially corrected identity by removing consumer `execution_id` from AW/dispatch **ingress** contracts while retaining `governance_approval_evidence` transport and pre-execution scope derivations.

**ARCH-R1 goal:** evidence-backed KEEP / REMOVE / REWIRE / DO NOT TOUCH map so **UCA-6C-R6** can restore canonical Governance/HITL + `ExecutionContinuationPort` without new identity, HITL, continuation, or governance mechanisms.

**Primary audit commit range (minimum):** `de8f8d5b52b1299a109a59f485c0d5561914ba94` … `2ed6d181628deffe3ff921d0bdeaad53255c36dd`  
**Notable commits in range:**

| SHA | Summary |
|-----|---------|
| `2ed6d1816` | fix(uca): close end to end approval evidence propagation |
| `62c533a3c` | fix(uca): restore platform owned execution identity (post-audit) |

---

## 2. Canonical target flow

```text
Worker recovery (RECOVERING → resume WorkItem responsibility)
  → acquisition (coordinate)
  → capability qualification (acquired subject)
  → binding (qualified + bound target)
  → canonical Execution request (UCA/AW stops here)
  → Execution Engine owns lifecycle + mints ExecutionId
  → binding handler (e.g. CodeCraft) under active execution identity
  → ToolRuntime + exact tool invocation (code.exec)
  → Governance evaluates exact invocation
  → if REQUIRE_HUMAN: canonical HITL → ExecutionContinuationPort
      PAUSE → WAITING_FOR_HUMAN → human resolution → RESUME_AUTHORIZED → RESUMED
  → same protected operation continues / re-evaluates
```

**Hard separations (frozen intent):**

```text
DISCOVERY != REALIZATION != ACQUISITION != QUALIFICATION != LIFECYCLE != EXECUTION
coordination != ownership
```

**Worker recovery resume ≠ execution resume**

| Concept | Owner | Meaning |
|---------|-------|---------|
| Worker capability recovery resume | Autonomous Work | Resume original work responsibility after recovery; not `ExecutionContinuationPort.resume()` |
| True execution pause/resume | Execution Engine via `ExecutionContinuationPort` | PAUSED → WAITING_FOR_HUMAN → RESUME_AUTHORIZED → RESUMED on **already-admitted** execution |

---

## 3. Incorrect pre-approval flow (R5-R4/R5-R5 drift)

```text
AW/Worker predicts future code.exec step
  → derives uca6c.bound:{execution_request_id} step/scope
  → attaches ToolInvocationGovernanceApprovalEvidence on resume/dispatch/intake
  → starts canonical execution already carrying approval
  → CodeCraft wiring passes evidence into ExecutionBoundCatalogToolInvokeRequest
  → Nexus adapter hydrates DeclarativeHitlApprovalGrant without live HITL pause on that invocation
```

**Why AW/UCA must not carry `governance_approval_evidence`:** the layer does not yet know the **exact** tool invocation under an admitted execution; approval authority belongs to Governance + canonical HITL at invoke time, correlated by `invocation_scope_id` on the real `ToolExecutionRequest`, not by pre-derived `execution_request_id` scopes.

---

## 4. Ownership matrix (GCF)

| Concern | Canonical owner |
|--------|-------------------|
| Need | Consumer |
| Discovery | Capability Catalog |
| Marketplace recommendation | Marketplace |
| Realization / lifecycle | Domain owner |
| Generic acquisition coordination | UCA acquisition |
| Acquired-subject qualification | Capability qualification |
| Provider / environment qualification | Core qualification |
| Binding | Qualification / domain handoff |
| Execution lifecycle | Execution Engine |
| Execution identity | ExecutionIdentityAuthority |
| Tool invocation | Tools / ToolRuntime |
| Policy decision | Governance |
| Human pause / resume | ExecutionContinuationPort |
| Code synthesis | CodeCraft |
| Sandbox backend | Sandbox |
| Nexus orchestration | Execution Engine internals only (not public) |

### GCF invariants (for future `GOVERNED_CAPABILITY_FULFILLMENT.md`)

- **GCF-INV-001** coordination != ownership  
- **GCF-INV-002** qualification != authorization  
- **GCF-INV-003** acquisition != lifecycle  
- **GCF-INV-004** binding != execution  
- **GCF-INV-005** capability growth != authority growth  
- **GCF-INV-006** no second HITL  
- **GCF-INV-007** no second Execution Engine  
- **GCF-INV-008** no public Nexus dependency  
- **GCF-INV-009** ToolRuntime mandatory for tool invocation  
- **GCF-INV-010** true gap only after complete discovery  

**Capability qualification boundary:**

```text
CapabilityQualification != ProviderQualification != Governance Authorization
!= Lifecycle Activation != Execution Admission
```

**Acquisition:** no direct registry mutation found in `intergrax/capability_acquisition/` (coordination only). Direct registry mutation would be a **BLOCKER**.

**CodeCraft path:** `CodeCraft → execution-bound public contract → ToolRuntime → code.exec → sandbox` — **KEEP**.

---

## 5. Approval evidence type comparison (semantics)

| Field / semantic | ToolInvocationGovernanceApprovalEvidence | DeclarativeHitlApprovalGrant | GovernedContinuationApprovalGrant | Duplicate? |
|------------------|------------------------------------------|------------------------------|-----------------------------------|------------|
| Grant / evidence id | `evidence_ref` | `grant_id` | `grant_id` | Parallel transport id; different minting authority |
| Tool invoke correlation | `invocation_scope_id` | `invocation_scope_id` | — (uses `side_effect_scope_id`) | TIGAE ↔ DHA grant: **same HITL tool scope**; GCG: **continuation scope** |
| Execution binding | task/run/step (strings) | task/run/step | typed TaskId/RunId/AttemptId/**ExecutionId** | GCG binds **paused execution**; TIGAE/DHA bind **tool invocation** |
| Tool/agent | `tool_id`, `agent_id` | same | — | Tool HITL only |
| Human workflow | `human_request_id`, `pause_id`, `approved_at` | same | same family | Shared HITL store semantics for tool path |
| Policy rules | `matched_rule_ids`, `policy_provenance_digest` | same | policy bundle fields on GCG | Overlap on tool path only |
| Tenant | `tenant_id` | — (tenant on request) | — | TIGAE carrier only |
| Continuation request | — | — | `continuation_request_id`, `operation_id` | GCG-only |
| Authority | Neutral **transport** from DHA grant; **not** an approval authority | **Authoritative** post-APPROVE tool grant | **Authoritative** post-APPROVE continuation grant | **Do not merge** into UniversalApprovalGrant |

**`invocation_scope_id` correlation:** already canonical on `DeclarativeHitlPendingApproval`, `DeclarativeHitlApprovalGrant`, and `ToolExecutionRequest.declarative_hitl_invocation_scope_id`. No replacement design.

---

## 6. Surgical matrix

| File / symbol | Current role | Problem? | Verdict | Reason | Future action (R6) |
|---------------|--------------|----------|---------|--------|-------------------|
| `WorkerQualifiedCapabilityResumeRequest.governance_approval_evidence` | AW ingress optional pre-approval | Yes — pre-authorizes future tool | **REMOVE** | AW must not know future tool approval | Drop field + validation; coordinator stops forwarding |
| `WorkerQualifiedCapabilityExecutionRequest.governance_approval_evidence` | Handoff to EE dispatch | Yes | **REMOVE** | Same | Drop field; adapter stops mapping |
| `QualifiedCapabilityExecutionDispatchRequest.governance_approval_evidence` | EE ingress | Yes | **REMOVE** | Pre-loaded approval at admission | Drop field; dispatch service stops copying to intake |
| `QualifiedCapabilityExecutionIntakePayload.governance_approval_evidence` | Runtime payload | Yes | **REMOVE** | Carries AW approval into boundary | Drop field; delegate/handler stop passing |
| `derive_qualified_capability_governance_step_id` | Pre-execution step id | Yes — predicts step before invoke | **REMOVE** | Conflicts with invoke-time governance | Delete or restrict to tests only if needed short-term |
| `derive_qualified_capability_governance_invocation_scope_id` | Pre-execution scope | Yes | **REMOVE** | Same | Same |
| `validate_governance_approval_evidence_for_execution_request` | Validates pre-derived scope | Yes | **REMOVE** | Encodes pre-approval model | Remove with fields |
| `WorkerQualifiedCapabilityExecutionResult.execution_id` | Outcome from EE | No | **KEEP** | EE-minted identity returned to AW | Keep as read-only outcome |
| `QualifiedCapabilityExecutionDispatchResult.execution_id` | Outcome | No | **KEEP** | Platform-owned output | Keep |
| `QualifiedCapabilityExecutionDispatchRequest.execution_id` (ingress) | Was consumer root id | Fixed at `62c533a3c` | **DO NOT TOUCH** (already removed) | Identity fix landed | Verify no reintroduction |
| `CodeCraftBoundCapabilityExecutionRequest.execution_id` | Active execution correlation | No | **KEEP** | Supplied by handler from `peek_active_execution_id()` | Keep; must match active context |
| `CodeCraftBoundCapabilityExecutionRequest.governance_approval_evidence` | Skip HITL at invoke | Yes when fed from AW | **REWIRE** | Invoke may carry grant **only** from runtime HITL resume | Stop sourcing from dispatch; optional field only for in-execution grant injection |
| `WiringCodeCraftBoundCapabilityExecution` + `ExecutionBoundCatalogToolInvokeRequest.governance_approval_evidence` | ToolRuntime path | No if evidence from HITL | **KEEP** wiring / **REWIRE** source | Canonical ToolRuntime boundary | Keep invoker; remove AW-fed evidence |
| `ToolInvocationGovernanceApprovalEvidence` (contract) | Neutral carrier | Drift when AW-filled | **REWIRE** | Legitimate as DHA→invoke adapter input | **REMOVE CANDIDATE** only if no invoke consumer remains; expect **KEEP** for ToolRuntime |
| `declarative_hitl_tool_invocation_approval_evidence.py` | DHA → TIGAE | No | **KEEP** | Bridges canonical grant to neutral invoke shape | DO NOT TOUCH |
| `governance_approval_evidence_adapter.py` (Nexus) | TIGAE ↔ DHA at invoke | No (internal) | **KEEP** | EE-internal ToolRuntime | DO NOT TOUCH semantics |
| `WorkerQualifiedCapabilityExecutionEngineAdapter` | AW→dispatch map | Propagates evidence | **REWIRE** | Remove evidence mapping | R6 |
| `QualifiedCapabilityExecutionDispatchService` | Root launch + dedup | Copies evidence to intake | **REWIRE** | Remove evidence branch | R6 |
| `QualifiedCapabilityExecutionRuntimeDelegate` | Handler under active id | Rebuilds dispatch with evidence | **REWIRE** | Stop evidence on rebuild | R6 |
| `CodeCraftQualifiedCapabilityExecutionHandler` | Handler | Passes request evidence to CodeCraft | **REWIRE** | Pass `None` / runtime-only grant | R6 |
| `worker_qualified_capability_resume_coordinator.py` | Orchestrates resume | Validates/forwards evidence | **REWIRE** | Remove pre-approval validation path | R6 |
| `worker_qualified_capability_resume_ports.py` | Port surface | If mirrors contracts | **REWIRE** | Align with contract removal | R6 |
| `test_uca6c_r5_r5_end_to_end_approval_evidence_propagation.py` | Proves AW→EE propagation | Encodes anti-pattern | **REMOVE/REWIRE** | Replace with canonical HITL E2E | R6 |
| `test_uca6c_r5_r2_strict_governance_composition.py` (approval fixtures) | Pre-load evidence | Drift | **REWIRE** | Test governance at ToolRuntime only | R6 |
| `execution_identity.py` / `identity_authority.py` | Identity core | Frozen | **DO NOT TOUCH** | Certified | — |
| `execution_continuation.py` / continuation runtime | HITL pause/resume | Frozen | **DO NOT TOUCH** | Canonical | Reuse in R6 E2E |
| `declarative_hitl.py` / `governed_continuation_grant.py` | Grants | Frozen | **DO NOT TOUCH** | Canonical | — |
| `QualifiedCapabilityExecutionDispatchService` root launch | Canonical admission | No | **KEEP** | Correct EE ingress | — |
| `nexus_execution_bound_catalog_tool_invoker.py` | ToolRuntime impl | Internal | **KEEP** | Mandatory boundary | — |
| Capability acquisition / qualification packages | Coordinate / qualify | Ownership OK | **KEEP** | No lifecycle takeover observed | — |
| `RootExecutionLaunchRequest` / global intake contracts | Platform admission | Not UCA consumer bug | **DO NOT TOUCH** | §21 scope | — |

---

## 7. `governance_approval_evidence` — per-layer answer

| Location | Why layer would need future approval? | ARCH-R1 answer |
|----------|--------------------------------------|----------------|
| `WorkerQualifiedCapabilityResumeRequest` | Pre-approve code.exec before execution exists | **It should not** → REMOVE |
| `WorkerQualifiedCapabilityExecutionRequest` | Pass worker-owned approval into EE | **It should not** → REMOVE |
| `QualifiedCapabilityExecutionDispatchRequest` / `IntakePayload` | Start execution with tool grant | **It should not** → REMOVE |
| `CodeCraftBoundCapabilityExecutionRequest` | Correlate invoke after pause/resume | **Only in-execution** grant from HITL → REWIRE |
| `ExecutionBoundCatalogToolInvokeRequest` | ToolRuntime enforcement | **KEEP** when grant produced by canonical HITL at invoke |

---

## 8. ExecutionId audit (UCA/AW path)

| Symbol | Direction | Verdict |
|--------|-----------|---------|
| `WorkerQualifiedCapabilityResumeRequest.execution_id` | Consumer ingress | **REMOVED** at `62c533a3c` — do not reintroduce |
| `WorkerQualifiedCapabilityExecutionRequest.execution_id` | Consumer ingress | **REMOVED** at `62c533a3c` |
| `QualifiedCapabilityExecutionDispatchRequest.execution_id` | Consumer ingress | **REMOVED** at `62c533a3c` |
| `WorkerQualifiedCapabilityExecutionResult.execution_id` | EE → AW outcome | **KEEP** (platform minted) |
| `QualifiedCapabilityExecutionDispatchResult.execution_id` | EE outcome | **KEEP** |
| `CodeCraftBoundCapabilityExecutionRequest.execution_id` | Handler supplies active id | **KEEP** |
| `QualifiedCapabilityExecutionRuntimeDelegate` | `peek_active_execution_id()` | **KEEP** |

---

## 9. Nexus matrix

| Layer | Nexus dependency |
|-------|-----------------:|
| Public contracts | NO |
| UCA / AW production | NO |
| CodeCraft public surface | NO |
| Marketplace | NO |
| Tools public API | NO |
| Execution Engine internals | INTERNAL ONLY |

Tests may import Nexus tool invoker fixtures; permanent gates should forbid **production** UCA/AW/CodeCraft public imports of Nexus.

---

## 10. R6 exact candidate scope (files)

**Contracts**

- `intergrax/contracts/autonomous_work/worker_qualified_capability_resume.py`
- `intergrax/contracts/execution/qualified_capability_execution_dispatch.py`
- `intergrax/contracts/execution/qualified_capability_execution_intake.py`
- `intergrax/contracts/codecraft/bound_capability_execution.py` (rewire `governance_approval_evidence` semantics only)

**Runtime / AW**

- `intergrax/autonomous_work/worker_qualified_capability_resume_coordinator.py`
- `intergrax/autonomous_work/worker_qualified_capability_resume_ports.py` (if applicable)
- `intergrax/runtime/execution/worker_qualified_capability_execution_adapter.py`
- `intergrax/runtime/execution/qualified_capability_execution_dispatch_service.py`
- `intergrax/runtime/execution/qualified_capability_execution_runtime_delegate.py`
- `intergrax/runtime/codecraft/qualified_capability_execution_handler.py`
- `intergrax/runtime/codecraft/wiring_bound_capability_execution.py` (source of evidence only)

**Tests (replace drift tests)**

- `tests/unit/autonomous_work/test_uca6c_r5_r5_end_to_end_approval_evidence_propagation.py`
- `tests/unit/autonomous_work/test_uca6c_r5_r2_strict_governance_composition.py`
- `tests/unit/autonomous_work/test_uca6c_r5_r4_execution_bound_approval_evidence.py`
- Related fixtures: `uca6c_r5_*`, `uca6c_r5_r2_strict_fixtures.py`

**Optional follow-up (not blocking R6 if invoke still needs carrier)**

- Re-evaluate `intergrax/contracts/tool_invocation_governance_approval_evidence.py` after AW fields removed — expect **KEEP** for ToolRuntime.

---

## 11. R6 forbidden scope (frozen families)

- `intergrax/contracts/execution_identity.py`
- `intergrax/contracts/execution_identity_authority.py`
- `intergrax/runtime/execution/identity_authority.py`
- `intergrax/contracts/execution_continuation.py`
- `intergrax/contracts/governed_continuation_grant.py`
- `intergrax/contracts/declarative_hitl.py`
- `intergrax/runtime/execution/continuation/**`
- `intergrax/runtime/human/governed_continuation_bridge.py`
- `intergrax/runtime/human/governed_continuation_grant.py`
- `intergrax/runtime/human/declarative_hitl_grant.py`
- `intergrax/runtime/policy/declarative_enforcer.py`
- `intergrax/runtime/policy/mse_hitl_effect_gate.py`
- Execution Engine core lifecycle / root admission (read-only integration only)
- New public Nexus adapters
- New approval / continuation / identity authorities

---

## 12. R6 test plan (design only)

### Primary E2E (§35)

```text
Worker recovery success
  → qualified + bound capability
  → canonical Execution starts WITHOUT preloaded tool approval
  → code.exec reaches ToolRuntime
  → Governance REQUIRE_HUMAN
  → canonical HITL pauses same Execution
  → human approval
  → governance re-evaluation
  → execution resumes
  → exact invocation succeeds
```

If this cannot run without changing frozen HITL core → **ARCHITECTURAL DECISION REQUIRED**. Current evidence: canonical contracts exist; R6 should wire tests through existing `ExecutionContinuationPort` + declarative HITL, not new paths.

### Negative tests (§36)

1. No approval → WAIT/HITL, not bypass  
2. DENY → blocked, no acquisition retry  
3. Wrong HITL scope → fail closed  
4. Stale approval → fail closed  
5. Wrong tenant → fail closed  
6. Wrong ExecutionId → fail closed  
7. Wrong `invocation_scope_id` → fail closed  
8. MSE DENY after human approval → blocked  
9. No gap reclassification after governance deny  
10. No rediscovery  
11. No reacquisition  
12. No requalification  

### Identity tests (§37)

- Worker does not mint/provide root `ExecutionId` on ingress  
- Execution Engine owns root admission  
- HITL binds current already-admitted Execution  

### Permanent architecture gates (§38) — future CI

- UCA core must not import Nexus  
- AW must not import Nexus  
- CodeCraft public boundary must not import Nexus  
- UCA must not import execution identity **mint** helpers  
- UCA/AW must not own `ExecutionContinuationService`  
- UCA/AW must not create approval grants  
- UCA must not mutate ToolRegistry / AgentRegistry  
- CapabilityQualification must not import core/provider qualification implementations  
- CapabilityQualification must not authorize execution  
- CodeCraft UCA execution must cross ToolRuntime  

---

## 13. Documentation plan

**Governed Capability Fulfillment META_ARCHITECTURE (`GOVERNED_CAPABILITY_FULFILLMENT.md`):** **YES** — after R6 lands, freeze cross-domain ownership using §4 matrix and GCF-INV-*; this reconciliation doc is the precursor, not the long-term hub.

**Terminology:** `WorkerQualifiedCapabilityResume` risks confusion with execution resume → add clarification in AUTONOMOUS_WORK satellite / this meta doc; **no rename** in R6 unless operator accepts migration cost.

**Doc drift (out of UCA scope):** pause/resume **same ExecutionId** vs resume segment **fresh root ExecutionId** in architecture docs (`RELIABILITY_FAILURE_AND_HITL.md`, `DECISION_APPROVAL_GOVERNANCE.md`) — **DO NOT MODIFY** in R6.

---

## 14. Unresolved decisions

| Item | Status |
|------|--------|
| Delete `ToolInvocationGovernanceApprovalEvidence` entirely vs keep invoke-only carrier | Favor **KEEP** invoke carrier until ToolRuntime uses DHA grant directly everywhere |
| Whether `derive_qualified_capability_governance_step_id` remains for non-HITL correlation | **REMOVE** with pre-approval; step_id at invoke should come from execution step context |
| Full E2E harness without touching frozen EE tests | Validate in R6; no ADR required if removal-only |

**ARCHITECTURAL DECISION REQUIRED?** **NO** for scoped R6 — correction is removal, rewiring existing public contracts, and test replacement without changing frozen owners.

---

## 15. Go / no-go for R6

| Gate | Result |
|------|--------|
| R6 achievable by removal + rewire + tests only | **YES** |
| Frozen identity / HITL / EE semantics change required | **NO** |
| New approval authority required | **NO** |

**ARCH-R1 go/no-go:** **R6 READY** (conditional on surgical scope above).

---

## 16. Known out-of-scope drift

- Frozen identity documentation inconsistency (same vs fresh ExecutionId on resume segment)  
- Legacy CodeCraft direct execution paths unrelated to UCA primary path  
- Distributed exactly-once, durable dedup, host consolidation  

---

## 17. Session goal check (Q)

| Question | Answer |
|----------|--------|
| One canonical capability fulfillment flow? | **YES** (target restored by R6) |
| Second execution / HITL / governance / identity / lifecycle / discovery / tool-runtime authority introduced? | **YES** (temporary drift via AW pre-approval — to be removed in R6) |
| Contract-first and provider-neutral? | **YES** |

Drift is **identified and scoped for removal**; not expanded in ARCH-R1.

---

**UCA-6C-ARCH-R1 — Canonical HITL Boundary & Surgical Reconciliation Plan**
