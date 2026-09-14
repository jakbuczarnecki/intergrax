# GR-2-R2 — Canonical Root Admission Trust Boundary Architecture

**Status:** Architecture design (no runtime implementation)  
**Audit HEAD (design):** `e7d08f846c9a4a3b3f63439f6fad7cb0e1e79a7a` (`development`)  
**Verdict:** `ARCHITECTURE_APPROVAL_RECOMMENDED`  
**Supersedes decision gap:** GR-2-R1 `ARCHITECTURAL_DECISION_REQUIRED` / GOV-GAP-013  
**Implementation:** GR-2-R3 (blocked until operator approves this design)

---

## 1. Context

Intergrax requires one **mandatory, strategy-neutral, contract-first** trust boundary so that **INFERENCE**, **AGENTIC**, and **ORCHESTRATION** root Executions cannot start without Governance admission, while preserving:

- frozen Execution Engine ownership of lifecycle and identity;
- tier boundaries (`intergrax/` must not import `agents/` or `applications/`);
- replaceable Governance/policy implementations via platform contracts;
- separate child authority (`ExecutionAuthorityPolicy`), capacity (`ExecutionCapacityAdmissionPort`), and HITL (GR-5).

GR-2-R1 proved code truth: the **canonical host path** reaches `ExecutionRuntime` without `RootExecutionAuthorityAdmissionPort`, `RuntimeExecutionPolicyAdmissionPort`, or `CanonicalExecutionIntakePort`.

---

## 2. Problem

**GOV-GAP-013 (P0):** Root admission coverage exists only on the Autonomous Work dispatch seam. The primary application/host path mints effective root authority from **untrusted** `task.execution_authority` via `resolve_root_parent_execution_authority`, then calls `Execution` → `ExecutionRuntime.execute`.

**Design principle (target):**

```text
No Governance admission → No trusted root authority → No legal root Execution
```

**Non-goal:** Reopen frozen Execution Engine semantics beyond the minimum contracts required to close bypass (GR-2-R3 scope).

---

## 3. Existing topology

### 3.1 Layer ownership (current)

```text
Application / Host composition
    │ owns wiring, Task, harness
    ▼
HostTaskExecutionPort / HostTaskExecution          [runtime/execution/host_task.py]
    │ builds StrategyExecutionRouter (INFERENCE | AGENT | ORCHESTRATION)
    │ resolve_root_parent_execution_authority(task.execution_authority)  ← UNTRUSTED
    ▼
Execution (facade)                                   [runtime/execution/facade.py]
    │ RootExecutionOptions.authority
    ▼
ExecutionRuntime.execute(request, root_context)      [runtime/execution/runtime.py]
    │ capacity: ExecutionCapacityAdmissionPort (optional)
    │ hooks: ExecutionAdmissionHook (validation, lineage)
    ▼
ExecutionBoundary → StrategyExecutionRouter → executors

Parallel (AW only):
WorkerExecutionDispatchService                     [autonomous_work/worker_execution_dispatch.py]
    │ WorkerExecutionAdmissionService (collaborative AW evidence)
    ▼
RootExecutionAuthorityAdmissionPort.authorize      [contracts/runtime_execution_admission.py]
    │ → RootExecutionAuthorityAdmissionService
    │ → RuntimeExecutionPolicyAdmissionPort.evaluate (WORKER_ROOT_EXECUTION_OPERATION)
    ▼ mints ParentExecutionAuthority
CanonicalExecutionIntakePort.dispatch                [contracts/execution_intake.py]
    ▼
CanonicalExecutionRuntimeAdapter                     [runtime/execution/canonical_intake_adapter.py]
    ▼
ExecutionRuntime.execute
```

**Dependency direction (valid today):**

- Contracts: `execution_intake`, `runtime_execution_admission`, `runtime_execution_policy_admission`, `delegation_authority` — no implementation imports.
- Governance service implements admission ports; AW consumes ports; intake adapter consumes `ExecutionRuntime` only.
- Execution Engine does **not** import Governance concrete classes.

### 3.2 ASCII — Current (two root paths)

```text
Host / Harness / LKW
 └─> HostTaskExecution
      └─> Execution
           └─> ExecutionRuntime          (NO Governance admission)

Worker (AW)
 └─> WorkerExecutionDispatchService
      └─> RootExecutionAuthorityAdmissionPort
           └─> CanonicalExecutionIntakePort
                └─> ExecutionRuntime
```

---

## 4. Root bypass proof (code truth)

| Path | Mechanism | Classification |
| --- | --- | --- |
| `HostTaskExecution.execute` → `Execution.execute` → `ExecutionRuntime.execute` | `RootExecutionOptions(authority=resolve_root_parent_execution_authority(task.execution_authority))` | **PUBLIC LEGAL ENTRY** (production host); **bypasses Governance** |
| Direct `Execution(runtime).execute(..., options=...)` | Any composition root may construct facade + pass arbitrary `ParentExecutionAuthority` | **COMPOSITION-ONLY** today, but **public types** — treated as **latent PUBLIC LEGAL ENTRY** |
| Direct `ExecutionRuntime.execute(request, root_context)` | Bypasses facade; same authority injection | **INTERNAL ENGINE API** in intent, but **reachable from composition** — **latent bypass** |
| `CanonicalExecutionRuntimeAdapter.dispatch` | Requires `CanonicalExecutionIntakeRequest.trusted_parent_execution_authority` | **COMPOSITION-ONLY** intake; **legal only when preceded by admission** (AW path) |
| Tests / lab direct runtime wiring | Same as direct runtime | **TEST-ONLY** / **COMPOSITION-ONLY** (must not define production legality) |

**Frozen root definition (unchanged):** `ExecutionRuntime.execute` with `RootExecutionContext` from `resolve_root_execution_context` / `mint_root_execution_identity`; lineage root `parent_execution_id is None`.

---

## 5. Ownership model

| Concern | Owner |
| --- | --- |
| Root execution request (intent, Task, payload) | Application / host adapter |
| Root Governance decision (ALLOW/DENY/REQUIRE_HUMAN/ESCALATE/UNAVAILABLE) | Governance plane (`RuntimeExecutionPolicyAdmissionPort` + `RootExecutionAuthorityAdmissionPort`) |
| Policy implementation (enterprise rules) | Composed plugin / `RuntimeExecutionPolicyAdmissionPort` implementation |
| Trusted root authority (`ParentExecutionAuthority` mint) | **Only** `RootExecutionAuthorityAdmissionService` (Governance runtime), after policy ALLOW |
| Root admission sequencing (collaborative evidence → policy → mint → intake) | **Composition** (`RootExecutionLaunchPort` — new GR-2-R3 contract) |
| Execution lifecycle / identity | Execution Engine (`ExecutionRuntime`, identity minting) |
| Execution identity types | Platform contracts (frozen primitives) |
| Capacity | `ExecutionCapacityAdmissionPort` (orthogonal) |
| Child authority narrowing | `ExecutionAuthorityPolicy` (unchanged) |
| HITL pause/resume | GR-5 (Governance disposition may block admission; lifecycle not designed here) |
| Evidence emission | GR-8 (hook points at admission result; no authority change) |
| Composition (wiring ports, plugin selection) | Application / platform bootstrap |

---

## 6. Architecture alternatives

### Option A — Mandatory external canonical intake (recommended)

```text
All root callers
    → RootExecutionLaunchPort (composition contract)
        → RootExecutionAuthorityAdmissionPort
        → CanonicalExecutionIntakePort
            → ExecutionRuntime
```

`ExecutionRuntime` remains Governance-unaware. Bypass closure = **no production-legal API** that accepts caller-supplied root authority without prior admission.

### Option B — Frozen admission hook inside `ExecutionRuntime.execute`

```text
Application → ExecutionRuntime → RootGovernanceAdmissionPort (neutral contract) → execute body
```

**Rejected:** Duplicates intake seam; pushes Governance *invocation* into engine startup (even via neutral port, every caller must still hit runtime — direct runtime remains bypass); conflates with `ExecutionAdmissionHook` (validation) unless new port added — larger frozen surface with weaker “single outer trust boundary” story.

### Option C — Sealed root launcher only (refinement of A)

Same as A, but one named contract `RootExecutionLaunchPort` wraps admission + intake so host, AW, and future inference-only hosts share one entry. AW `WorkerExecutionDispatchService` delegates to launcher after AW-local collaborative gates.

**Selected:** **Option C** (Option A + unified launcher contract). **No tie.**

---

## 7. Comparison matrix

| Criterion | Option A | Option B | Option C |
| --- | ---: | ---: | ---: |
| No bypass | High (if public APIs demoted) | Medium (runtime entry still public) | **High** |
| Frozen engine impact | Low | **High** (runtime.execute) | **Low** |
| Layering correctness | **Yes** | Risky (engine invokes admission) | **Yes** |
| Contract-first | **Yes** | Yes | **Yes** |
| Pluginability | **Yes** | Yes | **Yes** |
| Runtime coupling | **None** to Governance impl | Neutral port in engine | **None** |
| Security strength | High | Medium | **High** |
| Migration complexity | Medium | High | Medium |
| Testability | High | High | **High** |
| Backward compatibility risk | Medium (host refactor) | High | Medium |
| Enterprise suitability | High | Medium | **High** |

---

## 8. Recommended architecture

### 8.1 Target diagram

```text
INFERENCE / AGENTIC / ORCHESTRATION callers
        │
        ▼
┌───────────────────────────────────────┐
│  LEGAL ROOT ENTRY: RootExecutionLaunchPort │  (contracts — composition)
└───────────────────────────────────────┘
        │ 1. build RootExecutionAuthorityAdmissionRequest (strategy-neutral)
        ▼
RootExecutionAuthorityAdmissionPort.authorize
        │ 2. RuntimeExecutionPolicyAdmissionPort.evaluate(execution_operation)
        ▼
 trusted ParentExecutionAuthority (minted, narrowed scopes)
        │ 3. CanonicalExecutionIntakeRequest
        ▼
CanonicalExecutionIntakePort.dispatch
        ▼
ExecutionRuntime.execute   (engine — no Governance impl)
        ▼
ExecutionBoundary → strategies
```

### 8.2 Roles after GR-2-R3

| Component | Final status |
| --- | --- |
| `RootExecutionLaunchPort` | **LEGAL ROOT EXECUTION ENTRY** (public contract) |
| `HostTaskExecutionPort` | **Mandatory host adapter** for Task-shaped roots; **must** use launcher, not direct `Execution` |
| `WorkerExecutionDispatchService` | AW orchestration shell; **reuses launcher** after AW collaborative admission (no duplicate mint path) |
| `CanonicalExecutionIntakePort` | **Mandatory legal intake into engine** (trusted authority → runtime); not AW-only |
| `Execution` facade | **INTERNAL / composition-only** for root starts (host migrates to launcher) |
| `ExecutionRuntime.execute` | **Internal engine API** — root starts only via intake adapter in production |
| `CanonicalExecutionRuntimeAdapter` | **Composition-only** bridge intake → runtime |
| `ExecutionAdmissionHook` | **EXECUTION VALIDATION** (lineage, hooks) — **not** Governance |
| `ExecutionCapacityAdmissionPort` | **CAPACITY** — orthogonal, may deny after admission |

### 8.3 Why Option C

- **Bypass:** Single outer port; host path aligned with AW; public self-serve authority injection removed from legal host flow.
- **Layering:** Governance implements ports; engine consumes trusted authority at intake only; no `ConcreteGovernance` in `ExecutionRuntime`.
- **Pluginability:** `RuntimeExecutionPolicyAdmissionPort` remains swap-in; selection at composition/bootstrap (existing plugin infrastructure).
- **Migration:** Reuse `RootExecutionAuthorityAdmissionService` + `CanonicalExecutionRuntimeAdapter`; generalize admission request for non-AW principals; retire parallel “host mints authority” step.
- **Frozen impact:** Smaller than Option B (no mandatory policy call inside `runtime.py`).

---

## 9. Contract model (GR-2-R3)

| Contract | Existing/New | Owner | Purpose | Caller | Provider | Breaking |
| --- | --- | --- | --- | --- | --- | --- |
| `RootExecutionLaunchPort` | **New** | Platform contracts | Single legal root start | Host, AW (post-collab), apps | Composition module | N/A (additive) |
| `RootExecutionAuthorityAdmissionPort` | Existing | Governance | Mint trusted root authority | Launcher | `RootExecutionAuthorityAdmissionService` | **Extend request** for strategy-neutral host/inference (non-breaking if additive fields) |
| `RuntimeExecutionPolicyAdmissionPort` | Existing | Governance | Policy ALLOW/DENY | Admission service | Plugin / default engine | **Non-breaking** — new `execution_operation` values |
| `CanonicalExecutionIntakePort` | Existing | Runtime intake | Trusted authority → runtime | Launcher | `CanonicalExecutionRuntimeAdapter` | **Non-breaking** |
| `Execution` / public `execute` | Existing | Runtime | Developer facade | — | Demote from legal root | **Breaking for direct callers** (internal re-export) |

**Fail-closed:** Launcher **must not** call intake unless admission disposition is `ALLOWED`. Missing `RuntimeExecutionPolicyAdmissionPort` implementation → `UNAVAILABLE` → no start (existing `Denying` / `Unavailable` adapters).

---

## 10. Trust artifact model

**`ParentExecutionAuthority` is sufficient for authorization semantics** (scopes, unrestricted flag, child narrowing) **but insufficient alone for anti-forgery** (any caller can construct `ParentExecutionAuthority.scoped(...)` today).

**GR-2-R3 anti-forgery (no new primitive identity types):**

1. **Structural:** Production-legal root path is **only** `RootExecutionLaunchPort` → admission mint → `CanonicalExecutionIntakePort`. Applications must not call `Execution.execute` with caller-built `RootExecutionOptions.authority`.
2. **Minting monopoly:** Only `RootExecutionAuthorityAdmissionService.authorize` may attach `trusted_parent_execution_authority` on `RootExecutionAuthorityAdmissionResult` (already enforced by result invariants).
3. **Intake coupling:** `CanonicalExecutionIntakeRequest` continues to require `ParentExecutionAuthority`; launcher is the only production composer of intake requests after admission.
4. **Optional R3 hardening (if architecture tests insufficient):** package-private factory module `intergrax.runtime.governance.trusted_root_authority` exporting mint helper used **only** by admission service — intake accepts authority only when paired with admission result object (`RootExecutionAuthorityAdmissionResult`) in launcher closure (no new lineage/identity artifact).

**Do not** pluginize `TaskId` / `RunId` / `AttemptId` / `ExecutionId`.

---

## 11. Layer dependency model

```text
Application
    ↓ implements/wires
RootExecutionLaunchPort (contract)
    ↓ calls
RootExecutionAuthorityAdmissionPort + RuntimeExecutionPolicyAdmissionPort (contracts)
    ↑ implemented by
Governance runtime (policy plugins)
    ↓ mints
ParentExecutionAuthority
    ↓ passed to
CanonicalExecutionIntakePort (contract)
    ↓ implemented by
CanonicalExecutionRuntimeAdapter
    ↓ calls
ExecutionRuntime (Execution Engine)
```

**Forbidden (unchanged):** Execution Engine → Governance implementation; Governance → Nexus internals for admission; contracts → runtime implementations.

---

## 12. Pluginability model

| Field | Value |
| --- | --- |
| **contract** | `RuntimeExecutionPolicyAdmissionPort`, `RootExecutionAuthorityAdmissionPort`, `RootExecutionLaunchPort` |
| **consumer** | Host launcher, AW dispatch (via launcher), applications composition root |
| **composition root** | Application harness / `applications/_shared` wiring + platform plugin bootstrap |
| **default implementation** | `RootExecutionAuthorityAdmissionService` + configured policy evaluator |
| **external implementation** | Enterprise policy plugin implementing `RuntimeExecutionPolicyAdmissionPort` |
| **selection** | Existing platform plugin manifest / composition (no new registry) |
| **plugin provenance** | Compatible with GR-11 identity, manifest, version, capability admission |
| **fail-closed** | Missing/unavailable policy → `UNAVAILABLE` / DENY; no implicit ALLOW |
| **vendor coupling** | None in core contracts |
| **global registry** | Forbidden |

---

## 13. Root strategy coverage

| Strategy | Entry | Same boundary? | Special case |
| --- | --- | ---: | ---: |
| INFERENCE | `HostTaskExecution` → launcher → router → `InferenceExecutor` | **YES** | None (no provider-side admission) |
| AGENTIC | Host → launcher → `AgentEnginePort` | **YES** | No Governance inside `AgentExecutor` |
| ORCHESTRATION | Host or AW → launcher | **YES** | AW retains **collaborative** `WorkerExecutionAdmissionService` as **evidence** before launcher |

---

## 14. Public API / legal entry model

**LEGAL ROOT ENTRY (post R3):** `RootExecutionLaunchPort.launch(...)` (exact request type defined in R3-1).

**INTERNAL / composition-only:**

- `Execution.execute` (root) — host and apps migrate off
- `ExecutionRuntime.execute` — intake adapter + controlled test helpers only
- `CanonicalExecutionRuntimeAdapter` — wired at composition

**PUBLIC API control (R3):**

- Narrow `intergrax.runtime.execution` package exports; document `__all__`
- Architecture tests: `applications/` and `agents/` must not import `Execution` or `ExecutionRuntime` for root starts
- Optional: `import-linter` layer rule — launcher contract lives in `intergrax/contracts/`

---

## 15. Bypass prevention (per component, post R3)

| Component | May initiate root after R3? |
| --- | --- |
| `HostTaskExecution` | **Yes**, only via injected `RootExecutionLaunchPort` |
| `Execution` | **No** (legal root) — internal to launcher/intake wiring |
| `ExecutionRuntime` | **No** (legal root) — intake only |
| `CanonicalExecutionRuntimeAdapter` | **Yes**, as intake implementor (not for apps to call directly) |

---

## 16. Identity / authority semantics

| Input | Classification |
| --- | --- |
| `task.execution_authority` | **UNTRUSTED REQUEST INPUT** (evidence of desired scopes; not minted trust) |
| Collaborative / AW authority decision | **EVIDENCE** |
| `PolicyDecision` / admission disposition | **EVIDENCE** |
| `ParentExecutionAuthority` from admission ALLOW | **TRUSTED AUTHORIZATION ARTIFACT** |
| `RootExecutionOptions` from application | **UNTRUSTED** unless produced inside launcher |
| Execution request payload | **EXECUTION INPUT** |

**Child executions:** `ExecutionAuthorityPolicy` narrows from active parent authority — **unchanged**.

**Lineage:** Root remains `parent_execution_id = None`; no fake parent for authorization.

---

## 17. Resume / recovery implications

| Case | Admission |
| --- | --- |
| Fresh root execution | **Full** root admission required |
| Same-execution HITL resume (GR-5 continuation) | **Not** a new root Execution — **no repeat** root Governance admission if frozen lifecycle defines continuation; launcher not invoked for resume-only path |
| New `AttemptId` (retry) | **Fresh** root admission (security-first default) unless GR-7/recovery contract proves reuse |
| Checkpoint resume with same execution identity | **Continuation** — admission preserved; interact with `RecoveryAdmissionPort` separately (not merged into Governance) |
| Recovery / external effects | GR-7 — **interaction only**; do not merge into root admission |

**Boundary:** `restore_existing_execution` / active execution resume plans must not be classified as new root without explicit identity mint semantics.

---

## 18. Migration plan

1. Introduce `RootExecutionLaunchPort` + strategy-neutral `RootExecutionAuthorityAdmissionRequest` fields (`execution_operation`, principal, optional collaborative evidence).
2. Implement default launcher: admission → intake (existing services).
3. Wire `HostTaskExecution` to launcher; remove `resolve_root_parent_execution_authority` from root options path (use admission mint only).
4. Refactor `WorkerExecutionDispatchService` to call launcher after AW collaborative gates (delete duplicate intake wiring logic where redundant).
5. Demote `Execution` root usage; update harness (`harness_host_runtime.py`), lab, compensation paths to launcher or documented test-only shims.
6. Add qualification tests (§20) and architecture gates.
7. Remove temporary dual-path milestone in same release train (no “no admission → old path” fallback).

**`WORKER_ROOT_EXECUTION_OPERATION` replacement:** Use stable operation IDs derived from **`ExecutionCapability`** / `ExecutionStrategy`, e.g. `root.execution.inference`, `root.execution.agent`, `root.execution.orchestration` (contract constants in `runtime_execution_policy_admission.py` or dedicated `root_execution_operation.py` contract module). Policy rules match on `execution_operation`; AW dispatch passes `root.execution.orchestration` (or agent) from request capabilities — **not** worker-specific string as hardcoded default in admission service.

---

## 19. Frozen boundary reopening scope (GR-2-R3)

| Path | Reason | Semantic change | Risk | Unavoidable |
| --- | --- | --- | --- | --- |
| `intergrax/runtime/execution/host_task.py` | Route root via launcher; stop untrusted authority | Host root trust | Host regression | **Yes** |
| `intergrax/contracts/runtime_execution_admission.py` | Strategy-neutral admission request | Add fields / operation dimension | AW compatibility | **Yes** |
| `intergrax/contracts/runtime_execution_policy_admission.py` | Root operation constants | New operation strings | Policy rules | **Yes** |
| `intergrax/contracts/execution_intake.py` | Optional tighten intake invariants | Document trusted provenance | Low | Maybe |
| `intergrax/runtime/governance/root_execution_authority_admission.py` | Accept `execution_operation` from request | Stop hardcoding worker op | Policy coverage | **Yes** |
| `intergrax/autonomous_work/worker_execution_dispatch.py` | Delegate to launcher | Dispatch flow | AW tests | **Yes** |
| `intergrax/applications/_shared/harness_host_runtime.py` | Wire launcher + ports | Composition | App harness | **Yes** |
| `intergrax/runtime/execution/facade.py` | Export / doc “not legal root” | API surface | Callers | **Yes** |
| `intergrax/runtime/execution/canonical_intake_adapter.py` | Unlikely logic change | — | Low | No |
| `intergrax/runtime/execution/runtime.py` | **Avoid** unless intake enforcement hook | Prefer no change | Freeze | **Prefer no** |

---

## 20. Test / qualification plan (GR-2-R3)

| Scenario | Expect |
| --- | --- |
| INFERENCE + ALLOW / DENY | Launcher + policy |
| AGENTIC + ALLOW / DENY | Same |
| ORCHESTRATION + ALLOW / DENY | Same |
| UNAVAILABLE policy | No start |
| Scope widening | DENY |
| Forged authority direct to intake | Rejected or unroutable (no launcher) |
| Direct `HostTaskExecution` bypass | Fail arch test after refactor |
| Direct `Execution` / `Runtime` bypass | Arch test / import gate |
| Same-execution resume | No second root admission |
| New attempt | Fresh admission |

**Architecture tests (proposed):**

- `applications/` may not reference `ExecutionRuntime.execute` for roots
- Execution engine modules must not import `intergrax.runtime.governance.*` implementations (except allowed adapter packages per layer rules)

---

## 21. Risks

| Rank | Risk |
| --- | --- |
| **P0** | Incomplete migration leaves hidden `Execution(...)` callers |
| **P1** | Sync `RuntimeExecutionPolicyAdmissionPort.evaluate()` blocks on slow external plugins — document timeouts at adapter boundary (GR-11) |
| **P2** | Dual admission during migration if AW not refactored to launcher |

---

## 22. Explicit non-goals

- GR-2-R3 implementation in this task
- HITL lifecycle (GR-5), DecisionId (GR-6), evidence plane (GR-8)
- Changing `ExecutionAuthorityPolicy` child model
- Nexus coupling for inference/agentic admission
- New god-service orchestrator

---

## 23. GR-2-R3 implementation sequence

| Step | Work |
| --- | --- |
| R3-1 | Contracts: `RootExecutionLaunchPort`, neutral admission request, root `execution_operation` constants |
| R3-2 | `DefaultRootExecutionLauncher` composition (admission + intake) |
| R3-3 | `HostTaskExecution` integration + harness wiring |
| R3-4 | `WorkerExecutionDispatchService` → launcher; remove duplicate semantics |
| R3-5 | Public API demotion + architecture tests (bypass closure) |
| R3-6 | Inference qualification scenarios |
| R3-7 | Agentic qualification scenarios |
| R3-8 | Orchestration / AW qualification scenarios |
| R3-9 | Ledger update GOV-GAP-013 closure evidence |

**File budget (estimate):** contracts ≤4, runtime/governance ≤4, host/AW ≤3, apps wiring ≤2, tests ≤8.

---

## 24. Security threat model

| Threat | Mitigation |
| --- | --- |
| Caller bypasses Governance | Mandatory `RootExecutionLaunchPort`; demote direct facade/runtime |
| Forged `ParentExecutionAuthority` | Mint monopoly + intake-only root runtime entry |
| Stale admission across new attempt | Re-admit on new `AttemptId` |
| Scope widening | Existing narrowing in admission service |
| Plugin missing/crash/timeout | FAIL CLOSED → UNAVAILABLE/DENY; adapter timeouts (GR-11) |
| Malicious plugin | Provenance + deny; no widen |
| Direct runtime/facade call | Architecture tests + non-export |
| Strategy-specific bypass | Single launcher; operation from `ExecutionCapability` |
| Resume confused with new root | Lifecycle rules in §17 |

---

## 25. `ExecutionAdmissionHook` classification

Remains **execution validation** (lineage activation, physical hooks). **Not** repurposed for Governance root admission (would blur capacity/validation/policy). Root Governance stays **before** `ExecutionRuntime` via intake.

---

## 26. Async / sync policy contract

`RuntimeExecutionPolicyAdmissionPort.evaluate()` is **synchronous** today. Acceptable for in-process plugins; **external network policy** should be wrapped in a **composition-layer adapter** (timeout, fail-closed) implementing the same port. Changing to async would require **contract reopening** — defer unless GR-11 proves insufficient.

---

## 27. Invariant table

| ID | Invariant |
| --- | --- |
| INV-1 | No root execution without trusted admission |
| INV-2 | Governance cannot execute work |
| INV-3 | Execution Engine does not import concrete Governance implementation |
| INV-4 | Policy implementation replaceable via port |
| INV-5 | Authority cannot widen vs collaborative/upstream evidence |
| INV-6 | Child authority remains `ExecutionAuthorityPolicy` |
| INV-7 | Capacity remains orthogonal |
| INV-8 | Same-execution resume is not accidentally a new root |
| INV-9 | Direct internal runtime API is not a legal root entry |
| INV-10 | No vendor dependency in core |

---

## 28. Architectural verdict

**`ARCHITECTURE_APPROVAL_RECOMMENDED`** — pending operator acceptance before GR-2-R3.

---

*GR-2-R2 design artifact — independent re-audit required before implementation closure.*
