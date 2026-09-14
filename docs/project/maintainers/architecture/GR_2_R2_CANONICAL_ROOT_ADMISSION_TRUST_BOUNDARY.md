# GR-2-R2 — Canonical Root Admission Trust Boundary Architecture

**Status:** GR-2-R2-R1 corrected architecture (no runtime implementation)  
**Audit HEAD (design):** `722143bc37ff4459e126e9118c2e9596fdbbe9db` (`origin/development` baseline for R1)  
**Verdict:** `ARCHITECTURE_APPROVAL_RECOMMENDED` (candidate — independent audit required before GR-2-R3)  
**Supersedes decision gap:** GR-2-R1 `ARCHITECTURAL_DECISION_REQUIRED` / GOV-GAP-013  
**Implementation:** GR-2-R3 (next after independent audit of this document)

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
| Admission-provenanced root authority (`ParentExecutionAuthority` on legal path) | **Only** via `RootExecutionAuthorityAdmissionService` after policy ALLOW on certified launcher path (value object remains constructible — see §10) |
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

### Option B — Mandatory neutral root-admission hook inside `ExecutionRuntime.execute`

Conceptual flow:

```text
Application / composition
    → ExecutionRuntime.execute(...)
        → mandatory neutral RootGovernanceAdmissionPort (contract)
        → policy via RuntimeExecutionPolicyAdmissionPort (composed implementation)
        → ALLOW / block
        → execution body (identity mint, boundary, strategies)
```

**Bypass semantics (accurate):** If the hook is **mandatory** and **non-skippable** for every root `execute`, then **direct** `ExecutionRuntime.execute(...)` is **not** an admission bypass — the same code path runs admission before the execution body. Python visibility does not matter; runtime enforcement closes the bypass for any caller that reaches root `execute`.

**No-bypass strength (runtime-enforced):** **HIGH** — comparable to mandatory intake enforcement inside the engine. Disadvantages are **architectural**, not “callers can skip admission by calling runtime directly.”

**Rejected (valid trade-offs only):**

| Reason | Impact |
| --- | --- |
| Reopens frozen `ExecutionRuntime.execute` startup semantics | **HIGH** blast radius on Execution Engine |
| Root admission becomes engine lifecycle concern | Stronger coupling between authorization and execution identity mint |
| New mandatory neutral contract **inside** frozen engine | Contract + composition changes in `runtime.py` domain |
| Harder separation from `ExecutionAdmissionHook` | Validation vs Governance policy risk unless ports are strictly distinct |
| HITL / REQUIRE_HUMAN at root | Admission disposition must align with frozen resume/continuation (GR-5) at engine boundary |
| Direct internal child/root semantics | Child starts and engine-internal paths must not accidentally re-enter full root policy |
| Plugin failure domain | Missing/timeout/invalid policy → **ExecutionRuntime root start failure** (startup domain) |
| Single outer trust boundary story | Weaker than one explicit public launcher + demoted engine API for enterprise auditability |

**Not a valid rejection reason:** “Direct `ExecutionRuntime.execute` remains bypass” — **false** when admission is mandatory inside `execute`.

### Option C — Sealed root launcher only (refinement of A)

Same as A, but one named contract `RootExecutionLaunchPort` wraps admission + intake so host, AW, and future inference-only hosts share one entry. AW `WorkerExecutionDispatchService` delegates to launcher after AW-local collaborative gates.

**Selected:** **Option C** (Option A + unified launcher contract). **No tie.**

### 6.1 Primary security enforcement model (Option C only)

Option C does **not** rely on Python making `ExecutionRuntime` uncallable. Internal APIs remain **technically callable**; production legality is enforced separately.

**Chosen model: MODEL C1 — STRUCTURAL REPOSITORY ENFORCEMENT (architecture gates)**

```text
RootExecutionLaunchPort = only PUBLIC LEGAL PRODUCTION root start contract

ExecutionRuntime.execute / Execution facade (root) / direct intake construction =
INTERNAL CALLABLE APIs (illegal for production root starts outside allowlist)

Mandatory CI architecture gates = security enforcement (not documentation advice)
```

> **Root admission security is enforced by platform API boundaries plus mandatory static architecture gates.**

A root execution bypass must be **detectable deterministically before merge/deployment**. P0 bypass closure is **MANDATORY**; there is no “optional hardening if tests are insufficient.”

**Not selected for Option C:** MODEL C2 (runtime trust proof / unforgeable admission artifact) — provenance-by-path plus gates is sufficient without new cryptographic proof types; intake already requires launcher-composed trusted authority flow (see §10).

**Trust assumption:** Platform production source in `intergrax/**`, `agents/**`, `applications/**`, `platform_proofs/**` is trusted repository code. Untrusted extensions interact only through admitted plugin contracts (`RuntimeExecutionPolicyAdmissionPort`). Arbitrary malicious Python outside the repository is out of scope unless plugin sandboxing applies.

**If a new production module imports and calls `ExecutionRuntime.execute` for a root start:** the **import/invocation architecture gate fails**, merge is **blocked**, module is **outside the production allowlist** (unless explicitly allowlisted, e.g. engine intake adapter).

---

## 7. Comparison matrix

| Criterion | Option B (runtime hook) | Option C (launcher + gates) |
| --- | ---: | ---: |
| Runtime-enforced no bypass | **HIGH** (mandatory hook) | N/A (not runtime-enforced for internal APIs) |
| CI architecture-enforced no bypass | Optional (hook may reduce need) | **MANDATORY** |
| Frozen engine change | **HIGH** | **LOW** |
| Governance / runtime coupling | Higher (admission in `execute`) | Lower (admission before intake) |
| Layering | Valid if **neutral port only** (no Governance impl in engine) | **Clean** (engine unaware of Governance impl) |
| Pluginability (`RuntimeExecutionPolicyAdmissionPort`) | **High** | **High** |
| Failure-domain coupling | Higher (engine start fails on policy) | Lower (launcher/admission fails before engine) |
| Migration complexity | **High** | Medium |
| Security enforceability | **High** (runtime) | **High** (if gates mandatory) |
| Enterprise suitability | Medium (engine reopen + audit story) | **High** |
| Single public legal root API | Weak (runtime remains “the” entry) | **Strong** (`RootExecutionLaunchPort` only) |

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

### 8.3 Why Option C (after corrected Option B analysis)

- **Bypass closure:** Option B already achieves **high** runtime no-bypass; Option C achieves **high** enforceability via **mandatory** architecture gates + one **public legal** entry without reopening frozen `ExecutionRuntime.execute`.
- **Enterprise auditability:** Explicit `RootExecutionLaunchPort` vs demoted internal engine APIs; separation of Governance admission from engine validation hooks.
- **Layering:** Governance implements ports; engine consumes admission-provenanced authority at intake only; no concrete Governance in `ExecutionRuntime`.
- **Pluginability:** `RuntimeExecutionPolicyAdmissionPort` remains swap-in; selection at composition/bootstrap (existing plugin infrastructure).
- **Failure domain:** Policy/plugin failures block at launcher/admission, not inside engine root startup.
- **Migration:** Reuse `RootExecutionAuthorityAdmissionService` + `CanonicalExecutionRuntimeAdapter`; generalize admission request for non-AW principals; retire parallel “host mints authority” step.
- **Frozen impact:** **Lower** than Option B (no mandatory neutral admission port inside `runtime.py`).

### 8.4 Default launcher ownership (GR-2-R3)

| Property | Decision |
| --- | --- |
| **Contract** | `RootExecutionLaunchPort` in `intergrax/contracts/` |
| **Default implementation** | `DefaultRootExecutionLauncher` in `intergrax/runtime/governance/` (thin orchestrator — **no** policy engine, retry, HITL runtime, or evidence store) |
| **Dependencies** | **Only** `RootExecutionAuthorityAdmissionPort` + `CanonicalExecutionIntakePort` (ports, not concrete services) |
| **Not allowed** | Launcher executing business work; god-object lifecycle ownership |

Application/bootstrap wires port implementations; launcher does not import Nexus or application packages.

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

## 10. Trust artifact model (anti-forgery)

### 10.1 Value object vs trusted production provenance

| Concept | Meaning |
| --- | --- |
| `ParentExecutionAuthority` | Platform **value object** (scopes, narrowing) — **constructible** in Python today (`ParentExecutionAuthority.scoped(...)`, etc.) |
| **Trusted production root authority** | Authority whose **provenance** is certified: produced only on the launcher → `RootExecutionAuthorityAdmissionPort` → ALLOW path and passed to intake by the launcher |

Do **not** claim “mint monopoly,” “cryptographically trusted,” or “unforgeable” for the value type alone. Use **admission-provenanced authority** for trust on the legal production path.

### 10.2 Trust model: provenance-by-path (selected)

Authority is **trusted for production root start** only when created and forwarded on the certified path:

```text
RootExecutionLaunchPort
    → RootExecutionAuthorityAdmissionPort.authorize (ALLOW)
    → CanonicalExecutionIntakePort.dispatch (launcher-built request)
```

**Provenance-by-artifact** (separate `RootExecutionAdmissionProof` verified inside runtime) is **not** required for Option C when MODEL C1 gates are **mandatory**.

### 10.3 Anti-forgery rules (MANDATORY for R3)

1. **Legal API:** Production code may start root Execution **only** through `RootExecutionLaunchPort` (not by passing caller-built authority into `Execution` / `ExecutionRuntime` / intake).
2. **Admission service:** Only `RootExecutionAuthorityAdmissionService` (via port) produces `RootExecutionAuthorityAdmissionResult.trusted_parent_execution_authority` after policy ALLOW — this is the **certified mint on the legal path**, not a claim that the value type is unconstructible elsewhere.
3. **Intake coupling:** `CanonicalExecutionIntakePort` remains **mandatory internal engine intake** (trusted authority → runtime). Production modules outside the allowlist **must not** build `CanonicalExecutionIntakeRequest` for root starts.
4. **Authority gate:** Production code outside allowlist **must not** use `resolve_root_parent_execution_authority(task.execution_authority)` (or equivalent) to supply root `ParentExecutionAuthority` for a production root start.
5. **Architecture gates:** Any forbidden import/construction/invocation is a **deterministic CI failure** (§20).

**Forgery question (target state):** Can production code construct `ParentExecutionAuthority`? **YES** (type is constructible). Can it **legally** start production root Execution with a self-built authority? **NO** — no legal API accepts it; direct internal calls are **gate violations**.

**Do not** pluginize `TaskId` / `RunId` / `AttemptId` / `ExecutionId` (Execution Engine remains identity owner).

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

### 14.1 Terminology (architecture-defined; not Python visibility)

| Term | Definition |
| --- | --- |
| **PUBLIC LEGAL API** | The **only** production-approved contracts for the concern; CI gates enforce who may use them |
| **INTERNAL CALLABLE API** | Types/methods remain importable/callable in Python but **illegal** for production root starts outside allowlist |
| **TEST-ONLY API** | Direct engine/facade/intake access allowed under `tests/**`, `testing_support/**`, and explicit gate allowlists |
| **COMPOSITION-ONLY API** | Wired only at platform bootstrap / certified adapters (intake adapter, launcher default impl, harness allowlists) |

### 14.2 Classifications (post GR-2-R3)

| Surface | Classification |
| --- | --- |
| `RootExecutionLaunchPort` | **THE ONLY PUBLIC LEGAL PRODUCTION ROOT START CONTRACT** |
| `CanonicalExecutionIntakePort` | **Mandatory internal engine intake** (legal only as launcher downstream; not a second public root entry) |
| `HostTaskExecutionPort` / `HostTaskExecution` | **HOST ADAPTER** — delegates to `RootExecutionLaunchPort`; **must not** independently mint root authority |
| `WorkerExecutionDispatchService` | AW pre-admission / collaborative evidence → **generic `RootExecutionLaunchPort`**; no parallel canonical root start |
| `Execution` facade (root start) | **INTERNAL / ENGINE COMPOSITION FACADE** for root starts |
| `ExecutionRuntime.execute` (root) | **INTERNAL ENGINE API** — root starts only via `CanonicalExecutionRuntimeAdapter` in production allowlist |

**LEGAL ROOT ENTRY (post R3):** `RootExecutionLaunchPort.launch(...)` (exact request type defined in R3-1).

### 14.3 PUBLIC API control (MANDATORY in R3)

- Narrow `intergrax.runtime.execution` package exports; document `__all__`
- **Mandatory** architecture gates (§20) on `intergrax/**`, `agents/**`, `applications/**`, `platform_proofs/**`
- Prefer **AST / import dependency** checks over brittle substring grep where feasible
- Launcher contract remains in `intergrax/contracts/` (contract-first)

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
| `ParentExecutionAuthority` from admission ALLOW on launcher path | **ADMISSION-PROVENANCED AUTHORITY** |
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

**`WORKER_ROOT_EXECUTION_OPERATION` replacement:** Introduce platform-owned **`RootExecutionOperation`** (or equivalent contract enum) for **authorization operation identity** — distinct from `ExecutionCapability` in `execution_request.py`, which describes **semantic work requirements** (agent/tools/orchestration/streaming), not policy operation keys. Stable values e.g. `root.execution.inference`, `root.execution.agent`, `root.execution.orchestration` in `runtime_execution_policy_admission.py` or `root_execution_operation.py`. Policy rules match on `execution_operation`; launcher maps strategy/host context → operation; admission service **must not** hardcode worker-only default.

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

### 20.1 Mandatory architecture gates (GR-2-R3 — security enforcement)

Gate ownership: **`tests/unit/runtime/architecture/`** (or sibling qualification layer) — platform certification, not runtime.

| Gate type | Enforces |
| --- | --- |
| **Import gate** | Who may import `ExecutionRuntime`, root `Execution` facade paths, internal intake types |
| **Construction gate** | Who may construct `RootExecutionOptions`, `CanonicalExecutionIntakeRequest` for production root |
| **Authority gate** | Who may call `resolve_root_parent_execution_authority` / attach root `ParentExecutionAuthority` for production start |
| **Invocation gate** | Who may call `ExecutionRuntime.execute` / facade root `execute` for production root |

**Illustrative test modules (implement in R3):**

- `test_no_production_direct_execution_runtime_root_calls.py`
- `test_no_production_execution_facade_root_calls.py`
- `test_only_launcher_builds_root_intake.py`
- `test_only_admission_service_mints_trusted_root_authority_on_legal_path.py`

### 20.2 Production allowlist (design)

Modules/categories **allowed** to invoke internal root engine APIs or build root intake (non-exhaustive; refine in R3 against repo topology):

| Allowlisted role | Examples (repository paths) |
| --- | --- |
| Intake → runtime bridge | `intergrax/runtime/execution/canonical_intake_adapter.py` |
| Engine internal root wiring | `intergrax/runtime/execution/runtime.py`, `facade.py` (internal composition only) |
| Certified qualification / architecture tests | `tests/**`, `testing_support/**` (scoped fixtures) |
| Platform proofs (explicit subpaths only) | `platform_proofs/**` only where scenario manifest declares engine-direct shim (not general feature code) |

**Explicitly forbidden** for production root start via internal APIs:

- `applications/**` (including `_shared` harness after migration — harness must use launcher)
- `agents/**`
- General feature / business modules in `intergrax/**` outside engine + launcher + intake adapter
- Any module constructing root intake or root authority without going through `RootExecutionLaunchPort`

### 20.3 Option C acceptance conditions (all MANDATORY)

1. Only `RootExecutionLaunchPort` is legal public root entry  
2. Direct runtime/facade root usage classified internal  
3. CI static architecture gates prohibit forbidden production imports/calls  
4. Host migrates to launcher  
5. AW migrates to launcher  
6. No second authority mint path on legal production flow  
7. INFERENCE / AGENTIC / ORCHESTRATION share same legal entry  
8. Policy missing / exception / timeout / invalid output / incompatible version → **fail closed**  
9. Authority widening impossible (existing admission invariants)  
10. Architectural gates part of **mandatory** platform qualification (not optional)

**Invariant:** A root execution bypass must be detectable deterministically before merge/deployment.

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
| R3-5 | Public/internal API enforcement + **mandatory** architecture gates (import/construction/authority/invocation) |
| R3-6 | Inference qualification scenarios |
| R3-7 | Agentic qualification scenarios |
| R3-8 | Orchestration / AW qualification scenarios |
| R3-9 | Ledger update GOV-GAP-013 closure evidence |

**File budget (estimate):** contracts ≤4, runtime/governance ≤4, host/AW ≤3, apps wiring ≤2, tests ≤8.

---

## 24. Security threat model

| Threat | Mitigation |
| --- | --- |
| Caller bypasses Governance | **MANDATORY** `RootExecutionLaunchPort`; internal facade/runtime; **MANDATORY** CI gates |
| Forged `ParentExecutionAuthority` | Provenance-by-path; no legal API accepts caller-built root authority; gates block direct intake/runtime |
| Stale admission across new attempt | Re-admit on new `AttemptId` (§17) |
| Scope widening | Existing narrowing in admission service |
| Plugin missing / exception / timeout / invalid output / incompatible version | **FAIL CLOSED** → UNAVAILABLE/DENY; adapter timeouts (GR-11) |
| Malicious plugin | Typed `RuntimeExecutionPolicyAdmissionPort` result only; no raw runtime authority to plugin; provenance + deny |
| Direct runtime/facade call in production | **Deterministic architecture gate failure**; merge blocked |
| Strategy-specific bypass | Single launcher; `RootExecutionOperation` per strategy (not parallel entry points) |
| Resume confused with new root | Lifecycle rules in §17 |
| Reflection / dynamic calls in repo code | Same static gates on import/call patterns in production trees; malicious arbitrary Python out of scope (§6.1) |

**Plugin threat model:** External policy plugin never receives `ExecutionRuntime` or mint authority; it returns typed policy disposition through `RuntimeExecutionPolicyAdmissionPort` only.

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

**`ARCHITECTURE_APPROVAL_RECOMMENDED`** — GR-2-R2-R1 corrections applied (Option B bypass accuracy, MODEL C1 enforcement, anti-forgery nomenclature, mandatory gates). **Independent audit required** before GR-2-R3 implementation.

**GR-2-R2-R1 status:** **DONE** (architecture correction only).

---

*GR-2-R2 design artifact — independent re-audit required before implementation closure.*
