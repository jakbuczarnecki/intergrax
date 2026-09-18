# ADR-GOVERNED-EXECUTION-003: Root governance identity authority boundary

| Field | Value |
|-------|-------|
| **Status** | Proposed (design complete — OBS-DIAG-RECERT-P2C-R0A-ADR1) |
| **Date** | 2026-09-18 |
| **Design SHA** | `ba0e37f8b07f4e96a55749cf1c302b2c5da752b4` |
| **Trigger** | Commit `0d8fc977c8287f4ce6159f1a8ae6d8783ee27164` (DG-001 P3 functional restore; trust-boundary defect) |
| **Related** | [ADR-GOVERNED-EXECUTION-001](entries/2026-08-16/ADR-GOVERNED-EXECUTION-001.md) · [ADR-GOVERNED-EXECUTION-002](entries/2026-08-17/ADR-GOVERNED-EXECUTION-002.md) · [ADR-GR-10-001](entries/2026-09-18/ADR-GR-10-001.md) · [`GOVERNED_EXECUTION.md`](../../../architecture/GOVERNED_EXECUTION.md) |

## Context

Governed root execution binds `ActiveExecutionGovernanceIdentity` (runtime `ContextVar`) for PRE_MODEL policy and evidence (GR-10). After `0d8fc977`, production root paths populate `RootExecutionContext.tenant_id` / `workspace_id` / `principal_id` from `Task` fields via `host_workspace_id(task)` and `host_principal_id(task)` in `intergrax/runtime/execution/host_root_launch_evidence.py`, whose module docstring states these are **host upstream authority evidence (not trusted authority)**. `ExecutionRuntime.execute` (`intergrax/runtime/execution/runtime.py`) promotes all three non-`None` loose fields into `ActiveExecutionGovernanceIdentity` without an admission-bound identity contract.

**Invariant:** untrusted input must never become governance authority without an explicit trust / admission boundary.

**Ownership:** Execution owns lifecycle binding; Governance owns authority semantics; Host/Admission owns authenticated upstream input; Observability owns facts; Diagnostics owns interpretation.

## Problem

### Current unsafe flow (design SHA)

```text
Task (payload)
  → host_workspace_id / host_principal_id  [evidence helpers; fallback metadata→tenant→"default", user→tenant→"anonymous"]
  → RootExecutionContext (loose tenant_id / workspace_id / principal_id)
  → ExecutionRuntime.execute
  → bind_active_execution_governance_identity(ActiveExecutionGovernanceIdentity)
  → PRE_MODEL Governance (require / validate projection)
```

Parallel harness path (`execute_root_task` in `orchestration.py`) uses the same helpers and `task.tenant_id` for tenant — **not** a host admission product.

UAEP path (`_runtime_request_identity` in `uaep_step_bridge.py`) can synthesize `RequestIdentity(tenant_id=…"default")` when `canonical_identity` is absent; `resolve_agentic_pre_model_scope` (`pre_model_principal.py`) may mint a **new** `ActiveExecutionGovernanceIdentity` from request projection when no active identity exists and `production_mode` is false.

### Authority escalation risk

Any caller that can set `Task.metadata["workspace_id"]`, `Task.user_id`, or `Task.tenant_id` can steer PRE_MODEL subject identity without passing host authentication, violating enterprise fail-closed expectations and contradicting `host_root_launch_evidence` semantics.

### Fallback matrix (current → target)

| Missing value | Current behavior (`host_*` / UAEP) | Allowed target (governed production) |
|---------------|-------------------------------------|--------------------------------------|
| tenant | `task.tenant_id` (unvalidated as auth) | Fail closed unless admitted triple supplied |
| workspace | metadata → tenant → `"default"` | Fail closed; no metadata→authority |
| principal | `user_id` → tenant → `"anonymous"` | Fail closed; no synthetic principals |

## Existing contracts inventory (design SHA)

| Contract / type | Owner | Trusted by type alone? | Reusable for root authority? |
|-----------------|-------|------------------------|-----------------------------|
| `RequestIdentity` (`intergrax/contracts/agent_run.py`) | Contracts / ACP | **No** — doc says authenticated principal, but any layer may construct it | **Partial** — tenant + principal mapping; **no workspace** |
| `TaskEnvelope.canonical_identity` | Contracts intake | **No** — optional on envelope | Projection + host composition input |
| `Task` / `RuntimeRequest` | Execution payload | **No** | Projection only |
| `RootExecutionLaunchRequest` | Contracts (GR-2-R3) | **No** — validates non-empty strings, not auth provenance | **Carrier** after host supplies admitted identity |
| `CanonicalExecutionIntakeRequest` | Contracts | **No** — doc: after trusted **authority** admission (ParentExecutionAuthority), not identity admission | **Carrier** into `RootExecutionContext` |
| `RootExecutionAuthorityAdmissionService` | Runtime/Governance | Mints trusted `ParentExecutionAuthority` only | Policy on launch triple; does not authenticate identity source |
| `ResolvedWorkerPrincipal` | Contracts (Autonomous Work) | **No** — binding-resolved coordinates | **Pattern** for admitted triple (worker path already correct) |
| `ActiveExecutionGovernanceIdentity` | Runtime (GR-10) | **Yes** — only when bound from admitted input | Runtime carrier; not a host input contract |
| `validate_governance_identity_projection` | Runtime Governance | N/A | **Yes** — trusted vs projection |

### Bind / reset lifecycle

| Function | Module | Role |
|----------|--------|------|
| `bind_active_execution_governance_identity` | `active_execution_governance_identity.py` | Set `ContextVar` at root `ExecutionRuntime.execute` |
| `reset_active_execution_governance_identity` | same | Root completion `finally` |
| `peek_*` / `require_*` | same | PRE_MODEL consumers |

## Root entry matrix (production-relevant)

| Entry point | Caller layer | tenant source | workspace source | principal source | Trusted? | Creates `ActiveExecutionGovernanceIdentity`? |
|-------------|--------------|---------------|------------------|------------------|----------|---------------------------------------------|
| `HostTaskExecutionPort` → `launcher.launch(RootExecutionLaunchRequest)` | Applications / host | `task.tenant_id` | `host_workspace_id(task)` | `host_principal_id(task)` | **No** | Yes, if intake reaches `ExecutionRuntime` with full triple |
| `WorkerExecutionDispatch` → `RootExecutionLaunchRequest` | Autonomous Work | `resolved_principal.tenant_id` | binding-resolved workspace | binding-resolved principal | **Yes** (worker admission binding) | Yes |
| `execute_root_task` → `RootExecutionContext` | Harness / scheduler (`UnifiedTaskRunner`) | `task.tenant_id` | `host_*` | `host_*` | **No** (documented non-production) | Yes |
| `Execution` facade → `runtime.execute` | Internal continuation (`host_task` restore path) | `RootExecutionOptions` from caller | caller | caller | Depends on caller | Yes if triple present |
| `CanonicalExecutionRuntimeAdapter.dispatch` | Default launcher post-admission | launch request fields | launch request fields | launch request fields | Only if launch fields were admitted upstream | Yes |

**True host boundary (first authenticated identity):** Tier-3 application / API authentication and intake adapters that build `TaskEnvelope` (`canonical_identity`, `workspace_id`, `tenant_id`, `user_id`). For workers: `WorkerExecutionAuthorityAdmissionService` + `WorkerPrincipalBinding` → `ResolvedWorkerPrincipal`. **Not** `Task` construction inside Execution, **not** Nexus.

## Trust boundary

```text
External IdP / session auth
        ↓
Host adapter (HTTP, CLI, worker binding, lab harness injector)
        ↓
Admitted root governance identity (typed contract — see Decision)
        ↓
RootExecutionLaunchRequest / harness-equivalent explicit parameter
        ↓
RootExecutionAuthorityAdmissionService (policy + ParentExecutionAuthority)
        ↓
CanonicalExecutionIntakeRequest → RootExecutionContext
        ↓
ExecutionRuntime.bind_active_execution_governance_identity
        ↓
ActiveExecutionGovernanceIdentity
        ↓
Governance PRE_MODEL
```

**Separate projection path (must not feed the arrow above):**

```text
Task / RuntimeRequest fields (user_id, metadata, tenant_id, request.tenant_id)
        ↓
validate_governance_identity_projection(active trusted identity, …)
        ↓
PASS or FAIL-CLOSED
```

## UAEP / PRE_MODEL semantics (design decision)

| Mode | Behavior |
|------|----------|
| Agent step **inside** admitted root execution | **A** — `ActiveExecutionGovernanceIdentity` remains authoritative; `RuntimeRequest` / `RequestIdentity` are projection validated via `resolve_agentic_pre_model_scope` / `require_orchestration_pre_model_governance_scope` |
| **Standalone** agentic execution (no active root) | Requires explicit admitted identity at host boundary; `production_mode=True` → fail closed (already enforced); lab/tests → explicit typed injection (`testing_support.inference_governance_wiring`), **not** `_runtime_request_identity` synthesis |
| `_runtime_request_identity` fallback | **REMOVE FROM AUTHORITY PATH** in R1; keep only for non-governed diagnostics or delete synthesis of `"default"` |

`production_mode` semantics: when true, governed paths **must** have authoritative governance identity (active bind or explicit admitted input) — fail closed.

## Child execution

Without explicit delegation authority: child executions **inherit** the same tenant/workspace/principal; they **must not** escalate or replace governance identity. Delegation models (`ParentExecutionAuthority`, collaborative scopes) remain separate from principal identity.

## Governed vs ungoverned execution

Platform may run execution without PRE_MODEL governance. In that case: **do not bind** `ActiveExecutionGovernanceIdentity` (no fake identity). Governed production root **requires** admitted triple; partial triple on a governed path is a **configuration error** (fail closed), not silent omission of bind (current “all three non-None” skip is insufficient for governed hosts).

## Options

### Option A — Reuse `RequestIdentity` as authority input

Map `RequestIdentity` + host-supplied `workspace_id` at launch composition.

### Option B — New atomic admitted governance identity contract

Immutable `AdmittedRootGovernanceIdentity` (tenant, workspace, principal) in `intergrax/contracts/…`, constructible only at host/admission boundaries.

### Option C — Promote existing admission carrier (`ResolvedWorkerPrincipal` / launch intake)

Generalize worker-resolved triple to platform-neutral admitted identity; thread through `RootExecutionLaunchRequest` / `CanonicalExecutionIntakeRequest`; stop deriving from `Task` inside Execution.

### Comparison

| Criterion | A | B | C |
|-----------|---|---|---|
| authority correctness | Weak — type not admission-marked | Strong | Strong |
| reuse existing contracts | High | Medium | High (worker pattern exists) |
| public API impact | Low | Medium | Medium (replace loose strings) |
| migration cost | Low short-term; high risk | Medium | Medium |
| testability | Poor — easy to forge | Good | Good |
| pluginability | Host maps to RequestIdentity | Host maps to admitted type | Same as B |
| multi-host support | Adequate with discipline | Good | Good |
| duplicate truth risk | High (Task + RequestIdentity) | Low | Low |
| backwards compatibility | Silent forge continues | Breaks implicit fallbacks | Breaks implicit fallbacks |

## Decision

**RECOMMENDED OPTION: C (promote admitted triple contract; align with Option B shape)**

Adopt a **single vendor-neutral admitted identity contract** in Tier-0 contracts (proposed name: `AdmittedRootGovernanceIdentity`; fields mirror `ResolvedWorkerPrincipal` / atomic triple). **Host/admission layers alone** construct it after authentication or durable binding resolution. Execution and Governance **only consume** it; they **never** derive it from `Task`, metadata, or UAEP request fallbacks.

`RequestIdentity` remains the **principal + tenant claim** for agent runs and envelope intake; host adapters **compose** `AdmittedRootGovernanceIdentity` from authenticated session + workspace membership (not from `Task` round-trip).

`RootExecutionLaunchRequest` and `CanonicalExecutionIntakeRequest` should carry `AdmittedRootGovernanceIdentity` (or equivalent single field) instead of three independent strings in R1.

`ResolvedWorkerPrincipal` becomes a **worker-specific projection** into `AdmittedRootGovernanceIdentity` at the worker dispatch boundary (no duplicate semantics).

**Trust marker:** the admitted contract type denotes **host-authenticated / admission-bound** identity for root governance. `RequestIdentity` alone does **not**. `ActiveExecutionGovernanceIdentity` denotes **runtime-active** authority for the current root execution scope.

### Rejected alternatives

- **A alone:** `RequestIdentity` lacks workspace; forgeable without admission seam.
- **B without C linkage:** duplicates `ResolvedWorkerPrincipal` instead of unifying worker and application host paths.

## Target flow (post-R1)

Same as trust boundary diagram. `execute_root_task` gains required `admitted_governance_identity: AdmittedRootGovernanceIdentity | None` with `None` only for explicitly ungoverned harness profiles; governed qualification harnesses pass explicit admitted values.

## Helper disposition (R1)

| Helper | Disposition |
|--------|-------------|
| `host_workspace_id` | **KEEP AS PROJECTION/EVIDENCE** — diagnostics, OBS facts, projection validation input; **REMOVE FROM AUTHORITY PATH** |
| `host_principal_id` | **KEEP AS PROJECTION/EVIDENCE**; **REMOVE FROM AUTHORITY PATH** |
| `_runtime_request_identity` | **RESTRICT** — no `"default"` / metadata synthesis for governed paths; **REMOVE FROM AUTHORITY PATH** for PRE_MODEL |

## Failure semantics

- Governed production root without admitted identity → **fail closed** (launch denied or `PreModelPolicyConfigurationError` / admission `DENIED`).
- Projection mismatch → **fail closed** (existing `GovernanceIdentityProjectionMismatchError` path).
- Partial admitted triple → **invalid configuration** (do not partially bind).

## Migration impact (R1)

| Area | Required change |
|------|-----------------|
| `host_task.py` launch | Accept admitted identity from host port wiring, not `host_*` |
| `orchestration.execute_root_task` | Explicit admitted parameter; qualification injectors |
| `CanonicalExecutionIntakeRequest` | Single admitted field |
| Applications / lab FastAPI | Map auth session → admitted identity at composition |
| Worker dispatch | Map `ResolvedWorkerPrincipal` → admitted contract |
| Tests using `host_*` for authority | Replace with `testing_support` / explicit admitted fixtures |
| UAEP | Active identity authoritative in-root; remove non-production mint from request-only path |

**Call-site counts (design SHA):** production launch — `host_task.py` (1), `worker_execution_dispatch.py` (1); harness — `execute_root_task` via `unified_task_runner.py` (1) + unit tests (4 modules).

## Architecture gate (future)

Static/behavior gate: forbid `host_workspace_id` / `host_principal_id` / `Task.metadata` / `Task.user_id` arguments to `RootExecutionContext` / `bind_active_execution_governance_identity` outside allowlisted test harness modules.

## Testing / qualification plan (R1)

- Valid admitted identity → bind succeeds; DG-001 5/5 P3 preserved.
- Missing identity on governed production path → fail closed.
- Metadata workspace without admitted identity → must not become authority.
- `Task.user_id` without admitted identity → must not become authority.
- `"default"` / `"anonymous"` forbidden as authority.
- Projection mismatch → fail closed.
- Child / nested child preserves identity.
- UAEP PRE_MODEL uses active authority in-root.
- Standalone lab mode uses explicit test admission only.
- GR-10 PRE_MODEL wiring regression suite.

## Non-goals

- Nexus public identity ports; service locators; vendor auth in core.
- Implementing R1 in this ADR task.
- Changing runtime behavior in this commit.

## Security / correctness invariants

- NO METADATA-DERIVED AUTHORITY.
- NO DEFAULT PRINCIPAL AUTHORITY.
- NO DEFAULT WORKSPACE AUTHORITY.
- NO TENANT-AS-PRINCIPAL FALLBACK.
- NO TENANT-AS-WORKSPACE FALLBACK unless contractually explicit at host (documented lab-only profiles).
- NO SILENT GOVERNANCE DISABLE on governed paths.
- NO CHILD AUTHORITY ESCALATION.

## Compliance

- Tier boundaries preserved (`contracts` do not import runtime).
- Aligns with ADR-GR-10-001 active identity + projection validation model.
- Host pluginability: callers supply typed admitted value; no global provider.

## Implementation notes

Implementation tracked under **OBS-DIAG-RECERT-P2C-R0A-R1**. Diagnostics impact: **none** (interpretation only). Nexus impact: **none** (internal Execution Engine).
