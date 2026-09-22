# ADR-HARNESS-003: Execution-Bound Capability Invocation Ownership Model

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture freeze · HARNESS-01-R5-ADR3) |
| **Date** | 2026-09-22 |
| **Deciders** | Harness architecture / HARNESS-01-R5 |
| **Related** | [`ADR-HARNESS-001`](../2026-09-20/ADR-HARNESS-001.md) (does not supersede D5 public Tools ABI) · [`EXECUTION_ENGINE.md`](../../../../maintainers/architecture/EXECUTION_ENGINE.md) · [`UNIFIED_EXECUTION_ARCHITECTURE.md`](../../../../architecture/UNIFIED_EXECUTION_ARCHITECTURE.md) |

**Resolved by:** HARNESS-01-R5-ADR3

## Context

HARNESS-01 established that **Nexus is a private Execution Engine implementation**, not a public, plugin, agent, tool, UCA, integration, or host API ([`ADR-HARNESS-001`](../2026-09-20/ADR-HARNESS-001.md)). Platform consumers must depend on **neutral typed contracts**, not Nexus types.

After W3 work, multiple **invocation-shaped** contracts coexist:

| Contract | Module (current) |
|----------|------------------|
| `ToolInvocationInvokerPort` | `intergrax.tools.invocation_pattern` |
| `ExecutionBoundCatalogToolInvoker` | `intergrax.contracts.execution_bound_catalog_tool_invocation` |
| `ExecutionBoundDeclarativeToolInvoker` | `intergrax.contracts.execution_bound_declarative_tool_invocation` |
| `CatalogToolInvocationPort` | `intergrax.tools.catalog_tool_invocation_port` (alias) |
| `CompensationSideEffectExecutionPort` | `intergrax.contracts.compensation_side_effect_execution` |
| `RuntimeToolInvoker` | `intergrax.runtime.nexus.tools.invoker` |
| `CatalogDeclarativeToolInvoker` | `intergrax.runtime.nexus.agents.catalog_declarative_invoker` |

There was **no formal taxonomy** of which surface is public authoring ABI, domain-owned execution-bound contract, EE-internal adapter, or Nexus-internal implementation; nor a frozen rule for **execution identity propagation** (`bind_execution_identity` vs request fields).

Canonical identity carriers already exist and must be reused:

- `intergrax.contracts.execution_identity` — `RunId`, `AttemptId`, `ExecutionId`, `TaskId`, `ActiveExecutionIdentityState`, active identity `ContextVar` helpers (**mint / active propagation authority** lives with Execution Runtime / `ExecutionBoundary`, not with tool invokers).
- `intergrax.runtime.execution.identity_binding.ExecutionIdentityBinding` — frozen four-ID binding used **inside** `ExecutionBoundary` (EE-internal propagation view).

This ADR freezes **ownership**, **visibility**, a shared **Execution-Bound Capability Invocation (EBCI)** invariant, **identity model**, **RuntimeState projection** rules, incremental **migration**, and **qualification gates**. **No production implementation** is in scope for this ADR.

### Non-goals

- No universal execution-bound megacontract (`UniversalToolInvoker`, `ExecutionBoundCapabilityInvoker[Any]`, etc.).
- No superseding of [`ADR-HARNESS-001`](../2026-09-20/ADR-HARNESS-001.md) **D5** public Tool Invocation Pattern ABI without an explicit future ADR.
- No inventory / W3 gate / allowlist edits.
- No CodeCraft, UCA semantic, or Nexus implementation changes in the ADR3 delivery commit.

## Decision

### 1. Three-layer invocation model (frozen)

| Level | Owner | Purpose | Nexus import |
|-------|-------|---------|--------------|
| **L1 — Public authoring ABI** | Tools domain | Extension authors implement `ToolInvocationPattern` strategies | **Forbidden** |
| **L2 — Domain execution-bound capability contracts** | Owning domain (Tools catalog seam, Agents/ACP compensation, etc.) | Typed operations under **already-admitted** execution authority | **Forbidden** for external/domain implementers |
| **L3 — EE / Nexus implementation adapters** | Execution Engine | Map L2 contracts to `RuntimeToolInvoker`, Nexus runtime projection | **Allowed only inside** `intergrax/runtime/execution/**` and `intergrax/runtime/nexus/**` composition |

**Common invariant ≠ common megacontract.** Domains keep **separate** L2 contracts when input/output semantics or admission path differ.

### 2. Architecture decision matrix (variants)

| Variant | Description | Verdict |
|---------|-------------|---------|
| **A** | Reuse only `ToolInvocationInvokerPort` everywhere | **Rejected** — wrong semantics for compensation admission, catalog governance evidence, declarative replay; would expand public ABI into a god-port |
| **B** | Multiple domain execution-bound L2 contracts + shared EBCI invariant | **Accepted** |
| **C** | One universal execution-bound invoker | **Rejected** — violates domain typing, pluginability, and duplication controls |
| **D** | Single EE-only execution-bound port + domain adapters | **Rejected as sole model** — L2 must remain domain-owned and Nexus-free; L3 adapters stay EE-internal |

### 3. Contract classification (frozen)

#### 3.1 Contract matrix

| Contract | Owner | Consumer | Semantics | Public? | Identity source | Nexus allowed? |
|----------|-------|----------|-----------|--------:|-----------------|---------------:|
| `ToolInvocationInvokerPort` | Tools domain | `ToolInvocationPattern` plugins, reference enterprise plugin | Single prepared `ToolExecutionRequest` invoke inside a pattern loop | **Yes** (sole public Tools invocation ABI per ADR-HARNESS-001 D5) | Pattern context + request fields; **no** lifecycle ownership | Implementations: EE bridge only; **consumers: no** |
| `ExecutionBoundCatalogToolInvoker` | Platform contracts (Tools/UCA catalog seam) | CodeCraft, UCA composition, qualified capability wiring | Catalog tool gateway under active run; governance evidence on request | **No** — **domain/platform L2** | **Per-invoke request** (target); today also mutable `bind_*` (**debt**) | Impl: L3 only; **contract: no** |
| `CatalogToolInvocationPort` | Tools domain (alias) | Same as catalog invoker | Typedef alias to `ExecutionBoundCatalogToolInvoker` | **No** | Same as catalog invoker | Same |
| `ExecutionBoundDeclarativeToolInvoker` | Agents / ACP persistence plane | Compensation sessions, declarative replay | Async declarative tool invoke under bound run | **No** — **domain L2** | **Per-invoke** (target); today `bind_*` (**debt**) | Impl: L3; **contract: no** |
| `CompensationSideEffectExecutionPort` | Agents + Execution admission | `ExecutionRuntime` composition, queue worker | **Root-admitted** compensation side-effect execution | **No** — **domain L2 admission** | `CompensationSideEffectInput` (immutable work unit) | Runtime adapter: L3 |
| `CompensationToolInvokeSession` | Agents persistence | Wrapped by `RuntimeCompensationSideEffectExecution` | Session-scoped tool invoke inside admitted work | **No** | Explicit per-invoke parameters | Impl may use declarative invoker |
| `RuntimeToolInvoker` | EE / Nexus | Nexus loop, tool loop, attestation, internal patterns | Physical tool runtime invoke, scope policy, cancellation | **No** — **L3** | From `ToolExecutionRequest` + runtime state | **Yes** (internal) |
| `CatalogDeclarativeToolInvoker` | EE / Nexus (ACP) | ACP session host context, acceptance harness | Nexus declarative catalog dispatch for sessions | **No** — **L3** | Internal Nexus binding | **Yes** (internal) |
| `NexusExecutionBoundCatalogToolInvoker` | EE / Nexus | Host composition wiring to L2 catalog contract | L3 adapter: L2 → `RuntimeToolInvoker` + projection | **No** | Validates request vs binding (**transitional**) | **Yes** (internal) |

#### 3.2 Per-contract rationale (summary)

| Contract | Why it exists | Duplicate of another? |
|----------|---------------|------------------------|
| `ToolInvocationInvokerPort` | Stable plugin ABI for custom invocation **strategies** | **No** — strategy authoring, not domain admission |
| `ExecutionBoundCatalogToolInvoker` | Neutral catalog seam for domains that must not import Nexus | **No** vs invoker port — catalog + governance + typed `BaseModel` input |
| `ExecutionBoundDeclarativeToolInvoker` | Declarative JSON tool path for ACP/compensation | **No** vs catalog — async, `JsonObject`, replay semantics |
| `CompensationSideEffectExecutionPort` | **ExecutionRuntime admission** boundary for compensation jobs | **No** vs declarative invoker — owns **admission orchestration**, not tool dispatch |
| `RuntimeToolInvoker` | Single physical Nexus tool runtime | **No** — implementation engine, not a domain contract |

#### 3.3 Duplication verdict (pairs)

| Pair | Verdict |
|------|---------|
| `ToolInvocationInvokerPort` ↔ `ExecutionBoundCatalogToolInvoker` | **JUSTIFIED SEPARATION** (public strategy vs L2 catalog seam) |
| `ExecutionBoundCatalogToolInvoker` ↔ `ExecutionBoundDeclarativeToolInvoker` | **JUSTIFIED SEPARATION** (sync typed catalog vs async declarative) |
| `ExecutionBoundDeclarativeToolInvoker` ↔ `CompensationSideEffectExecutionPort` | **JUSTIFIED SEPARATION** — port is admission authority; invoker is tool session dependency |
| `CompensationSideEffectExecutionPort` ↔ `CompensationToolInvokeSession` | **JUSTIFIED SEPARATION** — admission vs invoke session |
| `ExecutionBoundCatalogToolInvoker` ↔ `NexusExecutionBoundCatalogToolInvoker` | **INTERNAL ADAPTER** |
| `CatalogToolInvocationPort` ↔ `ExecutionBoundCatalogToolInvoker` | **DEPRECATED ALIAS** (keep name only for compatibility; no second semantics) |
| Any L2 ↔ `RuntimeToolInvoker` | **INTERNAL ADAPTER** — never a substitute for L2 |

### 4. Authority and lifecycle (frozen)

**Execution-bound invocation uses authority; it does not own authority.**

| Concern | Owner |
|---------|-------|
| Mint `RunId` | `ExecutionIdentityAuthority` / root admission (EE) |
| Mint `AttemptId` | EE identity authority |
| Mint `ExecutionId` | EE identity authority |
| Bind **active** execution (`ContextVar`) | `ExecutionBoundary` / EE only |
| Propagate immutable scope on L2 invoke | **Caller + frozen request/session parameters** (target model) |
| Validate scope on L2 invoke | L2 contract implementer (fail-closed mismatch) |
| Build Nexus `RuntimeState` projection | **L3 adapter only** (`NexusExecutionBoundCatalogToolInvoker`, etc.) |

| Lifecycle concern | Owner |
|-------------------|-------|
| Root open / close | `ExecutionRuntime` |
| Retry decision / execution | EE strategies / runtime — **not** L2 invokers |
| Resume / recovery | EE continuation — **not** L2 invokers |
| Tool invocation (physical) | `RuntimeToolInvoker` (L3) |
| Compensation side-effect admission | `CompensationSideEffectExecutionPort` + `ExecutionRuntime` |
| Compensation tool dispatch | `CompensationToolInvokeSession` / declarative invoker (L2/L3) |

L2 contracts **MUST NOT**: mint identity, bind global active execution context, open/close root lifecycle, own retry/recovery/scheduling, or expand authority.

### 5. Execution identity model (frozen target)

**Decision: `PER-CALL IMMUTABLE` (Model A)** as the **canonical L2 contract plane**, with optional **immutable scoped factory** (Model B) **only** as a composition convenience that returns a **new** object per execution scope (no shared mutable rebind).

Rationale (selected after comparison):

| Criterion | Model A (per-call scope) | Model B (immutable bound instance) | Stateful `bind_*` (current) |
|-----------|--------------------------|-------------------------------------|-----------------------------|
| Correctness | Single source of truth on invoke | Single source if factory is per-scope | Duplicate A+B fields; mismatch checks |
| Concurrency | Shared stateless invoker OK | Shared factory OK; instances not shared | **Unsafe** on shared instance |
| Lifecycle ownership | Clear: EE owns; L2 validates | Clear if factory is pure | Blurs ownership |
| Pluginability | Fake impl without Nexus | Same | Encourages hidden mutable scope |
| Migration | Catalog request already carries IDs | Declarative can adopt scope parameter | Remove after M6 |

**Canonical projection (no new DTO class required for ADR3):**

- **Catalog:** `ExecutionBoundCatalogToolInvokeRequest` fields `tenant_id`, `run_id`, `task_id`, `agent_id` are the **authoritative per-invoke execution scope** for the catalog plane (plus step/correlation payload).
- **Declarative / compensation session:** per-invoke explicit identity parameters (as on `CompensationToolInvokeSession.invoke`) or a **frozen** scope parameter added in ADR3-IMP — reuse field names aligned with `CompensationSideEffectInput`, not a parallel identity mint type.
- **EE full four-ID binding:** `ExecutionIdentityBinding` remains **internal** to `ExecutionBoundary`; L2 tool contracts use the **admitted run/task/tenant/agent projection** appropriate to their domain, not mint `AttemptId`/`ExecutionId` unless already present in their work unit.

**`bind_execution_identity` verdict: `DEPRECATE` → remove from final L2 plane (`INTERNAL-ONLY` transitional in L3 until migration M6).** It may today mean only “store IDs for later validation,” but the name and mutable behavior imply authority binding and shared mutable scope. Fail-closed comparison is not a substitute for a concurrency-safe contract.

### 6. RuntimeState materialization (frozen)

**`LEGAL INTERNAL PROJECTION`** inside `intergrax/runtime/nexus/**` (and EE composition wiring), **not** a second execution-runtime context authority.

`NexusExecutionBoundCatalogToolInvoker` may construct `RuntimeConfig`, `RuntimeContext`, `RuntimeRequest`, `RuntimeState`, `SessionManager` **solely** to project an already-admitted execution into the legacy/current Nexus Tool Runtime surface.

L3 projection adapters **MUST NOT**:

- Mint `RunId`, `AttemptId`, or `ExecutionId`
- Open or close root lifecycle
- Own retry, recovery, or scheduling
- Expand authority beyond the admitting execution
- Be exposed publicly or consumed outside EE/Nexus composition

**Invariant (literal):** *An execution-bound invocation adapter must never evolve into a second `ExecutionRuntime`.*

Host composition may construct L3 adapters; the host **must not** become execution lifecycle owner, identity authority, or Tool Runtime authority.

### 7. Public surface matrix

| Surface | Public | Domain internal | EE internal | Nexus internal |
|---------|-------:|----------------:|------------:|---------------:|
| `ToolInvocationInvokerPort` | ✓ | | bridge | impl |
| `ToolInvocationPattern` / planner ports | ✓ | | | |
| `ExecutionBoundCatalogToolInvoker` | | ✓ (L2) | | |
| `CatalogToolInvocationPort` | | ✓ (alias) | | |
| `ExecutionBoundDeclarativeToolInvoker` | | ✓ (L2) | | |
| `CompensationSideEffectExecutionPort` | | ✓ (L2) | adapter | |
| `CompensationToolInvokeSession` | | ✓ | | |
| `RuntimeToolInvoker` | | | | ✓ |
| `CatalogDeclarativeToolInvoker` | | | | ✓ |
| `NexusExecutionBoundCatalogToolInvoker` | | | ✓ | ✓ |

### 8. CodeCraft, UCA, plugins

- **CodeCraft** consumes L2 `ExecutionBoundCatalogToolInvoker` only — not `RuntimeToolInvoker`, `RuntimeState`, or Nexus types.
- **UCA** retains UCA-specific capability semantics; this ADR only fixes the **invocation ownership seam**.
- **External L2 implementations** must be substitutable **without** `import intergrax.runtime.nexus.*`.

### 9. Strong typing, concurrency, governance

- L2/L1 contracts: no `Any` / `object` / `dict[str, Any]` for known semantics; no reflection-as-contract; no service locator registries for invocation.
- **Concurrency:** L2 invoker instances **may be shared** only when **stateless** or **immutable**; mutable `bind_*` is incompatible with sharing (transitional only).
- **Governance:** L2 may **carry** `ToolInvocationGovernanceApprovalEvidence` (catalog); it must not mint approvals or decide policy.

### 10. EBCI invariants (frozen)

| ID | Invariant |
|----|-----------|
| **EBCI-01** | Use **existing** execution authority only; never create a parallel admission path |
| **EBCI-02** | **No identity mint** on L2 invokers |
| **EBCI-03** | **No lifecycle ownership** (root open/close) on L2 |
| **EBCI-04** | **No scheduler ownership** |
| **EBCI-05** | **No retry/recovery ownership** |
| **EBCI-06** | **No authority expansion** |
| **EBCI-07** | **Immutable execution scope** on each invoke (per-call authoritative fields) |
| **EBCI-08** | **No public Nexus dependency** for L1/L2 |
| **EBCI-09** | **Typed** domain request/result |
| **EBCI-10** | **Replaceable** implementation (fake/in-memory/alternate provider) without core patch |
| **EBCI-11** | **Fail-closed** identity mismatch when validation is required |
| **EBCI-12** | **Thread/concurrency-safe contract semantics** — no shared mutable execution scope |

### 11. Migration plan (documentation only — ADR3-IMP-01)

| Step | Change | Owner | Breaking? | Gate |
|------|--------|-------|----------:|------|
| **M1** | Canonical per-call scope fields documented; `ExecutionIdentityBinding` stays EE-internal | Architecture | No | Gate 1–2 |
| **M2** | Catalog L2: request-only identity; deprecate `bind_execution_identity` on protocol | Tools/UCA contracts | Soft deprecate | Gate 4 |
| **M3** | Declarative L2: identity on `invoke` / frozen scope param | Agents/ACP | Soft deprecate | Gate 4 |
| **M4** | L3 Nexus adapters: stop requiring mutable bind; validate request only | EE/Nexus | Internal | Gate 5–6 |
| **M5** | Consumers (CodeCraft, compensation wiring) migrated | Applications/agents | Maybe | Gate 7 |
| **M6** | Remove `bind_execution_identity` from L2 protocols | Contracts | **Yes** (L2) | Gate 4 + regression |
| **M7** | Architecture catalog + qualification gates enforced in CI | Qualification | No | Gates 1–8 |

Temporary compatibility: **internal-only**, **deprecated**, **time-bounded**, **not documented as canonical**, removed before HARNESS-01 final closure.

### 12. Qualification plan (design only — not implemented in ADR3)

| Gate | Requirement |
|------|-------------|
| **Gate 1** | No new execution-bound invocation contract without owner classification in architecture catalog |
| **Gate 2** | Domain/public execution-bound contracts Nexus-free |
| **Gate 3** | No execution-bound contract mints Run/Attempt/Execution IDs |
| **Gate 4** | No stateful `bind_execution_identity` in **final** L2 contract plane |
| **Gate 5** | No execution-bound adapter owns root lifecycle |
| **Gate 6** | Nexus `RuntimeState` projection stays inside EE/Nexus implementation |
| **Gate 7** | Custom L2 implementation test without Nexus import |
| **Gate 8** | Negative synthetic duplicate-invoker detection via declared owner + semantic class catalog (not name-only) |

Consider a **qualification metadata catalog** (not a runtime service locator) listing invocation surfaces, owner, and semantic class.

## Rejected alternatives

| Alternative | Reason |
|-------------|--------|
| `UniversalExecutionBoundInvoker` / generic capability executor | Megacontract, authority blur, typing erosion |
| Public Nexus facade for tool invoke | Violates ADR-HARNESS-001 |
| `Any` / `object` shared execution context bag | Non-replaceable, non-auditable |
| Host-owned runtime / identity authority via invoker binding | Violates EE ownership |
| Collapsing all tool paths into `ToolInvocationInvokerPort` | God-port; breaks compensation admission and catalog governance |

## Consequences

### Positive

- Clear L1/L2/L3 ownership for all current invocation contracts
- Concurrency-safe target model without shared mutable execution scope
- CodeCraft / plugin paths remain Nexus-free at contract level
- RuntimeState projection explicitly bounded as L3 implementation detail

### Negative

- ADR3-IMP-01 required to align protocols and adapters with EBCI-07/12 and remove `bind_*` from L2
- Multiple L2 contracts remain (intentional — no universal invoker)

## Compliance

- Tier boundaries preserved (`intergrax/` does not import agents/applications; agents do not import applications)
- Does not silently supersede ADR-HARNESS-001 D5 (`ToolInvocationInvokerPort` remains sole public Tools invocation ABI)
- Linked from ADR index and Execution Engine maintainer hub

## Implementation notes

- **ADR3 delivery:** documentation + doc regression tests only
- **Next work:** ADR3-IMP-01 if contract reconciliation is scheduled; otherwise resume W3-R1-Q2 after independent audit of this ADR against GitHub code/docs

## Relationship to ADR-HARNESS-001

- **Retained:** Nexus privacy, EE owner zones, public `intergrax.tools.invocation_pattern` ABI
- **Extended:** execution-bound L2 taxonomy, EBCI invariants, identity propagation target, L3 projection rules
- **Not changed:** D5 sole public ABI for Tool Invocation Pattern without a future explicit supersession ADR
