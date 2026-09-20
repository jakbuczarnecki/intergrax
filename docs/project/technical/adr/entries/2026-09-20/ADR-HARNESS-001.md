# ADR-HARNESS-001: Execution Engine Public Boundary & Nexus Encapsulation Model

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture freeze · HARNESS-01-ADR2) |
| **Date** | 2026-09-20 |
| **Deciders** | Platform architecture · Execution Engine ownership |
| **Baseline HEAD** | `3dc68c32e2fdd7710d47dba6135e7a2bba964851` (`development`) |
| **R5 block lineage** | `358593fd6324cf4682eee9092ae0efac62ea2a2d` (HARNESS-01-R5 BLOCKED — architecture decision required) |
| **Related** | [`UNIFIED_EXECUTION_ARCHITECTURE.md`](../../../architecture/UNIFIED_EXECUTION_ARCHITECTURE.md) · [`EXECUTION_ENGINE.md`](../../../maintainers/architecture/EXECUTION_ENGINE.md) · ADR-TOOL-003 · ADR-AGENT-002 · NPSC-3C |
| **Supersedes (boundary interpretation)** | HARNESS-01-R4 path-by-path inventory legality for non-EE zones; any reading that `runtime/*` ≡ Execution Engine; any public Nexus facade proposal |
| **Does not reopen** | HARNESS-02 lifecycle authority · frozen Execution Engine identity/deadline/cancellation semantics · TR-01 · SESSION-01 · CE-02 domain ownership |

---

## Context

HARNESS-01-R5 (`Nexus Dependency Inversion & External Consumer Elimination`) stopped with:

```text
HARNESS-01-R5: BLOCKED
ARCHITECTURE DECISION REQUIRED
```

No production migration commit was made. As-built inventory under HARNESS-01 higher-layer discovery (excludes `intergrax/runtime/nexus/**`) at ADR baseline:

| Metric | Count |
|--------|------:|
| Higher-layer Nexus importers (production scope) | **191** |
| `intergrax/runtime/execution/**` | **23** |
| Outside strict EE owner zone (execution+nexus) | **168** |

Semantic groups of the 168 external importers (class-level; not a permanent allowlist):

| Semantic class | Approx. role today |
|----------------|--------------------|
| Host composition | `applications/*/host/**`, `intergrax/applications/_shared/**` — builds/wires `NexusLoop` |
| Agent / UAEP | `intergrax/agents/**` — `RuntimeRequest` / `RuntimeContext` / `RuntimeAnswer` |
| Tools / websearch | tool providers + invocation helpers |
| Context | CE providers reading `ContextProviderContext.runtime` |
| RAG / LLM / integrations | provider adapters touching Nexus config/session |
| Runtime non-EE | `task`, `wiring`, `observability`, `hooks`, `human`, `long_running`, `tools`, … |
| Debug / eval / CLI / lab | maintainer and developer tooling |

R4 classified many of these as `LEGAL` or `DEBT` under a broad “platform runtime / host composition” reading. That reading is **ambiguous** and **unsafe** as a final HARNESS-01 exit: it treats large parts of `runtime/*` and host trees as if they were Execution Engine internals, and leaves `DEBT` as an acceptable end state.

---

## Problem

1. **Nexus leaks as a de-facto public API** through host factories (`build_nexus_loop_from_environment` → `NexusLoop`), Agent/UAEP types, tool guides, and provider imports.
2. **Owner-zone ambiguity** — unclear whether `runtime/task`, `runtime/wiring`, `applications/_shared`, and `applications/*/host` may legally import Nexus.
3. **R4 error** — treating `runtime/*` as automatically Execution Engine.
4. **Host ≠ EE** — composition root may wire EE but must not expose Nexus implementation types to applications.
5. **Public guides instruct Nexus imports** (e.g. `TOOL_INVOCATION_PATTERN_AUTHOR_GUIDE.md`) while neutral Tools contracts already exist.
6. **`ContextProviderContext.runtime: Any | None`** masks known coupling (`base_messages`).
7. Path-by-path inventory of ~190 files cannot be the final qualification model.

---

## Constraints

- ADR-only task: **no** production class moves, host factory renames, Agent/UAEP rewrites, Nexus changes, or contract mutations in this decision package.
- Nexus remains **private orchestration implementation** inside Execution Engine (UEA §7; NPSC-3C).
- Execution Engine remains **sole execution authority** (HARNESS-02 lifecycle untouched).
- Prefer **reuse** of existing execution-semantic contracts over new facades.
- Forbidden: `NexusFacade`, `PublicNexusPort`, `NexusService`, `NexusRuntimeAPI`, public Nexus DTO copies, `Any`/`object` as coupling escape, service locator, getattr reflection, second EE/host/agent authority.
- Public API must be **execution-semantic**, never Nexus-semantic.
- Fail-closed owner-zone gate; **no** permanent path-by-path allowlist; **no** final `DEBT` class.
- Pluginability stays **domain-owned** capability contracts (not a universal plugin megacontract).

---

## Decision

### D0 — Hard invariant

```text
Nexus = private/internal orchestration implementation inside Execution Engine
```

There is **no** public Nexus entry. External layers depend only on neutral domain/platform contracts and Execution Engine public/internal boundaries.

Canonical dependency direction:

```text
Applications / Agents / Plugins / Tools / Integrations / RAG / LLM / Context / Schedulers / Workers
        ↓
neutral domain/platform contracts
        ↓
Execution Engine public / internal boundary
        ↓
Execution Engine implementation (runtime/execution/**)
        ↓
Nexus private internals (runtime/nexus/**)
```

Never: external/domain layer → Nexus.

### D1 — Execution Engine owner zones (freeze)

**Conservative preferred model (accepted):**

| Path zone | Formal owner | EE internal? | Direct Nexus allowed? | Public? | Reason / migration |
| --------- | ------------ | -----------: | --------------------: | ------: | ------------------ |
| `intergrax/runtime/execution/**` | Execution Engine | **YES** | **YES** | No (impl) | Canonical EE implementation; may adapt Nexus |
| `intergrax/runtime/nexus/**` | Execution Engine (private) | **YES** | **YES** (self) | **No** | Private orchestration only |
| `intergrax/runtime/task/**` | Platform task substrate (consumer of EE) | **No** | **No** (final) | No | Task models/registry are not Nexus; migrate off Nexus imports |
| `intergrax/runtime/wiring/**` | Platform composition helpers | **No** | **No** (final) | No | Composition helpers must call EE factories, not Nexus modules |
| `intergrax/runtime/observability/**` | Observability domain | **No** | **No** (final) | Partial contracts | Map via EE adapters to `RuntimeEvent` / Evidence |
| `intergrax/runtime/hooks/**` | Runtime hooks plane | **No** | **No** (final) | No | Hook contracts only |
| `intergrax/runtime/human/**` | HITL / human plane | **No** | **No** (final) | Contracts | Continuation via EE ports |
| `intergrax/runtime/long_running/**` | Long-running / checkpoint | **No** | **No** (final) | Contracts | Checkpoint/resume through EE authority |
| `intergrax/runtime/tools/**` | Runtime tool bridge (non-domain) | **No** | **No** (final) | No | Prefer `intergrax/tools/**` contracts |
| `intergrax/applications/_shared/**` | Host composition (Tier-0 shared) | **No** | **Composition-time only until Wave 6**; **No** at HARNESS-01 final for application-visible APIs | No | May construct EE internals **inside** designated composition owner; must not return/export Nexus types |
| `applications/*/host/**` | Application composition root | **No** | **No** (final) | Host-local | Receives EE handle / `HostTaskExecutionPort` only |
| `intergrax/agents/**` | Agent / UAEP | **No** | **No** (final) | Public agent contracts | Neutral request/result; EE adapter owns Nexus DTO mapping |
| `intergrax/tools/**` | Tools domain | **No** | **No** (final) | **Yes** (tools ABI) | Author-facing; Nexus-free |
| `intergrax/context/**` | Context Engineering | **No** | **No** (final) | CE contracts | Minimal typed assembly deps |
| `intergrax/rag/**` | RAG domain | **No** | **No** (final) | RAG contracts | Retriever/reranker contracts |
| `intergrax/llm_adapters/**` | LLM adapters | **No** | **No** (final) | Adapter contracts | No Nexus `RuntimeConfig`/`RunBudget` |
| `intergrax/integrations/**` | Integrations | **No** | **No** (final) | Integration contracts | No Nexus persistence/session internals |
| `intergrax/websearch/**` | WebSearch capability | **No** | **No** (final) | Provider contracts | Capability layer, not Nexus |
| `intergrax/debug/**`, `lab/**`, `eval/**`, `experiments/**` | Maintainer/debug tooling | **No** | **INTERNAL-ONLY** until reclassified; production-facing tooling → EE contracts | No | ≠ `tests/**` |
| `tests/**`, `testing_support/**` | TEST_ONLY | n/a | Allowed under TEST_ONLY class | No | Never production ABI |
| `intergrax/contracts/**` | Public contracts | **No** | **VIOLATION** | **Yes** | Hard ban |

**Additional EE-internal zones** beyond `execution/**` + `nexus/**` are **not** granted by this ADR. Any future expansion requires a new Accepted ADR with documented ownership proof.

**Composition-time exception (temporary, Wave 6):** designated EE composition modules under `intergrax/applications/_shared/` (and EE-owned factory modules under `runtime/execution/`) may construct Nexus **internally**. They must expose only execution-semantic results. Application hosts must not import Nexus.

### D2 — Canonical host entry

| Concern | Decision |
|---------|----------|
| Composition-time construction | Target public name: **`build_execution_engine(...)`** (or existing semantic equivalent living under EE/composition ownership). Current `build_nexus_loop_from_environment(...)` is **Nexus-named debt** — rename/wrap in R5 Wave 6; do **not** implement in this ADR task. |
| Composition responsibility | Build RuntimeConfig-derived EE graph; wire governance/budget/observability; **construct Nexus privately**; return execution-semantic handle. |
| Composition inputs | `ApplicationEnvironmentProfile`, registry reads, optional stores/adapters already accepted by host composition — typed; no `Any` for known seams. |
| Composition outputs | **Must not** return `NexusLoop` to application hosts. |
| Canonical runtime execution entry | Existing **`HostTaskExecutionPort`** (`intergrax.runtime.execution.host_task`) — reuse. |
| Supporting EE coordination | Existing **`ExecutionBoundary`**, **`ExecutionRuntime`**, **`Execution`** facade — reuse; do **not** invent `ExecutionEngineFacade` unless a later ADR proves `HostTaskExecutionPort` insufficient. |
| NexusLoop escape | **Forbidden** on public host/application surfaces after Wave 6 exit. |

Build-time vs run-time:

```text
composition-time: build_execution_engine / shared composition → EE handle
run-time:         HostTaskExecutionPort.execute(...) / ExecutionBoundary.execute(...)
```

### D3 — Agent / UAEP execution boundary

| Piece | Decision |
|-------|----------|
| Principle | **Agent ≠ Execution Engine**. Agents must not consume Nexus implementation API. |
| Request | Prefer existing **`AgentRunRequest`** / identity + options already in `intergrax.contracts.agent_run` (tenant/principal, objective/input, execution options, capability overrides). EE adapter maps to internal `RuntimeRequest`. **Do not** publish `PublicRuntimeRequest`. |
| Context | Agents receive **neutral projection / typed capability ports / immutable execution scope** — not full Nexus `RuntimeContext`. Materialization owned by EE (`AgentRuntimeContextMaterializer` lives under `runtime/execution`). |
| Result | Canonical public: **`AgentExecutionResult`** and/or **`AgentRunResult`**. Nexus `RuntimeAnswer` stays internal; mapping remains EE-owned (`runtime_answer_mapping` target: EE-internal). |
| Adapter owner | `intergrax/runtime/execution/**` (EE). |

### D4 — Context provider runtime dependency

`ContextProviderContext.runtime: Any | None` is **rejected** as final design.

Observed production need (builtin providers): **`base_messages`** only.

Target (R5 Wave 2/5 contract change — designed here, not implemented in ADR2):

- Replace `Any` with a **minimal typed Protocol** owned by Context/CE contracts (or a tiny EE-facing assembly port re-exported into CE without Nexus types), exposing only required assembly fields (at minimum read-only `base_messages: Sequence[ChatMessage]`), **or** move `base_messages` into `ContextProviderContext.sources` and **remove** `runtime`.
- No god-object “ContextExecutionRuntimeDependencies”.
- No `object` / `dict[str, Any]` for this known semantic.

### D5 — Tool invocation pattern public API

Canonical public extension surface (already exists — **freeze as sole public ABI**):

```text
intergrax.tools.invocation_pattern
  ToolInvocationPattern
  ToolInvocationPatternContext
  ToolInvocationInvokerPort
  ToolInvocationPlannerPort
  ToolInvocationPatternResult
```

Nexus may adapt these internally (`runtime/nexus/tools/**`). Author guides and examples **must not** import `intergrax.runtime.nexus.*`. Criterion for R5 Wave 1 exit.

### D6–D10 — Domain extension boundaries

| Domain | Ownership | Public contract direction | Nexus visibility |
|--------|-----------|---------------------------|------------------|
| Tools | Tools domain | `intergrax.tools.*` contracts | Internal EE adaptation only |
| RAG | RAG domain | retriever/reranker/search contracts | EE adapter maps into Nexus |
| LLM adapters | LLM domain | model/provider contracts | No Nexus RuntimeConfig/RunBudget in adapter code |
| Integrations | Integrations domain | provider/domain contracts | No Nexus persistence/session imports |
| WebSearch | Capability/provider | search contracts | Not Nexus implementation |

### D11 — Host composition object

Reuse existing:

- `ApplicationEnvironmentProfile`
- `ApplicationBuildContext` (where already composition-facing)
- **`HostTaskExecutionPort`** as canonical run-time execution entry
- `ExecutionBoundary` / `ExecutionRuntime` as EE coordination — not new public facades

Do **not** create `ExecutionEngineFacade` without proven gap.

### D12 — Compatibility policy

| Path / surface | Fate |
|----------------|------|
| `intergrax/agents/agent_runtime_context_materializer.py` (compat re-export) | **KEEP TEMPORARILY** · **DEPRECATE** · remove before HARNESS-01 final |
| `intergrax/runtime/nexus/context/context_budget.py` (re-export of contracts) | **KEEP TEMPORARILY** · **DEPRECATE** · remove before final; callers → `intergrax.contracts.context_budget` |
| `intergrax/runtime/nexus/execution/evaluator_loop_*.py` (legacy loop surfaces) | **DEPRECATE** · internal-only · remove or fully internalize before final; no public docs |
| `build_nexus_loop_from_environment` name / `NexusLoop` return | **DEPRECATE** · Wave 6 replace with execution-semantic factory/return |
| Public guide Nexus imports | **REMOVE IN R5** Wave 1 |
| Any new public Nexus re-export | **FORBIDDEN** |

Compat imports: temporary, internal-only, undocumented as public ABI, **removed before HARNESS-01 final closure**. Final closure **must not** leave public compat Nexus paths.

### D13 — Owner-zone gate (final qualification)

Prefer zone rules over path-by-path allowlists.

Final importer classes:

| Class | Meaning |
|-------|---------|
| `EE_INTERNAL` | Path under declared EE owner zones (`execution/**`, `nexus/**`) |
| `TEST_ONLY` | `tests/**` / explicit test support — never production ABI |
| `VIOLATION` | Everything else importing Nexus |

Fail-closed:

```text
new external Nexus importer → FAIL
unknown external importer → FAIL
```

**`DEBT` is not an acceptable final state.** R4/R5 transitional inventory may keep temporary classifications only until Wave 8 replaces them with the owner-zone gate.

### D14 — Indirect / transitive Nexus exposure

Direct import count = 0 is necessary but not sufficient.

```text
public API cannot expose Nexus types transitively
```

Forbidden example: `agents.foo` → public `InternalProtocol` → `RuntimeContext` annotation/re-export.

Require a **public signature gate** on public extension surfaces: annotations, imports, and re-exports must be Nexus-free.

### D15 — Pluginability

Each extension point: contract-defined, replaceable, runtime-selectable where appropriate, core-patch-free, **Nexus-free**.

Platform plugins coordinate discovery/package identity/admission/compatibility/lifecycle — they do **not** seize Tools/RAG/Memory/Execution semantics. No universal plugin megacontract.

### D16 — Failure semantics

Boundary adapters expose **typed** public errors/results. Raw Nexus exceptions must not cross public API.

### D17 — Lifecycle (HARNESS-02 intact)

Public boundary preserves: run identity, attempt identity, deadline, cancellation, recovery, checkpoint, governance, evidence — **without reminting authority**.

- Execution Engine = sole execution authority
- Nexus does not mint root identity
- Host does not mint a second authority
- Agent does not mint a second authority
- No second Execution Engine / host runtime / agent runtime authority

### D18 — Observability

Nexus internal telemetry may map to canonical `RuntimeEvent` / Evidence / execution reconstruction via EE adapters. Observability consumers must not require public Nexus types.

### D19 — Testability

External custom implementations (Agent, Tool, ToolInvocationPattern, ContextProvider, RAG, LLM, Integration) must be unit-testable **without** importing `intergrax.runtime.nexus.*`.

### D20 — Migration strategy (R5 resume)

Incremental waves — no big-bang rewrite. Each wave: typed contracts, tests green, no bypass, no new public Nexus.

| Wave | Scope | Preconditions | Changes | Qualification | Rollback constraints |
| ---- | ----- | ------------- | ------- | ------------- | -------------------- |
| 1 | Public extension surfaces / docs / examples / reference plugin | ADR2 Accepted | Remove Nexus imports from guides/examples; point to Tools/CE/RAG contracts | Doc + import gates | Docs-only revert safe |
| 2 | Agents / UAEP | Wave 1 green; request/result mapping design | Agent/UAEP → neutral contracts; EE adapter owns Nexus DTOs | Agent public surface Nexus-free; UAEP bridge tests | Keep temporary compat shims internal |
| 3 | Tools / websearch | Wave 1–2 contracts stable | Domain tools Nexus-free | Tools ABI gates | No Nexus reintroduction in tools |
| 4 | RAG / LLM / integrations | Domain contracts identified | Provider code off Nexus config/session | Domain import gates | Adapters stay EE-owned |
| 5 | Runtime non-EE | EE public ports sufficient | Strip Nexus from task/wiring/obs/hooks/human/long_running/… | Owner-zone scan | No widening EE zones without ADR |
| 6 | Host composition | EE factory design complete | `build_execution_engine`; host gets `HostTaskExecutionPort`; no `NexusLoop` escape | Host import + signature gates | Compatibility alias temporary only |
| 7 | Remove compatibility paths | Waves 1–6 consumers migrated | Delete deprecated re-exports/aliases | Compat path absence gate | No public compat left |
| 8 | Final owner-zone gate | Zero external debt | Enforce EE_INTERNAL / TEST_ONLY / VIOLATION fail-closed | HARNESS-01 final metrics | Gate is authoritative |

Wave order may be adjusted only if dependency analysis proves a safer partial order; exit metrics do not change.

---

## Decision matrix

| Concern | Current | Decision | Owner | Public contract | Nexus visibility |
| ------- | ------- | -------- | ----- | --------------- | ---------------- |
| Host entry | `build_nexus_loop_from_environment` → `NexusLoop` | Composition → `build_execution_engine`; run → `HostTaskExecutionPort` | EE composition + host | `HostTaskExecutionPort` | Private construct only |
| Agent execution | `RuntimeRequest`/`Context`/`Answer` in agents | Neutral `AgentRunRequest`/`AgentRunResult`/`AgentExecutionResult` + EE adapter | Agent domain + EE adapter | `intergrax.contracts.agent_*` | Internal only |
| Context runtime deps | `runtime: Any` | Minimal typed port or field removal | CE (+ EE assembly) | CE contracts | None |
| Tool invocation pattern | Guide imports Nexus | `intergrax.tools.invocation_pattern` only | Tools | TIP protocols | Internal adapter |
| RAG | Mixed Nexus touches | Retriever/reranker contracts | RAG | RAG contracts | EE map |
| LLM provider | Config/budget leakage | Model/provider contracts | LLM adapters | Adapter contracts | EE composition |
| Integration | Persistence/session imports | Provider contracts | Integrations | Integration contracts | None |
| Runtime task | Nexus imports | Consumer of EE; Nexus-free final | Task substrate | Task contracts | None final |
| Runtime wiring | Nexus imports | Calls EE factories only | Platform wiring | None public Nexus | Composition-time EE only |
| Observability | Possible Nexus types | `RuntimeEvent`/Evidence via EE | Observability | Event contracts | Mapped |
| Debug tooling | Nexus imports | Production-facing → EE; maintainer debug = INTERNAL-ONLY explicit class | Tooling | EE contracts | Restricted |

---

## Public surface matrix

| Surface | Canonical contract | Custom implementation possible? | Nexus-free? |
| ------- | ------------------ | ------------------------------: | ----------: |
| Host task execution | `HostTaskExecutionPort` | Host binds EE composition | Required |
| Agent run | `AgentRunRequest` / `AgentRunResult` / `AgentExecutionResult` | Yes (Agent authoring) | Required |
| Tool | `ToolPlugin` / `ToolContract` | Yes | Required |
| Tool invocation pattern | `ToolInvocationPattern` (+ context/ports/result) | Yes | Required |
| Context provider | CE provider protocols + typed assembly deps | Yes | Required |
| RAG provider | retriever/reranker contracts | Yes | Required |
| LLM adapter | LLM adapter contracts | Yes | Required |
| Integration | Integration provider contracts | Yes | Required |
| Platform plugin package | Platform plugin admission/lifecycle | Yes | Required |
| NexusLoop / RuntimeContext / RuntimeAnswer | — | — | **Not public** |

---

## Hard contract rules

Platform public/extension surfaces operate on: Protocol, ABC, immutable typed dataclass, typed result/error contracts, domain service definitions — **not** concrete Nexus types.

Forbidden escape hatches: `Any` for known semantics; `object` for known semantics; reflection (`getattr`/`setattr`/`hasattr`) as contract substitute; service locator; global registry as coupling escape; private-field cross-layer coupling; test-only production hooks; second EE/host/agent authority.

---

## Rejected alternatives

| ID | Alternative | Why rejected |
| -- | ----------- | ------------ |
| A | Expose Nexus as public API | Violates UEA private orchestration invariant |
| B | Create `PublicNexusFacade` / `NexusService` / `NexusRuntimeAPI` | Legalizes public Nexus under a new name |
| C | Allow host-wide Nexus imports permanently | Host ≠ EE; leaks implementation to applications |
| D | Allow agents/tools direct Nexus imports | Agent/Tool ≠ Execution; breaks plugin author golden path |
| E | Hide Nexus under `Any`/`object` | Untyped coupling; fails enterprise audit |
| F | Path-by-path permanent allowlist (~190 files) | Unmaintainable; not fail-closed |
| G | Duplicate Nexus DTOs into `contracts` (`PublicRuntimeRequest`…) | Second schema; drift; still Nexus-shaped |
| H | Make all `runtime/*` part of EE | Repeats R4 error; blurs ownership |

---

## Consequences

### Positive

- Unambiguous EE boundary; Nexus encapsulation enforceable by owner-zone gates.
- Clear reuse of `HostTaskExecutionPort` / Agent / Tools contracts.
- Incremental R5 migration path with exit metrics.
- Public author experience: implement extensions without knowing Nexus exists.

### Negative / costs

- Large migration (168+ external importers) across multiple waves.
- Temporary dual naming (`build_nexus_loop_*` alias) during Wave 6.
- Stricter composition discipline for `_shared` and hosts.

### Neutral

- Does not change frozen lifecycle, deadline, or cancellation ownership (HARNESS-02).
- Does not reopen TR-01 / SESSION-01 / CE-02 semantic freezes; only dependency direction into Nexus.

---

## Qualification requirements

Final HARNESS-01 metrics (frozen by this ADR):

```text
external production Nexus importers = 0
public Nexus imports = 0
public Nexus re-exports = 0
public Nexus annotations = 0
external Nexus debt = 0
unknown external importer = FAIL
```

Internal metric: Nexus direct imports allowed **only** inside explicitly declared EE owner zones.

Gate plan:

1. Owner-zone enforcement (fail-closed EE_INTERNAL / TEST_ONLY / VIOLATION)
2. No public Nexus type / re-export / annotation (signature gate)
3. No `Any` masking for known CE/runtime seams
4. Negative synthetic cases (attempted external import must fail CI)
5. Plugin author proof (reference plugin + guide Nexus-free)

Public author golden path (exit criterion):

```text
external developer can implement Agent, Tool, ToolInvocationPattern,
ContextProvider, RAG provider, LLM adapter, Integration
without importing intergrax.runtime.nexus.*
```

DeepSeek-level / Integrax enterprise checks supported by this model: contract-driven composition, replaceable components, pluginability, zero bypass, lifecycle safety, durability, observable execution, governance, tenant isolation, deterministic authority, evidence, recovery, version pinning, fail-closed behavior.

---

## Exit criteria

- ADR Accepted and indexed.
- Architecture hubs point here for Nexus encapsulation.
- Doc regression gate asserts ADR semantics.
- HARNESS-01-R5 may resume **under this ADR**; HARNESS-01 is **not** CLOSED until Waves 1–8 complete and final metrics pass.

Architecture decision required by R5:

```text
Resolved by HARNESS-01-ADR2 / ADR-HARNESS-001.
```

---

## Compliance

- Tier boundaries preserved (`intergrax` ↛ `agents`/`applications` as today; Nexus not a cross-tier public ABI)
- HARNESS-02 lifecycle authority unchanged
- Linked UEA + Execution Engine maintainer hub updated
- Production code unchanged in ADR2 package

## Implementation notes

- Resume: **HARNESS-01-R5** under ADR2 waves.
- Do not implement `build_execution_engine` in ADR2.
- Doc gate: `tests/unit/runtime/architecture/test_harness_01_adr2_documentation_regression_gates.py`
- ADR layout check: `python scripts/maintenance/check_harness_adr.py`
