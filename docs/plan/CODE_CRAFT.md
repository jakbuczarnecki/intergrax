# Ephemeral Code Craft — Implementation Plan

**Architecture (1:1):** [`architecture/CODE_CRAFT.md`](../architecture/CODE_CRAFT.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)  
**ADR:** [`adr/entries/2026-06-10/ADR-CODECRAFT-001.md`](../adr/entries/2026-06-10/ADR-CODECRAFT-001.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

**Status:** **ECC-0 Done** · **ECC-1 Done** (2026-06-13) · ECC-2 **Planned**  
**Default queue:** Pull ECC-1 after operator selects this domain; otherwise §6.1 gate maintenance continues in [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md).

---

## Delivery rules

1. **One ECC phase per PR** (or one cohesive sub-slice within a phase) → gate green → update this plan row.
2. **Contract first** — Pydantic models + Protocol before orchestrator wiring.
3. **Trace** — every state transition emits `CODECRAFT_*` (+ `RuntimeEvent` / `TraceEvent` where wired).
4. **Tests** — unit + integration; deterministic; no network in gate tests (mock sandbox).
5. **Reuse Tier-0** — extend sandbox, ToolRuntime, CVL; no parallel exec stacks.
6. **Fail closed** — deny paths must have policy tests.
7. **No product scope creep** — ECC harness only; no K.1/K.2 agents without §6.3 decision.

---

## Phase ECC-0 — Architecture canon (Done)

| ID | Deliverable | Status | Date |
|----|-------------|--------|------|
| ECC-0.1 | ADR-CODECRAFT-001 — separate domain decision | **Done** | 2026-06-10 |
| ECC-0.2 | `architecture/CODE_CRAFT.md` | **Done** | 2026-06-10 |
| ECC-0.3 | `plan/CODE_CRAFT.md` + full audit register (this doc §Audit) | **Done** | 2026-06-10 |
| ECC-0.4 | Hub + README + AGENTS.md + audit map routing | **Done** | 2026-06-10 |

---

# Audit Result: Ephemeral Code Craft (ECC)

**Audit date:** 2026-06-10  
**Method:** Vision vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` §3.6 · `AUDIT-IDEAL-11.1` · code `runtime/sandbox/` · `tools/providers/sandbox/` · skills `sandbox.*` · UAEP §42.12  
**Verdict:** Execution **substrate Done** (~40% of target); **harness orchestration Missing** — new domain **CODE_CRAFT** required.

---

## Audit §1 — Scope

What was audited:

- Operator requirement: dynamic code generation in isolated environment, configurable rules, iterative test/fix loop, observability, control — when catalog tools are insufficient.
- Intergrax Tier-0 sandbox tools (`sandbox.exec`, `code.exec`, `script.run`, `browser.run`).
- Tier-1 `SandboxSession`, `HostedSandboxSession`, `SandboxSessionManager`.
- Skills `sandbox.code_exec`, `sandbox.test_runner`, `sandbox.refactor_loop`.
- Policy: `ToolAccessPolicy`, `SANDBOX_REQUIRED_TOOLS`, UAEP `BoundToolGateway`.
- Cross-domain: CVL evaluator-loop, HITL, observability spine, cloud `sandbox_host` integrations.
- Ideal target: sandboxed execution for code and side-effectful actions (`IDEAL` §3.6).

Out of scope: implementation code (ECC-1+), product-specific agents (§6.3).

---

## Audit §2 — Target state (ideal architecture)

| Requirement | Ideal / operator target |
|-------------|-------------------------|
| Dynamic synthesis | Agent may generate helper code not in tool catalog |
| Isolation | OS-grade or cloud sandbox for untrusted codegen |
| Configurability | Host profile: languages, imports, modes, limits, egress |
| Iteration | generate → static check → exec → test → fix → repeat |
| Promotion | Typed result back to pipeline — not raw stdout only |
| Ephemeral semantics | Helpers die with task; no catalog pollution |
| Observability | Full trace of generation, gates, exec, verdict |
| Control | Policy, budgets, HITL, fail-closed |
| Tier discipline | Harness orchestrates; agents invoke tools only |

---

## Audit §3 — Current state

### §3.1 Done (substrate)

| ID | Item | Evidence |
|----|------|----------|
| CUR-01 | Local `SandboxSession` | `intergrax/runtime/sandbox/session.py` — allowlisted ops, audit log, cleanup |
| CUR-02 | Operations | `echo`, `write_file`, `read_file`, `list_files`, `run_python`, `run_script`, `browser_fetch` |
| CUR-03 | `AGENT_BUILDER_SANDBOX_OPERATIONS` | `sandbox_runtime.py` |
| CUR-04 | Catalog tools | `sandbox.exec`, `code.exec`, `script.run`, `browser.run`, `sandbox.list_operations` |
| CUR-05 | Tool providers | `intergrax/tools/providers/sandbox/` |
| CUR-06 | Cloud bridge | `HostedSandboxSession` + `sandbox_host` (e2b, modal, daytona) |
| CUR-07 | Session manager | `SandboxSessionManager` per tenant/task |
| CUR-08 | Task lifecycle cleanup | `cleanup_sandbox_for_task` in task finisher |
| CUR-09 | UAEP routing | `BoundToolGateway._invoke_sandbox` |
| CUR-10 | Policy constants | `SANDBOX_REQUIRED_TOOLS`, `requires_sandbox_tool()` |
| CUR-11 | Skills composition | `sandbox.code_exec`, `sandbox.test_runner`, `sandbox.refactor_loop` manifests |
| CUR-12 | Tier-3 wiring | `wire_sandbox_sessions`, `tool_profile_with_sandbox` |
| CUR-13 | AUDIT-IDEAL-11.1 | Sandboxed execution for side-effectful tools — **Done** in `AUDIT_IDEAL_2026.md` |
| CUR-14 | Security scan tool | `security.scan` via `security_scanner` integration (M.6 P6) |
| CUR-15 | Shadow workspace | Separate artifact isolation (`runtime/shadow/`) |

### §3.2 Partial

| ID | Item | Gap |
|----|------|-----|
| PAR-01 | `sandbox.refactor_loop` skill | Manifest only — no harness executor |
| PAR-02 | Evaluator-loop (CVL) | Pattern exists for graphs — not wired to code craft |
| PAR-03 | Local sandbox security | Workspace isolation only — not production-grade containment |
| PAR-04 | Trace | `TOOL_*` + `SandboxAuditEntry` — no `CODECRAFT_*` span |
| PAR-05 | Documentation | Sandbox split across TOOLS, RELIABILITY, PLATFORM_FOUNDATION §7.4.9 |

### §3.3 Missing (ECC target)

| ID | Item |
|----|------|
| MIS-01 | `CodeCraftOrchestrator` |
| MIS-02 | `CodeCraftSession` / `craft_id` lifecycle |
| MIS-03 | `CodeCraftProfile` on ApplicationEnvironmentProfile |
| MIS-04 | Craft modes (`disabled` … `autonomous`) |
| MIS-05 | `StaticCodeGate` (AST, imports, size, secrets) |
| MIS-06 | `codecraft.*` catalog tools |
| MIS-07 | Iteration API (`start`, `iterate`, `dispose`) |
| MIS-08 | `CraftTestRunner` integrated in loop |
| MIS-09 | `CraftResult` promotion contract |
| MIS-10 | `EphemeralToolRegistry` (task-scoped) |
| MIS-11 | Separate codegen LLM profile |
| MIS-12 | HITL gate before exec (supervised mode) |
| MIS-13 | `CODECRAFT_*` observability events |
| MIS-14 | Optional `CodeCraftNode` in execution graph |
| MIS-15 | AHI trigger for when to invoke craft |
| MIS-16 | Dedicated domain pair documentation — **addressed ECC-0** |

---

## Audit §4 — Gap list (consolidated)

| GAP-ID | Category | Description | Maps to |
|--------|----------|-------------|---------|
| GAP-ECC-01 | orchestration | No harness generate→test→fix loop | ECC-2 |
| GAP-ECC-02 | profile | No `CodeCraftProfile` | ECC-3 |
| GAP-ECC-03 | tools | No `codecraft.*` surface | ECC-1, ECC-2 |
| GAP-ECC-04 | security | No static code gate before exec | ECC-1 |
| GAP-ECC-05 | security | Local sandbox overstated as full isolation | ECC-4 |
| GAP-ECC-06 | semantics | No ephemeral tool registry | ECC-5 |
| GAP-ECC-07 | I/O | No typed promotion (`CraftResult`) | ECC-3 |
| GAP-ECC-08 | control | No craft-specific HITL path | ECC-3 |
| GAP-ECC-09 | observability | No `CODECRAFT_*` events | ECC-1 |
| GAP-ECC-10 | integration | CVL not specialized for code iterations | ECC-2, ECC-3 |
| GAP-ECC-11 | skills | `refactor_loop` without executor | ECC-2 |
| GAP-ECC-12 | graph | No `CodeCraftNode` | ECC-5 |
| GAP-ECC-13 | L4 | No adaptive craft trigger | ECC-6 |
| GAP-ECC-14 | docs | No canonical domain — **Closed ECC-0** | ECC-0 |
| GAP-ECC-15 | tier risk | Agents may implement own loops | ECC-1+ policy + docs |

**Coverage:** 15 gaps — 1 closed (ECC-0); 14 open → ECC-1…ECC-6.

---

## Audit §5 — Risk assessment

| Risk | Level | Mitigation phase |
|------|-------|------------------|
| RCE via generated code | **Critical** | ECC-4 cloud/container + ECC-1 static gate + ECC-3 supervised |
| Data exfiltration via network | **High** | ECC-3 egress policy + cloud sandbox |
| Unbounded token/iteration cost | **High** | ECC-2 budgets + `max_iterations` |
| Catalog pollution (ephemeral as permanent tools) | **Medium** | ECC-5 ephemeral registry design |
| Tier boundary erosion (agent subprocess) | **Medium** | ECC-1 UAEP-only path + gate tests |
| False Done on AUDIT-IDEAL-11.1 | **Medium** | This audit + layer 11b in hub |
| Operational complexity | **Medium** | Presets: `harness_codecraft_stack` (ECC-4) |
| Local sandbox in production | **High** | Document + profile default `cloud` for regulated hosts |

---

## Audit §6 — Architecture updates required

| Update | Status |
|--------|--------|
| New pair `CODE_CRAFT.md` | **Done** ECC-0 |
| ADR-CODECRAFT-001 | **Done** ECC-0 |
| Hub 21st domain + layer 11b | **Done** ECC-0 |
| `TOOLS.md` cross-link (not ownership) | **Done** ECC-0 |
| `PLATFORM_FOUNDATION.md` Tier-0 index row | **Done** ECC-0 |
| `INTEGRAX_HARNESS_AUDIT_MAP.md` layer 11b note | Optional follow-up |
| `AUDIT_IDEAL_2026.md` row AUDIT-IDEAL-11.4 | Optional follow-up |

---

## Audit §7 — Comparison with reference systems

| System | Pattern | Intergrax mapping |
|--------|---------|-------------------|
| OpenAI Code Interpreter | Session + promote result | `CodeCraftSession` + `CraftResult` |
| E2B / Modal | VM/container per session | `HostedSandboxSession` (Done) |
| Cursor / Devin | Plan→code→test→fix trace | `CodeCraftOrchestrator` (Missing) |
| LangGraph code nodes | Graph checkpoint | `CodeCraftNode` ECC-5 |
| CVL Evaluator-loop | critique→revise | Reuse — ECC specialization |

---

## Audit §8 — Domain placement decision

| Option | Verdict |
|--------|---------|
| TOOLS only | **Rejected** — atomic catalog ≠ orchestration engine |
| SKILLS only | **Rejected** — no runtime loop |
| RELIABILITY only | **Rejected** — sandbox is substrate |
| **CODE_CRAFT domain** | **Accepted** — ADR-CODECRAFT-001 |

Precedent: RAG (`intergrax/rag/` + `rag.*` tools), CVL (orchestrator + `eval.*` tools).

---

## Audit §9 — Definition of Done (domain)

ECC domain **implementation complete** (ECC-1…ECC-4 minimum) when:

1. `codecraft.run` and `codecraft.start/iterate/dispose` pass gate tests with mocked sandbox.
2. `StaticCodeGate` blocks forbidden imports in autonomous path.
3. `CodeCraftProfile` wired on lab host with `supervised` and `autonomous` presets.
4. `CODECRAFT_*` events appear in task trace.
5. Promotion returns typed `CraftResult` validated against schema.
6. Cloud `sandbox_host` path tested integration (optional network marker).
7. No Tier-2 reference implementation of craft loops in harness agents (lab uses ECC only).
8. Architecture + plan rows updated; ADR consequences satisfied.

ECC-5/ECC-6 are **depth** — not required for initial L2→L3 closeout.

---

## Audit §10 — Suggested implementation order

```text
ECC-0 (Done) → ECC-1 → ECC-2 → ECC-3 → ECC-4 → ECC-5 → ECC-6
```

---

# Implementation phases ECC-1 … ECC-6

## Phase ECC-1 — Single-shot craft (`codecraft.run`)

**Status:** **Done** (2026-06-13)

**Goal:** Smallest coherent slice — generate (or accept) code, static gate, sandbox exec, trace, deny paths.

| ID | Deliverable | Module | Acceptance |
|----|-------------|--------|------------|
| ECC-1.1 | Package scaffold `intergrax/codecraft/` contracts | `contracts.py` | `CodeCraftRunInput`, `CraftResult`, `StaticGateResult` |
| ECC-1.2 | `StaticCodeGate` | `static_gate.py` | Unit: blocks `import os`, oversize, forbidden patterns |
| ECC-1.3 | Tool `codecraft.run` | `tools/providers/codecraft/` | Invokes gate + `code.exec` via ToolWiringContext |
| ECC-1.4 | `CodeCraftTraceEmitter` stub | `runtime/codecraft/trace.py` | Emits `CODECRAFT_SESSION_OPENED`, `STATIC_GATE`, `EXEC`, `DISPOSED` |
| ECC-1.5 | Policy deny tests | tests | No sandbox session → DENIED; `mode=disabled` → DENIED |
| ECC-1.6 | Register tool in catalog bootstrap | `register_default_tools()` | `codecraft.run` in catalog |

**Closes:** GAP-ECC-03 (partial), GAP-ECC-04, GAP-ECC-09 (partial), GAP-ECC-15 (partial).

**ADR:** No new ADR — extends ADR-CODECRAFT-001.

---

## Phase ECC-2 — Session iteration loop

**Goal:** `codecraft.start`, `codecraft.iterate`, `codecraft.get_state`, `codecraft.dispose` + orchestrator.

| ID | Deliverable | Module | Acceptance |
|----|-------------|--------|------------|
| ECC-2.1 | `CodeCraftSessionManager` | `runtime/codecraft/session_manager.py` | craft_id per task; dispose cleanup |
| ECC-2.2 | `CodeCraftOrchestrator` | `runtime/codecraft/orchestrator.py` | Multi-iteration loop with budget |
| ECC-2.3 | `CodeGenerationAdapter` | `codecraft/codegen_adapter.py` | Separate LLM profile ref |
| ECC-2.4 | `CraftTestRunner` | `codecraft/test_runner.py` | Run pytest template in sandbox |
| ECC-2.5 | Tools start/iterate/get_state/dispose | `tools/providers/codecraft/` | E2E test with mock LLM + sandbox |
| ECC-2.6 | CVL L0 hook after iteration | bridge to `CriticOrchestrator` | Failed test → revise verdict |
| ECC-2.7 | Skill `codecraft.ephemeral_builder` | `skills/providers/codecraft/` | Bundles codecraft.* + workspace |

**Closes:** GAP-ECC-01, GAP-ECC-03, GAP-ECC-10 (partial), GAP-ECC-11.

---

## Phase ECC-3 — Modes, HITL, promotion

**Goal:** `CodeCraftProfile`, supervised/autonomous, `CraftResultPromoter`, `codecraft.promote`.

| ID | Deliverable | Module | Acceptance |
|----|-------------|--------|------------|
| ECC-3.1 | `CodeCraftProfile` model | `applications/contracts/` or `codecraft/profile.py` | Typed fields per architecture §6.2 |
| ECC-3.2 | `wire_application_codecraft()` | `applications/_shared/codecraft_wiring.py` | Lab + poc_template presets |
| ECC-3.3 | Mode enforcement | orchestrator | `dry_run` never exec; `assist_only` returns code only |
| ECC-3.4 | HITL before exec | PolicyEngine + HitlRunner | `supervised` pauses until resume |
| ECC-3.5 | `CraftResultPromoter` | `codecraft/promoter.py` | Pydantic promotion schema L0 |
| ECC-3.6 | Tool `codecraft.promote` | tools | Supervised explicit promotion |
| ECC-3.7 | Policy fragment `codecraft_governance` | UAEP bundle | Trace checklist in AGENT_CREATION_GUIDE |

**Closes:** GAP-ECC-02, GAP-ECC-07, GAP-ECC-08, GAP-ECC-10 (complete).

---

## Phase ECC-4 — Production isolation

**Goal:** Regulated hosts default cloud sandbox; optional `security.scan` pre-exec.

| ID | Deliverable | Module | Acceptance |
|----|-------------|--------|------------|
| ECC-4.1 | `isolation_tier` routing | orchestrator | `cloud` → `HostedSandboxSession` |
| ECC-4.2 | `security_scan_before_exec` | pre-exec hook | `security.scan` on written files |
| ECC-4.3 | Network egress policy | profile + sandbox bridge | deny-by-default documented |
| ECC-4.4 | Preset `harness_codecraft_stack()` | integration presets | e2b + semgrep optional |
| ECC-4.5 | HARNESS_ENVIRONMENT probe extension | health check | codecraft capability probe |

**Closes:** GAP-ECC-05.

---

## Phase ECC-5 — Ephemeral tools + graph node

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| ECC-5.1 | `EphemeralToolRegistry` | Tools visible only for `craft_id` |
| ECC-5.2 | `codecraft.list_ephemeral_tools` | Introspection tool |
| ECC-5.3 | `CodeCraftNode` spec | `NEXUS_EXECUTION_FLOW` graph pattern |
| ECC-5.4 | Graph executor bridge | Optional node in lab graph spec |

**Closes:** GAP-ECC-06, GAP-ECC-12.

---

## Phase ECC-6 — Adaptive trigger (L4)

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| ECC-6.1 | AHI signal: catalog tool miss | Suggest or auto-invoke craft when policy allows |
| ECC-6.2 | Budget-aware craft decision | No craft when budget exhausted |

**Closes:** GAP-ECC-13.

---

# Master register

| Phase | Status | Priority | Closes gaps |
|-------|--------|----------|-------------|
| ECC-0 | **Done** | P0 | GAP-ECC-14 |
| ECC-1 | **Done** | P0 | 03, 04, 09, 15 |
| ECC-2 | **Planned** | P0 | 01, 03, 10, 11 |
| ECC-3 | **Planned** | P1 | 02, 07, 08, 10 |
| ECC-4 | **Planned** | P1 | 05 |
| ECC-5 | **Planned** | P2 | 06, 12 |
| ECC-6 | **Planned** | P2 | 13 |

---

# Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-10 | ECC-0 | Domain pair CODE_CRAFT, ADR-CODECRAFT-001, full audit register, hub/README/AGENTS sync |
| 2026-06-13 | LAYER-AUDIT | Layer completion audit — zero implementation code; substrate ~40% Done; sprint plan §Sprints added; ECC-1 started |
| 2026-06-13 | ECC-1 | `codecraft.run`, StaticCodeGate, CodeCraftTraceEmitter, catalog plugin, gate tests |

---

# Layer completion audit (2026-06-13)

**Method:** Re-read domain pair + ADR + grep `intergrax/codecraft/`, `intergrax/runtime/codecraft/`, `intergrax/tools/providers/codecraft/` — **no Python modules exist** (confirmed).

**Verdict:** Documentation canon **Done** (ECC-0). Harness orchestration **Missing**. Substrate reuse path **validated** — `SandboxSession`, `code.exec`, `security.scan`, CVL `eval.judge`, `sandbox_host` integrations present.

**Doc inconsistencies closed this iteration:**

| Issue | Resolution |
|-------|------------|
| README layout lists `codecraft/` without status | README updated — `(ECC-1+ in progress)` |
| No file-level module map | Added architecture §16 |
| Sprint breakdown absent | Added §Sprints below |
| `TraceComponent` for CODECRAFT unspecified | Documented as `TraceComponent.CODECRAFT` diagnostic steps (ECC-1) |

**Open gaps unchanged:** GAP-ECC-01…13 (14 closed ECC-0 only).

---

# Sprints (layer completion)

Each sprint = one PR-sized slice → gate green → plan row update → commit.

## Sprint S1 — ECC-1 Single-shot craft

| Field | Value |
|-------|-------|
| **Scope** | `codecraft.run`, `StaticCodeGate`, `CodeCraftTraceEmitter`, catalog registration |
| **Goal** | Smallest governed path: accept code → L0 gate → sandbox exec → typed `CraftResult` |
| **DoD** | Unit tests for gate + deny paths; tool in catalog; `CODECRAFT_*` trace steps; gate tests green |
| **Files** | `intergrax/codecraft/{contracts,profile,static_gate}.py` · `intergrax/runtime/codecraft/trace.py` · `intergrax/tools/providers/codecraft/*` · `intergrax/tools/registry/shipped_plugins.py` · `intergrax/runtime/sandbox/sandbox_runtime.py` · `tests/unit/codecraft/` · `tests/unit/tools/providers/codecraft/` |

## Sprint S2 — ECC-2 Session loop

| Field | Value |
|-------|-------|
| **Scope** | `CodeCraftOrchestrator`, session manager, `start/iterate/get_state/dispose`, codegen adapter, test runner |
| **Goal** | Multi-iteration generate→gate→exec→test loop with budgets |
| **DoD** | E2E mock test; CVL L0 hook; skill `codecraft.ephemeral_builder` manifest |
| **Files** | `intergrax/runtime/codecraft/{orchestrator,session_manager}.py` · `intergrax/codecraft/{codegen_adapter,test_runner}.py` · extended `tools/providers/codecraft/` · `intergrax/skills/providers/codecraft/` |

## Sprint S3 — ECC-3 Modes + HITL + promotion

| Field | Value |
|-------|-------|
| **Scope** | `CodeCraftProfile` on `ApplicationEnvironmentProfile`, wiring, modes, HITL, `CraftResultPromoter` |
| **Goal** | Tier-3 profile drives mode enforcement; supervised HITL; typed promotion |
| **DoD** | Lab preset wired; mode matrix tests; `codecraft.promote` tool |
| **Files** | `intergrax/applications/contracts/environment_profile.py` · `intergrax/applications/_shared/codecraft_wiring.py` · `intergrax/codecraft/promoter.py` · UAEP policy fragment |

## Sprint S4 — ECC-4 Production isolation

| Field | Value |
|-------|-------|
| **Scope** | Cloud sandbox routing, `security.scan` pre-exec, egress policy, harness preset |
| **Goal** | Regulated hosts default `isolation_tier=cloud` |
| **DoD** | Integration test with mocked `HostedSandboxSession`; health probe extension |
| **Files** | orchestrator isolation routing · `applications/_shared` presets · health probes |

## Sprint S5 — ECC-5 Ephemeral registry + graph

| Field | Value |
|-------|-------|
| **Scope** | `EphemeralToolRegistry`, `codecraft.list_ephemeral_tools`, optional `CodeCraftNode` |
| **Goal** | Task-scoped tools never pollute global catalog |
| **DoD** | Registry isolation tests; graph spec example in lab |
| **Files** | `intergrax/runtime/codecraft/ephemeral_registry.py` · NEXUS_EXECUTION_FLOW cross-ref |

## Sprint S6 — ECC-6 Adaptive trigger

| Field | Value |
|-------|-------|
| **Scope** | AHI catalog-miss signal, budget-aware craft decision |
| **Goal** | L4 suggests or invokes craft when policy allows |
| **DoD** | Signal tests in adaptive domain integration |
| **Files** | `intergrax/runtime/adaptive/` hooks · ADAPTIVE_HARNESS_INTELLIGENCE plan row |

**Minimum L3 closeout:** S1–S4 (ECC-1…ECC-4). S5–S6 are depth.

---

# Explicitly out of scope

- Phase K business agents using ECC (§6.3 product decision).
- Container runtime implementation details before ECC-4 design spike.
- Replacing or removing `code.exec` / `sandbox.exec` primitives.
- Monolithic `docs/INTERGRAX_IMPLEMENTATION_PLAN.md` or `plan/phases/` folders.
