# CODE_CRAFT — implementation history + LC closeout

**Parent hub:** [`CODE_CRAFT.md`](../CODE_CRAFT.md)

> **Plan ownership:** Implementation phases and LC closeout below. Historical audit findings/verdicts archived at [docs/audit_results/legacy/plan-audit-history/CODE_CRAFT_implementation_history.md](../../../../audit_results/legacy/plan-audit-history/CODE_CRAFT_implementation_history.md).


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
**Method:** Vision vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` §3.6 · `AUDIT-IDEAL-11.1` · code `runtime/sandbox` · `tools/providers/sandbox` · skills `sandbox.*` · UAEP §42.12
**Verdict:** Execution **substrate Done** (~40% of target at audit open); **harness orchestration Done** (ECC-1…ECC-6, 2026-06-13) — domain **CODE_CRAFT** closed at L3.

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
| CUR-05 | Tool providers | `intergrax/tools/providers/sandbox` |
| CUR-06 | Cloud bridge | `HostedSandboxSession` + `sandbox_host` (e2b, modal, daytona) |
| CUR-07 | Session manager | `SandboxSessionManager` per tenant/task |
| CUR-08 | Task lifecycle cleanup | `cleanup_sandbox_for_task` in task finisher |
| CUR-09 | UAEP routing | `BoundToolGateway._invoke_sandbox` |
| CUR-10 | Policy constants | `SANDBOX_REQUIRED_TOOLS`, `requires_sandbox_tool()` |
| CUR-11 | Skills composition | `sandbox.code_exec`, `sandbox.test_runner`, `sandbox.refactor_loop` manifests |
| CUR-12 | Tier-3 wiring | `wire_sandbox_sessions`, `tool_profile_with_sandbox` |
| CUR-13 | AUDIT-IDEAL-11.1 | Sandboxed execution for side-effectful tools — **Done** in `AUDIT_IDEAL_2026.md` |
| CUR-14 | Security scan tool | `security.scan` via `security_scanner` integration (M.6 P6) |
| CUR-15 | Shadow workspace | Separate artifact isolation (`runtime/shadow`) |

### §3.2 Partial / depth (post ECC-0…ECC-6)

| ID | Item | Status | Notes |
|----|------|--------|-------|
| PAR-01 | `sandbox.refactor_loop` skill | **Superseded** | Use `codecraft.ephemeral_builder` + orchestrator (ECC-2) |
| PAR-02 | Evaluator-loop (CVL) | **Done** | `runtime/codecraft/cv_bridge.py` (ECC-2) |
| PAR-03 | Local sandbox security | **Accepted limitation** | Documented — use `isolation_tier=cloud` for regulated hosts |
| PAR-04 | CODECRAFT trace taxonomy | **Done** | S8 — full §10.1 taxonomy |
| PAR-05 | Documentation scatter | **Mitigated** | Domain pair + audit map §11b; cross-links in TOOLS/RELIABILITY |

### §3.3 Missing at audit open — closed by ECC-1…ECC-6 (2026-06-13)

| ID | Item | Closed by |
|----|------|-----------|
| MIS-01 | `CodeCraftOrchestrator` | ECC-2 |
| MIS-02 | `CodeCraftSession` / `craft_id` lifecycle | ECC-2 |
| MIS-03 | `CodeCraftProfile` on ApplicationEnvironmentProfile | ECC-3 |
| MIS-04 | Craft modes (`disabled` … `autonomous`) | ECC-3 |
| MIS-05 | `StaticCodeGate` (AST, imports, size, secrets) | ECC-1 |
| MIS-06 | `codecraft.*` catalog tools | ECC-1…ECC-5 |
| MIS-07 | Iteration API (`start`, `iterate`, `dispose`) | ECC-2 |
| MIS-08 | `CraftTestRunner` integrated in loop | ECC-2 |
| MIS-09 | `CraftResult` promotion contract | ECC-3 |
| MIS-10 | `EphemeralToolRegistry` (task-scoped) | ECC-5 |
| MIS-11 | Separate codegen LLM profile | ECC-2 (Protocol + template adapter; `codegen_llm_profile_ref` wiring → GAP-ECC-20) |
| MIS-12 | HITL gate before exec (supervised mode) | ECC-3 |
| MIS-13 | `CODECRAFT_*` observability events | ECC-1+ |
| MIS-14 | Optional `CodeCraftNode` in execution graph | ECC-5 |
| MIS-15 | AHI trigger for when to invoke craft | ECC-6 |
| MIS-16 | Dedicated domain pair documentation — **addressed ECC-0** | ECC-0 |

---

## Audit §4 — Gap list (consolidated)

| GAP-ID | Category | Description | Maps to |
|--------|----------|-------------|---------|
| GAP-ECC-01 | orchestration | No harness generate→test→fix loop | **Done** ECC-2 |
| GAP-ECC-02 | profile | No `CodeCraftProfile` | **Done** ECC-3 |
| GAP-ECC-03 | tools | No `codecraft.*` surface | **Done** ECC-1, ECC-2 |
| GAP-ECC-04 | security | No static code gate before exec | **Done** ECC-1 |
| GAP-ECC-05 | security | Local sandbox overstated as full isolation | **Done** ECC-4 |
| GAP-ECC-06 | semantics | No ephemeral tool registry | **Done** ECC-5 |
| GAP-ECC-07 | I/O | No typed promotion (`CraftResult`) | **Done** ECC-3 |
| GAP-ECC-08 | control | No craft-specific HITL path | **Done** ECC-3 |
| GAP-ECC-09 | observability | No `CODECRAFT_*` events | **Done** ECC-1 |
| GAP-ECC-10 | integration | CVL not specialized for code iterations | **Done** ECC-2, ECC-3 |
| GAP-ECC-11 | skills | `refactor_loop` without executor | **Done** ECC-2 |
| GAP-ECC-12 | graph | No `CodeCraftNode` | **Done** ECC-5 |
| GAP-ECC-13 | L4 | No adaptive craft trigger | **Done** ECC-6 |
| GAP-ECC-14 | docs | No canonical domain | **Done** ECC-0 |
| GAP-ECC-15 | tier risk | Agents may implement own loops | **Done** ECC-1+ policy + docs |

**Coverage:** 15 gaps — **all closed** (ECC-0…ECC-6, 2026-06-13). Remaining depth: metrics dashboards (architecture §10.2), container isolation tier.

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
| `INTEGRAX_HARNESS_AUDIT_MAP.md` layer 11b note | **Done** ECC doc-sync |
| `AUDIT_IDEAL_2026.md` row AUDIT-IDEAL-11.4 | Optional follow-up |

---

## Audit §7 — Comparison with reference systems

| System | Pattern | Intergrax mapping |
|--------|---------|-------------------|
| OpenAI Code Interpreter | Session + promote result | `CodeCraftSession` + `CraftResult` |
| E2B / Modal | VM/container per session | `HostedSandboxSession` (Done) |
| Cursor / Devin | Plan→code→test→fix trace | `CodeCraftOrchestrator` **Done** (ECC-2) |
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

Precedent: RAG (`intergrax/rag` + `rag.*` tools), CVL (orchestrator + `eval.*` tools).

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
| ECC-1.1 | Package scaffold `intergrax/codecraft` contracts | `contracts.py` | `CodeCraftRunInput`, `CraftResult`, `StaticGateResult` |
| ECC-1.2 | `StaticCodeGate` | `static_gate.py` | Unit: blocks `import os`, oversize, forbidden patterns |
| ECC-1.3 | Tool `codecraft.run` | `tools/providers/codecraft` | Invokes gate + `code.exec` via ToolWiringContext |
| ECC-1.4 | `CodeCraftTraceEmitter` stub | `runtime/codecraft/trace.py` | Emits `CODECRAFT_SESSION_OPENED`, `STATIC_GATE`, `EXEC`, `DISPOSED` |
| ECC-1.5 | Policy deny tests | tests | No sandbox session → DENIED; `mode=disabled` → DENIED |
| ECC-1.6 | Register tool in catalog bootstrap | `register_default_tools()` | `codecraft.run` in catalog |

**Closes:** GAP-ECC-03 (partial), GAP-ECC-04, GAP-ECC-09 (partial), GAP-ECC-15 (partial).

**ADR:** No new ADR — extends ADR-CODECRAFT-001.

---

## Phase ECC-2 — Session iteration loop

**Status:** **Done** (2026-06-13)

**Goal:** `codecraft.start`, `codecraft.iterate`, `codecraft.get_state`, `codecraft.dispose` + orchestrator.

| ID | Deliverable | Module | Acceptance |
|----|-------------|--------|------------|
| ECC-2.1 | `CodeCraftSessionManager` | `runtime/codecraft/session_manager.py` | craft_id per task; dispose cleanup |
| ECC-2.2 | `CodeCraftOrchestrator` | `runtime/codecraft/orchestrator.py` | Multi-iteration loop with budget |
| ECC-2.3 | `CodeGenerationAdapter` | `codecraft/codegen_adapter.py` | Separate LLM profile ref |
| ECC-2.4 | `CraftTestRunner` | `codecraft/test_runner.py` | Run pytest template in sandbox |
| ECC-2.5 | Tools start/iterate/get_state/dispose | `tools/providers/codecraft` | E2E test with mock LLM + sandbox |
| ECC-2.6 | CVL L0 hook after iteration | bridge to `CriticOrchestrator` | Failed test → revise verdict |
| ECC-2.7 | Skill `codecraft.ephemeral_builder` | `skills/providers/codecraft` | Bundles codecraft.* + workspace |

**Closes:** GAP-ECC-01, GAP-ECC-03, GAP-ECC-10 (partial), GAP-ECC-11.

---

## Phase ECC-3 — Modes, HITL, promotion

**Status:** **Done** (2026-06-13)

**Goal:** `CodeCraftProfile`, supervised/autonomous, `CraftResultPromoter`, `codecraft.promote`.

| ID | Deliverable | Module | Acceptance |
|----|-------------|--------|------------|
| ECC-3.1 | `CodeCraftProfile` model | `applications/contracts` or `codecraft/profile.py` | Typed fields per architecture §6.2 |
| ECC-3.2 | `wire_application_codecraft()` | `applications/_shared/codecraft_wiring.py` | Lab + poc_template presets |
| ECC-3.3 | Mode enforcement | orchestrator | `dry_run` never exec; `assist_only` returns code only |
| ECC-3.4 | HITL before exec | PolicyEngine + HitlRunner | `supervised` pauses until resume |
| ECC-3.5 | `CraftResultPromoter` | `codecraft/promoter.py` | Pydantic promotion schema L0 |
| ECC-3.6 | Tool `codecraft.promote` | tools | Supervised explicit promotion |
| ECC-3.7 | Policy fragment `codecraft_governance` | UAEP bundle | Trace checklist in AGENT_CREATION_GUIDE |

**Closes:** GAP-ECC-02, GAP-ECC-07, GAP-ECC-08, GAP-ECC-10 (complete).

---

## Phase ECC-4 — Production isolation

**Status:** **Done** (2026-06-13)

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

**Status:** **Done** (2026-06-13)

| ID | Deliverable | Acceptance |
|----|-------------|------------|
| ECC-5.1 | `EphemeralToolRegistry` | Tools visible only for `craft_id` |
| ECC-5.2 | `codecraft.list_ephemeral_tools` | Introspection tool |
| ECC-5.3 | `CodeCraftNode` spec | `NEXUS_EXECUTION_FLOW` graph pattern |
| ECC-5.4 | Graph executor bridge | Optional node in lab graph spec |

**Closes:** GAP-ECC-06, GAP-ECC-12.

---

## Phase ECC-6 — Adaptive trigger (L4)

**Status:** **Done** (2026-06-13)

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
| ECC-2 | **Done** | P0 | 01, 03, 10, 11 |
| ECC-3 | **Done** | P1 | 02, 07, 08, 10 |
| ECC-4 | **Done** | P1 | 05 |
| ECC-5 | **Done** | P2 | 06, 12 |
| ECC-6 | **Done** | P2 | 13 |

---

# Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-10 | ECC-0 | Domain pair CODE_CRAFT, ADR-CODECRAFT-001, full audit register, hub/README/AGENTS sync |
| 2026-06-13 | LAYER-CLOSE | Layer completion audit — ECC-0…ECC-6 **Done**; architecture §12 + plan register synced |
| 2026-06-13 | ECC-1 | `codecraft.run`, StaticCodeGate, CodeCraftTraceEmitter, catalog plugin, gate tests |
| 2026-06-13 | ECC-2 | Orchestrator, session tools, codegen adapter, test runner, CVL bridge, skill |
| 2026-06-13 | ECC-3 | Tier-3 profile wiring, HITL, promoter, codecraft.promote |
| 2026-06-13 | ECC-4 | Isolation routing, security scan hook, harness_codecraft_stack preset |
| 2026-06-13 | ECC-5 | EphemeralToolRegistry, list_ephemeral_tools, CodeCraftGraphBinding |
| 2026-06-13 | S7–S10 | Post-closeout — trace taxonomy, sandbox routing parity, health probe, CI gate |
| 2026-06-13 | S11 | Layer completion II — audit doc sync, exec budget enforcement, GAP-ECC-23 register |

---

# Layer completion audit II (2026-06-13)

**Trigger:** Operator note — ECC still reads as architecture + partial start in audit prompts despite shipped runtime.

**Method:** Re-verify 28 Python modules, `wire_application_codecraft()`, 7 `codecraft.*` catalog tools, gate + unit tests.

**Findings:**

| ID | Severity | Issue | Resolution |
|----|----------|-------|------------|
| DOC-ECC-01 | P1 | `audit/CODE_CRAFT.md` + `generate_domain_audit_prompts.py` still say ECC-1+ Planned | S11 doc-sync |
| DOC-ECC-02 | P1 | Known gaps list stale (orchestrator missing) | Regenerated audit prompt |
| RUN-ECC-01 | P1 | `max_total_exec_time_s` tracked but not enforced before exec | S11 orchestrator + `codecraft.run` cap |

**Verdict:** Runtime **Done** (ECC-0…ECC-6 + S7–S11 + §6.1av ECC-MAINT-01..04). No open P0/P1 in domain scope.

---

# Layer completion audit (2026-06-13)

**Method:** Re-read domain pair + ADR + grep `intergrax/codecraft`, `intergrax/runtime/codecraft`, `intergrax/tools/providers/codecraft` — **28 Python modules** shipped.

**Verdict:** Documentation canon **Done** (ECC-0). Harness orchestration **Done** (ECC-1…ECC-6). Substrate reuse path **validated** — `SandboxSession`, `code.exec`, `security.scan`, CVL bridge, `sandbox_host` integrations wired.

**Doc inconsistencies closed (2026-06-13 doc-sync iteration):**

| Issue | Resolution |
|-------|------------|
| Architecture §12 still listed orchestrator/profile as Missing | Updated to L3 closeout table |
| Plan audit §3.3 / §4 still open gaps | Marked **Done** with ECC phase mapping |
| PLATFORM_FOUNDATION `codecraft.* (planned)` | **Done** — synced 2026-06-13 doc-sync |
| Root `README.md` Overview `(planned)` vs ECC Done elsewhere | **Done** — P1-ARCH-03 sync (2026-06-17) |

**Remaining depth (not blocking L3):** none in domain scope — GAP-ECC-20…23 closed via ECC-MAINT-01..04 (2026-06-18). Accepted: local sandbox ≠ OS containment.

---

# Post-closeout gap register (2026-06-13)

Layer completion audit after ECC-0…ECC-6 — gaps blocking **production parity** within CODE_CRAFT domain:

| GAP-ID | Category | Description | Sprint | Priority |
|--------|----------|-------------|--------|----------|
| GAP-ECC-16 | observability | Trace taxonomy incomplete — missing generation/test/verdict/HITL/promote steps | **S8** **Done** | P1 |
| GAP-ECC-17 | routing | `codecraft.run` bypasses `resolve_craft_sandbox_session` (no cloud tier on single-shot) | **S9** **Done** | P1 |
| GAP-ECC-18 | ops | `health.check_codecraft` probe not registered in health bundle | **S10** **Done** | P2 |
| GAP-ECC-19 | CI | No `check_codecraft_layer.py` harness gate | **S10** **Done** | P2 |
| GAP-ECC-20 | codegen | `codegen_llm_profile_ref` unused — template adapter only | **ECC-MAINT-02** **Done** | P3 |
| GAP-ECC-21 | security | `container` isolation tier not implemented | **ECC-MAINT-03** **Done** (local fallback) | P3 |
| GAP-ECC-22 | observability | §10.2 metrics dashboards | **ECC-MAINT-04** **Done** | P3 |
| GAP-ECC-23 | control | `Task.metadata.codecraft_mode` per-task override not wired | **ECC-MAINT-01** **Done** | P2 |

**Coverage:** S8–S11 **Done** · §6.1av ECC-MAINT-01..04 **Done** · depth backlog **closed**.

---

# Post-closeout sprints S7–S10

## Sprint S11 — Audit doc sync + exec budget (**Done**)

| Field | Value |
|-------|-------|
| **Scope** | Regenerate audit prompt; enforce `max_total_exec_time_s`; register GAP-ECC-23 |
| **Goal** | Documentation matches shipped runtime; cumulative exec budget fail-closed |
| **DoD** | Audit prompt shows Done; unit test for budget deny; gate green |
| **Files** | `docs/audit_results/CODE_CRAFT.md`, `scripts/audit/generate_domain_audit_prompts.py`, `intergrax/codecraft/profile.py`, `intergrax/runtime/codecraft/orchestrator.py`, `intergrax/tools/providers/codecraft/service.py`, tests |

## Sprint S7 — Documentation sync (**Done** in this iteration)

| Field | Value |
|-------|-------|
| **Scope** | Architecture §6/§10/§16 + plan gap register + §3.2 refresh |
| **Goal** | Docs as source of truth before code changes |
| **DoD** | Domain pair aligned; post-closeout gaps enumerated |
| **Files** | `docs/project/architecture/CODE_CRAFT.md`, `docs/project/maintainers/plans/CODE_CRAFT.md` |

## Sprint S8 — Trace taxonomy parity (ECC-7) (**Done**)

| Field | Value |
|-------|-------|
| **Scope** | `CodeCraftTraceEmitter` + orchestrator hooks for generation, test, verdict, HITL, promote |
| **Goal** | Architecture §10.1 events emitted on real paths |
| **DoD** | Unit tests assert new steps; closes GAP-ECC-16 |
| **Files** | `intergrax/runtime/codecraft/trace.py`, `orchestrator.py`, `tests/unit/runtime/codecraft` |

## Sprint S9 — Single-shot sandbox parity (ECC-8) (**Done**)

| Field | Value |
|-------|-------|
| **Scope** | `codecraft.run` uses `resolve_craft_sandbox_session` + optional security scan |
| **Goal** | ECC-4 isolation routing on all exec paths |
| **DoD** | Test with cloud-tier profile mock; closes GAP-ECC-17 |
| **Files** | `intergrax/tools/providers/codecraft/service.py`, tests |

## Sprint S10 — Health probe + CI gate (ECC-9) (**Done**)

| Field | Value |
|-------|-------|
| **Scope** | Register `health.check_codecraft`; add `scripts/maintenance/check_codecraft_layer.py` |
| **Goal** | Ops probe + gate maintenance for catalog/wiring invariants |
| **DoD** | Gate green; closes GAP-ECC-18, GAP-ECC-19 |
| **Files** | `intergrax/tools/providers/health/*`, `scripts/maintenance/check_codecraft_layer.py` |

---

# Sprints (layer completion — ECC-0…ECC-6)

Each sprint = one PR-sized slice → gate green → plan row update → commit.

## Sprint S1 — ECC-1 Single-shot craft (**Done**)

| Field | Value |
|-------|-------|
| **Scope** | `codecraft.run`, `StaticCodeGate`, `CodeCraftTraceEmitter`, catalog registration |
| **Goal** | Smallest governed path: accept code → L0 gate → sandbox exec → typed `CraftResult` |
| **DoD** | Unit tests for gate + deny paths; tool in catalog; `CODECRAFT_*` trace steps; gate tests green |
| **Files** | `intergrax/codecraft/{contracts,profile,static_gate}.py` · `intergrax/runtime/codecraft/trace.py` · `intergrax/tools/providers/codecraft/*` · `intergrax/tools/registry/shipped_plugins.py` · `intergrax/runtime/sandbox/sandbox_runtime.py` · `tests/unit/codecraft` · `tests/unit/tools/providers/codecraft` |

## Sprint S2 — ECC-2 Session loop (**Done**)

| Field | Value |
|-------|-------|
| **Scope** | `CodeCraftOrchestrator`, session manager, `start/iterate/get_state/dispose`, codegen adapter, test runner |
| **Goal** | Multi-iteration generate→gate→exec→test loop with budgets |
| **DoD** | E2E mock test; CVL L0 hook; skill `codecraft.ephemeral_builder` manifest |
| **Files** | `intergrax/runtime/codecraft/{orchestrator,session_manager}.py` · `intergrax/codecraft/{codegen_adapter,test_runner}.py` · extended `tools/providers/codecraft` · `intergrax/skills/providers/codecraft` |

## Sprint S3 — ECC-3 Modes + HITL + promotion (**Done**)

| Field | Value |
|-------|-------|
| **Scope** | `CodeCraftProfile` on `ApplicationEnvironmentProfile`, wiring, modes, HITL, `CraftResultPromoter` |
| **Goal** | Tier-3 profile drives mode enforcement; supervised HITL; typed promotion |
| **DoD** | Lab preset wired; mode matrix tests; `codecraft.promote` tool |
| **Files** | `intergrax/applications/contracts/environment_profile.py` · `intergrax/applications/_shared/codecraft_wiring.py` · `intergrax/codecraft/promoter.py` · UAEP policy fragment |

## Sprint S4 — ECC-4 Production isolation (**Done**)

| Field | Value |
|-------|-------|
| **Scope** | Cloud sandbox routing, `security.scan` pre-exec, egress policy, harness preset |
| **Goal** | Regulated hosts default `isolation_tier=cloud` |
| **DoD** | Integration test with mocked `HostedSandboxSession`; health probe extension |
| **Files** | orchestrator isolation routing · `applications/_shared` presets · health probes |

## Sprint S5 — ECC-5 Ephemeral registry + graph (**Done**)

| Field | Value |
|-------|-------|
| **Scope** | `EphemeralToolRegistry`, `codecraft.list_ephemeral_tools`, optional `CodeCraftNode` |
| **Goal** | Task-scoped tools never pollute global catalog |
| **DoD** | Registry isolation tests; graph spec example in lab |
| **Files** | `intergrax/runtime/codecraft/ephemeral_registry.py` · NEXUS_EXECUTION_FLOW cross-ref |

## Sprint S6 — ECC-6 Adaptive trigger (**Done**)

| Field | Value |
|-------|-------|
| **Scope** | AHI catalog-miss signal, budget-aware craft decision |
| **Goal** | L4 suggests or invokes craft when policy allows |
| **DoD** | Signal tests in adaptive domain integration |
| **Files** | `intergrax/runtime/adaptive` hooks · ADAPTIVE_HARNESS_INTELLIGENCE plan row |

**Minimum L3 closeout:** S1–S4 (ECC-1…ECC-4). S5–S6 are depth.

---

# Explicitly out of scope

- Phase K business agents using ECC (§6.3 product decision).
- Container runtime implementation details before ECC-4 design spike.
- Replacing or removing `code.exec` / `sandbox.exec` primitives.
- Monolithic `docs/INTERGRAX_IMPLEMENTATION_PLAN.md` or `plan/phases` folders.

---

## Phase CODE_CRAFT-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — re-validates 2026-06-13 Layer Completion (ECC-0…ECC-6 + S7–S11); no open P0/P1  
**Prerequisites:** ECC phases **Closed** · `check_codecraft_layer.py` gate  
**Goal:** Formal Full Harness LC closeout — gate verification, journal  
**ADR:** **No ADR needed** — formal closeout; ADR-CODECRAFT-001 accepted

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| CODE_CRAFT-LC-S1 | **Re-audit** — ECC register + tier-0/1 verdict | **Done** | High | No P0/P1 |
| CODE_CRAFT-LC-S2 | **Plan/architecture sync** — Full Harness LC note | **Done** | High | Domain pair consistent |
| CODE_CRAFT-LC-S3 | **Gate verification** | **Done** | High | 7 unit tests · `check_codecraft_layer.py` |
| CODE_CRAFT-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** none — GAP-ECC-20…23 closed (ECC-MAINT-01..04). Accepted: local sandbox ≠ OS containment.

### 6.1av Harness implementation queue — Code Craft audit maintenance (closed)

**Source:** Layer 9 audit (2026-06-18) — `CODE_CRAFT` layer 11b · [`../audit_results/2026-06-18/CODE_CRAFT.md`](../audit_results/2026-06-18/CODE_CRAFT.md)  
**Priority ladder:** **Band 1** (§6.1) — depth backlog only; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **ECC-MAINT-01** | Code | P2 | **Done** | GAP-ECC-23 — wire `Task.metadata.codecraft_mode` per-task override in orchestrator | Unit test: task metadata overrides profile default |
| 2 | **ECC-MAINT-02** | Code | P3 | **Done** | GAP-ECC-20 — wire `codegen_llm_profile_ref` to dedicated codegen LLM adapter | Separate adapter identity in craft loop |
| 3 | **ECC-MAINT-03** | Code | P3 | **Done** | GAP-ECC-21 — design spike + `container` isolation tier implementation | ADR spike or extension; tier selectable in profile |
| 4 | **ECC-MAINT-04** | Observability | P3 | **Done** | GAP-ECC-22 — §10.2 metrics dashboards / trace panels for Code Craft | Metrics emitted; dashboard or trace explorer panel |

**Suggested PR order:** none — §6.1av queue closed (2026-06-18).

**Explicitly accepted (no MAINT):** local `SandboxSession` ≠ OS containment — documented canon constraint.

### 6.1aw Harness implementation queue — Code Craft audit maintenance (2026-06-19)

**Source:** Interactive layer audit (2026-06-19) — `CODE_CRAFT` layer 11b · [`../audit_results/2026-06-19/CODE_CRAFT.md`](../audit_results/2026-06-19/CODE_CRAFT.md) (pending) · prior: [`../audit_results/2026-06-18/CODE_CRAFT.md`](../audit_results/2026-06-18/CODE_CRAFT.md)  
**Priority ladder:** **Band 1** (§6.1) — doc sync + audit artifact; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **ECC-MAINT-DOC-01** | Docs | P3 | **Done** | Close §6.1av; sync GAP-ECC-20..23 register; fix architecture §6.3; regenerate audit prompt known gaps | Canon matches ECC-MAINT-01..04 |
| 2 | **ECC-MAINT-AUDIT-01** | Docs | P3 | **Done** | Persist Mode A2 audit result under `docs/audit_results/2026-06-19` | `CODE_CRAFT.md` + `progress.json`; L3+ verdict layer 11b |

**Suggested PR order:** none — §6.1aw queue closed (2026-06-19).

---

*End of Code Craft Implementation Plan.*
