# Tools — Implementation Plan

**Architecture (1:1):** [`architecture/TOOLS.md`](../../architecture/TOOLS.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Queue status (2026-06-23):** Phase **TOOL-ENG** **closed** (36/36). Catalog expansion (Phase O / T-EXPAND) **closed** at **200** tools · **49** bundles. Strategic backlog → **Phase TOOL-PRODUCT-ROI** (below). Default harness queue → **gate maintenance** in [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md).

**Cross-feature — Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md). TOOLS owns compact LLM-facing tool catalog presentation and schema-preserving `ToolSchemaOptimizer`; canonical `ToolContract` registry, tool call payloads, and tool result JSON must not be mutated by default. **TOKEN-3** defines `ToolSchemaOptimizer` as an internal LLM-facing tool catalog compaction helper in `intergrax/runtime/token_optimization/tool_schema.py`; it does not change executable tool schemas or runtime tool registry behavior yet.

**Layer completion mode (2026-06-12):** [§Layer completion audit](.#layer-completion-audit-2026-06-12) · [§Layer completion sprints](.#layer-completion-sprints-2026-06-12) · [§Final audit](.#layer-completion-final-audit-2026-06-12)

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (TOOLS plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites`](plan/satellites) satellites on demand. Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Token Optimization:** read feature pair + rows `TOKEN-TOOLS-1A` / `TOKEN-TOOLS-1B`; inspect only tool schema export / planner input path needed for compact catalog view.
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/TOOLS.md`](../../architecture/TOOLS.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/TOOLS_implementation_history.md`](plan/TOOLS_implementation_history.md) | implementation history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

---

## Phase TOKEN-TOOLS — Tool schema optimization for compact LLM catalog

**Feature:** [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md)
**Architecture:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md)
**Priority:** P1 after TOKEN-UER-1 shared contracts  
**Delivery rule:** one `TOKEN-TOOLS-*` row per PR; no schema semantics mutation.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **TOKEN-TOOLS-1A** | Code | P1 | **Done / Closed** | `intergrax/runtime/token_optimization/tool_schema.py` defines helper-only `ToolSchemaOptimizer` for deterministic LLM-facing compact catalog views | Canonical `ToolContract` registry unchanged; executable tool schemas unchanged; names/parameters/enums/required/type/properties preserved; protected-region validation, receipts, optional token_counter measurement; no runtime registry wiring; focused token optimization tests pass |
| **TOKEN-TOOLS-1B** | Code | P1 | Planned | Runtime wiring for compact LLM-facing tool catalog view in `ToolPlanningService` / `CatalogToolPlanner` / schema export path before `generate_with_tools` | Compact view enabled only by Token Optimization policy/profile; canonical registry unchanged; tool call payloads/results not compressed by default; fixture shows lower token count; runtime/nexus tool tests and `scripts/check_tool_schema_optimizer.py` once wiring exists |

**Explicit exclusions:** no payload compression, no result compression, no permission expansion, no replacement of `ToolAccessPolicy`, no change to tool runtime execution semantics.

**TOKEN-TOOLS-1A closeout:**

- helper-only `ToolSchemaOptimizer` added
- deterministic compact LLM-facing view added
- no runtime tool registry wiring
- no executable schema mutation
- runtime wiring deferred to `TOKEN-TOOLS-1B`

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.6 · baseline **32/32 L3**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (catalog layer) — engine gaps tracked in **Phase TOOL-ENG** (2026-06-10)

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-11.1 | §11 Tools | Sandboxed execution for code / side-effectful tools | P1 | **Done** |
| AUDIT-IDEAL-11.2 | §11 Tools | MCP / function-schema export for shipped tool catalog | P2 | **Done** |
| AUDIT-IDEAL-11.3 | §11 Tools | Oversized-tool lint enforcement in CI (adoption sweep) | P2 | **Done** |

**Follow-on (engine, not AUDIT-IDEAL id):** TOOL-ENG-1–10 — see [Phase TOOL-ENG](.#phase-tool-eng--tool-engine-hardening-2026-06-10-audit).

**Delivery rule:** One **AUDIT-IDEAL-*** ID per PR → update this table + master register → gate green.

### Architecture sync — UE-DOC-0.8 (2026-08-26)

**Status:** documentation frozen in architecture hub — **no runtime implementation in this slice**. Code transformation mapping: **UE-DOC-0.9**.

Architecture hub additions: five orthogonal axes; loop ownership (`bounded_react` under UAEP); monotonic selection; TOOLS-INV; implementation readiness. Cross-ref: [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md#execution-first-agentic-model-ue-doc-08).

---

## Phase TOOL-PRODUCT-ROI — Catalog extension by product value (Planned)

**Status:** **Planned** — architecture & implementation backlog only  
**Last updated:** 2026-06-23  
**Prerequisites:** Phase **TOOL-ENG** **Done** · catalog **200** shipped `tool_id`s · Full Harness LC unchanged  
**Architecture (1:1):** [`architecture/TOOLS.md`](../../architecture/TOOLS.md) — §Phase TOOL-PRODUCT-ROI
**Band:** 2af (post–Full Harness LC product depth)  
**Policy:** One implementation ID per PR; **do not** register planned `tool_id`s until the matching task PR; **do not** mark any TOOL-PRODUCT-ROI task **Done** until code ships.

### Scope split

| Layer | This update | Follow-up PRs |
|-------|-------------|---------------|
| Architecture canon | TOOL-PRODUCT-ROI boundaries, waves, deferred runtime features | — |
| Tool catalog | Document planned `tool_id`s only | Register bundle + contracts + tests per task |
| Integrations | Document backend deps (`local_git`, GitHub/GitLab) | INT-P8.5 / existing VCS integrations per tool |
| Runtime / policy | Document alignment for patch + critic hooks | Policy gate extensions in dedicated PR |

### Execution order (recommended)

```text
Wave 1 (P0):  code.repo_map → code.symbol_search → code.dependency_graph → code.boundary_check → code.diff_risk_analyze → code.test_impact
Wave 2 (P1):  git.branch_diff ∥ git.pr_context ∥ git.ci_status
Wave 3 (P2):  patch.preview → patch.apply_safe (policy + idempotency + audit + optional HITL)
Later:        browser automation suite · research evidence tools (product-gated)
Deferred:     hierarchical LLM category pass (default OFF) · optional L1 critic on high-risk tool output (default OFF)
```

**Parallelism:** Wave 2 tools may proceed in parallel after Wave 1 `code.repo_map` contract is agreed. Wave 3 **blocked** until Wave 2 read-only git context ships.

**Bundle vs namespace (Wave 1):** bundle id `code_intelligence`; public `tool_id`s remain `code.*` (`code.repo_map`, `code.symbol_search`, …).

**Backend vs tools (Wave 2):** `local_git` (INT-P8.5) may expose approval-gated write backend ops (`apply_patch`, `commit`); Wave 2 `git.*` tools are **read-only** only. Patch/commit ships later via `patch.*` + ToolRuntime policy.

---

### TOOL-PRODUCT-ROI master register

| ID | Title | Type | Priority | Status | Depends on | Acceptance criteria |
|----|-------|------|----------|--------|------------|---------------------|
| **TOOL-ROI-1.1** | `code.repo_map` | Code | **P0** | **Planned** | TOOL-ENG | Read-only tool; repo directory/module map; schema + handler + tests; ToolRuntime dispatch; no direct agent→integration |
| **TOOL-ROI-1.2** | `code.symbol_search` | Code | **P0** | **Planned** | TOOL-ROI-1.1 | Symbol index search (classes, functions, methods, protocols, constants); bounded result set |
| **TOOL-ROI-1.3** | `code.dependency_graph` | Code | **P0** | **Planned** | TOOL-ROI-1.1 | Module/layer dependency graph output; configurable depth |
| **TOOL-ROI-1.4** | `code.boundary_check` | Code | **P0** | **Planned** | TOOL-ROI-1.3 | Rule-driven boundary violations (tier imports, ToolRuntime bypass, etc.) |
| **TOOL-ROI-1.5** | `code.diff_risk_analyze` | Code | **P0** | **Planned** | TOOL-ROI-1.1 | Diff/working-tree risk score + rationale for pre-commit/PR |
| **TOOL-ROI-1.6** | `code.test_impact` | Code | **P1** | **Planned** | TOOL-ROI-1.3 | Map changed files → recommended test targets |
| **TOOL-ROI-2.1** | `git.branch_diff` | Code | **P1** | **Planned** | TOOL-ROI-1.1 | Read-only branch diff; local git and/or GitHub/GitLab backend |
| **TOOL-ROI-2.2** | `git.pr_context` | Code | **P1** | **Planned** | TOOL-ROI-2.1 | PR metadata, files, review context for audit agents |
| **TOOL-ROI-2.3** | `git.ci_status` | Code | **P1** | **Planned** | TOOL-ROI-2.1 | CI/check status for branch or PR |
| **TOOL-ROI-3.1** | `patch.preview` | Code | **P2** | **Planned** | TOOL-ROI-1.5, TOOL-ROI-2.1 | Show patch effect; path allow-list validation |
| **TOOL-ROI-3.2** | `patch.apply_safe` | Code | **P2** | **Planned** | TOOL-ROI-3.1 | Gated apply; policy + idempotency + audit; optional HITL |
| **TOOL-ROI-4.1** | Browser automation suite | Code | **P3** | **Planned** | Product gate | `browser.navigate`, `click`, `fill_form`, `screenshot`, `extract`, `network_requests`, `console_messages` — only with Tier-3 web-app driver |
| **TOOL-ROI-4.2** | Research evidence tools | Code | **P3** | **Planned** | Product gate | `research.evidence_pack`, `research.claim_verify`, `research.source_rank` — evidence layer above websearch/RAG |
| **TOOL-ROI-D.1** | Hierarchical LLM category pass (runtime) | Code | **P3** | **Planned** | TOOL-ENG | `RuntimeConfig.tool_selection_hierarchical_llm_pass` default **false**; allow-list only; no permission expansion |
| **TOOL-ROI-D.2** | Optional L1 critic on high-risk tool output | Code | **P3** | **Planned** | TOOL-ENG, CVL | Post-invoke hook: allow / suspicious / block / require_hitl; scoped to high-risk `tool_id`s only; default **false** |

### Explicit non-goals (TOOL-PRODUCT-ROI)

- General-purpose tools duplicating existing catalog families
- Git merge / approve / push / apply-patch before read-only git context (Wave 2)
- Global L1 critic on read-only tools
- Browser automation without product driver

---

<a id="tools-governed-boundary-integrity--protocol-v2-tools-2026-08-18"></a>

### TOOLS-GOVERNED-BOUNDARY-INTEGRITY — Permission intersection, effective timeout, pre-invoke budget (Protocol v2 · 2026-08-18)

**Status:** `PLANNED`
**Priority:** P1
**Type:** Arch / Wire / Proof
**Source:** [`AUDIT-20260818-TOOLS-01`](../../audit_results/2026-08-18/TOOLS.md), [`AUDIT-20260818-TOOLS-02`](../../audit_results/2026-08-18/TOOLS.md), [`AUDIT-20260818-TOOLS-03`](../../audit_results/2026-08-18/TOOLS.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Deliverable intent:**

- monotonic permission intersection — `resolve_allowed_tools_from_config` and canonical policy/tool-scope owners intersect explicit caller allow-lists with `RuntimePolicyBundle.tool_access`; no caller list expands stricter upstream authority
- real `ToolContract.timeout_ms` boundary — caller-visible latency cap; timeout handling does not synchronously wait for timed-out worker; explicit abandon/cancel semantics for in-flight external effects (no unsafe thread killing)
- pre-invoke hard tool-call budget — reserve/check before side-effect boundary; authoritative invocation accounting (not stale mid-loop `tool_traces`); hard abort/HITL budget violations preserve canonical semantics and are not swallowed as ordinary tools-context errors
- reuse `RunBudget` / `BudgetEnforcer` — no second budget subsystem

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- Prior **TOOL-ENG** **Done** / **closed** rows remain historical — do **not** mark them undone; this block owns Protocol v2 governed-boundary gaps beyond harness closeout.
- Side-effect idempotency/retry/outcome gaps owned by **TOOLS-SIDE-EFFECT-SAFETY** — not duplicated here.
- Cross-ref **PG-FIX** / policy spine where tool-scope intersection overlaps Governed Execution — reuse canonical owners, no parallel policy evaluator.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).
- **Not implemented** by audit persistence task AUDIT-20260818-TOOLS-PERSIST.

<a id="tools-side-effect-safety--protocol-v2-tools-2026-08-18"></a>

### TOOLS-SIDE-EFFECT-SAFETY — Idempotency identity, retry authorization, outcome states (Protocol v2 · 2026-08-18)

**Status:** `IMPLEMENTED` (`6746106f9`; R1 effect-certainty correction `9046cfeda`; governance-first ordering — independent verification pending)
**Priority:** P1
**Type:** Arch / Wire / Proof
**Source:** [`AUDIT-20260818-TOOLS-04`](../../audit_results/2026-08-18/TOOLS.md), [`AUDIT-20260818-TOOLS-05`](../../audit_results/2026-08-18/TOOLS.md), [`AUDIT-20260818-TOOLS-06`](../../audit_results/2026-08-18/TOOLS.md)
**Campaign:** [`docs/audit_results/2026-08-18/`](../../audit_results/2026-08-18/README.md)

**Deliverable intent:**

- canonical idempotency operation identity — typed key contract binding `(tenant_id, idempotency_key)` to logical operation (minimum `tool_id`; deterministic input/operation fingerprint when required); cross-tool/cross-operation collision fails closed
- side-effect retry safety — automatic retry of mutating tools requires positive authorization (idempotent semantics + scoped identity, explicit retry-safe metadata, or retryable error classification); unknown-outcome mutating failures not blindly retried
- idempotency outcome-state model — ledger distinguishes successful completion, failed-before-effect, and failed-with-unknown-external-outcome; `record_completed` MUST NOT treat all failures as safe replay `COMPLETED`
- do **not** claim universal exactly-once against external providers

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- Prior **TOOL-ENG** idempotency wrapper **Done** rows remain historical delivery facts.
- Governed-boundary ordering (budget/timeout/permissions) owned by **TOOLS-GOVERNED-BOUNDARY-INTEGRITY** — not duplicated here.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** per [`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).
- **Not implemented** by audit persistence task AUDIT-20260818-TOOLS-PERSIST.

---
