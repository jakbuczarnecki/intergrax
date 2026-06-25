# Tools — Implementation Plan

**Architecture (1:1):** [`architecture/TOOLS.md`](../architecture/TOOLS.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites/` satellites on demand).

**Queue status (2026-06-23):** Phase **TOOL-ENG** **closed** (36/36). Catalog expansion (Phase O / T-EXPAND) **closed** at **200** tools · **49** bundles. Strategic backlog → **Phase TOOL-PRODUCT-ROI** (below). Default harness queue → **gate maintenance** in [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md).

**Cross-feature — Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md). TOOLS owns compact LLM-facing tool catalog presentation and schema-preserving `ToolSchemaOptimizer`; canonical `ToolContract` registry, tool call payloads, and tool result JSON must not be mutated by default.

**Layer completion mode (2026-06-12):** [§Layer completion audit](#layer-completion-audit-2026-06-12) · [§Layer completion sprints](#layer-completion-sprints-2026-06-12) · [§Final audit](#layer-completion-final-audit-2026-06-12)

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (TOOLS plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites/`](plan/satellites/) satellites on demand. Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Token Optimization:** read feature pair + row `TOKEN-TOOLS-1`; inspect only tool schema export / planner input path needed for compact catalog view.
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/TOOLS.md`](../architecture/TOOLS.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/TOOLS.md`](../guides/audit_slices/TOOLS.md).
- **Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/TOOLS_audit_history.md`](plan/TOOLS_audit_history.md) | audit history |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

---

## Phase TOKEN-TOOLS — Tool schema optimization for compact LLM catalog (Planned)

**Feature:** [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)  
**Architecture:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md)  
**Priority:** P1 after TOKEN-UER-1 shared contracts  
**Delivery rule:** one `TOKEN-TOOLS-*` row per PR; no schema semantics mutation.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **TOKEN-TOOLS-1** | Code | P1 | Planned | `intergrax/runtime/nexus/tools/tool_schema_optimizer.py` produces compact LLM-facing tool catalog/schema view for `ToolPlanningService` / `CatalogToolPlanner` / schema export path | Canonical `ToolContract` registry unchanged; tool names, parameter names, enum values, required fields, and JSON schema semantics unchanged; tool call payloads/results not compressed by default; compact view enabled only by Token Optimization policy/profile; fixture shows lower token count; `uv run pytest tests/unit/runtime/nexus/tools/ -q`; `uv run python scripts/check_tool_schema_optimizer.py` |

**Explicit exclusions:** no payload compression, no result compression, no permission expansion, no replacement of `ToolAccessPolicy`, no change to tool runtime execution semantics.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.6 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (catalog layer) — engine gaps tracked in **Phase TOOL-ENG** (2026-06-10)

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-11.1 | §11 Tools | Sandboxed execution for code / side-effectful tools | P1 | **Done** |
| AUDIT-IDEAL-11.2 | §11 Tools | MCP / function-schema export for shipped tool catalog | P2 | **Done** |
| AUDIT-IDEAL-11.3 | §11 Tools | Oversized-tool lint enforcement in CI (adoption sweep) | P2 | **Done** |

**Follow-on (engine, not AUDIT-IDEAL id):** TOOL-ENG-1–10 — see [Phase TOOL-ENG](#phase-tool-eng--tool-engine-hardening-2026-06-10-audit).

**Delivery rule:** One **AUDIT-IDEAL-*** ID per PR → update this table + master register → gate green.

---

## Phase TOOL-PRODUCT-ROI — Catalog extension by product value (Planned)

**Status:** **Planned** — architecture & implementation backlog only  
**Last updated:** 2026-06-23  
**Prerequisites:** Phase **TOOL-ENG** **Done** · catalog **200** shipped `tool_id`s · Full Harness LC unchanged  
**Architecture (1:1):** [`architecture/TOOLS.md`](../architecture/TOOLS.md) — §Phase TOOL-PRODUCT-ROI  
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
