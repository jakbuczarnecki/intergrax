# Tool Library and ToolRuntime — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/TOOLS.md`](../../architecture/TOOLS.md) · [`plan/TOOLS.md`](../plans/TOOLS.md)
**Audit map layers:** 11 · compact slice: [`audit_slices/TOOLS.md`](../../technical/guides/audit_slices/TOOLS.md)
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with the repository available, but do not perform broad repository exploration. Read only the files listed in Context budget / Canonical reads, use path-filtered grep before opening files, and do not use semantic search, subagents, or full-repo scans unless the operator explicitly approves.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus` slice).
4. The agent must **read code, run tests, and re-validate known gaps** — not survey documentation alone.
5. Output: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

Regenerate after architecture/plan changes: `uv run python scripts/audit/generate_domain_audit_prompts.py`

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: TOOLS
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Tool Library and ToolRuntime (`TOOLS`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Tool Library and ToolRuntime** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit the **Tool Library** (190+ catalog tools) and **ToolRuntime** execution engine: selection/planning strategies, policy enforcement, idempotency, MCP export, catalog dispatch, and TOOL-ENG hardening queue — vs production tool-governance systems.

## Key symbols and contracts

ToolContract · ToolRegistry · ToolProfile · ToolWiringContext · ToolRequest/ToolResponse · ToolAccessPolicy · ToolSelectionStrategy · ToolPlanDecision · ToolRiskLevel · tools_mode · tools_context_scope

## Active plan phases (verify status vs code reality)

Phase O/T-EXPAND Done · **TOOL-ENG Closed** (2026-06-12, 36/36, S0–S8) · Phase V V-SEC/V-COST/V-EVAL

## Known open gaps — re-validate every item (closed / still open / partial)

Deferred: hierarchical LLM category pass (ADR-TOOL-005 v1) · optional L1 critic per-tool output (CVL) · ACP invoke_tool/gateway consistency across 190 tools (cross-domain). TOOL-ENG register closed.

---

## 0. Context budget (mandatory)

**Load first:** [`docs/project/technical/guides/audit_slices/TOOLS.md`](../../technical/guides/audit_slices/TOOLS.md) — compact slice (layers **11**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites` or `architecture/satellites`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/project/technical/guides/audit_slices/TOOLS.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/project/architecture/TOOLS.md` — hub read-scope + one `architecture/satellites` satellite max
3. `docs/project/maintainers/plans/TOOLS.md` — hub + one `plan/satellites` satellite max
4. `docs/project/maintainers/audit/README.md` — shared production Harness checklist
5. `@docs/project/technical/guides/AGENT_CREATION_GUIDE.md` **Appendix J** — on demand
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/project/technical/guides/audit_slices/TOOLS.md` — then inspect:

```text
intergrax/tools/core/contracts.py · intergrax/tools/registry/
intergrax/runtime/nexus/tools/tool_runtime.py · invoker.py · catalog_dispatch.py
intergrax/runtime/nexus/tools/tool_planning_service.py · catalog_tool_planner.py
intergrax/runtime/nexus/tools/tool_selection.py
intergrax/runtime/nexus/tools/tool_loop.py
intergrax/runtime/tools/idempotent_invoker.py · runtime_bound_catalog.py
applications/_shared/catalog_runtime_bridge.py · tool wiring
scripts/maintenance/check_legacy_tool_plan_booleans.py · check_tool_mcp_schema_export.py
scripts/maintenance/check_tool_injection_defense.py · check_agent_registry_bypass.py
```

Grep `tests/unit`, `tests/integration`, `tests/acceptance` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. All invocations via ToolRuntime → policy → RuntimeToolInvoker — no bypass.
2. Every tool: tool_id, input/output schema, risk level, description for LLM selection.
3. ToolSelectionStrategy wired before LLM tool call (ENG-5) — not post-hoc only.
4. Tool planning: catalog_tool_planner + tool_planning_service — allow-list respects AgentContract.
5. ToolScopePolicy / StaticToolScopePolicy enforced (ENG-3).
6. Catalog tool_id dispatch (ENG-1/2) — capability alias vs catalog id consistent.
7. Idempotency keys on side-effect tools (idempotent_invoker).
8. Concurrency, timeout, retry on invocation path.
9. ops:tool_audit and TOOL_* trace events emitted.
10. MCP export schema parity with OpenAI function schema — CI green.
11. Legacy boolean flags (use_rag, tool_gateway) deprecated — check_legacy_tool_plan_booleans green.
12. Injection defense middleware active.
13. Agents cannot bypass registry — check_agent_registry_bypass green.
14. Plugin model: ToolPlugin + entry points + bootstrap_catalogs.
15. Skills merge into tool allow-list correctly at resolution time.
16. HIGH-risk tools: post-tool verification (ENG-7 gap status).
17. ReAct / iterative tool loop bounded (ENG-6 gap status).
18. EnvironmentProfile tool_selection fields wired (recent catalog_runtime_bridge work).

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- 190 tools / 48 bundles registration at bootstrap.
- RunBudget.max_tool_calls (128 prod default) enforcement.
- Parallel read-only tool invocations (ENG-9 target).
- Large allow-list filtering performance.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

ToolProfile.enabled/enabled_bundles · ReasoningProfile.tool_planner_prompt_id · tool_selection_mode on EnvironmentProfile · RuntimePolicyBundle.tool_access · tools_mode on engine plan · external ToolPlugin

---

## 6. Cross-cutting checklist (mandatory)

Apply **every** section in `docs/project/maintainers/audit/README.md` §Shared production Harness checklist:

- Architecture & modularity
- Configuration & strategy selection
- Override & customization surfaces
- Observability, tracing & logging
- Security & governance
- Reliability & error handling
- Performance & scale
- Testing & verification
- Documentation alignment

---

## 7. Production baseline comparison

Compare against: **OpenAI function calling / MCP · enterprise tool allow-lists and audit · Cursor-scale tool routing with policy**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Direct handler invoke bypassing ToolRuntime · boolean use_* flags parallel to tools · vendor SDK in tool handlers ·unbounded tool loops in agents

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
python scripts/maintenance/check_legacy_tool_plan_booleans.py
python scripts/maintenance/check_tool_mcp_schema_export.py
python scripts/maintenance/check_tool_injection_defense.py
python scripts/maintenance/check_agent_registry_bypass.py
uv run pytest tests/unit/runtime/nexus/tools/ -q
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- **O1 terse** checkpoint unless operator requests full report.
- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7–§8 for final write-up.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update plan/arch gap rows; **no code** unless operator requests separately.

Begin the audit now.

---END PROMPT---
