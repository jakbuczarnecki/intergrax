# Audit result — `AGENT_CONTRACTS_AND_ASSEMBLY`

**Run:** 2026-06-18 · **Mode:** audit_only (interactive layer 6)  
**Auditor:** cursor-agent · **Verdict:** L3+ mature_revalidated

---

## Scores (0–100)

| Dimension | Score |
|-----------|-------|
| Architecture completeness | 96 |
| Production readiness | 94 |
| Documentation consistency | 95 |
| Implementation consistency | 95 |

---

## Findings

| ID | Severity | Finding | Evidence | Status |
|----|----------|---------|----------|--------|
| ACP-GAP-01 | P2 | `boundary_demo` uses forbidden author-time `allowed_tools` | `agents/boundary_demo/boundary_demo_agent.py:67,116`; `check_agent_skill_resolution.py` | **planned** (ACP-MAINT-01) |
| ACP-GAP-02 | P2 | AS-3 skill resolution gate not in ACP close CI bundle | `check_agent_acp_close_ci.py` OK while AS-3 fails | **planned** (ACP-MAINT-02) |
| ACP-GAP-03 | P3 | Audit prompt lists AUDIT-IDEAL-19.1/20.1/31.1 as Planned | `docs/guides/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` vs ACP-LC **Done** | **planned** (ACP-MAINT-03) |
| ACP-GAP-04 | P2 | COST-1 Nexus graph `RunBudget` cap | Per-agent ACP-TOK **Done**; graph env cap Partial | deferred (UAEP/TIER3) |
| ACP-GAP-05 | P2 | FAUDIT-REG.1 eval registry depth | `PLATFORM_FOUNDATION` master register | deferred |

No open P0/P1. GAP-ACP **37/37 Closed** · ACP-FINISH · ACP-LC **Done** · fleet 17/17 migrated.

---

## Plan sync

| Action | Target | Notes |
|--------|--------|-------|
| Plan row added/updated | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` §6.1av | ACP-MAINT-01..03 |
| Architecture sync needed | no | |

---

## Gates executed

```bash
uv run python scripts/check_agent_acp_close_ci.py
uv run python scripts/check_agent_skill_resolution.py
uv run python scripts/check_agent_pattern_conformance.py
uv run python scripts/check_agents_no_vendor_sdk_imports.py
uv run python scripts/check_agents_lifecycle_metadata.py
uv run python scripts/phase_v_capability_graph_guard.py
uv run pytest tests/unit/agents/authoring/ tests/unit/agents/test_acp_token_budget_enforcement.py -q
```

ACP close CI + pattern + vendor + lifecycle + capability graph: **OK**. AS-3: **FAIL** (`boundary_demo`). Authoring tests: **93 passed**.

---

## Backlog P2–P4 (planned / deferred)

- ACP-MAINT-01..03 — §6.1av
- COST-1 graph RunBudget cap — cross-domain
- FAUDIT-REG.1 — PLATFORM_FOUNDATION

---

## Recommendation

**Architecturally Mature (L3+)** — single P2 fleet hygiene item (`boundary_demo`) blocks AS-3 CI honesty until ACP-MAINT-01/02 land.
