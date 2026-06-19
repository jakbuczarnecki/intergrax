# Audit result — `AGENT_CONTRACTS_AND_ASSEMBLY`

**Run:** 2026-06-19 · **Mode:** audit_only + implement (ACP-MAINT-DOC-01/02)  
**Auditor:** cursor-agent · **Verdict:** mature_revalidated (L3+)

---

## Scores (0–100)

| Dimension | Score |
|-----------|-------|
| Architecture completeness | 96 |
| Production readiness | 94 |
| Documentation consistency | 94 |
| Implementation consistency | 95 |

---

## Maturity (layers 17–20, 31)

| Layer | Score |
|-------|-------|
| 18 Agent Assembly and Agent Contracts | **L3** |
| 19 Registry Architecture | **L3** |
| 31 Agent Lifecycle Governance | **L3** |
| **Domain overall** | **L3+** |

---

## Findings

| ID | Severity | Finding | Evidence | Status |
|----|----------|---------|----------|--------|
| ACP-DRIFT-01 | P3 | Plan §6.1av header `(planned)` | plan §6.1av | **closed** (ACP-MAINT-DOC-01) |
| ACP-DRIFT-02 | P3 | Audit prompt AUDIT-IDEAL rows stale Planned | `docs/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` | **closed** (ACP-MAINT-DOC-02) |
| ACP-GAP-01 | P2 | `boundary_demo` author-time allowed_tools | ACP-MAINT-01 | **closed** |
| ACP-GAP-02 | P2 | AS-3 not in ACP close CI | ACP-MAINT-02 | **closed** |
| ACP-GAP-04 | P2 | COST-1 graph RunBudget cap | cross-domain | deferred |
| ACP-GAP-05 | P2 | FAUDIT-REG.1 eval registry | PLATFORM_FOUNDATION | deferred |

No open P0/P1. GAP-ACP **37/37 Closed** · fleet **17/17** · production readiness mean **100%**.

---

## Gates executed

```bash
check_agent_acp_close_ci.py           → OK
check_agent_skill_resolution.py       → OK
check_agent_pattern_conformance.py    → OK
check_agents_no_vendor_sdk_imports.py → OK
check_agents_lifecycle_metadata.py    → OK
pytest agents/authoring slice         → 81 passed
```

---

## Plan sync

| Action | Target | Notes |
|--------|--------|-------|
| Plan row added/updated | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` §6.1ay | ACP-MAINT-DOC-01/02, ACP-MAINT-AUDIT-01 **Done** |
| Architecture sync | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §28.3 | ACP-MAINT-DOC-01 |
| Audit prompt sync | `docs/audit/AGENT_CONTRACTS_AND_ASSEMBLY.md` | ACP-MAINT-DOC-02 |

---

## Recommendation

**Architecturally Mature (L3+)** — ACP fleet revalidated; §6.1ay closed. Next domain: `LLM_ADAPTERS`.
