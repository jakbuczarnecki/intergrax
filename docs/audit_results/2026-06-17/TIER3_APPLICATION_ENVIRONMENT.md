# Audit result — `TIER3_APPLICATION_ENVIRONMENT`

**Run:** 2026-06-17 · **Mode:** audit_only  
**Auditor:** cursor-agent · **Verdict:** drift_detected

---

## Scores (0–100)

| Dimension | Score |
|-----------|-------|
| Architecture completeness | 94 |
| Production readiness | 92 |
| Documentation consistency | 94 |
| Implementation consistency | 93 |

---

## Findings

| ID | Severity | Finding | Evidence | Status |
|----|----------|---------|----------|--------|
| T3-AUDIT-01 | P1 | APP-EVOL-8.6 M3 `spec_version` 2.0 nested canonical wire — Planned | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` T3-BL-P1-01 | open |
| T3-AUDIT-02 | P2 | CFG-14 LKW hybrid deferred §6.3 | ORCHESTRATION cross-ref | deferred |

**p0_open:** 0 · **p1_open:** 1

---

## Gates executed

```bash
uv run pytest tests/unit/applications/ -q
uv run python scripts/gates/check_application_production_gates.py
```

---

## Backlog P2–P4 (deferred)

- APP-EVOL-8 M3 spec_version 2.0 — P2
- CFG-14 LKW hybrid — deferred §6.3
- MCP mount research app test — P3

---

## Recommendation

**Architecturally Mature**
