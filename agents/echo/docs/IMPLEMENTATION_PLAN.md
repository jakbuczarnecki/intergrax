# echo agent — Implementation Plan

**The implementation map** for this Tier-2 agent — phases, status, gaps, and verification.

Status: **Done** (harness reference baseline) — maintenance only

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Platform plan: [`docs/project/architecture/intergrax_runtime_architecture.md`](../../../../docs/project/architecture/intergrax_runtime_architecture.md) · Phase AA-ECHO
Agent workflow: [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md)

Principle: **stable harness reference** · **gate smoke must stay green** · **no Tier-3 imports**

---

## Documentation model

| Topic | Where |
|-------|--------|
| Purpose, capabilities, lab registration | **ARCHITECTURE.md** |
| Task status, phases | **This file** |
| Platform harness work | `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` (gate maintenance) |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| ECHO-1 | Harness reference UAEP smoke | **Done** | High | `tests/test_echo` via gate |
| ECHO-2 | `ARCHITECTURE.md` conformance | **Done** | High | Phase AA-ECHO.1 |
| ECHO-3 | Lab + POC manifest registration | **Done** | Medium | `LAB_INCLUDE_ECHO`, POC default |
| ECHO-4 | Domain expansion | Deferred | Low | Not a product agent — keep minimal |

---

## 2. Verification

```bash
uv run pytest agents/echo/tests -q
```
