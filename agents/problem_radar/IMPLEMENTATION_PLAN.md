# problem_radar agent — Implementation Plan

**The implementation map** for this Tier-2 placeholder — phases, status, gaps, and verification.

Status: **Frozen** — Band 3 (K.1) until explicit product reprioritization

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Platform plan: [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md`](../../docs/INTERGRAX_IMPLEMENTATION_PLAN.md) · K.1 · §6.3

---

## Documentation model

| Topic | Where |
|-------|--------|
| Placeholder purpose, I/O schemas | **ARCHITECTURE.md** |
| Scheduling / go-no-go | **This file** + platform **Appendix A** |
| Product scope | Platform plan **§6.3** only |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| PR-1 | UAEP scaffold + tier hygiene | **Done** | High | Phase AA-PR |
| PR-2 | Pydantic I/O schemas | **Done** | Medium | `schemas/` |
| PR-3 | ARCHITECTURE + notebook documented | **Done** | Medium | AA-PR.3 |
| PR-4 | K.1 product implementation | **Deferred** | — | End of plan — do not start silently |
| PR-5 | Tier-3 product host | **Deferred** | — | After K.1 decision |

---

## 2. Verification

```bash
uv run pytest agents/problem_radar/tests -q
```

Conformance-only maintenance until Band 3 is authorized.
