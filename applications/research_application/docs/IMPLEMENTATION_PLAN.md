# research_application — Implementation Plan

**The implementation map** for this Tier-3 multi-agent host — phases, status, gaps, and verification.

Status: **Harness baseline Done** — graph depth optional

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)
Agent plan: [`docs/project/technical/agents/research/IMPLEMENTATION_PLAN.md`](../../../docs/project/technical/agents/research/IMPLEMENTATION_PLAN.md)
Platform plan: [`docs/project/architecture/intergrax_runtime_architecture.md`](../../../docs/project/architecture/intergrax_runtime_architecture.md) · Phase AA-RESAPP

---

## Documentation model

| Topic | Where |
|-------|--------|
| Multi-agent HTTP host, settings | **ARCHITECTURE.md** |
| Host vs agent tasks | **This file** + `docs/project/technical/agents/research/IMPLEMENTATION_PLAN.md` |
| Deploy | `BUILD_AND_DEPLOY.md` |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| RESAPP-1 | Dual-agent manifest + environment | **Done** | High | AA-RESAPP.1 |
| RESAPP-2 | Nexus loop default (`RESEARCH_USE_NEXUS_LOOP`) | **Done** | High | Legacy flag removed |
| RESAPP-3 | Deploy triad | **Done** | High | Gate |
| RESAPP-4 | Graph-native delegation wiring | Planned | Medium | RES-3 |
| RESAPP-5 | Product research UX | Deferred | Low | Band 3 |

---

## 2. Verification

```bash
uv run pytest applications/research_application/tests -q
uv run pytest tests/unit/applications/test_research_manifest_wiring.py -q
```
