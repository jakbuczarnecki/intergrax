# research agents — Implementation Plan

**The implementation map** for Tier-2 research + summary agents — phases, status, gaps, and verification.

Status: **Harness baseline Done** — graph/delegation depth optional

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Host: [`applications/research_application/`](../../applications/research_application/)  
Platform plan: [`docs/intergrax_runtime_architecture.md`](../../docs/intergrax_runtime_architecture.md) · Phase AA-RES

---

## Documentation model

| Topic | Where |
|-------|--------|
| Two-agent layout, capabilities, host wiring | **ARCHITECTURE.md** |
| Task status, phases | **This file** |
| Orchestration / graph | `docs/guides/AGENT_CREATION_GUIDE.md` Appendix I |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| RES-1 | UAEP smoke for `ResearchAgent` + `SummaryAgent` | **Done** | High | `agents/research/tests/` |
| RES-2 | `research_application` manifest + environment | **Done** | High | AA-RESAPP.* |
| RES-3 | Graph delegation (`research.pipeline`) | Planned | Medium | Nexus graph intent documented in ARCHITECTURE |
| RES-4 | Skill packs on contracts | Planned | Medium | Per `docs/architecture/SKILLS.md` |
| RES-5 | Product research features | Deferred | Low | Band 3 unless reprioritized |

---

## 2. Verification

```bash
uv run pytest agents/research/tests -q
uv run pytest applications/research_application/tests -q
```
