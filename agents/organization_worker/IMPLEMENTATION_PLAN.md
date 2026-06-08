# organization_worker agent — Implementation Plan

**The implementation map** for this Tier-2 agent — phases, status, gaps, and verification.

Status: **Harness baseline Done** — HITL/long-running depth optional

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Platform plan: [`docs/intergrax_runtime_architecture.md`](../../docs/intergrax_runtime_architecture.md) · Phase AA-ORG

---

## Documentation model

| Topic | Where |
|-------|--------|
| HITL-oriented purpose, capability | **ARCHITECTURE.md** |
| Task status | **This file** |
| HITL authoring | `docs/guides/AGENT_CREATION_GUIDE.md` Appendix H |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| ORG-1 | UAEP agent + offline stub LLM | **Done** | High | Gate smoke |
| ORG-2 | ARCHITECTURE baseline | **Done** | Medium | AA-ORG.1 |
| ORG-3 | Optional lab roster entry | **Done** | Low | `LabApplicationSettings` |
| ORG-4 | Checkpoint-friendly multi-step domain | Planned | Medium | With Tier-3 checkpoint store |
| ORG-5 | Product vendor-report workflow | Deferred | Low | Band 3 |

---

## 2. Verification

```bash
uv run pytest tests/unit/agents/test_organization_worker_agent.py -q
```
