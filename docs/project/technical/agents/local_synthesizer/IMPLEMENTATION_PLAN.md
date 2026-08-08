# LocalSynthesizerAgent — Implementation Plan

**The implementation map** for this Tier-2 agent — phases, status, gaps, and verification.

Status: **Scaffold Done** — Wave **LKW.2** (after LKW.1 ingest/search)

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Host: [`applications/local_workspace_application/`](../../applications/local_workspace_application/)  
Platform plan: [`docs/project/architecture/intergrax_runtime_architecture.md`](../../docs/project/architecture/intergrax_runtime_architecture.md) · **LKW.2**

---

## Documentation model

| Topic | Where |
|-------|--------|
| Synthesis responsibilities, shadow workspace | **ARCHITECTURE.md** |
| Wave tasks | **This file** + platform **`docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` §6.3a LKW.*** |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| LKW-SYN-0 | UAEP scaffold + smoke | **Done** | High | LKW.0 |
| LKW-SYN-1 | Consume retrieval context from prior step | Planned | High | Graph handoff LKW.2 |
| LKW-SYN-2 | Artifact write via `workspace.*` tools | Planned | High | Shadow workspace only |
| LKW-SYN-3 | Template-driven outputs (mail/report) | Planned | Medium | Prompt + schema work |
| LKW-SYN-4 | Desktop/tray UI client | Deferred | Low | LKW.8 product shell |

---

## 2. Verification

```bash
uv run pytest agents/local_synthesizer/tests -q
```
