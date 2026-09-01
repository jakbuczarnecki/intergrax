# LocalSearchAgent - Implementation Plan

**The implementation map** for this Tier-2 agent - phases, status, gaps, and verification.

Status: **Scaffold Done** - Wave **LKW.1** active

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Host: [`applications/local_workspace_application`](../../../applications/local_workspace_application/)
Platform plan: [`docs/project/architecture/intergrax_runtime_architecture.md`](../../../docs/project/architecture/intergrax_runtime_architecture.md) · **LKW.1**

---

## Documentation model

| Topic | Where |
|-------|--------|
| Retrieval responsibilities, tools, I/O | **ARCHITECTURE.md** |
| Wave tasks | **This file** + platform **`docs/project/maintainers/plans/PLATFORM_FOUNDATION.md` §6.3a LKW.*** |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| LKW-SRC-0 | UAEP scaffold + smoke | **Done** | High | LKW.0 |
| LKW-SRC-1 | Query parsing + collection filter | Planned | High | Wave LKW.1 |
| LKW-SRC-2 | `rag.retrieve` integration | Planned | High | Ranked chunks in `StepOutput` |
| LKW-SRC-3 | Multi-collection search | Planned | Medium | Metadata filters |
| LKW-SRC-4 | Filesystem browse tools | Deferred | Medium | LKW.3 Tier-0 `filesystem.*` |

---

## 2. Verification

```bash
uv run pytest agents/local_search/tests -q
```
