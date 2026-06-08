# LocalIndexerAgent — Implementation Plan

**The implementation map** for this Tier-2 agent — phases, status, gaps, and verification.

Status: **Scaffold Done** — Wave **LKW.1** active (ingest + search smoke)

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
Host: [`applications/local_workspace_application/`](../../applications/local_workspace_application/)  
Platform plan: [`docs/intergrax_runtime_architecture.md`](../../docs/intergrax_runtime_architecture.md) · **LKW.1**

---

## Documentation model

| Topic | Where |
|-------|--------|
| Ingest responsibilities, tools, I/O | **ARCHITECTURE.md** |
| Wave tasks | **This file** + platform **`docs/plan/PLATFORM_FOUNDATION.md` §6.3a LKW.*** |
| LKW product architecture | `applications/local_workspace_application/ARCHITECTURE.md` |

---

## 1. Implementation queue

| ID | Task | Status | Priority | Notes |
|----|------|--------|----------|-------|
| LKW-IDX-0 | UAEP scaffold + smoke | **Done** | High | LKW.0 |
| LKW-IDX-1 | `validate_source_paths` step | Planned | High | Wave LKW.1 |
| LKW-IDX-2 | `rag.ingest_document` per path | Planned | High | Explicit `source_paths` metadata |
| LKW-IDX-3 | Ingest job summary `StepOutput` | Planned | High | Structured stats |
| LKW-IDX-4 | Background ingest queue | Deferred | Medium | LKW.4 — after LKW.2 |

---

## 2. Verification

```bash
uv run pytest agents/local_indexer/tests -q
```

Integration with host:

```bash
uv run pytest applications/local_workspace_application/local_workspace_application_tests -q
```
