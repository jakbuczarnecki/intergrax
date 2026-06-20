---
id: IJ-2026-06-17-001
date: 2026-06-17
tiers:
  - tier-0
  - tier-1
  - tier-3
scope: platform MEMORY layer completion
plan_ref:
  - MEM-VEC-3.1
  - MEM-VEC-3.2
  - MEM-DEPTH-5.2
  - MEM-DEPTH-4.2
  - MEM-OBS.1
status: completed
commit: 72e65f2b
adr: no ADR needed — extends ADR-MEM-002 wiring; no new architectural decision
---

# MEMORY layer completion — vector namespace, semantic search, org LTM

## Operator request

Complete the MEMORY layer per strategic architecture review: close P1 gaps (retrieval_service wiring, vector namespace, temporal validity, semantic search runtime, observability) and accepted hardening items (org LTM, entity graph indexing, explore delegation).

## Summary

Wired `RetrievalService` into `UserProfileManager`, enforced `vector_index_namespace` via `collection_name` metadata, added temporal validity filtering on LTM retrieval, shipped `memory.semantic_search` catalog tool and `SessionTurnIndexStorePlugin` discovery, connected LTM/episodic observability metrics, uplifted org profile memory entries, indexed entity graph on consolidation, and injected explore synthesis in graph executor delegation runs.

## Project impact

Memory layer reaches production-ready L4 for vector recall and agent-facing semantic search; Tier-3 hosts expose unified memory tooling without direct vector SDK access.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/MEMORY.md` |
| Plan | `docs/plan/MEMORY.md` — MEM-VEC-3.*, MEM-DEPTH-5.2, MEM-DEPTH-4.2 |
| ADR | `docs/adr/entries/2026-06-14/ADR-MEM-002.md` (extended wiring only) |
| Audit | MEM-AUDIT-3..6 closed or partial |

## Changed artifacts

- `intergrax/memory/memory_temporal.py` — temporal validity helpers
- `intergrax/memory/memory_vector_namespace.py` — collection namespace resolution
- `intergrax/memory/user_profile_manager.py` — retrieval_service + namespace + temporal filter
- `intergrax/tools/providers/memory/service.py` — `memory.semantic_search` tool
- `intergrax/core/memory_bootstrap.py` — SessionTurnIndexStorePlugin discovery
- `intergrax/runtime/nexus/execution/graph_executor.py` — explore delegation context
- `docs/architecture/MEMORY.md`, `docs/plan/MEMORY.md` — layer completion sync

## Verification

```bash
uv run pytest -m "gate and not no_ci" -q
uv run pytest tests/unit/memory/ tests/integration/applications/test_memory_vector_ltm_wiring.py -q
```

Result: 1409 gate tests passed.

## Risks and follow-ups

- Org memory on Mongo hosts uses in-memory org store fallback (no durable Mongo org store yet).
- Entity graph remains in-process; deep Zep-style graph queries are P2 backlog.
- Postgres multi-tenant memory backends remain P4 (MEM-PERS.3).
