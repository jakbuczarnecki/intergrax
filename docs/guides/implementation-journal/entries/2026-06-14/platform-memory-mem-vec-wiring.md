---
id: IJ-2026-06-14-003
date: 2026-06-14
tiers:
  - tier-0
  - tier-1
scope: MEMORY
plan_ref:
  - MEM-VEC-1.1
  - MEM-VEC-1.2
  - MEM-VEC-1.3
  - MEM-VEC-1.4
  - MEM-VEC-2.1
  - MEM-VEC-2.2
  - MEM-VEC-2.3
  - MEM-VEC-2.4
status: completed
commit: pending
adr: ADR-MEM-002
---

# MEM-VEC — LTM and episodic vector recall wiring

## Operator request

Close the open MEM-VEC phase for the Memory layer: wire LTM vector search end-to-end in Tier-3 hosts, ship episodic session turn indexing and semantic recall, and raise FAUDIT-32 past L3 vector-recall gap.

## Summary

Tier-3 wiring injects the integration RAG stack into `UserProfileManager` and `VectorSessionTurnIndexStore`. Session turns index on `append_message`; LTM and episodic hits populate CE provider metadata via `memory_context_invocation.py` (ToolRuntime + UAEP). Fail-closed validation when vector memory flags are set without a resolved backend at host entry points. `UserProfileManager` now uses tenant-scoped store access and `MetadataFilter` vector queries with `index_domain=ltm`.

## Project impact

Reference and lab hosts with `enable_long_term_memory` / `enable_session_vector_index` get working semantic recall instead of silent no-op vector paths. CE `LONGTERM_MEMORY` and `SESSION_HISTORY_SEMANTIC` fragments receive attributable backend hits. `ltm.search` catalog tools bind to the same vector-enabled manager instance.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/MEMORY.md` §5.3, §6.4–6.5, §17 |
| Plan | `docs/plan/MEMORY.md` Phase MEM-VEC MEMV1/MEMV2 |
| ADR | `docs/adr/entries/2026-06-14/ADR-MEM-002.md` |
| Audit / gap | FAUDIT-32 Memory L3→L4 vector recall |

## Changed artifacts

- `intergrax/applications/_shared/memory_vector_wiring.py` — RAG stack resolution + manager/index factories
- `intergrax/applications/_shared/memory_wiring.py` — rag_stack injection into SessionManager
- `intergrax/applications/_shared/runtime_config_bridge.py` — shared manager on ToolWiringContext
- `intergrax/applications/_shared/environment_wiring.py` — host fail-closed + user_profile_manager
- `intergrax/memory/session_turn_index_service.py` — episodic vector adapter
- `intergrax/memory/user_profile_manager.py` — tenant store + MetadataFilter LTM query
- `intergrax/runtime/nexus/context/memory_context_invocation.py` — recall population
- `intergrax/runtime/nexus/session/session_manager.py` — index on append + search API
- `tests/integration/applications/test_memory_vector_ltm_wiring.py` — MEM-VEC-1.3 gate

## Verification

```bash
uv run pytest tests/integration/applications/test_memory_vector_ltm_wiring.py tests/unit/applications/test_memory_vector_wiring.py -q
uv run pytest tests/integration/applications/test_memory_full_stack_lab.py -q
```

Result: pass (10+ tests).

## Risks and follow-ups

- MEM-VEC-3.1 plugin EP and MEM-VEC-3.2 `memory.semantic_search` skill runtime remain P2 backlog.
- LTM vector path uses direct vectorstore query (not shared `RetrievalService`) for three-domain isolation.
