# CONTEXT_ENGINEERING — embedded detail

**Parent hub:** [`CONTEXT_ENGINEERING.md`](../CONTEXT_ENGINEERING.md)

## Verification commands

```bash
# Domain unit tests (after CE-1)
uv run pytest tests/unit/context/ tests/unit/runtime/nexus/context/ -m gate -q

# Application wiring
uv run pytest tests/unit/applications/test_context_wiring.py tests/unit/applications/test_context_runtime_bridge.py -m gate -q

# Acceptance never-overflow
uv run pytest tests/acceptance/test_acceptance_context_compiler_long_session.py -q

# Context Tier-0 import boundary (CE-1.6)
python scripts/maintenance/check_context_tier0_import_boundary.py

# Builtin provider collect wiring (CE-PROV-GATE)
uv run python scripts/maintenance/check_context_builtin_providers.py

# Platform gates
uv run pytest -m gate -q
python scripts/docs/check_docs_domain_pairs.py
uv run python scripts/maintenance/check_observability_gates.py
```

---

## Explicitly out of scope (CE-EXT)

| Item | Owner |
|------|-------|
| L4 adaptive context learning loops | `ADAPTIVE_HARNESS_INTELLIGENCE` |
| Mem0 SaaS auto-ingest | MEMORY MEM-8 |
| Phase K business agents | PLATFORM_FOUNDATION §6.3 |
| RAG retrieval algorithm changes | `plan/RAG.md` |
| New memory store types | `plan/MEMORY.md` |

---

## Suggested PR order

```text
CE-DOC.* (Done)
→ CE-2.6 → CE-1.1 → CE-1.3 → CE-1.4 → CE-2.1 → CE-2.3 → CE-2.5
→ CE-3.1 → CE-3.9 → CE-3.10 → CE-3.7 → CE-3.4 → CE-3.11 → CE-3.8
→ CE-4.1 → CE-5.1 → CE-4.4 → CE-4.5 → CE-4.7
→ CE-VEC-1 (after MEM-VEC-2.1)
→ CE-7.1 → CE-7.2 → CE-7.3 → CE-7.5
→ CE-9.1 → CE-9.2 → CE-10.1
→ CE-11.1 → CE-11.4 → CE-12.1 → CE-12.4
```

---
