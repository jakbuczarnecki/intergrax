---
id: IJ-2026-06-14-001
date: 2026-06-14
tiers:
  - tier-0
  - tier-1
scope: CONTEXT_ENGINEERING
plan_ref:
  - CE-PROV-02
  - CE-PROV-04
  - CE-PROV-05
  - CE-PROV-06
  - CE-PROV-07
  - CE-PROV-09
  - CE-PROV-10
  - CE-PROV-11
  - CE-PROV-GATE
  - CE-PROV-INT
status: completed
commit: pending
adr: none — legacy bridge wiring; no new cross-cutting contract
---

# CE-PROV-WIRE B2–B4 — builtin provider collect closeout (GAP-CTX-20)

## Operator request

Close the remaining Context Engineering provider path gap (GAP-CTX-20): wire all builtin stub `collect()` methods to legacy collectors so the engine assembly spine is production-complete.

## Summary

Extended `legacy_bridge.py` with fragment adapters for RAG, LTM, websearch, tool output, system instructions, policy overlays, attachments, and shared context reads. Wired all remaining `BuiltinContextPlugin` collectors, extended `build_graph_provider_handles` / `ContextManager` for handle propagation, added `check_context_builtin_providers.py`, and integration coverage for RAG + graph_prior on the graph engine path. GAP-CTX-20 closed.

## Project impact

`DefaultNexusContextEngine.assemble()` can now emit provenance-tagged fragments for every §8.4 builtin source when runtime handles are populated — CE plugin collect path is handle-gated live, not stub-empty.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/CONTEXT_ENGINEERING.md` §8.3–§8.4, §16 |
| Plan | `docs/plan/CONTEXT_ENGINEERING.md` CE-PROV-WIRE sprints B2–B4 |
| ADR | none — reuse ADR-CTX-001 contracts |
| Audit / gap | GAP-CTX-20 **Closed** |

## Changed artifacts

- `intergrax/context/providers/legacy_bridge.py`
- `intergrax/context/providers/builtin.py`
- `intergrax/runtime/nexus/context/provider_handles.py`
- `intergrax/runtime/nexus/context/context_manager.py`
- `intergrax/runtime/nexus/context/uaep_assemble.py`
- `scripts/check_context_builtin_providers.py`
- `tests/unit/context/test_legacy_bridge_providers.py`
- `tests/integration/runtime/test_context_provider_wiring.py`
- `docs/architecture/CONTEXT_ENGINEERING.md`
- `docs/plan/CONTEXT_ENGINEERING.md`

## Verification

```bash
uv run pytest tests/unit/context/test_legacy_bridge_providers.py tests/integration/runtime/test_context_provider_wiring.py -m gate -q
python scripts/check_context_builtin_providers.py
python scripts/check_context_tier0_import_boundary.py
```

## Risks and follow-ups

- Handle population on UAEP/ACP hot paths still depends on upstream steps writing metadata keys — providers are live but handle-gated.
- CE-10.3 (`classify_candidates` metadata) remains deferred post CE-PROV-05 validation.
