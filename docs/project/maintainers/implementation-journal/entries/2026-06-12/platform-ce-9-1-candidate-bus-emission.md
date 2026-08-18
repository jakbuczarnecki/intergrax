---
id: IJ-2026-06-12-013
date: 2026-06-12
tiers:
  - tier-1
scope: CONTEXT_ENGINEERING
plan_ref:
  - CE-9.1
  - GAP-CTX-10
status: completed
commit: pending
adr: none — wires existing CE-9.1 event contracts to engine assemble path
---

# CE-9.1 — CONTEXT_CANDIDATE_* bus emission on engine assemble

## Operator request

Close documentation and implementation gaps: finish CE-9.1 bus emission, fix journal gaps, and align CE-2.3 stub provider wording with as-built behavior.

## Summary

Added `record_context_candidate_collected`, `record_context_candidate_dropped`, and `record_context_validation_failed` in `context_skill_recording.py`. `DefaultNexusContextEngine` emits typed events when `event_bus` is present in `ContextProviderContext.handles`; graph `ContextManager` now passes bus + node/agent ids. Extended `ContextCandidatePayloadV1` with `drop_reason`. Closed GAP-CTX-10 in architecture and plan canon.

## Project impact

Context assembly observability matches CE-9.1 plan claims: candidate collect/drop and validation failures are traceable on the runtime event bus, not counters-only.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/CONTEXT_ENGINEERING.md` §12.4, §16 |
| Plan | `docs/project/maintainers/plans/CONTEXT_ENGINEERING.md` CE-9.1, GAP-CTX-10 |
| ADR | `docs/project/technical/adr/entries/2026-06-12/ADR-CTX-001.md` (unchanged) |
| Audit / gap | GAP-CTX-10 **Closed** |

## Changed artifacts

- `intergrax/runtime/events/context_skill_recording.py` — candidate + validation record helpers
- `intergrax/runtime/nexus/context/context_engine.py` — bus emission on collect/dedup/validate
- `intergrax/runtime/nexus/context/context_manager.py` — pass `event_bus` to provider handles
- `intergrax/runtime/events/payloads/canonical.py` — `drop_reason` on candidate payload
- `intergrax/runtime/events/payload_registry.py` — event type → schema mapping
- `intergrax/runtime/events/phase_coverage.py` — phase + ops hints
- Journal sprint 2/3 entries — missing sections restored

## Verification

```bash
uv run pytest tests/unit/runtime/events/test_context_skill_recording.py tests/unit/runtime/nexus/context/test_context_engine.py -m gate -q
python scripts/maintenance/check_implementation_journal.py
python scripts/docs/check_docs_domain_pairs.py
```

Result: pass.

## Risks and follow-ups

- UAEP/ACP paths without `event_bus` in provider handles still rely on `CONTEXT_ASSEMBLED` only.
- Builtin catalog stub providers remain no-op until legacy collectors are wired per-source (out of CE-9.1 scope).
