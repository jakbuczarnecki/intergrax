---
id: IJ-2026-06-12-016
date: 2026-06-12
tiers:
  - tier-0
scope: CONTEXT_ENGINEERING
plan_ref:
  - CE-DOC.11
  - GAP-CTX-20
status: completed
commit: pending
adr: none — documentation-only register; no architecture contract change beyond GAP traceability
---

# CE-DOC.11 — CE-PROV-WIRE phase and GAP-CTX-20 register

## Operator request

After CE-ALIGN closeout, formalize the missing plan phase for wiring eleven builtin stub `collect()` providers to legacy Nexus collectors, and synchronize architecture canon with the as-built stub vs live state.

## Summary

Added phase **CE-PROV-WIRE** to the Context Engineering implementation plan with per-provider CE-PROV-01..11 rows, shared **CE-PROV-BRIDGE**, gate **CE-PROV-GATE**, and integration **CE-PROV-INT**. Registered **GAP-CTX-20** in plan gap matrix and architecture §16. Refreshed architecture §2, §3, §8.3–§8.4 (stub/live table + CE-PROV-WIRE column), §9.1 UAEP note, §17 planned `legacy_bridge.py`. Updated FAUDIT guide and hub status. Sprint register B0–B4 (B0 doc-only Done).

## Project impact

Closes the documentation gap between CE-2.3 (catalog stubs Done) and full §7.1 provider collect. Gives operators a sequenced delivery path (B1 graph core → B2 memory/RAG → B3 shared/tools → B4 gate) without conflating handle wiring (CE-PROV-CTX, Done) with legacy collector delegation.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/CONTEXT_ENGINEERING.md` §8.4, §16 |
| Plan | `docs/plan/CONTEXT_ENGINEERING.md` — CE-PROV-WIRE, CE-DOC.11, sprints B0–B4 |
| ADR | none — plan register only |
| Audit / gap | GAP-CTX-20 Open |

## Changed artifacts

- `docs/plan/CONTEXT_ENGINEERING.md` — CE-PROV-WIRE phase, GAP-CTX-20, CE-DOC.11, sprints B0–B4
- `docs/architecture/CONTEXT_ENGINEERING.md` — §2–§3, §8.3–§8.4, §9.1, §16–§17
- `docs/audit/CONTEXT_ENGINEERING.md` — phase status
- `docs/intergrax_runtime_architecture.md` — hub row

## Verification

```bash
python scripts/check_docs_domain_pairs.py
python scripts/check_implementation_journal.py
```

Result: pass (expected).

## Risks and follow-ups

- **B1–B4** implementation not started — stub providers still return `[]`.
- **CE-10.3** deferred until CE-PROV-05/08 validate fragment metadata shape.
- **GAP-CTX-08**, **GAP-CTX-12** unchanged.
