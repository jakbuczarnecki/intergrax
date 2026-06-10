---
id: IJ-YYYY-MM-DD-NNN
date: YYYY-MM-DD
tiers:
  - tier-0
scope: RAG
plan_ref:
  - M-RAG.23
status: completed
commit: pending
adr: none — one-line rationale when no ADR required
---

# Short title — imperative, matches primary plan_ref

## Operator request

Paraphrase the architect/operator question or instruction (1–3 sentences).
This is a **paraphrase**, not a verbatim chat transcript.

## Summary

What was implemented or delivered? Reference symbols, profiles, or wiring.

## Project impact

What did this unlock for the Harness, agents, or product hosts?

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/<DOMAIN>.md` or `agents/<slug>/ARCHITECTURE.md` |
| Plan | `docs/plan/<DOMAIN>.md` row / phase |
| ADR | `docs/adr/ADR-XXX-NNN.md` or `agents/<slug>/adr/` |
| Audit / gap | GAP-* or AUDIT-IDEAL-* if applicable |

## Changed artifacts

- `path/to/file.py` — brief role

## Verification

```bash
uv run pytest path/to/test.py -q
```

Result: pass / fail / not run — reason.

## Risks and follow-ups

- Remaining risk or deferred item (link plan row).
