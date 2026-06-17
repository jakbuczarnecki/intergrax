---
id: IJ-2026-06-17-016
date: 2026-06-17
tiers:
  - tier-0
scope: PLATFORM_FOUNDATION
plan_ref:
  - P1-ARCH-03
status: completed
commit: 3b3a9fc2
adr: none — documentation sync only; no contract or runtime change
---

# P1-ARCH-03 — Root README architecture canon sync

## Operator request

Verify and close the P1-ARCH-03 recommendation: root `README.md` Overview still listed Ephemeral Code Craft as `(planned)` while domain canon and lower README sections already reflect ECC-0…ECC-6 Done.

## Summary

Synchronized root `README.md` Overview and Project snapshot with `CODE_CRAFT` L3 closeout. Registered P1-ARCH-03 as **Closed** in the architecture debt register and PLATFORM_FOUNDATION Appendix B.

## Project impact

First-read onboarding now matches domain pairs and shipped `intergrax/codecraft/` runtime — no stale `(planned)` ECC signal in Overview.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/CODE_CRAFT.md` |
| Plan | `docs/plan/CODE_CRAFT.md` · `docs/plan/PLATFORM_FOUNDATION.md` P1-ARCH-03 |
| Debt | `docs/guides/ARCHITECTURE_DEBT_REGISTER.md` — P1-ARCH-03 |

## Changed artifacts

- `README.md` — ECC status in Overview; Project snapshot date and maturity wording
- `docs/guides/ARCHITECTURE_DEBT_REGISTER.md` — P1-ARCH-03 closed
- `docs/plan/CODE_CRAFT.md` — doc inconsistency row
- `docs/plan/PLATFORM_FOUNDATION.md` — Appendix B.1 P1-ARCH-03 row

## Verification

```bash
python scripts/check_implementation_journal.py
python scripts/check_docs_domain_pairs.py
```

Result: pass.

## Risks and follow-ups

- No automated gate for README Overview vs domain closeout rows — future doc-sync iterations should grep `(planned)` in root README after domain phase closeouts.
