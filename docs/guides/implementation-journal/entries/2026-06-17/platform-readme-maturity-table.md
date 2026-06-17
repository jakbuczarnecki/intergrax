---
id: IJ-2026-06-17-019
date: 2026-06-17
tiers:
  - tier-0
scope: PLATFORM_FOUNDATION
plan_ref:
  - P2-ARCH-03
status: completed
commit: pending
adr: none — onboarding docs only; L0–L4 canon unchanged
---

# P2-ARCH-03 — Root README platform maturity table

## Operator request

Add a credible per-layer maturity view to the root README (external audit P2-ARCH-03) — prefer evidence-backed L-levels over arbitrary percentages.

## Summary

Added README §Current platform maturity: 32/32 L3 harness baseline with links to `harness_maturity_report.py`, IDEAL_HARNESS_L3, and audit map §5; compact table for ACP, Tools, Tier-3, Memory, and AHI with plan evidence and named open gaps. Project snapshot now links to the section. Registered P2-ARCH-03 Done in PLATFORM_FOUNDATION and ARCHITECTURE_DEBT_REGISTER.

## Project impact

Architects evaluating the repo see auditable L0–L4 maturity per hero domain on first read, without misleading percentage bars or declaring the whole platform complete.

## Traceability

| Link | Target |
|------|--------|
| README | `README.md` §Current platform maturity |
| Scorecard | `scripts/harness_maturity_report.py` |
| Plan | `docs/plan/PLATFORM_FOUNDATION.md` P2-ARCH-03 |
| Debt | `docs/guides/ARCHITECTURE_DEBT_REGISTER.md` P2-ARCH-03 |

## Changed artifacts

- `README.md` — new maturity section + snapshot link
- `docs/plan/PLATFORM_FOUNDATION.md` — P2-ARCH-03 Done
- `docs/guides/ARCHITECTURE_DEBT_REGISTER.md` — P2-ARCH-03 closed

## Verification

```bash
python scripts/check_docs_domain_pairs.py
python scripts/check_implementation_journal.py
python scripts/harness_maturity_report.py
```

Result: pass (expected).

## Risks and follow-ups

- Update table rows when domain closeouts change — same drift class as P1-ARCH-03; prefer linking to plan pairs over duplicating status prose.
- Optional: generate hero-domain rows from plan parsers in a future script (not required for P2-ARCH-03).
