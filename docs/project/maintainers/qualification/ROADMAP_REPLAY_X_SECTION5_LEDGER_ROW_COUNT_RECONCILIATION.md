# ROADMAP-REPLAY-X — §5 ledger row-count reconciliation (232 vs R1 233)

**AUDITED_HEAD:** `a2ed11f0e811446ae67fdfd57b4c4997c546e10f`

## Mechanical parse (@ `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` §5)

| Metric | Value |
| --- | --- |
| Physical table data rows | **232** |
| 2-column QUAL-X-P0/P5 rows | **2** (SHA column absent — notes-only) |
| Placeholder row | **`remaining mandatory stages`** (category **P**) |

## R1 **233** vs repo **232**

R1 closed-world inventory counted **233** ledger units as **232 evidence rows + 1 structural placeholder**
(`remaining mandatory stages | — | Fill on closure.`). No accepted parent SHA is missing from the parse;
the delta is **not** a omitted mandatory stage closure row.

## Category totals (rule engine — no rebalance hack)

| Category | Count @ HEAD | R1 plan target | Δ |
| --- | --- | --- | --- |
| **A** | **26** | 26 | 0 |
| **B** | **109** | 99 | 10 |
| **C** | **7** | 12 | -5 |
| **D** | **26** | 29 | -3 |
| **E** | **47** | 22 | 25 |
| **F** | **2** | 30 | -28 |
| **G** | **5** | 4 | 1 |
| **I** | **9** | 12 | -3 |
| **P** | **1** | — | — |

## C / D / F explanation (source-based)

- **C** — rows whose notes contain *Independent audit rejected* on child remediation attempts (EBH-2F chain); accepted parent rows are **not** C.
- **D** — superseded/historical narrative rows (`superseded`, historical recert labels).
- **F** — scoped deferrals: EBH-5 **51 NOT YET**, DIST-SCALE WL-7 **14400s** deferred record.
- **P** — non-evidence placeholder; excluded from replay evidence denominator (**232** data rows).

Regenerate: `uv run python scripts/qualification/generate_roadmap_replay_x_matrices.py`.

