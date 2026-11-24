# PT-FRZ-RR — Path B contract amendment (AD-AF-PTFRZ-001)

**Status:** Architecture decision **ACCEPTED** (record only; no FRZ promotion, no node closure, no DS-4 admission).

| Field | Value |
| --- | --- |
| Decision ID | **AD-AF-PTFRZ-001** |
| Architecture authority | ChatGPT (ARCH-FREEZE program) |
| Accepted | 2026-11-24 |
| Execution batch | AF-02B |
| Audit SHA | `aaabfc6c8d3a0f7dd43ebd43f47d830bdf1c7a20` |
| Independent dossier | `.tmp/session/arch-freeze/frz-rr/FRZ_RR.md` (session artefact; AF-02A-G6) |
| Executor evidence | `.tmp/session/arch-freeze/frz-rr/AF02A/` |
| Independent audit verdict | **PASS_WITH_FINDINGS** (not unconditional PASS) |

## Binding rule (English rendering)

When the historical WP-4 primary artefacts for PT-FRZ-RR are **irrecoverably absent**, the substantive obligation is met by applying the **current, complete FRZ census** at the identified audit SHA together with the documented Path B assessment. This is the **Architecture-Authority-approved substitute substantive basis**. Requirements for **independence**, **reproducibility** and **formal closure** remain governed by the **existing contract** (independent audit, MEP/DAG semantics, DS-4 hard edge, no checklist promotion from this amendment alone).

## What this amendment changes

| Topic | Before | After (in force) |
| --- | --- | --- |
| PT-FRZ-RR substantive input | Independent re-read of WP-4 classifications (209 rows) | Independent **225-row** census + Path B protection audit at `aaabfc6c8…` |
| WP-4 row-level diff | Required where primary exists | **Not required** while WP-4 primary stays absent |
| Path B / L66 / DP-34 / OD-AF-18 | Unchanged rules | Unchanged; no architecture obligation moved to PROD-Q-only closure |
| `PT_FRZ_RR_CLOSED` | WP-4 re-read completion | Formal node closure still requires contract gates beyond this amendment |
| DS-4 | After PT-FRZ-RR | Unchanged prerequisite edge; **DS4_READY** not set by this decision alone |

## What this amendment does not do

- Does not set `PASS` on any FRZ row; **109 OPEN** rows stay OPEN.
- Does not change **PASS_WITH_FINDINGS** to unconditional PASS for AF-02A-G6.
- Does not set **DS4_READY=YES**, **PT_FRZ_RR_CLOSED=YES**, or **ARCH_FREEZE_CLOSED**.
- Does not edit the Operator-only FRZ checklist (NOT_CLAIMED checklist text remains PT-D10).
- Does not weaken plugin trust (**FRZ-SEC-06**); **OD-AF-18** excludes named sandbox **providers** only, not plugin admission gates.

## Evidence preserved (AF-02A-G6)

| Metric | Value |
| ---: | ---: |
| FRZ rows | 225 |
| OPEN | 109 |
| PASS (census `CURRENT_STATUS`, incl. 8 off-vocabulary labels) | 116 |
| LOW-confidence set audited | 14 |
| Historical hide-risk set | 28 |
| Protected from whole-row Path B | 38 |
| Architecture-scoped OPEN (MEP A.6 decomposition) | 104 |
| Live-scoped OPEN (split + whole live) | 5 |

## Findings disposition (amendment scope)

| ID | Severity | Disposition under AD-AF-PTFRZ-001 |
| --- | --- | --- |
| F-PT-001 | HIGH | **Closed for contract text** — Path B substitute basis recorded here and in MEP §0 / DAG / register |
| F-PT-002 | MED | **Open** — 104 architecture OPEN rows; out of scope for AF-02B (no FRZ promotion) |
| F-PT-003 | MED | **Partially closed in repo docs** — SEC-05/06 `NOT_CLAIMED` semantics aligned in matrix Appendix B/D and MEP split; checklist row text remains Operator-only (PT-D10) |

## Cross-references

- [`ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`](../ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md) — row **AD-AF-PTFRZ-001**
- [`MEP-AF2_v2.1.md`](./MEP-AF2_v2.1.md) — section 0 amendment (i)
- [`WP12_DAG.md`](./WP12_DAG.md) — node **PT-FRZ-RR**
- [`ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md`](../ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md) — Appendix B/D (SEC-05/06, flags 1–2)