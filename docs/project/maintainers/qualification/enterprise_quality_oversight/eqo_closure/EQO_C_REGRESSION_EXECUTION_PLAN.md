# EQO Campaign C — Regression and layer drift

**Parent:** `EQO-C` · **Pins:** `DOC_PIN` (A3-004), `TRAIN_PIN` (A6-004, drift diffs).

## Tasks (execution order)

| Task | Finding / subject | Execute now | Write set |
|------|-------------------|-------------|-----------|
| **C-01** | EQOA-A3-004 — HIST-85 / FRZ-BND-05 sample | Yes | `eqo_closure/c/A3-004/` |
| **C-02** | EQOA-A6-004 — ENT-AUDIT manifest vs 225 FRZ | Yes (after W2-05 read) | `eqo_closure/c/A6-004/` |
| **C-03** | 18 certification layers drift | Parallel with W3 | `EQO_C_DRIFT.jsonl` + `EQO_C_HISTORICAL_CERTIFICATION_REQUALIFICATION_MATRIX.jsonl` |
| **C-04** | `TARGETED_REPLAY` rows from C-03 | After C-03 class | run logs under `eqo_closure/runs/c/` |
| **C-05** | Bind PT-Q05..Q10 post-FCS | `WAIT_FOR_INTEGRATION` | SHA refs per layer |
| **C-CLS** | Campaign close | Gated | No `UNKNOWN` in C-03; C-01/02 dispositioned |

## C-03 layer checklist (18 rows)

Each row in `EQO_C_HISTORICAL_CERTIFICATION_REQUALIFICATION_MATRIX.jsonl` must end with:

`drift_class ∈ {NO_PRODUCT_DELTA, TARGETED_REPLAY, FULL_REQUAL_REQUIRED}` — **not** `UNKNOWN` at C-CLS.

Layers: TOOL-LIFE-X, SKILL-X, DIST-SCALE-X, TENANT-X, CONFIG-X, GOV-X, CTRL-X, TRACE-X, STATE-X, DUP-X, COMPAT-X, EBH-5/6/7, ROADMAP-REPLAY-X, ENT-AUDIT-X, IIAC, MCP-SERVER-X, PDC-X, CODEOWNERS_DELTA.

**Method:** `git diff A_BASELINE..TRAIN_PIN` and `A_BASELINE..DOC_PIN` on layer surface; historical SHA from roadmap ledger (**read-only**). Tooling: existing change-impact / EVOL-X — no new engine.

## C-01 procedure (A3-004)

1. Load dossier A3 HIST-85 disputed rows from `FINDINGS.jsonl` evidence paths on `DOC_PIN`.
2. Re-evaluate three disagreements; record disposition per row.
3. Verdict: does 30/33 sample still bind on `DOC_PIN`?
4. **Audit:** independent session E.

## C-02 procedure (A6-004)

1. On `TRAIN_PIN`, extract ENT-AUDIT manifest check list.
2. Diff against 225 FRZ IDs — list FRZ without matching check.
3. Compare to #669 “85→91” claim; gap is **evidence**, not narrative closure.

## Dependencies

- W2-05 → C-02.
- W0-01 → C-01, C-03.
- FCS → C-05 only.

## Acceptance (C-CLS)

Both Campaign C findings dispositioned; C-03 complete; C-05 done or exact blocker per `FULL_REQUAL_REQUIRED` row.