# W5-H1-FIX1 — NPSC-5F Freeze Baseline Provenance Repair

**Task:** W5-H1-FIX1  
**Type:** Certification integrity repair (no production feature work)

## Root cause

`NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA` and `R4_POST_QUALIFIED_BASELINE_SHA` were pinned to `8879dc8aa6b5be809b3081b61cd5d12b24b6183f`, a **local-only** commit with the same message as W5-H1 on GitHub (`b0a465fc0b8e9f1c9b9e2879f94510fc88b77516`) but **not** an ancestor of `origin/development`. Fresh `git fetch` + clone could not resolve certification provenance; gates could pass only when the orphan object existed locally.

## Invalid SHA

| Field | Invalid value | On `origin/development` |
|-------|---------------|-------------------------|
| NPSC-5F Final baseline | `8879dc8aa6b5be809b3081b61cd5d12b24b6183f` | **NO** (not ancestor) |
| R4 post-qualified baseline | same | **NO** |

## Investigation history

| Milestone | SHA | Remote-reachable |
|-----------|-----|------------------|
| NPSC-5F original freeze | `3500b757c` | YES |
| EE-FINAL-02 re-freeze (documented) | `7a3569c64e892588992635c9cee10c264a9fc200` | YES |
| EE-FINAL-02 reconcile commit | `9ff59d102` | YES |
| DS-E2E-15J closure | `3de9870a7` | YES |
| W5-H1 OTLP (GitHub) | `b0a465fc0b8e9f1c9b9e2879f94510fc88b77516` | YES |
| Erroneous local duplicate | `8879dc8aa6b5be809b3081b61cd5d12b24b6183f` | **NO** |
| Stage projection hardening | `de575ed90` | YES (current `origin/development` at repair) |

Flow: **original NPSC-5F freeze** → **EE-FINAL-02 re-freeze (`7a3569c64`)** → **parallel qualified drifts (VPI projection, CI OTLP defer, W5-H1)** → **W5-H1-FIX1 baseline repair**.

## Candidate baselines

| Candidate SHA | GitHub / object DB | Ancestor of `origin/development` | Meaning | Baseline? |
|---------------|-------------------:|----------------------------------|---------|-----------|
| `8879dc8aa6b5…` | local orphan only | NO | Erroneous W5-H1 pin | **REJECT** |
| `7a3569c64e892…` | YES | YES | EE-FINAL-02 Evidence Plane re-freeze | **SELECT (NPSC-5F Final)** |
| `3de9870a7f6e…` | YES | YES | DS-E2E closure | valid historical, not chosen |
| `b0a465fc0b8e9…` | YES | YES | W5-H1 on GitHub | **SELECT (R4 post-qualified pre-FIX1)** |
| `de575ed90b47…` | YES | YES | Current remote HEAD at repair | not used as freeze point |

## Chosen baseline

- **NPSC-5F Final:** `7a3569c64e892588992635c9cee10c264a9fc200` (EE-FINAL-02 enterprise re-freeze).
- **NPSC-5F/R4 post-qualified:** `b0a465fc0b8e9f1c9b9e2879f94510fc88b77516` (W5-H1 on GitHub; drift tooling paths removed from R4 production protected set).

Proof:

```bash
git cat-file -e 7a3569c64e892588992635c9cee10c264a9fc200^{commit}
git merge-base --is-ancestor 7a3569c64e892588992635c9cee10c264a9fc200 origin/development
```

## Drift classification (`7a3569c64` → `origin/development`)

**QUALIFIED_COMPATIBLE (Evidence Plane protected surfaces):**

- `intergrax/runtime/observability/exporters/**` — W5-H1 OTLP optional capability + CI import defer (`bc0e21782`, `b0a465fc0`).
- `intergrax/runtime/observability/application_execution_stage_signal.py` — VPI stage projection (`491f837ae`) + failure semantics hardening (`de575ed90`); no journal ordering / export bypass.
- Qualification docs, `testing_support/npsc5f_*`, W5-H1 and FIX1 docs, related tests (prefix rules in `npsc5f_final_evidence_plane_drift.py`).

**UNRELATED:** decision system, retrieval multichannel, platform proofs, `intergrax/runtime/execution/**`, applications/agents, core plugins, contracts/tracing (outside frozen exact paths), etc.

**BREAKING:** none accepted — sentinel `collect_breaking_evidence_plane_production_drift` must remain empty.

Protected surfaces `intergrax/runtime/events/**`, `intergrax/contracts/runtime_event.py`, `intergrax/contracts/historical_reconstruction.py` — no unqualified production drift in window.

## Clean-checkout reproducibility

Guards: `tests/unit/testing_support/test_npsc5f_baseline_provenance.py` + `testing_support/frozen_baseline_provenance.py` (`assert_frozen_baseline_reachable`). No reflog; requires `git fetch origin development`.

## W5-H1 status

W5-H1 OTLP dependency boundary, lazy SDK load, wiring checks, qualification docs, and tests remain **QUALIFIED_COMPATIBLE** — not re-baselined to HEAD.

## Final verdict

**PASS** — Freeze baseline restored to remote-reachable EE-FINAL-02 SHA; orphan `8879dc8` rejected; tri-classification preserved; enterprise provenance guards added.

**Production code changed:** NO (testing_support, tests, qualification docs only).
