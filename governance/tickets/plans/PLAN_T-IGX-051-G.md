# PLAN T-IGX-051-G — T-AF2-RW-TYP

**Status:** plan pending (`plan_pending`); unsigned. **Evidence pin:** `fc2ccb73dea161156d067019548f88e08f033fd9`. **Measure slice:** `.tmp/session/arch-freeze/execution/AF-20D/typ_measure.md` (sibling worktree path, read-only citation). Nothing here is approved, implemented, executed or audited; no test was run while drafting.

**MEP allocation (quote):**

| Field | Value |
|---|---|
| Node | T-AF2-RW-TYP |
| Wave | W3 |
| Predecessors | SCOPE, PT-WA, T-AF2-TYP-MEASURE, OD-AF-11 |
| Ticket | T-IGX-051-G |
| Sensitivity | PROD |
| Scope | frozen contract surfaces; DP-36 |

**OD-AF-11 (quote):** ratchet on `intergrax/contracts/` frozen surfaces; fail on any new `Any`/`object`; TYP-05 reduction target; owner T-IGX-051-G.

**Owner role (Delivery DevOS):** `implementer`.

## 1. Frozen surface

- **Path rule:** all tracked `intergrax/contracts/**/*.py` (**446** files @ pin, `typ_measure.md`).
- **Not in scope:** `intergrax/` outside `contracts/`, `agents/`, `applications/`, draft manifest roots in ECAP drafts.

## 2. Design rules

1. **Ratchet only** — no bulk deletion of existing `Any`/`object` to “match” baselines in one ticket.
2. Baseline JSON is **tracked** after this ticket; today `typ_ratchet_baseline.v1.json` has 0 tracked paths (`typ_measure.md` §Ratchet test).
3. Promote measurers from gitignored `.tmp/session/arch-freeze/` only into paths listed in §3 (or equivalent tracked maintenance scripts).
4. Merges **after** the last train ticket that edits any `intergrax/contracts/**` path (RW-TYP rule in `PLAN_T-IGX-051-B.md` / sibling plans).

## 3. Write set (exact)

### 3a. Measurement and gate scripts (firm)

| Path | Intent |
|---|---|
| `scripts/maintenance/measure_frozen_contract_typ.py` | **New** (or promote from `.tmp/session/arch-freeze/closed-world/typ_measure.py`) — counts Any, object, reflection on frozen surface |
| `scripts/maintenance/measure_frozen_contract_typ05.py` | **New** (or promote from `.tmp/.../measure_typ05.py`) — string-key dispatch / TYP-05 tables |
| `scripts/maintenance/check_frozen_contract_typ_ratchet.py` | **New** — compares HEAD to tracked baseline; exit non-zero on regression |
| `governance/qualification/typ_ratchet_baseline.v1.json` | **New tracked** — baseline metrics + file fingerprints @ operator-approved pin |

### 3b. Tests (firm)

| Path | Marker |
|---|---|
| `tests/unit/architecture/test_rw_typ_ratchet_new_any_fails.py` | `NT-RW-TYP-RATCHET-ANY` |
| `tests/unit/architecture/test_rw_typ_ratchet_new_object_fails.py` | `NT-RW-TYP-RATCHET-OBJECT` |
| `tests/unit/architecture/test_rw_typ_ratchet_baseline_unchanged_passes.py` | steady-state pass at baseline |
| `tests/unit/architecture/test_rw_typ05_string_dispatch_reduction.py` | `NT-TYP-05-2` (string-keyed dispatch tables; not general Any ratchet) |

### 3c. Out of write set

Product feature code outside maintenance/gate paths; `intergrax/contracts/**` content edits except baseline-driven fixes explicitly listed in a future amendment (default: **none** in this ticket).

## 4. Baseline metrics (operator choice required before implement)

| Metric | OD-AF-11 packet 3 | Mechanical @ fc2ccb73 (`typ_measure.md`) |
|---|---:|---:|
| Any | 138 | 164 |
| object | 357 | 352 |
| getattr | (RW baseline) 11 | 11 |
| setattr | 0 | 0 |
| hasattr | 3 | 3 |
| TYP-05 tables | 5 (train cite) | not run |

The tracked `typ_ratchet_baseline.v1.json` must record the **operator-selected** authoritative counts at promotion time. This plan does not pick between packet vs mechanical totals.

**TYP-05 reduction target:** numeric target is an operator decision per OD-AF-11; implementation records it in baseline JSON metadata only after approval.

## 5. Negative tests (required)

| Id | Setup | Expected |
|---|---|---|
| `NT-RW-TYP-RATCHET-ANY` | Add one new `: Any` (or equivalent annotation) in a frozen contract file in test fixture / temporary branch | `check_frozen_contract_typ_ratchet.py` or pytest **FAIL** |
| `NT-RW-TYP-RATCHET-OBJECT` | Add one new `: object` annotation in frozen contract file | **FAIL** |
| Steady control | Clean tree at approved baseline | **PASS**; no metric above baseline |

## 6. Verification (for promoted ticket)

- One `uv run --frozen pytest` invocation: all `tests/unit/architecture/test_rw_typ_ratchet_*.py`.
- `uv run --frozen python scripts/maintenance/check_frozen_contract_typ_ratchet.py` exit 0 on train head after merge.
- CI gate selector: `pytest.mark.gate` where applicable (mirror `T-IGX-051-H` pattern).

## Plan digest

PENDING_COMPUTE