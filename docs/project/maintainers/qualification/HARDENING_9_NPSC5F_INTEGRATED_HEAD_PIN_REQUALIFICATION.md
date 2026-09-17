# HARDENING-9 — NPSC-5F Integrated H1 Head Pin Requalification

**Task:** `HARDENING_9_NPSC5F_INTEGRATED_HEAD_PIN_REQUALIFICATION`  
**Status:** `REQUALIFIED / RE-FROZEN / PASS`  
**START_HEAD (`origin/development` at session open):** `bde792ba6b1d226c57979ad7c06c322d2613b2a7`  
**FINAL_QUALIFIED_HEAD (code pin; pre qualification commit):** `62fdceac2122738751a8a1caeffe16c986dfe47d`  
**HEAD movement after open:** `62fdceac2` — docs/test gate only (`OUTSIDE_H1_SURFACE`); included in pin.  
**Previous H1 baseline:** `ad1a1e57fc70529aedcbfa27808fffdbfe5fdd14`  
**New H1 baseline:** `62fdceac2122738751a8a1caeffe16c986dfe47d`  
**Previous H1 qualification record:** `145bbd74e6d4175a5b868ed71ed1a6353b2c2c3b`  
**New H1 qualification record (proof commit):** `c94ba8ffe16e80466637a842e8ce71af854ccb67`

## Executive summary

Integrated H1 re-freeze after V2/V3/V4 lower-level sentinel advancement. All post–`ad1a1e57` protected drift on R1/R2/R3/Final surfaces is formally covered by prior requalification tasks; post-V4 commits (`be1b8df82`, `3fec92f48`, `bde792ba6`) are docs/tests/Nexus-only (no new R1/Final BREAKING drift). H1 event-surface paths since old baseline classify to bucket **K** (qualified OBS evolution).

**Production code in this requalification commit:** NO — H1 baseline, qualification record, and evidence only.

## Qualified lower-level records

| Record | SHA |
|--------|-----|
| V2 requalification | `64148ef27` (`5bf824c98` docs) |
| V3 requalification | `5bf824c98` |
| V4 requalification | `23781790d0ce1ca5f9a16170d1adc3e7c3ff2e22` |
| R1 / Final sentinel (V4 pin) | `33576b80521dda7dfc0e5895c943f91dfebffa94` |

## Commit classification summary (`ad1a1e57..bde792ba`)

| Commit / zakres | Klasyfikacja | H1 impact | Wynik |
|-----------------|--------------|-----------|-------|
| `145bbd74` … `7582a8ef4` | ALREADY_QUALIFIED_DRIFT | Maturity / F31 pin semantics | Absorbed at H1 advance |
| `64148ef27`, `5bf824c98` | ALREADY_QUALIFIED_DRIFT | V2/V3 R1/R2/Final sentinels | PASS |
| `39265e263`, `82d7989bf`, `33576b805`, `23781790d` | ALREADY_QUALIFIED_DRIFT | V4 OBS-R3 + sentinel re-freeze | PASS |
| `be1b8df82` | OUTSIDE_H1_SURFACE / SUPPORTING | Nexus metric R4; no R1/Final production drift | PASS |
| `3fec92f48`, `bde792ba6` | OUTSIDE_H1_SURFACE | Docs/tests only | PASS |
| `62fdceac2` | OUTSIDE_H1_SURFACE | Marketplace docs test gate only | PASS |
| Remaining `ad1a1e57..bde792ba` | ALREADY_QUALIFIED_DRIFT | Qualified via V2–V4 + domain tasks outside H1 ownership | PASS |

## R1 / R2 / R3 / Final Evidence Plane

```text
R1: QUALIFIED / NO NEW DRIFT
R2: QUALIFIED / NO NEW DRIFT
R3: QUALIFIED / NO NEW DRIFT
Final Evidence Plane: QUALIFIED / NO NEW BREAKING DRIFT
```

Lower-level baselines **UNCHANGED** (V4 pins retained).

| Sentinel | SHA |
|----------|-----|
| `R1_POST_R2_QUALIFIED_BASELINE_SHA` | `33576b80521dda7dfc0e5895c943f91dfebffa94` |
| `R2_POST_QUALIFIED_BASELINE_SHA` | `22c4793da4ba751fff6c93f780a7f6848650a5d9` |
| `R3_POST_QUALIFIED_BASELINE_SHA` | `48a33db23fafab89b5fdb4ff217dfcb113dd6cc5` |
| `NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA` | `33576b80521dda7dfc0e5895c943f91dfebffa94` |

## H1 post-baseline event surface (pre advance)

| Path | Bucket |
|------|--------|
| `intergrax/runtime/events/event_bus.py` | K |
| `intergrax/runtime/events/persistence_contract.py` | K |
| `intergrax/runtime/events/runtime_event_history.py` | K |
| `intergrax/runtime/events/runtime_event_history_validation.py` | K |
| `intergrax/runtime/events/runtime_event_metric_scope.py` | K |

After H1 baseline advance to `62fdceac2`: empty post-baseline event surface vs `origin/development`.

## Architecture invariants (@ FINAL_QUALIFIED_HEAD)

| Invariant | Wynik |
|-----------|-------|
| Contract-first preserved | **PASS** |
| Pluginability preserved | **PASS** |
| No layer violation | **PASS** |
| No bypass flow | **PASS** |

## Executed gates

See operator report section F (targeted pytest).

## GitHub audit note

> Wprowadzone zmiany muszą zostać zaudytowane na podstawie kodu z GitHub przed uznaniem zadania za ostatecznie zamknięte.
