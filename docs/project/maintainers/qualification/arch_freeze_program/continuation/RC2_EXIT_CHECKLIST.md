# RC-2 exit checklist (tor `development`) — nie CLOSED

**Data:** 2026-11-21 (autonomia fala 5)  
**Poprzednik:** RC-1 (`G4` / `CHECKPOINT.md`, `RW_PLAN_AUDIT_STATUS.md`)  
**Cel:** warunki wyjścia z rundy kontynuacji RC-2 na torze `development` bez ogłaszania ARCH-FREEZE CLOSED, FCS ani G2 PASS.

## Legenda

| Symbol | Znaczenie |
|--------|-----------|
| [x] | Spełnione w repo / logu sesji |
| [ ] | Otwarte |
| BLOCKED | Zależność zewnętrzna (operator, audyt, CI, B-1) |

## A. Tor development (parallel G1–G5)

| Id | Kryterium | Stan fala 5 | Dowód |
|----|-----------|---------------|-------|
| RC2-A1 | FC-3 core bundle §6/`regression_scope_fc3_bundle.txt` — lokalnie **260 passed**, 3 skipped (fala 4) | [x] | `CONTINUATION_AUTONOMY_CHECKPOINT.md`, `.tmp/session/arch-freeze/REGRESSION-78-fala4/` |
| RC2-A2 | G-09 PT-WF na `development` (NT-WF-G09-01..03 green) | [x] | `PLAN_T-IGX-043-J.md` §7.1 @ `aac4b083b` |
| RC2-A3 | Trust slice 1–2 (NT-TRUST-01, 04, 05 green) | [x] | `test_trust_fail_closed.py`, fala 4 commit `7b1818f5f` |
| RC2-A4 | WF E2–E4 — **BLOCKED** (B-1 / node ids TBD); udokumentowane, bez product slice | [x] | `WF_E2_E4_STATUS.md`, plan §7.1 |
| RC2-A5 | T-IGX-043-M — smoke NT-043M-S1 green → lifecycle `in_progress` (manifest Part E nadal BLOCKED) | [x] | ticket + `test_changed_since_impact_map.py` (fala 5 smoke) |
| RC2-A6 | RW-PLAN-AUDIT — pozostaje **BLOCKED** (brak PASS w repo) | BLOCKED | `RW_PLAN_AUDIT_STATUS.md` |
| RC2-A7 | `051-A`, FCS, freeze G2, `ARCH_FREEZE_CLOSED` — **nie dotykane** | [x] STOP | `OPERATOR_DECISIONS_AUTONOMY.md` OD-AF-AUTO-04..05 |

## B. Train (`arch-freeze/rw-train`) — cherry-pick / PR

| Id | Kryterium | Stan fala 5 | Dowód |
|----|-----------|---------------|-------|
| RC2-B1 | Kolejka slice: FC-3 delta, G-09, trust (commit list) zapisana | [x] | `TRAIN_CHERRY_PICK_STATUS.md` |
| RC2-B2 | Merge train **tylko** przy zielonym `governance-fast` **lub** wąskim ACT-N11 (OD-AF-AUTO-01 / KR-08) | [ ] | CI: `Regression gate` na `development` — **STOP** do green |
| RC2-B3 | PR `development` → `rw-train` utworzony i zmergowany | [ ] | Nie wykonano w fali 5 (bramka RC2-B2) |

## C. Dokumentacja i pin

| Id | Kryterium | Stan fala 5 |
|----|-----------|-------------|
| RC2-C1 | `CONTINUATION_AUTONOMY_CHECKPOINT.md` zaktualizowany (fala 5) | [x] |
| RC2-C2 | `CHECKPOINT.md` — G4 RC-1 → RC-2 checklist done | [x] |
| RC2-C3 | Push `origin/development` | [x] | `e4f7c4999` |

## Werdykt RC-2 (fala 5)

**PARTIAL** — tor `development` spełnia A1–A5 i A7; formalny „exit” RC-2 na trainie (**B2–B3**) czeka na zielony `governance-fast`. To **nie** jest zamknięcie programu ARCH-FREEZE.