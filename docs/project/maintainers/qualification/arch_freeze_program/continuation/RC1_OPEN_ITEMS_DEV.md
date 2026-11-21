# RC-1 — otwarte pozycje (tor `development`, G4 docs)

**Data:** 2026-11-21 (fala 5–6 parallel)  
**Pin sesji G4:** `origin/development` @ `dc083f6eadf0f7dbe501fbcf93659e2dd158b825`  
**Pakiet dowodowy train (SSOT RC-1 admission):** branch `af2/rc1-continuation` @ `36c470249` — **nie** scalony w całości na `development` (kolizja `CHECKPOINT.md` + numeracja ticketów PT-G01/G02).

## Legenda

| Symbol | Znaczenie |
|--------|-----------|
| [x] | Udokumentowane na torze `development` |
| [ ] | Otwarte |
| BLOCKED | Wymaga osobnej sesji / operatora |

## Pozycje planu RC-1 (sekcja J)

| Id | Kryterium | Stan na `development` | Dowód / następny krok |
|----|-----------|------------------------|------------------------|
| RC1-J1 | `T-AF2-RW-PLAN-AUDIT` — niezależny PASS | BLOCKED | [`RW_PLAN_AUDIT_STATUS.md`](RW_PLAN_AUDIT_STATUS.md); prep [`RW_PLAN_AUDIT_PREP_DOSSIER.md`](RW_PLAN_AUDIT_PREP_DOSSIER.md) (**T-IGX-050-N**) |
| RC1-J2 | PT-G01 / PT-G02 — merged lub PR | [ ] OPEN | Szkice na trainie (`T-IGX-046-F`/`G` w `36c470249` — **train PT-G01/G02**, nie EQO-A) |
| RC1-J3 | `OD-AF-RC-KR08` — mandat tak/nie | [ ] PENDING | [`OD-AF-RC-KR08.md`](OD-AF-RC-KR08.md); OD-AF-AUTO-01 częściowo na dev |
| RC1-J4 | EDER-DEC map | [ ] cite train | `EDER_DECISION_MAP.json` @ `36c470249` (brak kopii na dev) |
| RC1-J5 | KR-UNKNOWN proposals | [ ] cite train | `KR_UNKNOWN_RC1.json` @ `36c470249` |
| RC1-J6 | CLOSED-WORLD replay | [x] summary | [`CLOSED_WORLD_RC1_SUMMARY.json`](CLOSED_WORLD_RC1_SUMMARY.json) (diagnostic; nie PASS) |
| RC1-J7 | T-AF2-TYP-MEASURE | BLOCKED | `RC1_EVIDENCE.md` @ train — brak `typ_measure.py` |
| RC1-J8 | T-AF2-BND-CLOSURE | BLOCKED | `RC1_EVIDENCE.md` @ train — skrypt sesyjny untracked |
| RC1-J9 | G4 checkpoint + parallel G1–G5 | [x] | [`CHECKPOINT.md`](CHECKPOINT.md) |

## Werdykt G4

**PARTIAL** — tor `development` dokumentuje stan RC-1 i powiązanie z RC-2 **PARTIAL** / RC-3 prep; bramki audytu i PT-G01/G02 product pozostają otwarte lub na trainie.