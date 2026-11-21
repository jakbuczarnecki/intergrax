# G2 readiness checklist (draft) — **nie pin G2, nie Q02 PASS**

**Banner:** To jest **robocza** lista gotowości do runga `G2` (MEP-AF2 W3 exit → W4 entry). **Nie** oznacza G2 PASS, **nie** zastępuje PT-Q02, **nie** aktywuje RB-2 ani FCS.

**Data:** 2026-11-21 (fala 6 / TOR C)  
**Pin development:** `dc083f6eadf0f7dbe501fbcf93659e2dd158b825` (`dc083f6ea`)  
**Pin train:** `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` (`dbaebfa8`)

## Legenda

| Symbol | Znaczenie |
|--------|-----------|
| [x] | Spełnione w repo / logu (tylko pre-W4 development) |
| [ ] | Otwarte |
| BLOCKED | Zależność zewnętrzna (audyt, RB-2, CI, operator) |

## A. W3 — RW product train

| Id | Kryterium (MEP W3) | Stan fala 6 | Dowód |
|----|-------------------|-------------|-------|
| G2-A1 | Wszystkie tickety W3 train zmergowane na `arch-freeze/rw-train` | [ ] | `TRAIN_DEV_GAP.md` — **241** commitów dev-only |
| G2-A2 | RW head train zamrożony jako kandydat G2 | [ ] | Brak rekordu G2 w ladder |
| G2-A3 | Train CI green poza **exact expected-red** set | [ ] | KR-08 / known-red — nie zwalidowano @ G2 |
| G2-A4 | SCOPE + T-AF2-RW-PLAN na trainie (W2 exit) | BLOCKED | W2 poza torrem development-only |
| G2-A5 | Cherry-pick slice z `development` (FC-3, G-09, trust) | [x] | PR **#699** merged; replay CI **STOP** |

## B. W2 gates (PT-G01 / PT-G02) — na trainie

| Id | Kryterium | Stan fala 6 | Dowód |
|----|-----------|-------------|-------|
| G2-B1 | PT-G01 (BND reconciliation) | [ ] | MEP §G — train backlog |
| G2-B2 | PT-G02 (ownership / STATE-X handoffs) | [ ] | MEP §G |
| G2-B3 | T-AF2-RW-PLAN published + **T-AF2-RW-PLAN-AUDIT** | BLOCKED | `RW_PLAN_AUDIT_STATUS.md` — **nie** PASS |

## C. W4 — wave audit, RB-2, FCS

| Id | Kryterium (MEP W4) | Stan fala 6 | Dowód |
|----|-----------|-------------|-------|
| G2-C1 | **PT-Q02** niezależny wave audit @ G2 — werdykt **dokładnie PASS** | [ ] | **nie Q02 PASS** |
| G2-C2 | IIAC delta audit `a73d15ab3..G2` w Q02 | [ ] | — |
| G2-C3 | **PT-Q03** RB-2 (certifier + activation delegate) | BLOCKED | `IIIC_GOVERNANCE_FAST_BLOCKED.md` |
| G2-C4 | Rung `R`: `git diff G2 R` docs only | [ ] | — |
| G2-C5 | **T-AF2-RW-MERGE** → FCS; product tree ≡ G2 | [ ] | **nie FCS** |
| G2-C6 | `governance-fast` green @ G2/FCS | BLOCKED | ACT-N11 stale pin |
| G2-C7 | Klasyfikacja **12-commit** `development` przed FCS (MEP N2 / CHANGED-SINCE) | [ ] | Nie wykonano — wymaga G2 |

## D. Tor `development` (parallel G1–G5) — kontekst, nie G2

| Id | Kryterium | Stan fala 6 | Dowód |
|----|-----------|-------------|-------|
| G2-D1 | RC-2 tor development A1–A5 | [x] | `RC2_EXIT_CHECKLIST.md` |
| G2-D2 | RW-PLAN-AUDIT BLOCKED bez zmian | BLOCKED | `RW_PLAN_AUDIT_STATUS.md` |
| G2-D3 | `051-A`, ARCH_FREEZE CLOSED — STOP | [x] | `OPERATOR_DECISIONS_AUTONOMY.md` OD-AF-AUTO-04..05 |

## Werdykt (draft)

Wszystkie pozycje A2–C7 otwarte lub **BLOCKED** (poza A5 landing slice i D1/D3). **Nie** utworzono G2. **Nie** ogłoszono Q02 PASS, FCS ani **ARCH_FREEZE CLOSED**.