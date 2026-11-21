# T-AF2-W8-INDEP-CONFIRM prep (MEP W8) — **nie independent PASS**

**Banner:** Przygotowanie **nowej niezależnej sesji audytora** (bez kontekstu implementacji). **Nie** jest werdyktem PASS, **nie** zastępuje audytu ECAP-2C ani PT-Q02.

**Węzeł:** `T-AF2-W8-INDEP-CONFIRM` | **Fala:** W8 | **Pin:** `d043a21c6`

## Przodkowie

| Przodek | Stan |
|---------|------|
| `T-AF2-POST-BASELINE` | [ ] |
| `T-B7-PFP-VERIFY` | [ ] |

## Zakres audytu (do odhaczenia w osobnej sesji)

| Id | Zakres | Stan prep |
|----|--------|-----------|
| IC-1 | Wiersze FRZ-FRZ-**01, 02, 09, 10, 13** — verdict niezależny | [ ] |
| IC-2 | ECAP §15 — **12-item** confirmation (mapa → N1–N13 w MEP) | [ ] |
| IC-3 | Audytor **≠** PT-Q02 i **≠** T-B7-ECAP-2C | BLOCKED — brak ECAP-2C |
| IC-4 | Brief: SHA + scope file + evidence index only | [ ] — szablon |
| IC-5 | Explicit statement FRZ-FRZ-13 / RM499 (OD-AF-17 B) | [ ] |

## ECAP §15 → N (skrót do potwierdzenia)

| §15 | N rows (MEP) |
|-----|----------------|
| 1 stages closed | N3 |
| 2 FRZ | N4 |
| 3 vendor | N5 |
| … (12 items) | N1–N13 |

**Werdykt prep:** DRAFT — **brak** `PASS` / `INDEPENDENT_AUDIT_ACCEPTED`.