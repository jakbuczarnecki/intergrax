# T-B7-PFP-VERIFY prep (MEP W8) — **nie verified, nie FRZ-FRZ-10 PASS**

**Banner:** Przygotowanie węzła `T-B7-PFP-VERIFY` (CHANGED-SINCE smoke @ FREEZE_SHA). **Nie** oznacza aktywacji ani weryfikacji post-freeze policy.

**Węzeł:** `T-B7-PFP-VERIFY` | **Fala:** W8 | **Pin:** `d043a21c6`

## Przodkowie (DAG)

| Przodek | Stan @ pin |
|---------|------------|
| `T-B7-PFP-ACTIVATE` (W5) | [ ] — nie wykonano na torze |
| `T-B7-L3-FREEZE` / `M3` | [ ] — **brak M3** |
| `FREEZE_CERTIFICATION_SHA` | [ ] — TBD |

## Kryteria wykonania (prep)

| Id | Kryterium | Stan | Dowód / następny krok |
|----|-----------|------|------------------------|
| PFPV-1 | Certifying CHANGED-SINCE probe `--since-sha <M3>` | [ ] | Skrypt governance — replay po M3 |
| PFPV-2 | FRZ-FRZ-10 — policy verified active @ FREEZE_SHA | [ ] | Wymaga ACTIVATE + lock |
| PFPV-3 | Evidence dossier FRZ-FRZ-10 (N11) | [ ] | — |
| PFPV-4 | Uruchomienie **przed** `T-B7-FRZ-FRZ` | [ ] | Runbook krok 12 |
| PFPV-5 | Brak product diff po M3 poza allowlist EBH-7 | BLOCKED | M3 nie istnieje |

**Werdykt prep:** DRAFT — lista otwarta; **nie** „policy verified”.