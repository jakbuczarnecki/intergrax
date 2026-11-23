# W7 — ENTRY prep dossier (**prep only, not L499 entry PASS**)

**Data:** 2026-11-21 (fala MEP W7, worktree `af-dev-mep-w7`, base `d043a21c67cb57475a4f36a2baff7020718ec7cb`)  
**Węzeł WP-12:** `ENTRY` — Operator entry check vs roadmap **L499**  
**Werdykt wykonania:** **DRAFT / NOT STARTED** — brak spełnionych prerequisite `PT-Q13` + `PT-VBN`; brak Operator L499 sign-off.

## Definicja ENTRY (bez obniżania poprzeczki)

- Mechaniczna weryfikacja warunków z mapy RM499 @ **FREEZE_SHA** (nie @ `development` HEAD).
- **FRZ promotions** wyłącznie przez Operatora (plan nie promuje sam).
- Po ENTRY: `T-B7-COORD-ARCH-FRZ`, potem W8 (`T-AF2-POST-BASELINE` ← COORD + PT-Q13 per reguła 050-L).

Źródło: [`WP12_DAG.md`](../WP12_DAG.md), [`RM499_ENTRY_MAP.md`](../RM499_ENTRY_MAP.md), MEP §J rung `ENTRY`.

## Twarde prerequisites

| Węzeł | Stan @ prep |
|-------|-------------|
| `PT-Q13` | [ ] NOT STARTED |
| `PT-VBN` | [ ] NOT STARTED |
| `T-B7-COORD-ARCH-FRZ` | [ ] po REQUAL-HIST-PASS + 13× PROMOTE — W7 OPEN |
| `M3` / FREEZE_SHA | [ ] brak |

Ścieżka krytyczna MEP: `… → PT-Q13 → ECAP-2C → PT-VBN → REQUAL/PROMOTE → COORD → ENTRY`.

## Checklist Operator ENTRY (robocza)

| Krok | Źródło warunku | Uwaga prep |
|------|----------------|------------|
| E-1 | Regeneracja / przejście `RM499_ENTRY_MAP` @ FREEZE_SHA | Mapa @ baseline historyczny — **wymaga refresh** przed ENTRY |
| E-2 | RM499-M01..M29 — każdy wiersz `OPEN` zamknięty dowodem @ M3 lub wyjątek PROD-Q | 41+ OPEN @ stary pin — zob. mapa |
| E-3 | VBN §2.7 + annex (N5) | → PT-VBN, nie ENTRY prep |
| E-4 | Enterprise audit M25a–f | → PT-Q13 |
| E-5 | `architecture-scoped OPEN = 0` (MEP W7 exit) z wyjątkiem zbioru W8 | dokumentować reszidual W8 jawnie |
| E-6 | Operator podpis / OD decyzje entry (bez DS-ADM steps — one **po** ENTRY) | osobny dossier |

**OUT:** `ARCH_FREEZE_CLOSED`, roadmap row L499 checked, global FRZ PASS = 0 claim.

## Siblings

| Plik | |
|------|--|
| [`W7_DS_ADM_PREP_DOSSIER.md`](W7_DS_ADM_PREP_DOSSIER.md) | Kroki admission **post-ENTRY** (DR-02) |
| [`G2_CHECKLIST_DRAFT.md`](G2_CHECKLIST_DRAFT.md) | G2 jeszcze nie PASS — ENTRY nieosiągalne |

## Werdykt prep ENTRY

| Warstwa | Werdykt |
|---------|---------|
| Prep W7 | **DRAFT** |
| ENTRY / L499 | **NOT STARTED** |