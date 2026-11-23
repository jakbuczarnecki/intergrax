# W7/W8 — DS-ADM prep dossier (**prep only, not §15 admission**)

**Data:** 2026-11-21 (fala MEP W7, worktree `af-dev-mep-w7`, base `d043a21c67cb57475a4f36a2baff7020718ec7cb`)  
**Węzeł WP-12:** `DS-ADM` — Operator admission steps **not closable before ENTRY**  
**DAG:** `DS-ADM` zależy od `ENTRY` (POST-ENTRY by design — DR-02); **nie** jest prerequisite ENTRY.

**Werdykt wykonania:** **DRAFT / NOT STARTED** — brak ENTRY, brak `OD-ARCH-FREEZE-01`, brak EDER-D-0170/0172/0173 disposition @ admission SHA.

## Zakres DS-ADM (MEP / WP-12)

| Krok | Opis | EDER / OD |
|------|------|-----------|
| ADM-1 | Promote accepted baseline (ENT B6, FREEZE mode) | EDER-D-0170 |
| ADM-2 | ENT stage closure record | EDER-D-0172 |
| ADM-3 | ARCH-FREEZE admission (§15) | EDER-D-0173, **OD-ARCH-FREEZE-01** |
| ADM-4 | Powiązane W8: `T-B7-FRZ-FRZ`, `T-AF2-W8-INDEP-CONFIRM` | przed formalnym CLOSED |

Plan W8: admission dossier + roadmap closure — **poza** tym prep (fala W8 osobno).

## Kolejność względem ENTRY / W7

```text
PT-Q13 → ECAP-2C → PT-VBN → (REQUAL, PROMOTE-*) → COORD → ENTRY → DS-ADM → … → CLOSED
```

W7 prep obejmuje **wyłącznie** dokumentację wejścia operatora; wykonanie DS-ADM następuje po pozytywnym ENTRY i W8-INDEP-CONFIRM wg MEP §N.

## Checklist prep operatora (bez podpisów)

| Id | Przygotować przed sesją admission | Stan |
|----|-----------------------------------|------|
| P-1 | Pakiet FREEZE evidence @ M3 (`T-AF2-POST-BASELINE` ścieżka OD-AF-16b/17) | [ ] OPEN W8 |
| P-2 | OD-ARCH-FREEZE-01 szablon decyzji (Operator-only) | [ ] nie podpisany |
| P-3 | Mapowanie EDER 0170/0172/0173 → wiersze register | [ ] INGESTED only |
| P-4 | `verify_iiac_policy_activation` / IIAC @ admission SHA | [ ] zależne od FCS/RB-2 |
| P-5 | Brak promocji FRZ przez agenta / orchestratora | mandat stały |

## STOP

- Nie wykonywać DS-ADM przed ENTRY.
- Nie deklarować §15 admission PASS, `OD-ARCH-FREEZE-01` signed, ani CLOSED.
- Nie edytować roadmap L499 / checklist FRZ w imieniu Operatora (PT-D09/D10 ticket scope).

## Werdykt prep DS-ADM

| Warstwa | Werdykt |
|---------|---------|
| Prep dokumentacja | **DRAFT** |
| DS-ADM wykonanie | **NOT STARTED** — czeka na ENTRY + W8 bramki |