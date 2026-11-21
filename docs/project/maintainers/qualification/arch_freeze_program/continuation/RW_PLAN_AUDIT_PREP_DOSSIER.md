# T-AF2-RW-PLAN-AUDIT — prep dossier (development tor)

**Data:** 2026-11-21 (worktree `af-dev-rw-plan`)  
**Werdykt tej paczki:** **PREP DONE** — artefakty wejścia dla osobnej sesji audytora; **nie** PASS, **nie** symulacja audytu.  
**Werdykt węzła `T-AF2-RW-PLAN-AUDIT`:** **BLOCKED** — bez zmian do czasu niezależnej sesji (`RW_PLAN_AUDIT_STATUS.md`).

## Pin snapshot

| Ref | SHA (40) | Krótki |
|-----|----------|--------|
| Tor `development` (baza fali) | `e69e1ff5819b98c316377bcadf749f61fe00abc2` | `e69e1ff58` |
| `origin/arch-freeze/rw-train` (cite RW-PLAN) | `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` | `dbaebfa8` |
| RC-1 admission (train-only evidence) | `36c470249` | cite-only |

## Mandat operatora

| Id | Stan | Dowód |
|----|------|-------|
| AR-12-07 — niezależna sesja `T-AF2-RW-PLAN-AUDIT` | **GRANTED** | [`ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`](../../ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md) |
| OD-AF-AUTO-06 | **BLOCKED** bez symulacji | [`OPERATOR_DECISIONS_AUTONOMY.md`](OPERATOR_DECISIONS_AUTONOMY.md) |
| Ticket prep | **T-IGX-050-N** | `governance/tickets/T-IGX-050-N.md` |

## Wejścia obowiązkowe (MEP / DAG)

| Wejście | Rola | Lokalizacja @ pin |
|---------|------|-------------------|
| T-AF2-RW-PLAN | DOC — macierz overlap, kolejność merge, NT per CODE_FIX | Train: `docs/project/maintainers/qualification/arch_freeze_program/` (publish na `arch-freeze/rw-train`; brak pełnej kopii na `development` @ `e69e1ff58`) |
| T-AF2-FRZ-MATRIX | Appendix C / macierz FRZ | [`ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md`](../../ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md) |
| REV-MASTER RM-03 | Kryterium audytu (phase check 3, strict) | [`MEP-AF2_v2.1.md`](../MEP-AF2_v2.1.md) §F, [`WP12_DAG.md`](../WP12_DAG.md) węzeł `T-AF2-RW-PLAN-AUDIT` |
| Plan kontynuacji | Macierz AF-RC | Sesja operatora / plan ARCH-FREEZE continuation (cite-only) |

## Drabinka sesji (RC3-J7 + RC1-J1)

| Krok | Rola | Artefakt / komenda | Bramka | Stan prep |
|------|------|--------------------|--------|-----------|
| J7-0 | orchestrator | Ten dossier + ticket `T-IGX-050-N` | — | **DONE** |
| J7-1 | `independent_auditor` | Nowa sesja Cursor, worktree **read-only** od `development` lub HEAD po merge paczek RC; **bez** edycji product paths | **D6=BLOCKED** do startu sesji | OPEN |
| J7-1a | auditor | Przeliczenie WP-12 **phase check 3** (strict) na RW-PLAN + wyjście FRZ-MATRIX | RM-03 | OPEN |
| J7-1b | auditor | **Red-first:** każdy test negatywny pada na pre-fix SHA; „gate już czerwony” nie liczy się | MEP §G | OPEN |
| J7-1c | auditor | Mapowanie regression scope per ticket → 18 CODE_FIX (`FIND-AF2-23`) | FRZ matrix | OPEN |
| J7-1d | auditor | Właściciele 6 konfliktów audit-flag (A4-005, A4-018, A6-005, …) | WP12_TICKETS_DRAFT | OPEN |
| J7-2 | orchestrator | Aktualizacja [`RW_PLAN_AUDIT_STATUS.md`](RW_PLAN_AUDIT_STATUS.md) — **tylko** po sesji; bez PASS z tej fali prep | RC1-J1 | OPEN (prep nie dotyka werdyktu) |
| J7-3 | architect | `SCOPE` train → `DONE_REQUALIFICATION_REQUIRED` po audycie | Po J7-1 | BLOCKED |

## Wyjścia oczekiwane od sesji audytora (nie z tej paczki)

- Werdykt niezależny: `PASS` lub `FINDINGS` — **wyłącznie** w sesji J7-1; brak wpisu PASS w repo = nadal **BLOCKED**.
- Raport phase check 3 (strict) z mapą NT i regression scope.
- Rekord konfliktów audit-flag z przypisanymi ownerami lub eskalacją.

## Werdykt warstwowy

| Warstwa | Werdykt |
|---------|---------|
| Prep dokumentacja + ticket `T-IGX-050-N` | **DONE** @ `e69e1ff58` |
| Niezależna sesja audytora | **NOT STARTED** |
| `T-AF2-RW-PLAN-AUDIT` / RC1-J1 / G2-B3 | **BLOCKED** — bez PASS |

Zgodnie z **OD-AF-AUTO-05** i **OD-AF-AUTO-06**: brak G2 PASS, brak SCOPE closure, brak symulacji audytu w sesji implementacyjnej.