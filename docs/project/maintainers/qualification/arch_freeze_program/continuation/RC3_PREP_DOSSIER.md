# RC-3 prep dossier (tor `development`) — **prep only, not RC-3 exit**

**Data:** 2026-11-21 (autonomia fala 6 / TOR C — G4, worktree `af-dev-rc1-docs`)  
**Werdykt tego dokumentu:** **DRAFT / NOT STARTED** — nie jest pinem G2, nie jest wyjściem RC-3, nie ogłasza FCS ani ARCH_FREEZE CLOSED.

## Poprzednik

| Runda | Stan | Dowód |
|-------|------|-------|
| RC-2 exit | **PARTIAL** | [`RC2_EXIT_CHECKLIST.md`](RC2_EXIT_CHECKLIST.md) — A1–A5 spełnione; B2–B3 **STOP** (governance-fast) |
| RC-1 checkpoint | Done (fala 5) | [`CHECKPOINT.md`](CHECKPOINT.md) G4 |

RC-3 **nie rozpoczęto** — ten dossier mapuje wejście (W3→W4 / G2 readiness) bez uznania spełnienia kryteriów wyjścia.

## Pin snapshot (2026-11-21)

| Ref | SHA (40) | Krótki |
|-----|----------|--------|
| `origin/development` (fetch / TOR C) | `f551abb495ad2316a4b5ca0a90dd9aef2a4f28d9` | `f551abb49` |
| Pin sesji G4 / fala 6 parallel | `dc083f6eadf0f7dbe501fbcf93659e2dd158b825` | `dc083f6ea` |
| `origin/arch-freeze/rw-train` | `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` | `dbaebfa8` |
| Merge-base dev ↔ train | `70d73f0f08325a39c6b2fd0e3b00f4cab7e83033` | `70d73f0f0` |

## G2 — **nie istnieje jeszcze**

Na torze `development` **brak** zapisanego runga `G2` (MEP-AF2: `G2` = RW head = wave-audited SHA na trainie). HEAD `development` ≠ G2; nie wolno używać języka „G2 PASS” ani pinować G2 w tym commicie.

## Kryteria RC-3 (plan kontynuacji §J) — prerequisites **nie spełnione**

Źródło ladder: [`MEP-AF2_v2.1.md`](../MEP-AF2_v2.1.md) rungi `G2` / `R` / `FCS` oraz fala **W4** (PT-Q02, PT-Q03, T-AF2-RW-MERGE).

| Id | Kryterium RC-3 (wejście W4) | Stan fala 6 |
|----|-----------------------------|-------------|
| RC3-J1 | RW train kompletny (W3); head train = kandydat G2 | [ ] — cherry-pick **QUEUED** / **STOP** |
| RC3-J2 | Train CI green (poza mechanicznie oczekiwanym red set) | [ ] — merge **#699** przy **red** CI |
| RC3-J3 | **PT-Q02** — niezależny wave audit @ G2, werdykt dokładnie PASS (RB-2 delta) | [ ] — **nie Q02 PASS** |
| RC3-J4 | **PT-Q03 / RB-2** — re-pin IIAC; `verify_iiac_policy_activation --require-in-force` @ FCS | BLOCKED | [`IIIC_GOVERNANCE_FAST_BLOCKED.md`](IIIC_GOVERNANCE_FAST_BLOCKED.md) |
| RC3-J5 | **FCS** — merge train → `development`; product tree ≡ G2 | [ ] — **nie FCS** |
| RC3-J6 | `governance-fast` green @ G2/FCS (N9) | BLOCKED | ACT-N11 @ stale `969cbd964` |
| RC3-J7 | **T-AF2-RW-PLAN-AUDIT** — niezależny PASS | BLOCKED | [`RW_PLAN_AUDIT_STATUS.md`](RW_PLAN_AUDIT_STATUS.md) |

Szczegółowa checklista robocza: [`G2_CHECKLIST_DRAFT.md`](G2_CHECKLIST_DRAFT.md) (**nie** jest pinem G2).

## Pliki kontynuacji (siblings)

| Plik | Rola |
|------|------|
| [`TRAIN_DEV_GAP.md`](TRAIN_DEV_GAP.md) | Luka train ↔ development + kolejka slice |
| [`TRAIN_CHERRY_PICK_STATUS.md`](TRAIN_CHERRY_PICK_STATUS.md) | SSOT kolejki (nie zmieniana w TOR C) |
| [`IIAC_GOVERNANCE_FAST_BLOCKED.md`](IIAC_GOVERNANCE_FAST_BLOCKED.md) | governance-fast / RB-2 |
| [`RC2_EXIT_CHECKLIST.md`](RC2_EXIT_CHECKLIST.md) | Poprzednik PARTIAL |
| [`CONTINUATION_AUTONOMY_CHECKPOINT.md`](CONTINUATION_AUTONOMY_CHECKPOINT.md) | Fala 6 SHA |
| [`OPERATOR_DECISIONS_AUTONOMY.md`](OPERATOR_DECISIONS_AUTONOMY.md) | OD-AF-AUTO-04..05 — bez CLOSED / G2 w autonomii |

## Werdykt RC-3 prep (fala 6)

**DRAFT / NOT STARTED** — dokumentacja wejścia; wszystkie twarde bramki W4 pozostają otwarte lub BLOCKED. Zgodnie z **OD-AF-AUTO-05**: brak ogłoszenia ARCH_FREEZE CLOSED, FCS, G2 PASS ani symulacji RW-PLAN-AUDIT.
