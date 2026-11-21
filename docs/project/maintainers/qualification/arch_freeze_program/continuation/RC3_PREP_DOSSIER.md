# RC-3 prep dossier (tor `development`) — **prep only, not RC-3 exit**

**Data:** 2026-11-21 (TOR C fala 6 entry prep; MEP W4 exit prep @ `af-dev-mep-w4`)  
**Werdykt RC-3 (program W4):** **DRAFT / NOT STARTED** — nie jest pinem G2, nie jest wyjściem RC-3, nie ogłasza FCS ani ARCH_FREEZE CLOSED.  
**Werdykt TOR C fala 6 (entry prep):** **DONE** @ `e61e5136aa0617eea69b97ceedd8da38a4d0e796` — ten dossier, `G2_CHECKLIST_DRAFT.md`, `TRAIN_DEV_GAP.md`, G4 w `CHECKPOINT.md`.  
**Werdykt MEP W4 (exit prep docs):** **IN PROGRESS** @ `d043a21c67cb57475a4f36a2baff7020718ec7cb` — [`RC3_EXIT_PREP_CHECKLIST.md`](RC3_EXIT_PREP_CHECKLIST.md); bez claim G2 / Q02 / FCS / S04 LIVE.

## Poprzednik

| Runda | Stan | Dowód |
|-------|------|-------|
| RC-2 exit | **PARTIAL** | [`RC2_EXIT_CHECKLIST.md`](RC2_EXIT_CHECKLIST.md) — A1–A5 spełnione; B2–B3 **STOP** (governance-fast) |
| RC-1 checkpoint | Done (fala 5) | [`CHECKPOINT.md`](CHECKPOINT.md) G4 |

RC-3 **nie rozpoczęto** (wyjście W4) — ten dossier mapuje wejście (W3→W4 / G2 readiness) bez uznania spełnienia kryteriów wyjścia.

## Pin snapshot (2026-11-21)

| Ref | SHA (40) | Krótki |
|-----|----------|--------|
| `development` (MEP W4 worktree) | `d043a21c67cb57475a4f36a2baff7020718ec7cb` | `d043a21c6` |
| TOR C entry prep (historyczny) | `e61e5136aa0617eea69b97ceedd8da38a4d0e796` | `e61e5136a` |
| `origin/arch-freeze/rw-train` | `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` | `dbaebfa8` |
| Merge-base dev ↔ train | `70d73f0f08325a39c6b2fd0e3b00f4cab7e83033` | `70d73f0f0` |
| Dev-only commits (od merge-base) | **320** | `git rev-list --count rw-train..d043a21c6` |

## G2 — **nie istnieje jeszcze**

Na torze `development` **brak** zapisanego runga `G2` (MEP-AF2: `G2` = RW head = wave-audited SHA na trainie). HEAD `development` ≠ G2; nie wolno używać języka „G2 PASS” ani pinować G2 w tym commicie.

## Kryteria RC-3 (plan kontynuacji §J) — prerequisites **nie spełnione**

Źródło ladder: [`MEP-AF2_v2.1.md`](../MEP-AF2_v2.1.md) rungi `G2` / `R` / `FCS` oraz fala **W4** (PT-Q02, PT-Q03, T-AF2-RW-MERGE).

| Id | Kryterium RC-3 (wejście W4) | Stan fala 6 |
|----|-----------------------------|-------------|
| RC3-J1 | RW train kompletny (W3); head train = kandydat G2 | [ ] — cherry-pick **QUEUED** / **STOP** |
| RC3-J2 | Train CI green (poza mechanicznie oczekiwanym red set) | [ ] — merge **#699** przy **red** CI |
| RC3-J3 | **PT-Q02** — niezależny wave audit @ G2, werdykt dokładnie PASS (RB-2 delta) | [ ] — **nie Q02 PASS** |
| RC3-J4 | **PT-Q03 / RB-2** — re-pin IIAC; `verify_iiac_policy_activation --require-in-force` @ FCS | BLOCKED | [`IIAC_GOVERNANCE_FAST_BLOCKED.md`](IIAC_GOVERNANCE_FAST_BLOCKED.md) |
| RC3-J5 | **FCS** — merge train → `development`; product tree ≡ G2 | [ ] — **nie FCS** |
| RC3-J6 | `governance-fast` green @ G2/FCS (N9) | BLOCKED | ACT-N11 @ stale `969cbd964` |
| RC3-J7 | **T-AF2-RW-PLAN-AUDIT** — niezależny PASS | BLOCKED | [`RW_PLAN_AUDIT_STATUS.md`](RW_PLAN_AUDIT_STATUS.md) |
| RC3-J8 | **PT-S04** Stage 3 — `governance-fast` required (AR-48; po FCS + ≥ 20 green runs) | BLOCKED | W4 exit prep §D w `RC3_EXIT_PREP_CHECKLIST.md` |

Szczegółowa checklista robocza: [`G2_CHECKLIST_DRAFT.md`](G2_CHECKLIST_DRAFT.md) (**nie** jest pinem G2). Exit prep W4 (Q02/Q03/FCS/S04): [`RC3_EXIT_PREP_CHECKLIST.md`](RC3_EXIT_PREP_CHECKLIST.md).

## Pliki kontynuacji (siblings)

| Plik | Rola |
|------|------|
| [`TRAIN_DEV_GAP.md`](TRAIN_DEV_GAP.md) | Luka train ↔ development + kolejka slice |
| [`TRAIN_CHERRY_PICK_STATUS.md`](TRAIN_CHERRY_PICK_STATUS.md) | SSOT kolejki (nie zmieniana w TOR C) |
| [`IIAC_GOVERNANCE_FAST_BLOCKED.md`](IIAC_GOVERNANCE_FAST_BLOCKED.md) | governance-fast / RB-2 |
| [`RC3_EXIT_PREP_CHECKLIST.md`](RC3_EXIT_PREP_CHECKLIST.md) | MEP W4 exit prep (nie RC-3 CLOSED) |
| [`RC2_EXIT_CHECKLIST.md`](RC2_EXIT_CHECKLIST.md) | Poprzednik PARTIAL |
| [`CONTINUATION_AUTONOMY_CHECKPOINT.md`](CONTINUATION_AUTONOMY_CHECKPOINT.md) | Fala SHA (Sesja A) |
| [`OPERATOR_DECISIONS_AUTONOMY.md`](OPERATOR_DECISIONS_AUTONOMY.md) | OD-AF-AUTO-04..05 — bez CLOSED / G2 w autonomii |

## Werdykt RC-3 prep (fala 6)

| Warstwa | Werdykt |
|---------|---------|
| TOR C dokumentacja (fala 6) | **DONE** — zamknięcie deliverables planu prep; pin `e61e5136a` |
| RC-3 exit / wejście W4 | **DRAFT / NOT STARTED** — wszystkie twarde bramki W4 otwarte lub **BLOCKED** |
| MEP W4 exit prep (docs) | **IN PROGRESS** — `RC3_EXIT_PREP_CHECKLIST.md` @ `d043a21c6`; nadal **nie** exit |

Zgodnie z **OD-AF-AUTO-05**: brak ogłoszenia ARCH_FREEZE CLOSED, FCS, G2 PASS ani symulacji RW-PLAN-AUDIT.