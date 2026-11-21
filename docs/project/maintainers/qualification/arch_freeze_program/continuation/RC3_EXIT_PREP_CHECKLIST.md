# RC-3 exit prep checklist (MEP W4) — **nie RC-3 exit, nie G2 PASS**

**Banner:** Przygotowanie dokumentacyjne wyjścia RC-3 / wejścia fali **W4** (PT-Q02 → PT-Q03 → FCS → PT-S04). **Nie** oznacza RC-3 CLOSED, **nie** PT-Q02 PASS, **nie** FCS, **nie** aktywacji RB-2 ani Stage 3.

**Data:** 2026-11-21 (MEP W4, worktree `af-dev-mep-w4`)  
**Pin development (lokalny tor):** `d043a21c67cb57475a4f36a2baff7020718ec7cb` (`d043a21c6`)  
**Pin train:** `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` (`dbaebfa8`)  
**Poprzednik:** TOR C fala 6 — RC-3 **entry prep** @ `e61e5136a` ([`RC3_PREP_DOSSIER.md`](RC3_PREP_DOSSIER.md))

## Legenda

| Symbol | Znaczenie |
|--------|-----------|
| [x] | Udokumentowane w repo (prep only) |
| [ ] | Otwarte — wymaga sesji / operatora |
| BLOCKED | Zależność zewnętrzna |

## A. PT-Q02 — wave audit @ G2 (MEP W4)

Źródło: [`MEP-AF2_v2.1.md`](../MEP-AF2_v2.1.md) W4; WP-12 `PT-Q02` / AR-58, AR-15.

| Id | Kryterium prep | Stan W4 | Dowód / uwagi |
|----|----------------|---------|----------------|
| RC3X-A1 | G2 rung zapisany; RW head = audytowany SHA | [ ] | G2 **nie istnieje** na torze `development` |
| RC3X-A2 | Niezależna sesja audytora (≠ implementer / reviewer) | [ ] | AR-58 — **nie** rozpoczęto |
| RC3X-A3 | Delta IIAC `a73d15ab3..G2` w zakresie Q02 | [ ] | E-pin `a73d15ab3`; G2 **pending** |
| RC3X-A4 | Werdykt wave audit **dokładnie PASS** (RB-2 prerequisite) | [ ] | **nie Q02 PASS** |
| RC3X-A5 | AUD-ENF-2 / AUD-ENF-3 pakiety w evidence | [ ] | — |

## B. PT-Q03 — RB-2 rebind (MEP W4)

Źródło: WP-12 `PT-Q03`; AR-36, AR-60; DP-58, DP-08. Certifier DRAFT tylko jako **cite**, nie signoff.

| Id | Kryterium prep | Stan W4 | Dowód / uwagi |
|----|----------------|---------|----------------|
| RC3X-B1 | PT-Q02 **PASS** (twardy przodek) | BLOCKED | Q02 otwarte |
| RC3X-B2 | Certifier session + manifest (osobna sesja) | [ ] | [`IIAC_SESSION_HANDOFF_RECORD_ARCH_FREEZE_RB2_CERTIFIER_DRAFT.md`](../../iiac_evidence/ARCH-FREEZE/RB-2/certifier/IIAC_SESSION_HANDOFF_RECORD_ARCH_FREEZE_RB2_CERTIFIER_DRAFT.md) — **DRAFT / NOT PASS** |
| RC3X-B3 | Activation delegate (DP-08); **nie** edycja rekordu bez delegacji | [ ] | — |
| RC3X-B4 | `verify_iiac_policy_activation --require-in-force` @ FCS | BLOCKED | [`IIAC_GOVERNANCE_FAST_BLOCKED.md`](IIAC_GOVERNANCE_FAST_BLOCKED.md) |

## C. FCS — `T-AF2-RW-MERGE` (MEP W4)

| Id | Kryterium prep | Stan W4 | Dowód / uwagi |
|----|----------------|---------|----------------|
| RC3X-C1 | Rung `R`: `git diff G2 R` docs only | [ ] | — |
| RC3X-C2 | Merge train → `development`; product tree ≡ G2 | [ ] | **nie FCS** |
| RC3X-C3 | Klasyfikacja **12-commit** `development` przed FCS (MEP N2) | [ ] | Wymaga G2 |
| RC3X-C4 | `governance-fast` green @ G2/FCS (N9) | BLOCKED | ACT-N11 stale pin |

## D. PT-S04 — Stage 3 (`governance-fast` required)

Źródło: WP-12 `PT-S04` / AR-48; po FCS, ≥ 20 consecutive green PR runs + 3 pushes; IIIC re-pin (RB-2) done.

| Id | Kryterium prep | Stan W4 | Dowód / uwagi |
|----|----------------|---------|----------------|
| RC3X-D1 | PT-Q03 / RB-2 complete @ FCS | BLOCKED | B1–B4 |
| RC3X-D2 | Operator authorisation Stage 3 (C9 / AR-48) | [ ] | — |
| RC3X-D3 | ≥ 20 green PR runs + 3 pushes (post-FCS) | [ ] | — |
| RC3X-D4 | AUD-ENF-A part 1 (Stages 1–2) before Stage 3 | BLOCKED | `PT-AUD-ENF12` |

## E. Dokumentacja RC-3 (tor `development`)

| Id | Kryterium | Stan W4 | Dowód |
|----|-----------|---------|-------|
| RC3X-E1 | RC-3 entry prep (fala 6) | [x] | `RC3_PREP_DOSSIER.md` @ `e61e5136a` |
| RC3X-E2 | G2 readiness draft | [x] prep | [`G2_CHECKLIST_DRAFT.md`](G2_CHECKLIST_DRAFT.md) — **nie pin G2** |
| RC3X-E3 | RC-3 exit prep checklist (ten plik) | [x] | MEP W4 @ `d043a21c6` |
| RC3X-E4 | `CHECKPOINT.md` — G4 W4 row | [x] | `CHECKPOINT.md` — MEP W4 |

## Werdykt (exit prep only)

| Warstwa | Werdykt |
|---------|---------|
| MEP W4 dokumentacja exit prep | **IN PROGRESS** — checklisty bez claimów zamknięcia |
| RC-3 exit / program W4 gates | **DRAFT / NOT STARTED** — A1–D4 otwarte lub **BLOCKED** |

Zgodnie z **OD-AF-AUTO-05**: brak G2 PASS, Q02 PASS, FCS, RB-2 activation, PT-S04 LIVE, `ARCH_FREEZE CLOSED`.