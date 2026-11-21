# W7 — PT-Q13 prep dossier (**prep only, not M3 / not L3-FREEZE PASS**)

**Data:** 2026-11-21 (fala MEP W7, worktree `af-dev-mep-w7`)  
**Węzeł WP-12:** `PT-Q13` (= `T-B7-L3-FREEZE` — Enterprise audit FULL @ M3, ECAP FULL, FRZ-AUD-10 LIVE, EDER populated)  
**Werdykt wykonania:** **DRAFT / NOT STARTED** — brak `M3`, brak certyfikującego FULL/FREEZE @ FREEZE_SHA. Ten plik jest checklistą wejścia, nie raportem audytu.

## Poprzednik (stan toru `development`)

| Ref | SHA (40) | Krótki |
|-----|----------|--------|
| Bazowy pin fali W7 | `d043a21c67cb57475a4f36a2baff7020718ec7cb` | `d043a21c6` |
| RC-3 / W4 prep | [`RC3_PREP_DOSSIER.md`](RC3_PREP_DOSSIER.md) | **DRAFT** — G2/FCS/PT-Q02 nie spełnione |
| M3 (= `T-B7-LOCK-FREEZE-SHA`) | — | **nie istnieje** — wymaga `PT-Q11` (W6) |

**Nie wolno:** ogłaszać M3, FREEZE PASS, „EDER closed”, FRZ-AUD-10 LIVE bez operatora i niezależnej sesji certyfikującej.

## Mapowanie MEP / WBS

| Id | Rola |
|----|------|
| `PT-Q13` | AUD W7 — certyfikujący audit @ M3 |
| `T-B7-L3-FREEZE` | Alias w WBS — L1–L3 + FULL + FREEZE @ jednym checkout |
| `M3` | Milestone — EBH-7 pin = `FREEZE_CERTIFICATION_SHA` |

Źródło: [`MEP-AF2_v2.1.md`](../MEP-AF2_v2.1.md) fala W7, [`WP12_DAG.md`](../WP12_DAG.md) sekcja W7.

## Twarde prerequisites (DAG) — stan prep @ bazowy pin

| Id | Wymaganie | Stan @ prep |
|----|-----------|-------------|
| `M3` | Pin po `PT-Q11` | [ ] **brak** |
| `PT-Q12` | Family suites @ FCS | [ ] W6 OPEN |
| `PT-Q08`..`PT-Q07`, `PT-Q06` | Requal W6 | [ ] W6 OPEN |
| `PT-S04` | Stage 3 (≥20 green) | [ ] W4 OPEN |
| `PT-R01`, `PT-BA`, `PT-C01`, `PT-PB` | Material / batch | [ ] częściowo W3–W6 |
| `PT-AUD-ENFA`, `PT-AUD-ENF1` | ENF audyty | [ ] W5/W3 |
| `PT-D06`, `PT-R03` | Docs gates | [ ] W2 |
| `DS-Q` | Decyzje pre-ENTRY | [ ] W0 — record-only część |
| `PT-FRZ-RR` | FRZ re-read | [ ] W0 |
| `PT-Q10` | Opcjonalny (DP-04/09 = B) | soft → hard jeśli aktywowany |

Opcjonalny twardy edge: `PT-Q10` → `PT-Q13` przy DP-04 = B lub DP-09 = B ([`WP12_DAG.md`](../WP12_DAG.md)).

## Zakres wykonania PT-Q13 (gdy odblokowany)

Checklist robocza — **nie** claim PASS:

| Krok | Artefakt / komenda (plan) | Owner |
|------|---------------------------|--------|
| Q13-1 | Clean checkout @ `M3`; produkt ≡ FCS | tester |
| Q13-2 | `run_enterprise_audit.py` — tryb FULL + FREEZE certifying | tester+auditor |
| Q13-3 | ECAP FULL @ M3 (AR-10, AR-25) | independent session |
| Q13-4 | Weryfikacja FRZ-AUD-10 **LIVE** (nie COMPENSATING) | operator + auditor |
| Q13-5 | EDER populated / qualification registry zgodny z MEP §G | orchestrator |
| Q13-6 | Raport + SHA w ladder; brak self-referential pin | certifier |

Findings mapowane do węzła (WP-12): 21 ID (m.in. A5-015, A4-010, A2-017) — [`WP12_TICKETS_DRAFT.md`](../WP12_TICKETS_DRAFT.md).

## Siblings / SSOT

| Plik | Rola |
|------|------|
| [`RM499_ENTRY_MAP.md`](../RM499_ENTRY_MAP.md) | RM499-M25a..f → zależne od PT-Q13 |
| [`continuation/CHECKPOINT.md`](CHECKPOINT.md) | Stan kontynuacji |
| [`IIIC_GOVERNANCE_FAST_BLOCKED.md`](IIIC_GOVERNANCE_FAST_BLOCKED.md) | RB-2 / FCS blocker |

## Werdykt prep PT-Q13

| Warstwa | Werdykt |
|---------|---------|
| Dokumentacja prep (fala MEP W7) | **DRAFT** — dostarczona checklista; wykonanie **NOT STARTED** |
| PT-Q13 / M3 | **BLOCKED** — łańcuch W4→W6 + brak M3 |

Zgodnie z **OD-AF-AUTO-05**: żadnego PASS audytu enterprise, FREEZE ani M3 w commicie prep.