# ARCH-FREEZE continuation checkpoint (development train)

**Updated:** 2026-11-21 (MEP W6 prep, worktree `af-dev-mep-w6`)  
**Base branch:** `development` @ `d043a21c67cb57475a4f36a2baff7020718ec7cb` (`d043a21c6`)  
**Arch-freeze train reference:** `origin/arch-freeze/rw-train` @ `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` — PR **#699** **MERGED** (slice do `1cec68222`)

## Train merge — fakt operatora / automation

| Pole | Wartość |
|------|---------|
| PR | **#699** `af-dev-train-sync` → `arch-freeze/rw-train` |
| Merge SHA | `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` |
| CI przy merge | `governance-fast` **FAILURE (ACT-N11)**; `ci-smoke` **red**; `rag-gate` **red** |
| SSOT slice | `TRAIN_CHERRY_PICK_STATUS.md` |

Merge przy czerwonych checkach — zapisany stan; **nie** spełnia OD-AF-AUTO-01 jako dowód green gate. **RC-2** pozostaje **PARTIAL** (`RC2_EXIT_CHECKLIST.md` — B2 replay **OPEN**; B3 merge fact; replay `governance-fast` zależne od operatora).

## RC-1 (G4 — tor development)

**Werdykt:** **PARTIAL** — otwarte pozycje: [`RC1_OPEN_ITEMS_DEV.md`](RC1_OPEN_ITEMS_DEV.md).

| Pakiet | Plik |
|--------|------|
| RW-PLAN-AUDIT BLOCKED | [`RW_PLAN_AUDIT_STATUS.md`](RW_PLAN_AUDIT_STATUS.md) |
| KR-08 pending | [`OD-AF-RC-KR08.md`](OD-AF-RC-KR08.md) |
| CLOSED-WORLD diagnostic | [`CLOSED_WORLD_RC1_SUMMARY.json`](CLOSED_WORLD_RC1_SUMMARY.json) |
| Pełny admission train | branch `af2/rc1-continuation` @ `36c470249` (cite-only) |

## Parallel workstreams (development tor)

| Group | Slug / worktree | Branch | Ticket | First step | Status |
|-------|-----------------|--------|--------|------------|--------|
| G1 | `af-dev-pt-fc3` | `af-dev-pt-fc3` | T-IGX-043-C | Regression §8.6 follow-up merged (`413fb9186`) | Merged fala 2 |
| G2 | `af-dev-pt-wf` | `af-dev-pt-wf` | T-IGX-043-J | Plan §7; G-09 merged; WF E2–E4 product **BLOCKED** | merged @ `dc083f6ea` |
| G3 | `af-dev-rw-trust` | `af-dev-rw-trust` | T-IGX-051-C | §7 `NT-TRUST-01`..`06` green; FC-3 bundle replay @ `e61e5136a` | **merged** fala 5 |
| G4 (RC-1) | `af-dev-rc1-docs` | `af-dev-rc1-docs` | ticket-less docs | RC-1 open items + RC-2/3 prep sync | **merged this wave** |
| G4 (RC-3 prep) | `af-dev-fala6-close` | `af-dev-fala6-close` | ticket-less docs | RC-3 prep dossier + gap + G2 draft (TOR C fala 6) | **DONE** fala 6 |
| G5 | `af-dev-043m` | `af-dev-043m` | T-IGX-043-M | Slices 1–3 + fala 5 close **9 passed**; Part E manifest **BLOCKED** | in_progress |
| G6 (W6 prep) | `af-dev-mep-w6` | `af-dev-mep-w6` | T-IGX-052-F/G/H scaffold | Campaign + requal + M3 prep checklists | **IN PROGRESS** prep @ `d043a21c6` |
## RW-PLAN-AUDIT (`T-AF2-RW-PLAN-AUDIT`)

**Status: BLOCKED** — brak niezależnej sesji audytora i werdyktu PASS w repozytorium (plan sekcja A.3). To twardy przodek `SCOPE` / G2 na trainie; na torze `development` dokumentujemy stan, nie implementujemy audytu w tej sesji.

## Divergence warning

Po merge **#699** train zawiera slice FC-3 / G-09 / trust, ale **nie** pełny `development` (**286** dev-only / **217** train-only od merge-base `70d73f0f0`). Szczegóły — [`TRAIN_DEV_GAP.md`](TRAIN_DEV_GAP.md). Product remediation on `development` nadal **cherry-pick** / rebase z train tam, gdzie slice się rozjechał — **nie** merge `rw-train` bez mandatu operatora.

## Dependency notes (G1)

`T-IGX-043-C` Part A na `development` importuje `wiring_fails_closed` z `wiring_posture.py` (owner **T-IGX-043-B** / FC-2). Po cherry-pick FC-2 na torze sesji sprawdź HEAD pinu — starsze opisy mogą być **stale**.

## STOP / out of scope

- `ARCH_FREEZE_CLOSED`, FCS bez operatora, **RC-2 CLOSED**, G2 PASS, merge `rw-train` bez mandatu, TENANT 051-A bez FC3+WF+TRUST+approval.
- Edycja `CONTINUATION_AUTONOMY_CHECKPOINT.md` — **inny tor** (poza G4 tą falą).

## G4 — RC-3 prep (fala 6)

Deliverables: `RC3_PREP_DOSSIER.md`, `G2_CHECKLIST_DRAFT.md`, `TRAIN_DEV_GAP.md`; checkpointy w tym katalogu. TOR C fala 6 dokumentacja prep: **DONE** @ `e61e5136a`. Werdykt RC-3 exit (W4): **DRAFT / NOT STARTED** — nie wyjście RC-3.

## G6 — W6 prep (MEP requal / campaigns / M3 entry)

Deliverables: [`W6_PREP_DOSSIER.md`](W6_PREP_DOSSIER.md), [`W6_CAMPAIGN_PREP_CHECKLIST.md`](W6_CAMPAIGN_PREP_CHECKLIST.md), [`W6_REQUAL_PREP_CHECKLIST.md`](W6_REQUAL_PREP_CHECKLIST.md), [`M3_PREP_CHECKLIST.md`](M3_PREP_CHECKLIST.md); ticket scaffolds `T-IGX-052-F`..`H`. **Nie** M3 pin, **nie** kampanie wykonane.
