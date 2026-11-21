# Train ↔ development gap (MEP W3 — pin `d043a21c6`)

**Snapshot:** 2026-11-21 (worktree `af-dev-mep-w3`, fala MEP W3)  
**Disposition:** **MERGED / replay OPEN (QUEUED / STOP)** — PR **#699** **MERGED** @ `dbaebfa8`; replay `governance-fast` na train pin **OPEN**. **Operator STOP:** bez green `governance-fast` — **brak** kolejnego cherry-pick / merge train.

## Piny (MEP W3)

| Ref | SHA (40) | Krótki |
|-----|----------|--------|
| `origin/development` (MEP W3) | `d043a21c67cb57475a4f36a2baff7020718ec7cb` | `d043a21c6` |
| `origin/arch-freeze/rw-train` | `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` | `dbaebfa8` |
| Merge-base dev ↔ train | `70d73f0f08325a39c6b2fd0e3b00f4cab7e83033` | `70d73f0f0` |
| Slice tip (cherry-pick na PR) | `1cec68222751a128feab00e7b80063a09983296e` | `1cec68222` |
| Referencja późniejszego tipu dev (fala 5–6) | `e61e5136aa0617eea69b97ceedd8da38a4d0e796` | `e61e5136a` |

## Liczniki divergence (od merge-base `70d73f0f0`)

| Kierunek | @ `d043a21c6` | @ `e61e5136a` (referencja) | Znaczenie |
|----------|---------------|---------------------------|-----------|
| `rw-train` … not in `development` | **217** | **217** | Historia train / EQO — poza merge mandate |
| `development` … not in `rw-train` | **0** | **286** | Remediacja na torze primary — cherry-pick / cite |

Polecenie SSOT: `git rev-list --count 70d73f0f0..<dev-pin>`; train-only: `git rev-list --count 70d73f0f0..dbaebfa8`.

Na pinie MEP W3 tor `development` **nie wyprzedza** merge-base — późniejsze commity remediacji (do `e61e5136a`) są **poza** tym pinem i nie zmieniają faktu merge #699 na trainie.

Product na `development` musi **cherry-pick/rebase** z trainu tam, gdzie slice już istnieje — **nie** pełny merge `rw-train` bez mandatu operatora (`CHECKPOINT.md`, OD-AF-AUTO-01).

## Kolejka slice (SSOT narracji)

[`TRAIN_CHERRY_PICK_STATUS.md`](TRAIN_CHERRY_PICK_STATUS.md) — seria 1–4 **done**; merge **#699** **MERGED**. **Następny krok:** replay CI @ train HEAD — **nie** nowy slice bez green gate.

### Bramka CI (następny krok)

| Workflow | Stan | Uwagi |
|----------|------|-------|
| `governance-fast` @ train HEAD `dbaebfa8` | **FAILURE (ACT-N11)** | RC2-B2 replay **OPEN** |
| `governance-fast` @ `development` | **BLOCKED** | [`IIIC_GOVERNANCE_FAST_BLOCKED.md`](IIIC_GOVERNANCE_FAST_BLOCKED.md) — stale pin vs drzewo |

**Werdykt:** cherry-pick seria — **on train**; formalny green @ train — **STOP**.

## Stan MEP W3

Worktree **`af-dev-mep-w3`** @ `d043a21c6`; aktualizacja SSOT gap — **bez** claim G2, FCS, green `governance-fast`, **bez** train merge w tej fali.