# EQO-X-01 — A4-003 / A4-004 gate counter read (TRAIN_PIN)

**Task:** `EQO-X-01` · **Findings:** EQOA-A4-003 (BLOCKER, `MERGED_TO_TRAIN`), related tier-boundary cluster including A4-004 context  
**Owner:** ARCH-FREEZE (FRZ-BND-03) / `EQO-W2-02` replay obligation

## Pins

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

## Read-only matrix extract (`EQO_A_REM_MASTER_FINDING_MATRIX.jsonl`)

| Finding | Severity | `current_status` | `effective_arch_freeze_impact` | Train evidence |
|---------|----------|------------------|-------------------------------|----------------|
| EQOA-A4-003 | BLOCKER | `MERGED_TO_TRAIN` | `BLOCKS_ARCH_FREEZE` | PR #642 merge on rw-train; **revalidation on TRAIN_PIN still open** (`revalidation_requirement`: true) |
| EQOA-A2-002 | BLOCKER | `BLOCKED` | `BLOCKS_ARCH_FREEZE` | No merge SHA; paired X-02 OWNER_DECISION |

## Method (wave-2)

Read-only inspection of published closure artefacts — **no** product replay, **no** gate PASS claim.

- Impact matrix: `EQO_ARCH_FREEZE_IMPACT_MATRIX.jsonl` rows for A4-003 / A2-002
- Disposition: `EQO_B2_DISPOSITION.jsonl` — A4-003 → `EQO-W2-02` (`OWNER_DECISION`)
- W2 train verification parent: `EQO-W2-02` depends on band-C replay on TRAIN_PIN (deferred to wave-3 AUD + dedicated W2 session)

## Interim disposition

`NOT_CLOSED` — documentation records train merge fact for A4-003; **independent gate replay @ TRAIN_PIN** remains mandatory before ARCH-FREEZE impact downgrade. A2-002 stays exact BLOCKER.

**Audit:** session E pending (wave-3).

## Wave-4 refresh

Post wave-3 AUD packets: gate replay still tied to `EQO-W2-02` + queued session E for WB05 tier-boundary slice; no change to `MERGED_TO_TRAIN` fact for A4-003 @ `TRAIN_PIN`.

**Recorded:** 2026-11-21 · `Ref: governance/tickets/T-IGX-046-F.md` · lane `eqo/wave-4/x`

## Wave-8 read pass (`EQO-X-01`)

**Land:** `569254840`. `MERGED_TO_TRAIN` fact for A4-003 @ `TRAIN_PIN` **unchanged**; tier-boundary replay still blocked on WB05 (`AUD_SESSION_E` **BLOCKED**). W2-02 `MERGED_GATE_PENDING` carries forward.

**Recorded:** 2026-11-21 · lane `eqo/wave-8/x`
## Wave-12 disposition read (EQO-X-01)

**Land:** `292ca3348` (PR #704). `MERGED_TO_TRAIN` for A4-003 @ `TRAIN_PIN` **unchanged**. Wave-6 WB09 pytest exit `0` does **not** satisfy W2-02 tier-boundary gate replay; WB05 session E remains **BLOCKED**.

**Recorded:** 2026-11-21 · lane `eqo/wave-12/x` · `Ref: governance/tickets/T-IGX-046-F.md`
