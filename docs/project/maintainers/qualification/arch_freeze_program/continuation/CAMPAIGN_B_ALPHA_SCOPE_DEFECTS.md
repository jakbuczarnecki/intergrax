# Campaign B alpha — SCOPE defect ID list (Package B)

**Pin:** `51edb2a7c3399cc03f82173b255602efcb208ecf` (execution @ `deea67621120` after REV-MASTER commit in the same worktree)  
**Logs:** `.tmp/session/arch-freeze/campaign-b-alpha/pkg-b-51edb2a/` (not tracked)  
**Nodes:** PT-B01 (B-1), PT-B02 (B-2 alpha)  
**Nature:** ID-only SCOPE input. No reproduction steps. Non-certifying diagnostic.

## Defect IDs (alpha FAIL / ERROR at HEAD diagnostic)

| ID | Campaign case | Owner node |
| --- | --- | --- |
| DEF-CB-B2-S03-01 | PT-B02 / S03 | PT-WF |
| DEF-CB-B2-S03-02 | PT-B02 / S03 | PT-WF |
| DEF-CB-B2-S03-03 | PT-B02 / S03 | PT-WF |

B-1 defined cases: **0** defects (`incomplete=0/9`).

## Package C consumer index

Handoff defect IDs (ID-only; no reproduction steps): `DEF-CB-B2-S03-01`, `DEF-CB-B2-S03-02`, `DEF-CB-B2-S03-03`. Owner node PT-WF. Logs under `.tmp/session/arch-freeze/campaign-b-alpha/pkg-b-51edb2a/`.

## Disposition

Does not set SCOPE, omega, or FCS. Feeds SCOPE predecessor PT-B02 alpha evidence only.

**Package C (train @ `e9bebe0ac`):** [`DEF_CB_B2_S03_PT_WF_DISPOSITION.md`](DEF_CB_B2_S03_PT_WF_DISPOSITION.md) — DEF-CB-B2-S03-01..03 do **not** activate PT-WF E2–E4 product paths.