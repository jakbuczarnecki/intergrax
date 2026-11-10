# PRE-ENT Wave 3 — resume after IIAC ACT-N11 clearance

**Campaign:** `pre_ent_enterprise_certification`  
**Ticket:** `governance/tickets/T-IGX-034-A.md`  
**Blocked by:** **ACT-N11** — `verify_iiac_policy_activation --require-in-force` **FAIL** (7 product paths)  
**ENT recovery SSOT:** [`../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md`](../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md)  
**Date:** 2026-11-10  

PRE-ENT **does not** run Wave 3 independent audits, edit IIAC material paths, fabricate activation, or weaken ACT-N11.

---

## Current gate (factual)

| Field | Value |
| --- | --- |
| `origin/development` | `b7856fd2a95c2368acfb1fdf4d73725f0cd6514c` |
| Product `AUDIT_HEAD` | `03aa10959acf6ec3d81a7a0de8ac8224c8126973` (PR #434 merge) |
| IIAC verify | **FAIL** — same seven paths as ENT recovery doc §2 |
| Wave 3 queue | [`PRE_ENT_WAVE3_AUDIT_HANDOFF_QUEUE.md`](PRE_ENT_WAVE3_AUDIT_HANDOFF_QUEUE.md) — rows **scheduled but STOP** until verify green |

---

## Operator decision **D-IIAC-01** (shared with ENT)

Record once: `Approved: D-IIAC-01=a` (recommended) | `b` | `c` — table in [`IIAC_ACT_N11_RECOVERY_BATCH.md`](../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md) §5.

Until **D-IIAC-01=a** executes through activation, Wave 3 **must not** start.

---

## What ENT / MASTER must do (outside PRE-ENT chats)

| Step | Owner | Output |
| --- | --- | --- |
| 1 | Operator | `Approved: D-IIAC-01=a` (or documented alternative with acceptance of red CI) |
| 2 | Independent IIAC auditor (new session) | Audit dossier + Session B manifest for **7-path** product Δ since `0946b1157` |
| 3 | ENT implementer governance lane | Evidence publication commits; **no** activation record edit without Operator `Approved:` in OD doc |
| 4 | Operator | Personal line: `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <ISO-8601>` |
| 5 | CI / anyone | `verify_iiac_policy_activation --require-in-force` → **exit 0** on post-activation `development` HEAD |
| 6 | CI | `governance-fast` **green** on same SHA (evidence id for PRE-ENT sync) |
| 7 (optional, same batch) | MASTER / ENT | Q-12 / `T-IGX-021-B` validator alignment — only inside approved IIAC re-audit batch |

PRE-ENT **waits** at steps 1–6; **no** parallel PRE-ENT-only IIAC campaign ([`PRE_ENT_IIAC_ENT_HANDOFF_ACT_N11.md`](PRE_ENT_IIAC_ENT_HANDOFF_ACT_N11.md) §3).

---

## Wave 3 resume checklist (exact — run in order after step 6)

All items required unless marked optional.

- [ ] **G0** — `git fetch`; checkout `origin/development` @ documented post-IIAC SHA (new `AUDIT_HEAD`).
- [ ] **G1** — `python -m scripts.governance.verify_iiac_policy_activation --require-in-force` → exit **0** (record command, SHA, exit code).
- [ ] **G2** — Attach green **`governance-fast`** on that SHA (GitHub run URL + job id).
- [ ] **G3** — Update [`PRE_ENT_CAMPAIGN_PROGRESS.md`](PRE_ENT_CAMPAIGN_PROGRESS.md) `AUDIT_HEAD` + IIAC row.
- [ ] **G4** — Update [`PRE_ENT_WAVE3_AUDIT_HANDOFF_QUEUE.md`](PRE_ENT_WAVE3_AUDIT_HANDOFF_QUEUE.md) checkout pin on every row.
- [ ] **G5** — Refresh auditor briefs if SHA changed materially: [`PRE_ENT_WAVE3_AUDITOR_SESSION_BRIEFS.md`](PRE_ENT_WAVE3_AUDITOR_SESSION_BRIEFS.md).
- [ ] **G6** — Re-run Wave 2 serial qual spot checks **@ new AUDIT_HEAD** where tickets require fresh replay (see queue per batch).
- [ ] **G7** — Operator **D-PREENT-05**: open **five separate** `independent-quality-auditor` chats — **one batch per row** in Wave 3 queue; implementer of that batch **must not** PASS.
- [ ] **G8** — **Do not** claim campaign B6/G6 PASS, FRZ promotion, or stage `CLOSED` (unchanged ceiling).

**Still blocking final matrix (Wave 3 may proceed with residuals documented):** G-SKILL defer (D-PREENT-02), F10 block (D-PREENT-04), OD-HIST04 defer (D-PREENT-06/07), Population E → ENT (D-PREENT-03=b).

---

## PRE-ENT actions immediately after G2 (sync only)

1. Publish progress + queue updates (G3–G5).  
2. Notify operator that Wave 3 scheduling (G7) is **unblocked**.  
3. **Do not** merge or revert #434 — already on `development`; only refresh evidence docs if SHA moved.

---

## SSOT links

- IIAC handoff: [`PRE_ENT_IIAC_ENT_HANDOFF_ACT_N11.md`](PRE_ENT_IIAC_ENT_HANDOFF_ACT_N11.md)  
- Formal checkpoint §7: [`PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md`](PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md)  
- ENT ticket W4-03: `governance/tickets/T-IGX-024-B.md`
