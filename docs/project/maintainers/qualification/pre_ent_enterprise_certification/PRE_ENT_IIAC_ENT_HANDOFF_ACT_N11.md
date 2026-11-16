# PRE-ENT → ENT / IIAC handoff — ACT-N11 (cleared)

**From:** T-IGX-034-A (PRE-ENT orchestrator)  
**To:** ENT-AUDIT-X stream (`T-IGX-024-A` / `T-IGX-024-B`) — IIAC lane **complete** for Wave 3 scheduling prereq  
**Disposition:** **D-PREENT-08=a** — operator **autoryzowane** 2026-11-10  
**Date:** 2026-11-10 (post-activation sync)  

**Status:** **ACT-N11 CLEARED** — `verify_iiac_policy_activation --require-in-force` **exit 0** @ `development` HEAD below.

---

## Published activation (Operator + ENT recovery)

| Field | Value |
| --- | --- |
| Operator decision | `Approved: D-IIAC-01=a` + personal activation line |
| OD file | [`IIAC_OD_ACTIVATION_ENT_ACT_N11_2026-11-10.md`](../IIAC_OD_ACTIVATION_ENT_ACT_N11_2026-11-10.md) |
| Operator line | `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ 2026-11-10T13:20:00+01:00` |
| IIAC phase-1 commit | `661978df191d9356058a1312a712fcfa3c890e9e` |
| IIAC phase-2 / `effective_from_sha` | `661978df191d9356058a1312a712fcfa3c890e9e` |
| `development` merge (PR #437) | `1e83ade3f5fb951a9eedd85d4dae89ead48955ee` |
| Session B evidence (faza 1) | `e2bb6e9c0dab0b603f98fb6ffa11ff463da19c0b` |
| Product merge anchor (PR #434) | `03aa10959acf6ec3d81a7a0de8ac8224c8126973` |
| Independent auditor | `24eaa26b-34d7-451e-8acf-7e4949e5d348` — `PASS_SCOPED` |
| Gate evidence | [`PRE_ENT_POST_IIAC_GATE_EVIDENCE.md`](PRE_ENT_POST_IIAC_GATE_EVIDENCE.md) |

## Gate results @ `AUDIT_HEAD`

| Check | Result |
| --- | --- |
| `verify_iiac --require-in-force` @ `1e83ade3f` | **exit 0** — `IN_FORCE` from `661978df1…` |
| `governance-fast` | **SUCCESS** — run `42814981916` / job `129825667490` (tree identical to `1e83ade3f`) |

PRE-ENT **does not** re-open IIAC material paths. Wave 3 scheduling (G7) is **unblocked** for operator **D-PREENT-05** — five separate auditor chats only; **not** started in this batch.

---

## Historical failure signature (pre-activation)

| Field | Value |
| --- | --- |
| CI run (reference) | `42793932297` |
| Job | `governance-fast` / `129754569140` |
| Mechanism | **ACT-N11** — 7 product paths post-`0946b1157` |

Superseded by activation above. Detail retained in [`PRE_ENT_PR434_MERGE_READINESS.md`](PRE_ENT_PR434_MERGE_READINESS.md).

---

## ACT-N11 path inventory (unchanged — audit reference)

Seven-path delta vs `0946b11576d2ee617321fb099ed794f8b21bf5ba` — see [`../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md`](../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md) §2 and Session B evidence under `iiac_evidence/ENT-AUDIT-X/act_n11_recovery/`.

---

## Wave 3 resume

Checklist: [`PRE_ENT_WAVE3_RESUME_AFTER_IIAC.md`](PRE_ENT_WAVE3_RESUME_AFTER_IIAC.md) — G0–G6 **complete** @ `1e83ade3f`; **G7 STOP** (no auditor sessions in this publisher turn).

Queue + briefs: [`PRE_ENT_WAVE3_AUDIT_HANDOFF_QUEUE.md`](PRE_ENT_WAVE3_AUDIT_HANDOFF_QUEUE.md), [`PRE_ENT_WAVE3_AUDITOR_SESSION_BRIEFS.md`](PRE_ENT_WAVE3_AUDITOR_SESSION_BRIEFS.md).

---

## SSOT cross-links

- Recovery batch: [`../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md`](../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md)  
- Campaign progress: [`PRE_ENT_CAMPAIGN_PROGRESS.md`](PRE_ENT_CAMPAIGN_PROGRESS.md)  
- Ticket: `governance/tickets/T-IGX-034-A.md`, `governance/tickets/T-IGX-024-B.md`
