# PRE-ENT → ENT / IIAC handoff — ACT-N11 (`governance-fast` blocker)

**From:** T-IGX-034-A (PRE-ENT orchestrator)  
**To:** ENT-AUDIT-X stream (`T-IGX-024-A` / `T-IGX-024-B`) + MASTER Q-12 alignment  
**Disposition:** **D-PREENT-08=a** — operator **autoryzowane** 2026-11-10 (chat: „daję zgodę wykonaj”). Await IIAC re-qualification on `development` before PR #434 merge; PRE-ENT executes wait/sync only.  
**Evidence HEAD (PR):** `9ad4fbae47b869edaeae1d19945a6390b37db76a`  
**PR:** [#434](https://github.com/jakbuczarnecki/intergrax/pull/434) — frozen until `governance-fast` green after IIAC  
**Date:** 2026-11-10  

PRE-ENT **will not** edit material IIAC paths, weaken ACT-N11, or merge #434 while `governance-fast` is red.

---

## 1. Failure signature

| Field | Value |
| --- | --- |
| CI run (reference) | `42793932297` |
| Job | `governance-fast` / `129754569140` |
| Step | Governance unit suite (DCP) |
| Local verifier | `uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force` |
| Mechanism | **ACT-N11** — product paths changed after IIAC audited SHA while policy claims `IN_FORCE` |

Current activation record (read-only reference): [`IIAC_POLICY_ACTIVATION_RECORD.json`](../IIAC_POLICY_ACTIVATION_RECORD.json) — **do not edit from PRE-ENT**.

---

## 2. ACT-N11 path inventory (exhaustive @ verify)

### 2.1 Pre-existing on `origin/development` (ENT / enterprise audit slice)

Observed locally with detached checkout of remote `development` tip (post ENT W4 #435 family; 3 paths only):

| # | Path | Likely owner stream |
| --- | --- | --- |
| E1 | `intergrax/platform/enterprise_audit/baseline.py` | ENT-AUDIT-X — documented known regression `governance/tickets/T-IGX-024-B.md` (W4-01) |
| E2 | `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json` | ENT-AUDIT-X W4 evidence / impact map |
| E3 | `intergrax/platform/enterprise_audit/data/schemas/enterprise_audit_report.v1.schema.json` | ENT-AUDIT-X publication schema |

### 2.2 Added on PR #434 branch (`pre-ent/formal-closure-checkpoint-034a`)

Incremental vs `origin/development` (PRE-ENT **T-IGX-034-B** BND-03B):

| # | Path | PRE-ENT ticket |
| --- | --- | --- |
| P1 | `applications/shipped_product_manifest_catalog.py` | T-IGX-034-B |
| P2 | `intergrax/applications/_shared/product_manifest_registry.py` | T-IGX-034-B |
| P3 | `intergrax/cli/doctor_diff_app.py` | T-IGX-034-B (caller wiring) |
| P4 | `intergrax/cli/doctor_health_app.py` | T-IGX-034-B (caller wiring) |

### 2.3 Already on `development` tree but covered by HIST product batch (not extra PR delta)

Wave 0 confirmed on HEAD (P8 `427646f41…` ancestor): **T-IGX-030-B**, **030-C**, **032-A**, **033-A** product paths are **in** the integrated tree. They are **in scope** for any IIAC re-activation that legalizes `development` after DR-12=B, even though PR #434 does not re-touch them.

Per [`HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md`](../historical_certification_campaign/HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md) lanes C1–C4 (030-B, 030-C, 032-A, 033-A).

---

## 3. Existing IIAC corrective batch — do not duplicate

| Artifact | What it already covers | Status |
| --- | --- | --- |
| **HIST_IIAC_SHARED_REACTIVATION_AGREEMENT** | Single shared re-activation after HIST product batch C1–C4; acceptance gate `verify_iiac --require-in-force` exit 0 | **PROPOSED_NO_CONSENT**; Operator P2-WAIT supersession only — **not** executed activation |
| **DR-12 = B** (`HISTORICAL_AUTHORITY_AND_OPERATOR_DECISIONS.md`) | Product edits batched with **next** IIAC re-activation; ENT stream must confirm one-batch vs two-stream | **ANSWERED**; execution held |
| **T-IGX-024-B** | ENT W4 known `governance-fast` FAIL / ACT-N11 on baseline family — track as **separate IIAC re-activation**, not W4-01 blocker | **In flight** on `development` |
| **Q-12** (`HISTORICAL_REMAINING_EXACT_BLOCKERS.md`) | Validator / delegate alignment including baseline promotion + activation line semantics | **MASTER / ENT** — batched with IIAC re-audit/re-activation |
| **PRE-ENT 034-B** | +4 paths (§2.2) | **Held on PR #434** until ACT-N11 cleared |

**Orchestrator conclusion:** PRE-ENT must **not** open a parallel IIAC micro-campaign. Extend or complete the **existing ENT + MASTER IIAC re-activation / re-audit batch** so one activation covers:

1. ENT paths §2.1  
2. HIST product batch already on `development` (§2.3)  
3. PRE-ENT BND paths §2.2 when PR merges **after** green `governance-fast`

---

## 4. Required outputs from ENT/IIAC stream (acceptance)

1. Independent IIAC audit on the **combined** product delta since audited SHA (fresh sessions — not PRE-ENT implementer).
2. Operator **personal** activation line (`operator @`) per protocol — no fabricated signatures in PRE-ENT.
3. `verify_iiac_policy_activation --require-in-force` → exit **0** on the **new** activated SHA on `development`.
4. `governance-fast` green on that SHA (CI evidence for PRE-ENT sync).

Material paths (six IIAC paths + activation record `8045915a` / `f8c29fb8` family) — **ENT/Operator only**; PRE-ENT forbidden.

---

## 5. PRE-ENT actions after IIAC publishes (sequenced)

1. **Rebase or merge** `pre-ent/formal-closure-checkpoint-034a` onto current `origin/development` @ post-activation SHA.  
2. Re-run ACT-N11 verify and full PR **Regression gate** on updated head.  
3. Update [`PRE_ENT_PR434_DIFF_REVIEW.md`](PRE_ENT_PR434_DIFF_REVIEW.md) + [`PRE_ENT_PR434_MERGE_READINESS.md`](PRE_ENT_PR434_MERGE_READINESS.md) with new SHAs.  
4. Operator **D-PREENT-01=b** final merge when mandatory gates green.  
5. Post-merge: compute new **AUDIT_HEAD**, refresh [`PRE_ENT_WAVE3_AUDIT_HANDOFF_QUEUE.md`](PRE_ENT_WAVE3_AUDIT_HANDOFF_QUEUE.md) and briefs — **then** five independent auditor sessions (D-PREENT-05).

---

## 6. Wave 3 resume condition (exact)

All of:

- Formal operator `Approved:` for **D-PREENT-08=a** (or successor) **and** prior §7 line remains valid.  
- PR #434 merged **or** rebased head equals post-IIAC `development` + PRE-ENT commits with **green** Regression gate including `governance-fast`.  
- `AUDIT_HEAD` documented @ merge tip product tree.  
- No open ACT-N11 on `AUDIT_HEAD` (`verify_iiac --require-in-force` exit 0).  
- Campaign blockers unchanged: G-SKILL defer, F10 block, OD-HIST04 defer, Population E @ ENT — still block **final matrix**, not necessarily Wave 3 slice on material batches.

---

## SSOT cross-links

- **ENT recovery batch (supersedes wait-only routing):** [`../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md`](../ent_audit_x/IIAC_ACT_N11_RECOVERY_BATCH.md) (`development` @ `8d757b6ad`) — **D-IIAC-01**  
- Wave 3 resume checklist: [`PRE_ENT_WAVE3_RESUME_AFTER_IIAC.md`](PRE_ENT_WAVE3_RESUME_AFTER_IIAC.md)  
- Merge wait state: [`PRE_ENT_PR434_MERGE_READINESS.md`](PRE_ENT_PR434_MERGE_READINESS.md)  
- Checkpoint §7: [`PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md`](PRE_ENT_FORMAL_CLOSURE_CHECKPOINT.md)  
- ENT handoff: [`ENT_AUDIT_X_HIST_CAMPAIGN_HANDOFF.md`](../historical_certification_campaign/ENT_AUDIT_X_HIST_CAMPAIGN_HANDOFF.md)  
- Ticket: `governance/tickets/T-IGX-034-A.md`, `governance/tickets/T-IGX-024-B.md`
