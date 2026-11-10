# IIAC ACT-N11 recovery batch — ENT-AUDIT-X / shared `development`

**Campaign:** IIAC continuity (ACT-N11 unblock for `governance-fast` and PRE-ENT Wave 3)  
**Ticket anchors:** `governance/tickets/T-IGX-024-B.md` (W4-03 / T7b), `governance/tickets/T-IGX-034-A.md` (handoff only)  
**Policy:** `IIAC-2026-11-05.1` — record claims **`IN_FORCE`**; verifier **must not** be weakened.  
**Target branch:** `origin/development`  
**Activation audit base (ACT-N11):** `audit_execution_sha` = `0946b11576d2ee617321fb099ed794f8b21bf5ba` (from [`IIAC_POLICY_ACTIVATION_RECORD.json`](../IIAC_POLICY_ACTIVATION_RECORD.json))  
**Published `development` HEAD (2026-11-10):** `b7856fd2a95c2368acfb1fdf4d73725f0cd6514c` (docs atop merge `03aa10959acf6ec3d81a7a0de8ac8224c8126973`)  
**Product AUDIT_HEAD:** `03aa10959acf6ec3d81a7a0de8ac8224c8126973` (merge PR #434; product tree unchanged in `b7856fd2a`)  
**Date:** 2026-11-10  
**Publisher:** PRE-ENT orchestrator documentation batch (no product edits, no activation record edit)

---

## 1. Verification snapshot

| Check | Command | Result |
| --- | --- | --- |
| `git fetch` | `git fetch origin` | OK |
| `origin/development` | `git rev-parse origin/development` | `b7856fd2a95c2368acfb1fdf4d73725f0cd6514c` |
| IIAC in-force gate | `.venv/Scripts/python.exe -m scripts.governance.verify_iiac_policy_activation --require-in-force` | **FAIL exit 1** — **7× ACT-N11** |
| Alias | `verify_iiac --require-in-force` | same module as above |

**Mechanism (read-only):** `scripts/governance/iiac_activation_gate.py` — product loop @ `audit_execution_sha` → `HEAD`; `MATERIAL_GOVERNANCE_PATHS` (six paths) — **no PRE-ENT edits without Operator `Approved:` in a published decision doc**.

---

## 2. ACT-N11 path inventory (exhaustive @ verify)

Introducer = latest merge commit on `origin/development` that changed the path since `0946b1157`; blame tip commit on `development`.

| # | Path | Introducer commit | Subject | Owner stream |
| --- | --- | --- | --- | --- |
| 1 | `applications/shipped_product_manifest_catalog.py` | `01b3e1961f01e1ad578b84f3a20fbde7e16755f6` | Remove Tier-0 dynamic imports of Tier-3 product manifests (BND-03B) | PRE-ENT **T-IGX-034-B** |
| 2 | `intergrax/applications/_shared/product_manifest_registry.py` | `01b3e1961f01e1ad578b84f3a20fbde7e16755f6` | (same) | PRE-ENT **T-IGX-034-B** |
| 3 | `intergrax/cli/doctor_diff_app.py` | `01b3e1961f01e1ad578b84f3a20fbde7e16755f6` | (same) | PRE-ENT **T-IGX-034-B** |
| 4 | `intergrax/cli/doctor_health_app.py` | `01b3e1961f01e1ad578b84f3a20fbde7e16755f6` | (same) | PRE-ENT **T-IGX-034-B** |
| 5 | `intergrax/platform/enterprise_audit/baseline.py` | `786918b7ea921791b685726566b4ddbbe1bb1cd4` | Preserve check_manifest_digest through baseline promotion (C0) | ENT-AUDIT-X **T-IGX-024-B** (W4 / C0) |
| 6 | `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json` | `0e578ea8830642365a480f319ecf46665b26d09f` | ENT-AUDIT-X W4 loop-2: FRZ-AUD-03 unmapped neg + CHANGED-SINCE schema map | ENT-AUDIT-X **T-IGX-024-B** |
| 7 | `intergrax/platform/enterprise_audit/data/schemas/enterprise_audit_report.v1.schema.json` | `8d43143c69ef116ff1c96270b76edf8f79f36086` | ENT-AUDIT-X W4: FRZ-AUD neg tests and drift schema enum | ENT-AUDIT-X **T-IGX-024-B** (W4 corrective family) |

**Note:** HIST lanes C1–C4 (030-B/C, 032-A, 033-A) are on the tree but **not** in the current seven-path delta vs `0946b1157` — they were absorbed before Session B / audit_execution pin. Do **not** re-implement those batches; include them only in narrative scope if a new auditor re-baselines the whole product tree.

---

## 3. Material paths and activation record (hands off)

| Rule | SSOT |
| --- | --- |
| Six `MATERIAL_GOVERNANCE_PATHS` | `scripts/governance/iiac_activation_gate.py` L40–48 |
| Activation record | `docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json` (`effective_from_sha` `8045915a…`, `operator_activation_line` present — **not editable here**) |
| Q-12 queued alignment | `docs/project/maintainers/qualification/historical_certification_campaign/HISTORICAL_REMAINING_EXACT_BLOCKERS.md` row Q-12 — `iiac_baseline_promotion.py`, gate L34/L211/L262, protocol L77/L97 — **T-IGX-021-B**, batched with re-audit |
| Shared re-activation agreement | [`HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md`](../historical_certification_campaign/HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md) — P2-WAIT Operator limb cleared; **no** ENT counter-signature; **no** executed P5 activation |
| PRE-ENT handoff | [`PRE_ENT_IIAC_ENT_HANDOFF_ACT_N11.md`](../pre_ent_enterprise_certification/PRE_ENT_IIAC_ENT_HANDOFF_ACT_N11.md) |

**Forbidden for this batch author:** fake `IN_FORCE`, weaken ACT-N11, edit six material paths or activation record without Operator **`Approved:`** line in a governing doc.

---

## 4. Minimal corrective work (no duplication)

| Completed (do not redo) | Evidence |
| --- | --- |
| HIST product integration P3/P4 | `427646f41…` ancestor of HEAD |
| PRE-ENT BND-03B implementation | `01b3e1961` on `development` |
| ENT W4 FREEZE CLEAN + evidence @ `0ad6511ac…` | T-IGX-024-B §W4 loop-2 |
| D-2 delegated promotion **script** | `scripts/enterprise_audit/iiac_baseline_promotion.py` (outside `intergrax/`) |
| P2-WAIT supersession | `DELEG-HIST-IIAC-P2-SUPERSESSION-2026-11-09.md` |

| Still required (single gap) | Why |
| --- | --- |
| **One IIAC independent re-audit** on product Δ = **exactly the seven paths above** @ target `development` HEAD | Only legal way to advance `audit_execution_sha` / manifest pins |
| **Operator personal activation** `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ …` | Gate rejects delegate-only lines |
| **Optional same batch:** Q-12 validator edits (`T-IGX-021-B`) | Material-path changes must be in **same** audited activation (ACT-N11 material loop) |
| **ENT baseline promotion execution** (C065 / OD-ENT-01) | Uses `iiac_baseline_promotion.py` **after** policy activation covers current `baseline.py` bytes — not a substitute for re-activation |

**Baseline promotion alone** does **not** clear ACT-N11: four PRE-ENT product paths and ENT JSON/schema/baseline bytes remain post-audit until a new activation pins them.

---

## 5. Operator decision package **D-IIAC-01**

Single table — choose **one** option; record `Approved: D-IIAC-01=<a|b|c>` in chat + ticket checkpoint.

| Option | Action | Clears ACT-N11 @ `development`? | Duplicates completed work? | Wave 3 / `governance-fast` |
| --- | --- | --- | --- | --- |
| **(a) Shared full re-activation** | ENT-AUDIT-X leads **one** fresh independent IIAC session on the **7-path** product Δ; publish Session B evidence; Operator signs activation; optional Q-12 in same batch | **Yes** (when activation pins new audit SHA) | **No** — reuses landed ENT + PRE-ENT code | Unblocks after exit 0 verify + green CI |
| **(b) Baseline promotion / ENT-only micro-batch** | Run delegated baseline promotion tooling **without** new IIAC audit + activation | **No** — leaves 4+ PRE-ENT paths + unresolved pins | **Partial duplicate** of W4 work without fixing gate | Stays **FAIL** |
| **(c) Product revert / split PRE-ENT-only IIAC** | Revert `01b3e1961` and/or ENT C0 schema edits to green-window SHAs instead of re-audit | Theoretically could shrink Δ | **Yes** — rejects merged #434 + ENT W4 evidence | **Rejected** — conflicts DR-12=B and operator merge of #434 |

**Orchestrator recommendation:** **(a)** — aligns [`HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md`](../historical_certification_campaign/HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md), T-IGX-024-B W4-03, and PRE-ENT D-PREENT-08 intent (IIAC before Wave 3 audits).

---

## 6. ENT stream deliverables (separate auditor sessions — not PRE-ENT)

1. **Independent IIAC auditor** — new session; audited surface = seven paths + dependency blast radius per protocol; manifest under `docs/project/maintainers/qualification/iiac_evidence/`.
2. **Implementer (ENT ticket)** — publish evidence tree only; **no** self-PASS; **no** activation record edit without Operator approval line in OD file.
3. **Operator** — personal `operator @` activation line; if Q-12 ships, separate Operator approval for `T-IGX-021-B` material edits.
4. **Verification** — `verify_iiac_policy_activation --require-in-force` exit **0** @ new `development` tip; attach CI `governance-fast` run id.
5. **W4-02** — ENT independent auditor on B6 evidence (`T-IGX-024-B`) continues **after** evidence merge; distinct from IIAC Session B.

---

## 7. Cross-links

- PRE-ENT resume checklist: [`PRE_ENT_WAVE3_RESUME_AFTER_IIAC.md`](../pre_ent_enterprise_certification/PRE_ENT_WAVE3_RESUME_AFTER_IIAC.md)  
- Campaign progress: [`PRE_ENT_CAMPAIGN_PROGRESS.md`](../pre_ent_enterprise_certification/PRE_ENT_CAMPAIGN_PROGRESS.md)  
- ENT ticket: `governance/tickets/T-IGX-024-B.md`
