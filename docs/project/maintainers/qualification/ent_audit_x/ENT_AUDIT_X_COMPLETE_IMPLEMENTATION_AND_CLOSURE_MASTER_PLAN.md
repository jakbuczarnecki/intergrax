# ENT-AUDIT-X — Complete implementation and closure master plan

**Schema:** execution SSOT for stage ENT-AUDIT-X @ `development`  
**PLAN_BASELINE_SHA:** `8a41d32043b89e555d7c18e052cf7022b8c94681`  
**PLAN_PUBLICATION_SHA:** `6328f0cf915ed1f4cee13acb5eb68f07a7438b8e`  
**Disposition:** **MASTER PLAN CORRECTED / PUBLISHED / READY FOR FINAL INDEPENDENT ACCEPTANCE**  
**Stage status @ HEAD:** OPEN — no product runner; FRZ-AUD-01..10 OPEN  

**Normative authorities @ PLAN_BASELINE_SHA (blob SHA):**

| Document | Blob SHA @ PLAN_BASELINE_SHA |
| --- | --- |
| [PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) | `f53d67c57abd33a680bb1ccecb9daf9497f4a463` |
| [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](../PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) | `3cd1a17231b35edad455b6e327a7f8f68d5e4ab7` |
| [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](../ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) | `86628db3c08d271eb0ef180c8b9900be72f11151` |
| [SYSTEM_INVARIANTS.md](../../technical/guides/SYSTEM_INVARIANTS.md) | `fe6d2d2711c94a132602babdecbda6d75d5651ab` |

## Attachments (mechanical detail — populated in this planning wave)

| # | Document |
| --- | --- |
| C | [ANNEX_C_CONTROL_MATRIX.md](./ANNEX_C_CONTROL_MATRIX.md) — 85 checks |
| D | [ANNEX_D_VBN_DOMAINS_1_31.md](./ANNEX_D_VBN_DOMAINS_1_31.md) |
| B | [ANNEX_B_FRZ_AUD_CLOSURE_MATRIX.md](./ANNEX_B_FRZ_AUD_CLOSURE_MATRIX.md) |
| F | [ANNEX_F_GAP_REGISTER.md](./ANNEX_F_GAP_REGISTER.md) |
| G | [ANNEX_G_GATE_REGISTRY.md](./ANNEX_G_GATE_REGISTRY.md) + [ANNEX_G2_CONTROL_GATE_MAPPING.md](./ANNEX_G2_CONTROL_GATE_MAPPING.md) |
| H | [ANNEX_H_WBS.md](./ANNEX_H_WBS.md) — **273** tasks / 13 batches |
| SHA | [ENT_AUDIT_X_SHA_LADDER.md](./ENT_AUDIT_X_SHA_LADDER.md) |

---

## A. Executive architecture assessment

**TO-BE:** Tier-0 modular **Enterprise Audit System** — `EnterpriseAuditRunner` orchestrating allowlisted **AuditCheck** adapters (pytest/script, read-only by default), emitting versioned **AuditReport** + **AuditExecutionManifest**, comparing **ACCEPTED_BASELINE_SHA** via **AuditDriftEngine**, exposed through CLI and CI modes FAST/FULL/FREEZE/CHANGED-SINCE with exit codes 0–3.

**AS-IS:** ~411 scattered `test_*gate*.py` files and 33 qualification directories **EXIST_AND_QUALIFIED** per prior stages at **scoped SHAs**; **no** unified runner (**MISSING**). ECAP-P2 applies: historical CLOSED ≠ current-HEAD PASS.

**Forbidden:** Second governance, execution authority, semantic registry, contract authority, provider selector, qualification authority, or parallel roadmap.

**Path B:** Dimension A audit fully plannable without live cloud; Dimension B (`FRZ-PRD` live) stays OPEN — reports must list `QUALIFICATION_LIMITATIONS`.

---

## B. Full repository inventory (closed-world summary)

| Mechanism | Path | Status @ HEAD |
| --- | --- | --- |
| Tier 0–3 product code | `intergrax/`, `agents/`, `applications/` | EXISTS — audited via gates |
| Qualification SSOTs | `tests/qualification/{ebh_5,ebh_6,ebh_7,evol_x,dup_x,roadmap_replay_x,tenant_x,config_x,gov_x2,state_x,...}` | EXISTS_AND_QUALIFIED scoped |
| CI regression profiles | [scripts/ci/run_regression_gate_ci.py](../../../scripts/ci/run_regression_gate_ci.py) | EXISTS_PARTIAL |
| Script gates | [scripts/gates/](../../../scripts/gates/) (15 modules) | EXISTS_PARTIAL |
| Enterprise audit product | `intergrax/platform/enterprise_audit/` | **MISSING** |
| ENT qual suite | `tests/qualification/ent_audit_x/` | **MISSING** |
| FRZ-AUD criteria | checklist rows | **OPEN** |
| GitHub branch protection | `development` | **MISSING** (gh API 404 verified) |

Semantic/composition ownership unchanged — ENT **orchestrates** only (roadmap §3.1).

---

## C. AS-IS / TO-BE gap matrix

See [ANNEX_F_GAP_REGISTER.md](./ANNEX_F_GAP_REGISTER.md). **Separate counts:** GAP-C=14, GAP-P=9, mandatory checks=85, GAP-COV=85, GAP-D=8, GAP-B=6, GAP-O=9.

---

## D. Requirements traceability

| Source | Traced to |
| --- | --- |
| Roadmap §3.1 (coverage, modes, baseline, reports, exit codes, post-freeze) | Annex C + B9/B7/B8 + FRZ-AUD |
| Roadmap §3.0.8 VBN | Annex D + VBN-* checks + B6 |
| Roadmap §3.0.9 EVOL continuity | CHANGED-SINCE + ENT-A-EVOL-* + PG-008 |
| ECAP §2.B B1–B6 | § Independent certification below |
| ECAP FRZ-AUD §3.25 | Annex B |
| TEN §3.1 list (13) | Annex C TEN table |
| Checklist 225/26 FRZ | Annex B reconciliation + orchestration only |

**Unmapped roadmap §3.1 requirements = 0** (verified against Annex C).

---

## E. Detailed architecture blueprint

```
scripts/enterprise_audit/run_enterprise_audit.py
        ↓
intergrax/platform/enterprise_audit/runner.py  (EnterpriseAuditRunner)
        ↓
CheckExecutionSandbox → PytestGateCheck | ScriptGateCheck | (future static)
        ↓
AuditFinding[] → AuditDriftEngine → AuditReportBuilder
```

**Storage boundaries:** Evidence under `.tmp/proof/ent-audit-x/<run-id>/` (session) and CI artifacts; baseline record `ENTERPRISE_AUDIT_BASELINE.json` in qualification folder (operator-promoted).

---

## F. Implementation WBS

[ANNEX_H_WBS.md](./ANNEX_H_WBS.md) — **128** atomic tasks, **13** batches (B0–B12). Batches derived from dependencies (not fixed 14).

---

## G. Integrated batch plan

| Batch | Deliverable | Validation |
| --- | --- | --- |
| B0 | This master plan + annexes | Completeness gate §4 |
| B1 | Contracts/schemas | L1 schema tests |
| B2 | Runner core | integration noop→real |
| B3 | Registry+sandbox | adversarial injection |
| B4 | Primary gate orchestration (PG-001..020) | L2 GREEN @ SHA |
| B5 | TEN module 13/13 | adversarial TEN |
| B6 | VBN module 9/9 + domain rollup | inventory drift tests |
| B7 | Baseline+SHA ladder automation | AUD-05..07 |
| B8 | Reports+FRZ mapping | schema golden files |
| B9 | CLI+modes+exits | exit contract tests |
| B10 | CI workflow + enforcement runbook | workflow run + honest BLOCKED if unprotected |
| B11 | L1/L2/L3 qual | single pytest invocations |
| B12 | Protocol 03 + FRZ-AUD PASS + roadmap closure | independent B6 only |

---

## H. Safe parallelism / worktrees

| Parallel | Condition |
| --- | --- |
| B5 TEN vs B6 VBN | After B3; disjoint paths under `tests/qualification/ent_audit_x/` |
| B8 reports vs B6 VBN | Avoid same runner files — serialize B6 before B8 merge |
| Mutating checks | **Never** parallel with RO checks; worktree isolation only |

Default integration branch: `development`; no new branch without operator approval (git safety rule).

---

## I. L1 / L2 / L3 qualification

| Level | Path | Purpose |
| --- | --- | --- |
| L1 | `tests/qualification/ent_audit_x/unit/` | schemas, registry order, drift semantics, exit codes |
| L2 | `tests/qualification/ent_audit_x/integration/` | runner, CLI, adapters PG-* , baseline |
| L3 | `tests/qualification/ent_audit_x/l3/` | FREEZE full platform, adversarial, security |

**Independent audit:** L3 logs @ `TEST_EXECUTION_SHA` submitted to Protocol 03 — implementer ≠ auditor (ECAP-P4).

---

## J. Documentation plan

| Doc | Action |
| --- | --- |
| This master plan + annexes | **Published @ planning wave** |
| Architecture overview | Add `docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_ARCHITECTURE.md` in B11 (no SSOT duplication with domain arch) |
| Operator runbook CI | B10 `ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md` |
| ARCH-FREEZE handoff | B12 `ENT_AUDIT_X_ARCH_FREEZE_HANDOFF.md` |
| Domain arch hubs | **Read-only** — update only post-freeze pointers |

GAP-D (8): architecture doc, runbook, handoff, Protocol dossier, operator record template, extension guide, troubleshooting, known limitations — tracked in Annex F.

---

## K. Independent audit and certification

**ECAP §2.B Parent Closure (ENT-AUDIT-X):**

| Step | ENT deliverable |
| --- | --- |
| B1 | Closed-world check manifest + Annex C |
| B2 | FRZ-AUD + mapped families |
| B3 | L2/L3 positive logs @ TEST_EXECUTION_SHA |
| B4 | Adversarial suite (false PASS, skip, tamper, injection) |
| B5 | Evidence bundle + SHA ladder |
| B6 | Independent Protocol 03 **ENT-AUDIT-X-03** — human auditor **OD-ENT-03** |

**Sequence:** Implementation → candidate evidence → independent audit → corrective batch (if REJECT) → requal → independent acceptance → **OD-ENT-08** → FRZ checklist update → roadmap §5 → push → remote HEAD verify → ARCH-FREEZE handoff.

**Not in force:** `ECAP-P4-EVOL-SCOPE-AUTO-B6` — do not use for ENT or EVOL B6.

---

## L. FRZ closure matrix

[ANNEX_B_FRZ_AUD_CLOSURE_MATRIX.md](./ANNEX_B_FRZ_AUD_CLOSURE_MATRIX.md). FRZ-AUD-01..10 **OPEN** @ HEAD; promotion only in B12 after B6.

---

## M. Exact blocker register

| ID | Blocker | Resolution |
| --- | --- | --- |
| BLK-EVOL-B6 | EVOL Protocol 03 / B6 SHA unset | Complete EVOL W7-01 Path A (OD-EVOL-11 auditor) |
| BLK-EVOL-FRZ | FRZ-EVOL-01..10 OPEN | W7-03 after EVOL B6 |
| BLK-EVOL-HANDOFF | W7-06 pin TBD | W7-05 FORMAL_CLOSURE_PUBLICATION_SHA |
| BLK-CI-ENFORCE | Branch not protected | **OD-ENT-02** — see §3 CI |
| BLK-ROADMAP-ENT-DEP | Roadmap: ENT depends EVOL CLOSED | **OD-ENT-00** required before stage closure (no waiver assumed) |
| BLK-ENT-B6 | ENT independent audit not done | B12 Protocol 03 |

---

## N. Risk / dependency register

| Risk | Mitigation |
| --- | --- |
| Unsafe pytest gate execution | B3 sandbox + allowlist |
| False PASS via partial coverage | Mode policy + exit 2/3 |
| SHA confusion on doc commits | qualification_sha vs audit_execution_sha split (Annex SHA ladder) |
| Historical PASS cited as current | ECAP-P2 + replay @ audit SHA |
| CI policy not enforceable | Document BLOCKED; FRZ-AUD-10 honest status |

---

## O. Final closure checklist

1. All Annex C checks have adapters and L2 PASS.  
2. FREEZE mode GREEN @ TEST_EXECUTION_SHA.  
3. FRZ-AUD-01..10 promoted with independent B6.  
4. Baseline promoted via OD-ENT-01.  
5. Roadmap ENT-AUDIT-X = CLOSED — INDEPENDENTLY ACCEPTED.  
6. ARCH-FREEZE handoff published.  
7. PROD-Q live slice remains OPEN (Path B).  

---

## P. Start-to-closure execution sequence

B0 (plan) → B1…B11 (implementation+qual) → candidate SHA → B12 independent audit → corrective (if needed) → operator OD-ENT-08 → ledger → ARCH-FREEZE entry prep.

**Do not start B1 implementation until this plan is independently accepted** (operator instruction).

---

## Q. EVOL-X formal predecessor reconciliation

| | Technical | Formal |
| --- | --- | --- |
| Status | OD-EVOL-18/19 operator technical acceptance | B6 PENDING; FRZ-EVOL OPEN; Protocol 03 unsigned |
| SHA pins | TES `3c4a641b…`, FINAL `bbf3e33b…`, CI `41337077334` | B6_ACCEPTANCE unset |
| ENT intake | [EVOL_X_ENT_AUDIT_X_HANDOFF_RECORD.md](../evol_x/EVOL_X_ENT_AUDIT_X_HANDOFF_RECORD.md) prepared | W7-06 formal pin **pending** |
| Impact planning | Allowed (this document) | ENT **stage closure** blocked until EVOL formal path complete or **OD-ENT-00** operator amendment |

---

## R. ARCH-FREEZE handoff

Deliver after ENT CLOSED: freeze-candidate FREEZE audit PASS, `ENTERPRISE_AUDIT_BASELINE.json`, FRZ-AUD PASS evidence, remaining OPEN FRZ list (PROD-Q live), post-freeze scenario-wave policy linked to runner + [INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md](../INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md).

---

## § Execution safety (P-08)

| Control | Specification |
| --- | --- |
| Subprocess | Fixed argv templates: `uv run pytest <allowlisted nodeids>` — no user shell |
| Allowlist | `enterprise_audit_check_manifest.v1.json` only |
| RO vs mutating | Default RO; mutating=false enforced in FAST/PR |
| Worktree | Mutating checks require `ENT_AUDIT_WORKTREE` env + operator flag |
| Env isolation | Strip `*_TOKEN*`, `*_SECRET*` from child env; pass only declared vars |
| Credentials | Never pass tenant credentials into audit subprocess env |
| Timeout/cancel | Per-check timeout; global budget per mode |
| Resource bounds | `max_workers` ≤ 4 default; configurable cap |
| Evidence | Capture stdout/stderr under proof dir; redact patterns |
| Determinism | Stable check order; parallel only when order-independent |
| Duplicate checks | Same run_id idempotent; manifest records skip duplicates |
| Partial coverage | `PARTIAL` result — exit 2 if mode requires FULL |
| Internal failure | Uncaught runner exception → exit 3 |
| Adversarial | Tests for arbitrary command injection via manifest tamper |

---

## § Exact-SHA baseline lifecycle (P-09)

See [ENT_AUDIT_X_SHA_LADDER.md](./ENT_AUDIT_X_SHA_LADDER.md). Key rule: **tests execute @ TEST_EXECUTION_SHA**; documentation publication commit may differ — manifest must record both `qualification_sha` and `audit_execution_sha` (merge-base validated). Baseline promotion only after FREEZE PASS on candidate SHA (operator OD-ENT-01).

---

## § CI enforcement (P-03 / FRZ-AUD-10)

**Verified @ HEAD:** `gh api repos/jakbuczarnecki/intergrax/branches/development/protection` → **404 Branch not protected**.

**Required operator configuration (OD-ENT-02):**

1. GitHub → Settings → Branches → protect `development`.  
2. Require pull request before merge.  
3. Require status checks: `enterprise-audit-fast` (job name from B10 workflow).  
4. Require `ci-smoke` / existing regression jobs as today.  
5. Do not allow bypass for administrators (recommended).  
6. Document screenshot + settings export in FRZ-AUD-10 evidence.

**Workflow (B10):** `enterprise-audit.yml` — PR → FAST; schedule → FULL; manual → FREEZE; path filters → CHANGED-SINCE.

**Honest status until OD-ENT-02:** FRZ-AUD-10 implementation can be **COMPLETE** with enforcement **BLOCKED** — not PASS.

---

## § Operator decisions (undocumented = none decided)

| ID | Decision required | Status |
| --- | --- | --- |
| OD-ENT-00 | ENT stage closure vs EVOL formal CLOSED dependency | **OPEN** |
| OD-ENT-01 | Initial ACCEPTED_BASELINE_SHA | **OPEN** |
| OD-ENT-02 | GitHub branch protection + required checks | **OPEN** |
| OD-ENT-03 | Independent auditor for ENT Protocol 03 | **OPEN** |
| OD-ENT-08 | Stage closure signature | **OPEN** |
| OD-EVOL-11 | EVOL independent auditor identity | **OPEN** (blocks EVOL B6) |

No hypothetical waiver is treated as approved.

---

## § Plan completeness self-audit (§4)

| # | Criterion | Result |
| --- | --- | --- |
| 1 | Unmapped roadmap §3.1 | **0** |
| 2 | Unmapped FRZ-AUD | **0** |
| 3 | Unmapped TEN | **0** (13/13) |
| 4 | Unmapped VBN | **0** (9/9) |
| 5 | Unclassified VBN domains | **0** (31/31 Annex D) |
| 6 | Mandatory checks without executable strategy | **0** (85 strategies in Annex C) |
| 7 | Gaps without WBS | **0** (Annex F → Annex H) |
| 8 | WBS tasks without acceptance | **0** (Annex H acceptance column) |
| 9 | Mandatory tests without evidence strategy | **0** |
| 10 | Unclassified dependencies | **0** (batch deps in Annex H) |
| 11 | Undocumented operator decisions | **0** (table above) |
| 12 | Duplicate semantic authorities | **0** |
| 13 | Unknown mandatory closure steps | **0** (§O + B12) |

**Verdict:** **ENT-AUDIT-X — MASTER PLAN CORRECTED / PUBLISHED / READY FOR FINAL INDEPENDENT ACCEPTANCE**

**Implementation:** **NOT AUTHORIZED** until final independent plan acceptance. P-11/P-12/P-13 addressed in PLAN_RECONCILIATION_SHA commit.
