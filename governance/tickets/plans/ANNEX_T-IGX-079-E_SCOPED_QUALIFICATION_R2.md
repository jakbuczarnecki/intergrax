# ANNEX — T-IGX-079-E scoped qualification (R2)

**Parent plan (immutable):** `governance/tickets/plans/PLAN_T-IGX-079-E.md`  
**Parent plan digest (binding):** `b40163343535234fc2c241faa4f2b423232147e0463e4e6d77cb7f35bce9632c`  
**Parent plan blob (binding):** `409fcd2b2c3d59389650979dbf844472640515fd`  
**Ticket:** `governance/tickets/T-IGX-079-E.md`  
**Package:** II-W0-05 (II-W0-B documentation adoption)  
**Frozen corrective-scope candidate (not integrated in this batch):** `3e79c9e850065a3110282b9caed76b6bda74f7dc`  
**Historical local qualification candidate:** `9e1a202fe2dcb641a84cdbb3da1a438894ba7ad5`  
**Integration baseline:** `4aae44a7647a18da740eaa3b8295352294941b85`  
**Prior annex R1 blob (immutable historical proposal):** `0c10a9af7903ed134dcc5b9328f6ab01844e60c2`  
**Annex R2 status:** `PROPOSED_PENDING_INDEPENDENT_REVIEW_AND_EXACT_OPERATOR_APPROVAL`

This annex is a **proposed replacement** for the R1 qualification proposal (`ANNEX_T-IGX-079-E_SCOPED_QUALIFICATION_R1.md`). It is **not** an active qualification contract. Creating or publishing R2 does **not** activate amended qualification authority.

---

## A. Scope and preserved obligations

The amendment applies **exclusively** to the documentary adoption package **II-W0-05** under ticket T-IGX-079-E.

**Preserve without change:**

- All five existing documentary deliverables (D1–D5 contract).
- All original functional acceptance requirements (Q01–Q06).
- Exact approved documentary 50-path execution write-set per parent plan.
- Protected normative blobs (architecture hub, master plan WBS, W0-A closed plan blob, UEA frozen disposition).
- Independent review.
- Formal independent G6 (separate session).
- Separate Operator closure authority.
- All external program qualification requirements and canonical technical/procedural gates.

**No permission in this proposal for:** runtime architecture changes, product code changes, qualification mechanism changes, external program predicate changes, global EVOL-X baseline promotion, accepted manifest creation, or weakening of EVOL-X A8 fail-closed behavior.

---

## B. Proposed proportional Q07 contract (documentation-only II-W0-05)

When R2 is independently accepted and digest-bound approved, the **mandatory documentary qualification** for integration of the frozen W0-B candidate consists of exactly these twelve checks:

1. Exact plan and amendment approval validation (`plan_digest.py`, `verify_plan_approval.py` with bound plan and annex digest).
2. Canonical ticket lifecycle validation (`verify_ticket_lifecycle.py`).
3. Full authorized candidate write-set verification (50-path execution set; zero paths outside set).
4. Protected normative blob verification (unchanged blobs vs binding values on `development`).
5. Actual **186-path** D4 inventory consistency (49 core + 137 additional record-only paths).
6. All introduced and modified local Markdown links and anchors in candidate range.
7. Documentation domain-pair validation (`check_docs_domain_pairs`).
8. PDC focused feature documentation tests (P6R / feature-docs scope).
9. PDC inventory validation (`pdc_x_inventory.py --check`).
10. Candidate-range `git diff --check`.
11. Canonical Governance Smoke (`igx_gov` / CI smoke tier as defined for this batch).
12. Zero unauthorized product-code changes (`intergrax/`, `agents/`, `applications/`, `scripts/` delta vs integration baseline).

### Historical local qualification (binding reported SHA; not re-executed in R2 preparation)

**Retain** the actual historical local qualification bound to:

`9e1a202fe2dcb641a84cdbb3da1a438894ba7ad5`

**Reported historical results** (orchestrator-recorded; remain historical until evidence is independently bound and accepted at integration):

| Check | Reported summary |
|---|---|
| P6R feature docs | 4 passed |
| Governance Smoke | 1894 passed |
| Plan digest validator | exit 0 |
| Approval validator | exit 0 |
| Lifecycle validator | exit 0 |
| Domain pairs | exit 0 |
| PDC inventory | exit 0 |

These results **must not** be relabeled as newly executed at another SHA during preparation or integration planning without a fresh candidate-bound run.

---

## C. External-domain qualification disposition (proposal only)

**Preserve** historical EVOL-X and ENT-AUDIT-X verdicts without changing their meaning.

### EVOL-X (recovered evidence)

| Item | Value |
|---|---|
| Workflow stages A1–A7 | PASS (at recovered run) |
| A5 pytest population | 354 executed, 309 passed, 45 failed |
| A8 freshness | **FAIL** — `no accepted baseline manifest supplied` |
| Workflow overall exit | 2 (A8 failed) |
| Accepted manifest prerequisite | **Unresolved** |
| P03 artifact-integrity failures | **Unresolved** (45 failures in `test_evol_x_p03_integrity.py` population) |
| TEN-FINAL | **Unresolved** — `test_tenant_x_final_current_head.py::test_tx_final_head_q01_final_ready_for_audit` failed in A5 population |

**No global EVOL-X PASS. No A8 PASS. No accepted manifest creation. No global baseline promotion.**

### ENT-AUDIT-X (recovered evidence)

| Item | Value |
|---|---|
| Overall status | **BLOCKED** (exit 1) |
| Failed controls (executed) | 2 |
| ENT-A-PRD-01 | FAIL — gate non-zero exit (`ENT-A-PRD-01:FAIL`) |
| TEN-IDENTITY | FAIL — `test_tx_r1_head_q01_inventory_reconciles_twenty_two_mechanisms` |

Matched-baseline notes from recovered direct reruns: TEN-IDENTITY failure signature present at both integration baseline and candidate in recovered captures; ENT-A-PRD-01 exhibits `KeyError: 'catalog'` at candidate in recovered capture (baseline capture also non-zero).

### Proposed disposition (not asserted as satisfied)

The proposed amendment requests that **independently demonstrated** external-domain failures be carried as **explicit unresolved external obligations**, rather than automatically preventing acceptance of a **documentation-only** package, **only where all** of the following are demonstrated in a later authorized integration batch:

1. Failure is outside the actual changed semantic scope of II-W0-05.
2. Candidate-induced regression is excluded by adequate evidence.
3. Applicable higher-order governance permits the disposition.
4. Required competent domain-owner authority is recorded.
5. Independent review accepts the exact evidence and limitation.

**A failure without sufficient evidence remains BLOCKING.** This proposal does **not** assert that existing external findings already meet these conditions.

**No global EVOL-X PASS. No global ENT-AUDIT-X PASS. No global baseline promotion. No accepted manifest creation. No weakening of A8.**

---

## D. Amendment authority and precedence

| Rule | Record |
|---|---|
| Parent plan | Remains the binding execution contract |
| Annex R1 | Remains historical proposed amendment (byte-identical on `development`) |
| Annex R2 | Proposed **replacement** for the R1 **proposal** only |
| R2 effectiveness | Does **not** become effective by creation or publication |
| Higher-order gates | R2 cannot replace a mandatory higher-order gate |
| Plan change | Any change to the approved plan requires exact digest-bound approval |
| Original approval reuse | Original plan approval cannot be silently reused for a revised qualification contract |
| Implementer self-approval | Implementer cannot approve its own plan or annex |
| Formal G6 | Remains independently executed in a separate session |

**Proposal status:** `PROPOSED_PENDING_INDEPENDENT_REVIEW_AND_EXACT_OPERATOR_APPROVAL`

---

## Evidence matrix (recovered; preparation batch — no re-run)

Evidence recovered mechanically from W0-B worktree `D:/Projekty/_wt/intergrax/ii-w0-b-documentation-adoption-01` (branch `ii/w0-b-documentation-adoption-01` @ `3e79c9e850065a3110282b9caed76b6bda74f7dc`). Locators are **not** tracked in git (`.tmp/proof`).

| ID | Binding / execution SHA | Command / artifact | Exit / summary | Evidence locator | Temporal | Applicability |
|---|---|---|---|---|---|---|
| E-PLAN-DIGEST | `9e1a202fe…` (historical bind) | `uv run python scripts/governance/plan_digest.py governance/tickets/plans/PLAN_T-IGX-079-E.md` | 0; digest `b4016334…` | `.tmp/proof/II-W0-B-PHASE-A-RECOVERY-01/Q01_plan_digest.log` | historical | applicable at bind SHA |
| E-PLAN-APPROVAL | `9e1a202fe…` | `verify_plan_approval.py` | 0; OK | `.tmp/proof/…/Q01_verify_plan.log`, `Q01_verify_lifecycle_with_plan.log` | historical | applicable at bind SHA |
| E-LIFECYCLE | `9e1a202fe…` | `verify_ticket_lifecycle.py` (+ plan) | 0; OK with `--plan` | `.tmp/proof/…/Q01_verify_lifecycle_with_plan.log` | historical | applicable at bind SHA |
| E-LIFECYCLE-NOPLAN | recovery HEAD | `verify_ticket_lifecycle.py` (no `--plan`) | non-zero (stale digest guard) | `.tmp/proof/…/Q01_verify_lifecycle.log` | historical | informational only |
| E-DOMAIN-PAIRS | `9e1a202fe…` | `check_docs_domain_pairs` | 0; 23 domain / 8 feature pairs | `.tmp/proof/…/Q03_domain_pairs.log` | historical | applicable at bind SHA |
| E-PDC-INV | `9e1a202fe…` | `pdc_x_inventory.py --check` | 0; 361 items | `.tmp/proof/…/Q_pdc_inventory.log` | historical | applicable at bind SHA |
| E-P6R | `9e1a202fe…` | P6R feature docs pytest | 0; 4 passed | `.tmp/proof/…/Q03_p6r_feature_docs.log` | historical | applicable at bind SHA |
| E-SMOKE | recovery context | `igx_gov` CI smoke | 0; 1894 passed | `.tmp/proof/…/smoke.log` | historical | requires re-bind at integration SHA |
| E-EVOL-WF | `564ae9b722b7ec4046351dd57d56b56b43252034` | `evol_x_change_workflow.py` Phase A | 2; A8 FAIL | `.tmp/proof/…/evol_x_workflow.log`, `evol-x-phase-a/summary.json` | historical | external; independent disposition required |
| E-EVOL-A5 | `564ae9b722…` | A5 selected pytest population | 1; 309 passed, 45 failed | `.tmp/proof/…/a5_pytest_full.log` | historical | external; BLOCKING until disposition |
| E-ENT-AUDIT | `564ae9b722…` | `run_enterprise_audit.py` CHANGED-SINCE certifying | 1; BLOCKED; 2 failed controls | `.tmp/proof/…/ent_audit_run.log`, `ent-audit/enterprise_audit_outcomes.json` | historical | external; independent disposition required |
| E-ENT-PRD-01 | candidate vs `4aae44a76…` | direct gate capture | non-zero both sides (recovered) | `.tmp/proof/…/candidate/ENT-A-PRD-01_direct.txt`, `baseline/ENT-A-PRD-01_direct.txt` | historical | matched-baseline subject to verification |
| E-TEN-IDENTITY | candidate vs `4aae44a76…` | direct pytest capture | non-zero both sides | `.tmp/proof/…/candidate/TEN-IDENTITY_direct.txt`, `baseline/TEN-IDENTITY_direct.txt` | historical | matched-baseline subject to verification |
| E-D4-186 | `3e79c9e85…` (frozen candidate) | D4 register materialization | present in candidate tree | `docs/project/maintainers/qualification/intelligence_x/w0/II-W0-05_DOCUMENTATION_ADOPTION_REGISTER.md` @ frozen SHA | current artifact | integration verification required |
| E-LINK-AUDIT | — | full link sweep at preparation | **EVIDENCE_UNAVAILABLE** | — | — | integration batch |

### EVOL-X A5 failure signatures (recoverable subset)

All 45 failed nodes in recovered `a5_pytest_full.log` are in:

- `tests/qualification/evol_x/gates/test_evol_x_p03_integrity.py` (44 parameterized cases + core tests), and
- `tests/qualification/tenant_x/test_tenant_x_final_current_head.py::test_tx_final_head_q01_final_ready_for_audit` (TEN-FINAL).

Primary failure theme for P03 population: `EVIDENCE_INTEGRITY_FAIL` / `artifact missing @ evidence HEAD` for evol_x_p03_independent_replay artifacts.

### ENT-AUDIT-X failure signatures

- `ENT-A-PRD-01:FAIL` — check_id `ENT-A-PRD-01`, exit 1  
- `TEN-IDENTITY:FAIL` — check_id `TEN-IDENTITY`, exit 1; node `test_tx_r1_head_q01_inventory_reconciles_twenty_two_mechanisms`

---

## Integration contract (subsequent authorized batch only)

Execute **exactly** in order after R2 independent acceptance. **Do not** execute during R2 preparation.

1. Independent acceptance of this revised qualification proposal (R2).
2. Exact digest-bound Operator approval and any required domain-owner acceptance records.
3. Materialization of the approved qualification amendment through canonical governance (ticket/plan annex binding per digest rules).
4. Integration of the frozen W0-B candidate (`3e79c9e850065a3110282b9caed76b6bda74f7dc`) onto current `development` in an isolated worktree.
5. Verification of exact accepted documentary content and integration scope (50-path + D4 186-path + protected blobs).
6. One consolidated candidate-bound qualification batch (Q07 twelve checks + external disposition evidence as required).
7. Canonical `quality_gate` checkpoint.
8. Commit and push to `development` (non–force-push).
9. Remote SHA verification on `origin/development`.
10. Authorized `ready_for_audit` handoff.
11. Formal independent G6 in a separate session.
12. Operator closure authority.
13. Formal closure of II-W0-05 and roadmap reconciliation to **5/69** formal closures.
14. II-W0-06 becomes eligible for separately authorized execution.

No optional alternative implementation strategies are authorized in this contract.

---

## Digest

Compute with:

`C:\Users\XPS\.local\bin\uv.exe run python scripts/governance/plan_digest.py governance/tickets/plans/ANNEX_T-IGX-079-E_SCOPED_QUALIFICATION_R2.md`

**Recorded after publication commit:** see terminal report `R2_DIGEST` / `R2_BLOB`.