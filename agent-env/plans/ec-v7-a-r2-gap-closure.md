# ec-v7-a-r2-gap-closure

Continuation of [`ec-v7-a-product-completion.md`](ec-v7-a-product-completion.md). EC-V7-A wave closeout preserved candidate `7ff094a24` (A05 Part D only vs `b5724c3c`). This plan closes authority, residual, and evidence gaps without reopening full wave execution. **Do not declare EC-02 CLOSED.** Expected wave verdict while A01/A02 lack **Approved** write-sets: **`EC-V7-A-R2_PARTIAL`** — not `PRODUCT_COMPLETE`.

Main checkout stays `development`. Product edits only on worktree `D:\Projekty\_wt\intergrax\ec-v7-a-product-int`, branch `factory/ec-v7-a-product/int`. Documentation publication is not product publication.

---

- [ ] **R2-00** Candidate recovery
  - objective: Confirm `7ff094a24b7e7d59957eda7c1a01716a6dcf2f40` is the factory candidate HEAD; verify ancestry from `b5724c3c`; record containment vs `origin/development`.
  - evidence: `.tmp/session/ec-v7-a-r2/candidate-recovery.md`
  - integration: Do not reset or force-push candidate branch.

- [ ] **R2-01** A01 / A02 authority gate
  - objective: No product edit to seam ledger or closed-world paths unless a new **Approved** line names an explicit write-set. If none exists on `development`, one Operator Decision Packet (ODP) only.
  - prior: `.tmp/session/ec-v7-a/a01-a02-authority-packet.md`; tickets `T-IGX-046-F` (Approved = EQO docs only), `T-IGX-046-H` (`plan_pending`, G2 AST gate only).
  - write: _(empty until Approved)_
  - integration: A01/A02 must not merge into candidate without approval.

- [ ] **R2-04** A03 residuals (S04)
  - objective: Disposition `load_agent_class` and `agent_certification_wiring` compat resolver against `T-IGX-051-I` — fix only if defect is inside that ticket's Approved write-set; else log EC-V7-B residual.
  - ticket: `governance/tickets/T-IGX-051-I.md`
  - base: `7ff094a24`
  - write: conditional per 051-I write-set only
  - tests: `uv run pytest tests/unit/applications/test_a4007_caller_import_refusal.py tests/unit/applications/test_a4007_import_policy_derivation.py tests/unit/scripts/test_a4007_migration_gate_allowlist.py`

- [ ] **R2-05** A05 residuals (S05D Part E)
  - objective: Disposition schedule-store ports (SQLite/PostgreSQL) and three enqueue sites outside A1 against `T-IGX-051-E`; **preserve** Part D commit on candidate (two-file delta vs `b5724c3c`).
  - ticket: `governance/tickets/T-IGX-051-E.md`
  - write: _(empty unless Approved scope explicitly includes ports/enqueues)_
  - tests: five-module `test_rw_recovery_*` contract on candidate

- [ ] **R2-06** A06 E2–E4 (S06B)
  - objective: Evidence and status read only unless activation condition met (`WF_E2_E4_STATUS.md`, Campaign B-1 ancestor of base, named in-write-set defect).
  - ticket: `T-IGX-043-J.md`
  - write: _(empty — external dependency)_
  - tests: independent slice only if re-run requested

- [ ] **R2-07** A07 manifest / evidence / baseline consumers
  - objective: Record disposition vs `T-IGX-043-I` (engine qualified) and `T-IGX-043-M` (Part E consumers, baseline promotion blocked per plan §8.4).
  - write: _(empty — EC-V7-B unless new Approved scope)_
  - tests: four `test_wd_*` engine modules (re-qualify if candidate moves)

- [x] **A04** preservation
  - objective: No lane work; `harness_host_runtime.py` / FC-1 wiring remains qualified per prior wave (`T-IGX-043-N`).
  - integration: A06 must not edit shared harness path.

- [ ] **R2-08** Final qualification
  - objective: Combined pytest on candidate after any in-scope merges; separate pytest exit from CI smoke / outer shell exit 1; `igx_gov ticket` only if product commit exists.
  - verdict cap: cannot be `PRODUCT_COMPLETE` while A01/A02 lack **Approved** write-sets.

- [ ] **R2-09** Publication handoff
  - objective: Operator or authorized publisher only; candidate not contained in `origin/development` until integrated. Final reconciliation updates plan checkboxes and `.tmp/session/ec-v7-a-r2/` evidence; does not close EC-02.

---

## R2 terminal dispositions

R2 gap-closure closeout is recorded in [`.tmp/session/ec-v7-a-r2/final-report.md`](../../.tmp/session/ec-v7-a-r2/final-report.md) (sections A–G). **Wave verdict:** `EC-V7-A_PARTIAL_EXACT_BLOCKERS` — `PRODUCT_COMPLETE` not met (Seam Ledger and CW-11/CW-12 uncorrected). Candidate remains `factory/ec-v7-a-product/int` @ `7ff094a24`; not published to `origin/development`. A01, A02, and A06 E2–E4 remain open per the disposition matrix; deferrals and EC-V7-B inputs are listed in the final report.