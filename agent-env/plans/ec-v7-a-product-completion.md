# ec-v7-a-product-completion

EC-V7-A product completion factory plan. Candidate base `development` @ `b5724c3c285fe27fa3d20716ab1710a34c1def77`. Integration branch `factory/ec-v7-a-product/int` (worktree via `/igx-worktree`). Main checkout remains on `development`. Do not merge `arch-freeze/rw-train`. Do not resurrect corrective commits `ecf45eb76`, `07d81c206`, `de9d8e7f`, `05f368ec`. Do not declare EC-02 CLOSED or PRODUCT_COMPLETE.

Default wave verdict (without new Approved authority for A01/A02): `EC-V7-A_PARTIAL_EXACT_BLOCKERS`.

Assignment branches (only when a real diff exists): `ec-v7a/a05-recovery`; conditionally `ec-v7a/a03-callers`, `ec-v7a/a07-engine`. Merge order into candidate: A05 first, then any A03/A07 fixes. A01/A02 never merge into the candidate.

Shared-path reservation: `harness_host_runtime.py` is A04-only (A06 must not edit). `impact_map.py`, `digests.py`, and impact-map JSON are A07-only. Do not rewrite `package_wiring.py`, `sandbox_wiring.py`, `strict_product_manifest_ci_gates.py`, `runtime_event_delivery_wiring.py`.

- [ ] A01 Seam Ledger (evidence and escalation only) — terminal: `BLOCKED_BY_EXACT_AUTHORITY`
  - objective: Prove which mandatory gate behaviors are absent on HEAD; do not restore missing-seam scan, parent-package import expansion, or scan jurisdictions without Approved write-set authority (PT-G01, AR-26, R2-04, F03–F04).
  - ticket: PT-G01 (owner); related `governance/tickets/T-IGX-046-F.md` (EQO docs only), `T-IGX-046-H.md` (`plan_pending`, write-set `check_agent_registry_bypass.py` only); no `T-IGX-046-I`.
  - existing_disposition: EC-V6 R2-04 authority-blocked; skeleton ledger and import rules on HEAD; `detect_unregistered_cross_boundary_import` not invoked from `run_seam_ledger_check`; replay `1de7e69ea` (`T-IGX-050-N`) removed live missing-seam enforcement.
  - base: `b5724c3c285fe27fa3d20716ab1710a34c1def77`
  - write: _(empty — exact authority)_
  - deps: -
  - tests: `uv run pytest tests/unit/architecture/test_seam_ledger_gate.py` (neighbor evidence only; green does not prove parent-package bypass, similar-namespace, or missing-scan jurisdiction)
  - integration: Do not merge into `factory/ec-v7-a-product/int`.
  - qualification: Orchestrator records gap versus `5ef37c`, file list (`governance/architecture/seam_ledger.yaml`, `scripts/maintenance/seam_ledger_lib.py`, `scripts/maintenance/check_seam_ledger.py`, `tests/unit/architecture/test_seam_ledger_gate.py`), and resumption condition (Approved ticket whose write-set names those files).

- [ ] A02 Closed-World (evidence and escalation only) — terminal: `BLOCKED_BY_EXACT_AUTHORITY`
  - objective: Separate canonical tier classification from CW-11 test hole and CW-12 broken baseline pin; longest-prefix tier-0 rule in `testing_support/change_safety/discovery/closed_world.py` stays canonical; no architecture swap to explicit package lists.
  - ticket: PT-G01 / EVOL-X (same authority gap as A01); no Approved write-set for `closed_world.py` or `closed_world_gates.py`.
  - existing_disposition: EC-V6 R2-05 authority-blocked; CW-11 detects only `intergrax/unscoped/` test prefix; CW-12 `baseline_accepted_exclusion_ids()` reads live `exclusion_register.v1.json` instead of pin `tests/qualification/evol_x/data/exclusion_register_cw12_baseline.v1.json`.
  - base: `b5724c3c285fe27fa3d20716ab1710a34c1def77`
  - write: _(empty — exact authority)_
  - deps: Parallel with A01; does not block A03–A07.
  - tests: `uv run pytest tests/unit/architecture/test_closed_world_tier_gate.py` (neighbor evidence; green does not prove baseline pin)
  - integration: Do not merge into candidate; shared escalation packet with A01.
  - qualification: Document CW-11/CW-12 gaps; no product edits this wave.

- [x] A03 S04 dynamic import allowlist (qualification-first) — terminal: `VERIFIED_EXISTING_AND_QUALIFIED`
  - objective: Close S04 with caller census proof, not mere presence of `require_import_allowed()`; LAB unrestricted; policy before `import_module` on authorized callers.
  - ticket: `governance/tickets/T-IGX-051-I.md` (Approved `operator-delegate @ 2026-11-18`, digest `646fdfb58226876a55fc643f6cc90a977a6b4d1485d109e2f532c96025e33c70`; product @ `a766d2a7d` on ancestry)
  - existing_disposition: EC-V6 S04 open (`ec-v6-ec02-closure.md`); implementation largely on HEAD; residual callers (`wiring.load_agent_class`, compat resolver in `agent_certification_wiring.py`) deferred to EC-V7-B.
  - base: `b5724c3c285fe27fa3d20716ab1710a34c1def77`
  - write: _(empty unless census finds defect in file already within T-IGX-051-I write-set)_ — conditional files per S04 plan: `intergrax/applications/_shared/import_policy.py`, `agent_resolution.py`, `wiring.py`, `environment_conformance.py`, `roster_agent_contract_authority.py`, `environment_diff_wiring.py`, `scripts/maintenance/check_application_migrations.py`, `scripts/gates/check_application_production_gates.py`, `tests/unit/applications/test_a4007_*.py`, `tests/unit/scripts/test_a4007_migration_gate_allowlist.py`
  - deps: None vs A04–A07; do not touch `sandbox_wiring.py` or package lifecycle gates.
  - agent: composer-2.5-fast (qualification lane)
  - tests: `uv run pytest tests/unit/applications/test_a4007_caller_import_refusal.py tests/unit/applications/test_a4007_import_policy_derivation.py tests/unit/scripts/test_a4007_migration_gate_allowlist.py`
  - integration: Merge only if commit with in-write-set fix; else no merge.
  - qualification: Single pytest command; allowed/denied-before-import, empty allowlist, bad prefix, similar namespace covered by existing tests.

- [x] A04 S05A Human Decision Store (qualification-first) — terminal: `VERIFIED_EXISTING_AND_QUALIFIED`
  - objective: Confirm single persistence authority on harness path (`build_harness_host_runtime` → `wire_application_environment` → `build_application_tool_wiring`); STRICT without store still refuses.
  - ticket: `governance/tickets/T-IGX-043-N.md` (Approved `operator @ 2026-11-23`, lifecycle `quality_gate`; commits `3e7bb5773`, `08efdc7d6` ancestors of base)
  - existing_disposition: EC-V6 S05A open; wiring on HEAD per ticket intent.
  - base: `b5724c3c285fe27fa3d20716ab1710a34c1def77`
  - write: _(empty unless NT-FC1-26 case 2 fails)_ — reserved: `intergrax/applications/_shared/tool_wiring.py`, `environment_wiring.py`, `harness_host_runtime.py` (do not extend FC-1 files from `T-IGX-043-A`)
  - deps: No parallel edit of `harness_host_runtime.py` by A06.
  - agent: composer-2.5-fast
  - tests: `uv run pytest tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py` plus adjacent FC-1 regression named in PLAN 043-N (one command)
  - integration: No merge when qualification green.
  - qualification: Fail-closed STRICT persistence wiring only; no second store in `_shared`.

- [x] A05 S05D durable recovery (Part D EX-D1-07 — only planned code change) — terminal: `EXECUTED_AND_QUALIFIED` @ `7ff094a24`; residuals: Part E schedule-store ports (SQLite/PostgreSQL), three enqueue sites outside A1 (EC-V7-B)
  - objective: Complete Part D lifecycle/bindings comparison in authorized store and test module; qualify Parts A–C, F, G A1 already on HEAD; stop without weakening CAS comparison if pinned test fails.
  - ticket: `governance/tickets/T-IGX-051-E.md` (Approved `operator-delegate @ 2026-11-17`, digest `40040bf51bd46e10f0e5c89b7d235eda4fb516a2c1796ad17ec9bbd81820dc4c`)
  - existing_disposition: EC-V6 S05D open; Parts A, B, C, F, G A1 on HEAD; Part E reconciler present; store ports and three non-A1 enqueues residual EC-V7-B.
  - base: `b5724c3c285fe27fa3d20716ab1710a34c1def77`
  - write: `intergrax/applications/_shared/durable_registry_projection_input_store.py`, `tests/unit/applications/test_rw_recovery_durable_cas_projection.py` (add `test_rw_rec_05_lifecycle_and_bindings_are_compared`); no other paths unless five-module qualification shows defect elsewhere in 051-E write-set
  - deps: PT-FC2 on HEAD satisfies queue order; do not edit `queue_worker_wiring.py` (`active_work_sources` stays empty per plan).
  - agent: composer-2.5; branch `ec-v7a/a05-recovery` from pinned base
  - tests: `uv run pytest tests/unit/queueing/test_rw_recovery_no_double_execution.py tests/unit/hosting/test_rw_recovery_drain_wait_for_idle.py tests/unit/runtime/long_running/test_rw_recovery_uncertain_reconciler.py tests/unit/applications/test_rw_recovery_durable_cas_projection.py tests/unit/queueing/test_rw_recovery_producer_identity.py`; then `uv run ruff check` and `uv run pyright` on changed files
  - integration: Merge to candidate only when five modules green or Q-RC-02 stop recorded without weakened comparison.
  - qualification: EX-D1-07 executed and qualified; NT-RW-REC for A–C, F, A1 → `VERIFIED_EXISTING_AND_QUALIFIED` if those tests pass.

- [ ] A06 S06B PT-WF (qualification-first; E2–E4 external) — terminal: `PARTIAL_WITH_EXACT_REMAINING_WORK` (independent slice qualified; E2–E4 `BLOCKED_BY_EXACT_EXTERNAL_DEPENDENCY`)
  - objective: Qualify independent harness evaluation / debug / experiment slice; do not implement E2–E4 without current Campaign B-1 evidence; no new Campaign B run.
  - ticket: `governance/tickets/T-IGX-043-J.md` (Approved `operator @ 2026-11-21`; G-09, F01–F04 on HEAD `aac4b083b`, `2c6d7af1a`)
  - existing_disposition: EC-V6 S06B open; E2–E4 blocked on Campaign B-1; `af-fc/b-alpha` @ `e4857f515` not ancestor of base.
  - base: `b5724c3c285fe27fa3d20716ab1710a34c1def77`
  - write: _(empty unless lane start reads `docs/project/maintainers/qualification/arch_freeze_program/continuation/WF_E2_E4_STATUS.md` and finds verified replay ancestor of base plus named defect in `runtime.py`, `documents.py`, `experiments/workflow.py`, `experiments/store.py`, or `debug/router.py` only)_
  - deps: External Campaign B worktrees (`af-fc/b-alpha`, `af-fc/b02-rerun`) not owned by this wave; `harness_host_runtime.py` not edited.
  - agent: composer-2.5-fast
  - tests: `uv run pytest tests/unit/applications/test_harness_evaluation_wiring.py tests/unit/debug/test_wf_debug_tenant_authority.py tests/unit/experiments/test_experiment_workflow.py`
  - integration: No merge on green independent slice; no owner-batch outside plan.
  - qualification: Independent slice green → `VERIFIED_EXISTING_AND_QUALIFIED`; E2–E4 → `BLOCKED_BY_EXACT_EXTERNAL_DEPENDENCY`.

- [x] A07 S07A Enterprise Audit Engine (qualification-first) — terminal: `VERIFIED_EXISTING_AND_QUALIFIED`
  - objective: Qualify changed-since, impact map, FRZ attribution, FAST promotion disclosure; do not recover full executed-checks product surface; Part E manifest/baseline consumers mostly EC-V7-B.
  - ticket: `governance/tickets/T-IGX-043-I.md` (Approved `operator-delegate @ 2026-11-18`, digest `4ec30bbd721583c546186c03035dcfb3ed3cd84e397bd6567ebf075b552e2721`); Part E cross-check `governance/tickets/T-IGX-043-M.md` (Approved) for FRZ/FAST behaviors only this wave.
  - existing_disposition: EC-V6 S07A open; engine Parts A–D and first wave-diff installment in tree; manifest/evidence/baseline out of 043-I write-set.
  - base: `b5724c3c285fe27fa3d20716ab1710a34c1def77`
  - write: _(empty if four `test_wd_*` green; on red, only the 043-I write-set file the failing test names — not manifest)_
  - deps: None blocking parallel lanes; manifest reserved.
  - agent: composer-2.5-fast
  - tests: `uv run pytest tests/qualification/ent_audit_x/unit/test_wd_changed_since_default_deny.py tests/qualification/ent_audit_x/unit/test_wd_evol_x_tree_mapping.py tests/qualification/ent_audit_x/unit/test_wd_frz_family_attribution.py tests/qualification/ent_audit_x/unit/test_wd_fast_promotion_disclosure.py`
  - integration: Merge only on in-write-set engine fix branch `ec-v7a/a07-engine`.
  - qualification: Locate current qualification log if needed; engine defects in write-set fixed; product/check debt logged for EC-V7-B without inventing six failed check names.

## Wave qualification (parent orchestrator — not setup)

After lane integration on candidate SHA: combined pytest (A03, A04, A05, A06, A07 modules plus neighbor A01/A02 gates), `ruff`/`pyright` on A05 (and conditional) changes, `uv run python -m scripts.governance.igx_gov ticket` if product commit exists, one `scripts/ci/run_ci_smoke_pytest.py` run. Confirm protective commits remain ancestors and do not appear in candidate diff except conscious Part D.

Update `agent-env/plans/ec-v6-ec02-closure.md` checkboxes only at final disposition. R2-04/R2-05 remain authority-blocked, not QUALIFIED.