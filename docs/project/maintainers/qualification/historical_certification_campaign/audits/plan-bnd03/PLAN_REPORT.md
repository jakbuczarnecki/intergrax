# PLAN_REPORT - T-IGX-033-A (planner, plan-bnd03)

Branch `hist-cert/plan-bnd03`, base `origin/development` af323c005. Commit 92eb2c276c81c6778a210df10041b2fe50171142 (local, not pushed). Files: `governance/tickets/T-IGX-033-A.md`, `governance/tickets/plans/PLAN_T-IGX-033-A.md`. Lifecycle `plan_pending`, no `Approved:` line. Digest ae167672d4075e3e4dc190ac93c78a937d7ad0c435dfc3971e615143e4273775. No tests run, no product/test edit.

## Findings
- Thesis confirmed: one static edge, `intergrax/cli/external_work.py:338`, in `_retry_attestation`; reached by every `external-work retry-attestation` call (imports run before the signer check). Only `cli/main.py` imports the module.
- F-002 confirmed: 7 importlib sites + sys.path injection in the same module. Thesis incomplete: the same Tier-0 also has `intergrax/applications/_shared/product_manifest_registry.py:16-22` hard-coding 4 Tier-3 module names (not in HIST-02) - left out of scope, Q4.
- Gates tolerate it: ebh5 `test_ebh5_rpl_g_tier0_no_agents_import_applications` skips `intergrax/cli/`; EBH-7 F10 C1 ratifies it via `TIER_IMPORT_EXCEPTIONS` (substring check of an audit doc). Removing the import makes C1 red ("stale tier exception") and requires regenerating `EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md` (Q3).

## Recommendation
(b2) injected command set: Tier-0 `intergrax/cli/command_set.py` + `extra_command_sets` in `build_parser`/`main`; Tier-3 `host/cli_commands.py` (~320 lines moved verbatim) + `__main__.py` launcher; `receipt verify` stays Tier-0. Moving is feasible, so ACCEPTED_LIMITATION is not needed. Rejected: (a) pure form infeasible (single console script, wheel = `intergrax` only, tests use `-m intergrax.cli.main`); (b1) entry points (new governed plugin group, install-metadata risk; fallback if Q1 = NO); (b3) env string; (c) limitation.
- Cost: 7 product-path files (3 intergrax/, 4 applications/, 0 agents/), Tier-0 -about 300 lines, Tier-3 +about 380; 3 test/gate files, 1 regenerated matrix, 5 docs. None of the six MATERIAL_GOVERNANCE_PATHS.
- Red-first gate: new `tests/unit/architecture/test_bnd03_tier0_upward_edges_gate.py` (strict zero, no allow-list, tier roots computed from disk, negative controls); two tolerances removed.

## Operator questions (plan section 10)
Q1 accept the invocation change (`python -m applications.governed_contractor_application external-work|demo`)? Q2 stub or invalid-choice for the old names. Q3 regenerate the EBH-7 matrix row. Q4 separate finding for product_manifest_registry. Q5 join the single IIAC re-activation with 030-B/030-C. Q6 decision id only if (c).

## Not verified
Nothing executed: gate red/green, launcher import resolution without PYTHONPATH, G0/arch-freeze inventories on new files, render determinism, exact ACT-N11 set, whether ebh_3/2a/2i flag the edge (not read), pin of the F10-C1-TIER-EXC-01 prose in test_ebh_7_w8_ecap_l3_gates.py (grep only). Side effect: the commit hook created an untracked-ignored `.venv` in the worktree.
