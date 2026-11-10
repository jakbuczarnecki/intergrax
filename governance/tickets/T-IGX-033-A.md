# T-IGX-033-A - HIST02-F-001 / F-002: remove the Tier-0 -> Tier-3 edges from the `intergrax` CLI (FRZ-BND-03, Operator DR-8 option 2) (W1-PLAN step 3)

**HELD - ACT-N11 sequencing (DR-12=B): do not approve or implement until MASTER confirms the single IIAC re-audit/re-activation batch.**

## Scope

`intergrax/cli/external_work.py:338` imports `applications.governed_contractor_application.host.collaborative_work_local_fixture` inside `_retry_attestation`, and the same module reaches Tier-2/Tier-3 through 7 `importlib.import_module` sites and a `sys.path` injection (`:32-46`, `:219,252,270,320,323,326,329`). `AGENTS.md:11-14` says `intergrax/` MUST NOT import from `agents/` or `applications/`. The "ratified tier exception" has no decision id (HIST02-F-001, F-002). Operator decision DR-8 option 2 asks for a ticket that moves the CLI demo composition to Tier-3; an `ACCEPTED_LIMITATION` only if moving proves infeasible.

Recommended fix (alternative b2 in the plan; moving is feasible, so the limitation fallback is not needed): Tier-0 gets a name-free command-set seam (`intergrax/cli/command_set.py`, `build_parser`/`main` take `extra_command_sets`); the `external-work` and `demo governed-contractor` command groups (about 320 lines, moved verbatim) become a Tier-3 module `applications/governed_contractor_application/host/cli_commands.py` with a Tier-3 launcher `applications/governed_contractor_application/__main__.py`; `receipt verify` stays Tier-0. Public effect: those two command groups are invoked as `python -m applications.governed_contractor_application ...` instead of `intergrax ...` (Operator question Q1 in the plan; the surface-preserving fallback and the rejected options are costed there). Red-first gate `tests/unit/architecture/test_bnd03_tier0_upward_edges_gate.py` (strict zero, no allow-list); the ebh5 `intergrax/cli/` skip and the EBH-7 `TIER_IMPORT_EXCEPTIONS` entry are removed, i.e. tolerances become enforcement; no gate is weakened.

Exact files under `intergrax/`, `agents/`, `applications/` (7, none under `agents/`; none of the six `MATERIAL_GOVERNANCE_PATHS`): `intergrax/cli/command_set.py` (new), `intergrax/cli/main.py`, `intergrax/cli/external_work.py`, `applications/governed_contractor_application/__main__.py` (new), `applications/governed_contractor_application/host/cli_commands.py` (new), `applications/governed_contractor_application/host/offline_demo.py`, `applications/governed_contractor_application/tests/host/test_cli_verification_hardening.py`. Other files: the new gate, `tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py`, `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py`, the regenerated `docs/project/maintainers/qualification/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`, and five command-line docs listed in the plan. Full evidence, cost table with alternatives, test contract, negative probe, rollback, ACT-N11, risks and Operator questions: [`plans/PLAN_T-IGX-033-A.md`](plans/PLAN_T-IGX-033-A.md).

No PASS, closure, FRZ promotion (FRZ-BND-03 stays OPEN: the dynamic Tier-0 edge in `intergrax/applications/_shared/product_manifest_registry.py:16-22` is outside this ticket) or stage-parent change is implied. Independent audit follows in a fresh session.

## Plan digest

Plan digest: ae167672d4075e3e4dc190ac93c78a937d7ad0c435dfc3971e615143e4273775

## Approved

Approved: operator-delegate hist-sign-plans @ 2026-11-08T21:26:00+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-PRODUCT-PLANS-2026-11-08.md@9bfa07fa97c5054b07ff1f1dc9799d9ca77471f5

Authority cited for the scope (given in chat and recorded in `docs/project/maintainers/qualification/historical_certification_campaign/decisions/OPERATOR_DECISION_RECORDS_W0.md`, not signature lines of this ticket): DR-8 option 2 (move the Tier-0 -> Tier-3 demo composition), DR-12 = B (HELD for ACT-N11 sequencing; the fix is necessarily a product-path change), W1-PLAN step 3 (`OPERATOR_DECISION_RECORDS_W1.md`).

## Test contract

Phase 2 only, one pytest / ruff / uv process at a time, each under the heavy lock `D:/Projekty/intergrax/.tmp/session/hist-cert-campaign/locks/heavy.lock`, pytest's own exit code captured, logs under `.tmp/session/t-igx-033-a/`:

```text
uv run --extra dev-ci pytest -p no:xdist tests/unit/architecture/test_bnd03_tier0_upward_edges_gate.py tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py -q   # red proof before the product edit, green after; gate files byte-identical between the two runs
uv run --extra dev-ci pytest -p no:xdist <EBH-7 F10 modules that evaluate real inputs; find with git grep -l evaluate_c1 -- tests> -q                                       # red before, green after the matrix re-render
uv run --extra dev-ci pytest -p no:xdist applications/governed_contractor_application/tests/host/test_cli_verification_hardening.py applications/governed_contractor_application/tests/host/test_platform_completion.py applications/governed_contractor_application/tests/host/test_partner_attested_execution_demo.py -q
uv run --extra dev-ci pytest -p no:xdist <apps/health/env-diff scripts tests + certify/trace/evidence CLI tests listed in the plan 4.4> -q
uv run --extra dev-ci pytest -p no:xdist <ebh_3 / ebh_2a / ebh_2i / hardening_3 / EBH-7 F10 / ecap_l3 gates listed in the plan 4.5> -q
uv run --extra dev-ci ruff check <changed .py files> ; uv run --extra dev-ci ruff format --check <changed .py files>
git diff --name-only origin/development..HEAD    # only the allowed files; the six MATERIAL_GOVERNANCE_PATHS print nothing
```

Negative probe (recorded once, never committed): a function-local `from applications...` import and an `importlib.import_module("external_contractor_adapter...")` added to `intergrax/cli/main.py` in a scratch worktree must each turn the new gate red. Informational and expected red (not to be fixed here): `verify_iiac_policy_activation --require-in-force` (ACT-N11).

## Out of scope

Editing, weakening, skipping or allow-listing any gate beyond the two named tolerance removals; the `intergrax/integrations/providers/` skip; the `product_manifest_registry.py` Tier-0 -> Tier-3 dynamic edge; `AGENTS.md`, roadmap, FRZ checklist and IIAC record edits; the historical EBH-7 audit report prose; PASS/closure claims or FRZ promotion; Docker, vendor or production configuration (C10/C15); push, merge, rebase, PR.

## Lifecycle state

plan_approved

## Checkpoint

**operator-delegate pre-ent-wave2 (2026-11-10):** pytest pass @ HEAD `f2682d651f436c19f7000f940647be8fd134afa4` — still **NOT** `ready_for_audit`, no stage closure. Evidence: [`WAVE2_SERIAL_QUAL_LOG.md`](../../docs/project/maintainers/qualification/pre_ent_enterprise_certification/WAVE2_SERIAL_QUAL_LOG.md). Primary BND-03 CLI gate **not** in Wave 2 serial slice #1–#4; re-run `test_bnd03_tier0_upward_edges_gate.py` before audit routing.
