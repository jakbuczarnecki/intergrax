## PLAN `T-IGX-033-A` - HIST02-F-001 / F-002: remove the Tier-0 -> Tier-3 edges from the `intergrax` CLI (FRZ-BND-03; Operator decision DR-8 option 2)

**HELD - ACT-N11 sequencing (DR-12=B): do not approve or implement until MASTER confirms the single IIAC re-audit/re-activation batch.** Lifecycle `plan_pending`; no approval line exists or is requested by the planner.

Campaign: Integrax Historical Certification Closure, W1-PLAN step 3 (product batch). Planner session only: no product or test edit, no test run. This plan claims no PASS, no closure and no FRZ promotion; FRZ-BND-03 stays OPEN. Implementer != auditor != certifier != plan approver.

Basis: base `origin/development` = `af323c0057748bcaf472ae5290ceaeeba9bac97d`. All file:line below were read at that HEAD (`git grep` and file reads; nothing executed).

### 1. Evidence (verified vs thesis)

**1.1 The import chain.**

| Fact | Where | Status |
|---|---|---|
| Console script `intergrax = intergrax.cli.main:main` | `pyproject.toml:157`; wheel ships only `packages = ["intergrax"]` (`pyproject.toml:6`) | verified |
| `main.py` imports `register_parser` of `external_work` while building the parser | `intergrax/cli/main.py:31,43`; dispatch of `external-work`, `demo`, `receipt` at `main.py:105-116` | verified |
| `external-work` and `demo governed-contractor` run Tier-3 code | `run_external_work` `external_work.py:173-188`, `run_demo` `:191-198`, `_run_full_demo` `:217-247`, `_show_execution` `:250-265`, `_show_receipt` `:268-282` | verified |
| The static edge F-001 | `intergrax/cli/external_work.py:338`: `from applications.governed_contractor_application.host.collaborative_work_local_fixture import build_seeded_in_memory_external_work_authorization_boundary` (function-local, inside `_retry_attestation`, `:317`) | verified; it is the ONLY static `agents`/`applications` import in `intergrax/` (a `git grep -nE` for line-leading `from`/`import` of `agents` or `applications` under `intergrax/` gives 1 hit) |
| Dynamic edges F-002 | 7 `importlib.import_module` sites with bare Tier-2/Tier-3 names (`external_work.py:219,252,270,320,323,326,329`: `governed_contractor_application.host.{offline_demo,stores,orchestrator}`, `external_contractor_adapter.external_work_adapter`) plus `sys.path` injection of `.`, `applications`, `agents` by `_ensure_tier_paths` (`:32-46`) | verified (HIST02-F-002 said "7 sites": confirmed) |
| Which command reaches `:338` | `intergrax external-work retry-attestation <execution_id> [--store] [--demo-key]` -> `main.py:105-108` -> `run_external_work` `:181` -> `_retry_attestation` `:317`. The import statements at `:320-343` run BEFORE the fail-closed signer check at `:345`, so every invocation of that sub-command (including the "signing_key_source_required" refusal path) triggers it | verified by reading |
| Module identity quirk | `:338` imports with the `applications.` prefix; the other sites and the host code use bare `governed_contractor_application.` (both roots are put on `sys.path` by `_ensure_tier_paths`). `offline_demo.py:27` also uses the `applications.` prefix for the same fixture. Two names, two module objects; no cross-name `isinstance` was found | verified; pre-existing, NOT fixed here |
| Who else imports `intergrax/cli/external_work.py` | only `intergrax/cli/main.py` (`:31,43,106,110,114`). Tests reach it only through `intergrax.cli.main` (`applications/governed_contractor_application/tests/host/test_cli_verification_hardening.py:26,55`) or as a subprocess `-m intergrax.cli` (`:61-75`). No test imports the module directly | verified (`git grep` over `tests applications agents scripts`) |
| Operator-facing strings that name the commands | `offline_demo.py:417-420` (printed `recovery_command`, `verification_command` for `receipt verify` stays Tier-0); docs: `docs/project/integrations/impeachmentright_platform_completion.md:70,78,83`, `impeachmentright_full_validation.md:57`, `impeachmentright_validation_readiness.md:126,154`, `docs/project/technical/platform/execution_evidence_and_host_attestation.md:165,169`, `docs/project/technical/platform/governed_external_work_host_lifecycle.md:86,113,118` (`impeachmentright_cli_verification_hardening.md:23-30` is a recorded past result table, left untouched) | verified |

**1.2 The "ratified exception" and its authority (thesis: only `EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md:46`).** Partly verified, with more detail:
- `docs/project/maintainers/qualification/EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md:46` (`F10-C1-TIER-EXC-01`) is the prose.
- The executable form is `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py:200-206` (`TIER_IMPORT_EXCEPTIONS`, one entry: path `intergrax/cli/external_work.py`, prefix `applications.governed_contractor_application.host.collaborative_work_local_fixture`, evidence doc `docs/project/maintainers/audits/AGENT_PLATFORM_AC1_AC6_FINAL_ARCHITECTURE_AUDIT.md`). The audit doc says "One CLI lazy import ... PASS (CLI boundary only)" (line 118) and "CLI-only; not production host path" (line 178). The check that the evidence doc "ratifies" the entry is only `exc.path in doc.read_text()` (`tests/qualification/ebh_7/_ebh7_f10_ecap_normative_matrix.py:~226-230`), i.e. a substring match.
- No decision id exists anywhere (`git grep TIER-EXC` finds only the files above plus the HIST-02 dossier and the W0 decision record). Confirmed UNKNOWN.

**1.3 Gates that cover (or do not cover) BND-03 here.**

| Gate | What it does about this edge | Verdict |
|---|---|---|
| `tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py:48-60` `test_ebh5_rpl_g_tier0_no_agents_import_applications` | text search for `"from applications"` in `intergrax/**/*.py`, but `skip_prefixes = ("intergrax/cli/", "intergrax/integrations/providers/")` (`:50-57`) | TOLERATES: skips the whole CLI package; blind to `import applications...`, `from agents...` and every dynamic import |
| `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py:362-381` `scan_tier_import_violations` + `_ebh7_f10_ecap_normative_matrix.py:~215-237` `_tier_violations` + `:321-336` stale-exception check | AST scan of every import node in Tier-0 (function-local included); the violation is "ratified" by the single `TIER_IMPORT_EXCEPTIONS` entry; an exception with no matching violation is reported as `stale tier exception` and fails C1-TIER-0 | TOLERATES by allow-list; will turn RED the moment the import disappears unless the entry is removed (coupling that this plan uses) |
| The rendered matrix `docs/project/maintainers/qualification/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md:53` (C1-TIER-0 row carries "ratified: intergrax/cli/external_work.py -> applications.governed_co...") | C5-MATRIX-SYNC compares disk to the deterministic render (`_ebh7_f10_ecap_normative_matrix.py:826-830`) | after the exception is removed the render changes, so this file MUST be regenerated (`scripts/qualification/render_ebh7_f10_normative_matrix.py`, named in `_ebh7_f10_ecap_normative_matrix.py:79`) |
| `tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py` | contracts -> runtime only | not relevant to Tier-0 -> Tier-3 |
| HIST-02 candidates `test_ebh_3_dependency_ownership_gate.py`, `test_ebh_2a_*`, `test_ebh_2i_*` | listed by HIST-02 as scoped BND gates, PASS at `ba07eba` | NOT read by the planner; not verified that any of them would flag Tier-0 -> Tier-3 |

No gate today enforces "Tier-0 has no edge to agents/applications" without a skip or an allow-list. That is the red-first target of section 4.

**1.4 Observations outside the thesis (reported, NOT in this ticket's scope):**
- `intergrax/applications/_shared/product_manifest_registry.py:16-22` hard-codes four Tier-3 module names (`legal_application.manifest`, `research_application.manifest`, `dispute_sim_application.manifest`, `local_workspace_application.manifest`) and imports them with `importlib` (`:36-37`). It is a Tier-0 -> Tier-3 dynamic edge that HIST-02 F-002 did not list (F-002 counted only the CLI). Read by grep and a short file read; not executed.
- `intergrax/cli/apps.py:26-30` (`_ensure_paths`) also inserts `applications` and `agents` into `sys.path` (registry CLI); not an import.
- Tier-0 has an unrelated, governed plugin framework (`intergrax/core/plugins/discovery.py:22-40`, `EP_*` entry-point groups, closed-world catalogs `tests/qualification/plug_03|plug_04|plug_final/catalog.py`). See alternative (b1).

**1.5 What `_retry_attestation` does with the fixture (the composition to move).** It builds `RuntimePolicyBundleEvaluator`, calls `build_seeded_in_memory_external_work_authorization_boundary(policy, tenant_id="offline-demo-tenant", workspace_id="workspace-a", principal_id="offline-demo-user", decision_requirement_policy=PermissiveDecisionRequirementPolicy())` (`:355-361`), wires `ExternalWorkAdapter(_RefuseProvider(), authorization_boundary=...)` and `GovernedExternalWorkOrchestrator` against the persisted filesystem store (`:363-376`), calls `orch.retry_attestation` (`:378`), exports the receipt and verifies it (`:390-407`). All of it is demo/recovery composition over Tier-2/Tier-3 types; none of it is a Tier-0 capability. By contrast `receipt verify` and the key resolver (`external_work.py:134-170,426-609`) use only `intergrax.contracts` / `intergrax.runtime.execution_evidence` and are legitimately Tier-0.

### 2. Cost of moving, with alternatives

Moved code, measured: `external_work.py` is 609 lines. Moves to Tier-3: `_ensure_tier_paths` (`:32-46`), the `external-work` and `demo` parsers (`:61-132`), `run_external_work`, `run_demo`, `_run_full_demo`, `_show_execution`, `_show_receipt`, `_RefuseProvider`, `_resolve_retry_attestor`, `_retry_attestation` (`:173-198`, `:217-423`): about 320 lines. Stays in Tier-0: the `receipt verify` parser (`:134-170`), `run_receipt` (`:201-214`), key-file and key-resolver helpers (`:426-609`), about 290 lines.

Public CLI surface today: `intergrax external-work {demo-create,demo-accept,show,receipt,retry-attestation}`, `intergrax demo governed-contractor`, `intergrax receipt verify`.

| Option | Shape | Files / lines (estimate) | Surface | Verdict |
|---|---|---|---|---|
| **(a) move composition into `applications/`, entry point lives there** | `[project.scripts] intergrax` would have to point into `applications/` | root `pyproject.toml`, 4+ moved modules | **Infeasible in pure form**: `intergrax` is the ONLY entry point of the whole Tier-0 CLI (scaffold, run, doctor, apps, certify, trace, evidence, mvp, init, envs), the wheel contains only `intergrax` (`pyproject.toml:6`), and `python -m intergrax.cli.main apps ...` is relied on by `tests/unit/scripts/test_check_application_registry.py:33,42,58`, `test_check_application_health_score.py:36`, `test_check_application_environment_diff.py:36`. A Tier-3 application cannot own the platform CLI. The variant "separate Tier-3 launcher with NO seam" has to reach into argparse's private sub-parser action or duplicate the Tier-0 command table; it is B (below) without the clean seam | rejected |
| **(b1) inversion by entry points** (new group, e.g. `intergrax.cli`, declared in root `pyproject.toml`, loaded by `intergrax/core/plugins/discovery.py`) | keeps `intergrax external-work` unchanged | `pyproject.toml`, `discovery.py`, new Tier-3 module, plug catalogs, tests | Surface unchanged. Costs: it adds a new platform plugin group to a governed, closed-world surface (`EP_*` constants, PLUG-03/04/final catalogs, `tests/contract/core/plugins`) and so widens the PLG family that HIST-02 still has open (F-004); depends on the editable-install metadata being regenerated (stale `entry_points.txt` silently removes the commands); Tier-3 is `package = false` (`applications/governed_contractor_application/pyproject.toml`), so `applications/` must still be put on `sys.path` by Tier-0, keeping the F-002 `sys.path` injection | rejected as default; the fallback if the Operator refuses a surface change (Q1) |
| **(b2) inversion by injection (RECOMMENDED)** | Tier-0 owns a name-free seam; Tier-3 owns the commands and a launcher that injects them | see section 3: 3 files under `intergrax/` (1 new, 2 edited), 4 under `applications/` (2 new, 2 edited); Tier-0 shrinks by about 300 lines, Tier-3 grows by about 380 | **Changes the invocation**: `intergrax external-work ...` / `intergrax demo ...` become `python -m applications.governed_contractor_application external-work ...` / `demo ...` (the launcher also serves every Tier-0 command, including `receipt verify`). Tier-0 `intergrax` keeps `receipt verify` | recommended |
| **(b3) module string from env/flag** (`INTERGRAX_CLI_HOST=...`) | Tier-0 imports whatever string it is given | 1-2 files | adds ambient configuration and keeps a string-driven import in Tier-0 that no static or literal-string gate can tell apart from the violation; the surface still changes (the env must be set) | rejected |
| **(c) ACCEPTED_LIMITATION** | code unchanged | 0 product files; a decision record | Needs an Operator decision id (DR-8 allows it only if moving is infeasible). Moving is feasible (b2), so the fallback condition is not met. Leaves F-001, F-002 and the `intergrax/cli/` skip in the ebh5 gate open, and `AGENTS.md:11-14` ("never violate") needs a written exception | rejected unless the Operator chooses it after Q1 |

**Recommendation: (b2)**, because (1) it is the only option that gives `intergrax/` zero knowledge of any `agents`/`applications` name without adding a platform plugin group or ambient config; (2) the moved code is verbatim, so behaviour risk is small and checkable with `git diff --color-moved`; (3) it lets the gate be a strict zero with no allow-list; (4) the only price is a documented change of invocation for two command groups in a demo/recovery CLI, which is why it is Operator question Q1.

Infeasible: (a) in its pure form (reasons in the table). Nothing else is infeasible.

What (b2) does NOT achieve: the project-wide "reverse dependencies = 0" for DYNAMIC edges. After this ticket Tier-0 has zero static edges (whole of `intergrax/`) and zero Tier-2/Tier-3 module-name strings in `intergrax/cli/`; the `product_manifest_registry.py` edge (1.4) remains and needs its own finding/ticket (Q4).

### 3. Exact edit scope

**3.1 Product paths (these are the ACT-N11 triggers; section 6):**

| Path | Action |
|---|---|
| `intergrax/cli/command_set.py` | NEW, about 40 lines. `CliCommandSet` Protocol: `register_parsers(sub: argparse._SubParsersAction) -> None`; each registered leaf parser calls `set_defaults(cli_run=<callable taking Namespace, returning int>)`. No import of `agents`/`applications`; no Tier-2/3 name |
| `intergrax/cli/main.py` | `build_parser(extra_command_sets=())` and `main(argv=None, *, extra_command_sets=())`; each set registers after the built-in ones; dispatch ends with `run = getattr(args, "cli_run", None); if run: return run(args)` before the existing `parser.error`. Remove the `external-work` and `demo` branches (`:105-112`); keep `receipt`. A duplicate command name must fail closed (argparse raises on a conflicting sub-parser; a test pins it) |
| `intergrax/cli/external_work.py` | delete the moved block (about 320 lines, 1.5), keep the `receipt verify` parser and helpers; docstring updated; `_ensure_tier_paths`, all 7 `import_module` sites and `:338` disappear; drop imports that become unused (ruff) |
| `applications/governed_contractor_application/host/cli_commands.py` | NEW, about 340 lines: the moved code VERBATIM (same flags, same JSON payloads, same exit codes, same stderr text), exposed as `GOVERNED_CONTRACTOR_COMMANDS` (a `CliCommandSet`). Module-level imports are light (`argparse`, `json`, `pathlib`, `sys`); heavy imports stay inside the functions so `--help` stays fast. Because it is Tier-3, the `importlib` indirection becomes plain imports of `governed_contractor_application.host.*` and `external_contractor_adapter.*`; the fixture keeps the exact `applications.`-prefixed path it has now (module identity unchanged) |
| `applications/governed_contractor_application/__main__.py` | NEW, about 30 lines: puts `applications/` and `agents/` on `sys.path` (the checkout convenience `_ensure_tier_paths` provided, now owned by Tier-3), then `return intergrax.cli.main.main(argv, extra_command_sets=(GOVERNED_CONTRACTOR_COMMANDS,))`. Run as `uv run python -m applications.governed_contractor_application <args>` from the repo root (`applications/__init__.py` and `agents/__init__.py` exist; `applications/governed_contractor_application/__init__.py` is empty) |
| `applications/governed_contractor_application/host/offline_demo.py` | only the printed `recovery_cmd` string (`:417-420`) -> the launcher form. `verify_cmd` (`receipt verify`) is unchanged. The existing `from applications.... collaborative_work_local_fixture import` at `:27` is Tier-3 -> Tier-3 and stays |
| `applications/governed_contractor_application/tests/host/test_cli_verification_hardening.py` | route the `demo` / `external-work` calls (the list at `:144-152`, `:163`, `:307-326`, `:335-360`, `:396`, plus any of `:368-408` that are not `receipt verify`; phase 2 sorts them) through a new in-process helper and a subprocess helper that use the Tier-3 launcher; `receipt verify` calls keep `intergrax.cli.main`. Assertions unchanged. Add: Tier-0 alone rejects `external-work` and `demo` (exit 2); the launcher accepts them; a duplicate command-set name fails closed; every `--help` stays ASCII-safe |

No file under `agents/`. No change to `pyproject.toml`, `uv.lock`, any dependency, or any `[project.scripts]` entry.

**3.2 Non-product paths:**

| Path | Action |
|---|---|
| `tests/unit/architecture/test_bnd03_tier0_upward_edges_gate.py` | NEW, the red-first gate (section 4) |
| `tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py` | remove the `"intergrax/cli/",` entry from `skip_prefixes` (`:51`); leave the `intergrax/integrations/providers/` entry alone (no finding about it; the repo-wide AST count shows it is clean, so removing it is a separate, optional hardening) |
| `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py` | `TIER_IMPORT_EXCEPTIONS` becomes `()`; nothing else |
| `docs/project/maintainers/qualification/EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md` | regenerated by `scripts/qualification/render_ebh7_f10_normative_matrix.py`, never hand-edited (required by C5-MATRIX-SYNC). Stage rows for EBH-7 stay as they are |
| `docs/project/integrations/impeachmentright_platform_completion.md`, `impeachmentright_full_validation.md`, `impeachmentright_validation_readiness.md`, `docs/project/technical/platform/execution_evidence_and_host_attestation.md`, `docs/project/technical/platform/governed_external_work_host_lifecycle.md` | command lines updated to the launcher form (docs-only commit) |
| `docs/project/maintainers/qualification/EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md:46` | NOT edited by this ticket: it is a historical audit record. The auditor notes that row F10-C1-TIER-EXC-01 is superseded; if `test_ebh_7_w8_ecap_l3_gates.py` pins that row, phase 2 STOPs and asks (pre-check c) |

Commits: C1 = everything under `intergrax/`, `applications/`, `tests/` and the regenerated matrix (ONE atomic commit: removing the exception without the matrix, or the product change without the exception removal, is red by construction, so C1 is never pushed apart from itself); C2 = docs. Trailer on both: `Ref: governance/tickets/T-IGX-033-A.md`.

### 4. Test contract

Rules: phase 2 only; ONE `uv`/pytest/ruff process at a time under `D:/Projekty/intergrax/.tmp/session/hist-cert-campaign/locks/heavy.lock`; always `uv run --extra dev-ci pytest -p no:xdist`; pytest's own exit code captured with `; echo EXIT=$?` into the log; logs under `.tmp/session/t-igx-033-a/`; no `--basetemp` outside the repo; no diagnostic broad suites (C10/C15; `intergrax-test-process-budget`).

**The new gate** `tests/unit/architecture/test_bnd03_tier0_upward_edges_gate.py` (no allow-list, no skip, no xfail, no tolerance constant):
1. `test_tier0_has_no_static_upward_imports`: AST over every `intergrax/**/*.py`; every `Import`/`ImportFrom` node anywhere (module level, function-local, `TYPE_CHECKING`, `try`); a violation is a top-level module in `{"agents", "applications"}`. Expected set: empty. RED at HEAD with exactly `intergrax/cli/external_work.py -> applications.governed_contractor_application.host.collaborative_work_local_fixture`.
2. `test_tier0_cli_has_no_tier2_tier3_module_strings`: AST over `intergrax/cli/**/*.py`; any string constant that is a dotted module path whose first segment is in `TIER23_ROOTS`, where `TIER23_ROOTS = {"agents", "applications"} | {every directory directly under agents/ and applications/ that contains __init__.py}`, computed from the filesystem at test time (so a new application is covered automatically), plus any call of `import_module` / `__import__` whose first argument is not a string literal. RED at HEAD with the F-002 sites.
3. Negative controls (permanent tests, synthetic trees under `tmp_path`, scan functions take a root argument): a function-local `from applications.x import y` is detected; `import agents.x` is detected; `importlib.import_module("governed_contractor_application.x")` is detected; `"intergrax.applications._shared"` and `"intergrax.agents.x"` are NOT flagged (Tier-0 packages that share the names); the scan fails if the root does not resolve (no vacuous pass).

**The edited gates:** `test_ebh5_rpl_g_tier0_no_agents_import_applications` without the CLI skip; the EBH-7 F10 C1 evaluation with `TIER_IMPORT_EXCEPTIONS = ()`. Neither is weakened: both lose a tolerance. No assertion is removed.

**Commands (phase 2):**
1. Red proof BEFORE any product edit (gate files and test edits staged in the tree, product untouched):
   `uv run --extra dev-ci pytest -p no:xdist tests/unit/architecture/test_bnd03_tier0_upward_edges_gate.py tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py -q` -> expected: the new tests 1 and 2 FAIL, the ebh5 Tier-0 test FAILS (`tier boundary violation: intergrax/cli/external_work.py`), negative controls PASS. Record the log.
   Then the EBH-7 F10 module(s) that evaluate the real inputs (find with `git grep -l evaluate_c1 -- tests`; record the names): expected RED on C1-TIER-0 (unratified violation) and C5-MATRIX-SYNC only after the exception is removed.
2. Implement C1. Same commands as 1: all GREEN. Gate files byte-identical between steps 1 and 2 (`git diff` empty for them).
3. Behaviour preservation (one run): `applications/governed_contractor_application/tests/host/test_cli_verification_hardening.py`, `test_platform_completion.py`, `test_partner_attested_execution_demo.py`. The existing offline demo, `--simulate-signing-failure`, fresh-process `retry-attestation`, idempotent second retry and `receipt verify` cases are the semantic proof of the verbatim move.
4. Tier-0 CLI consumers (one run): `tests/unit/scripts/test_check_application_registry.py`, `test_check_application_health_score.py`, `test_check_application_environment_diff.py`, `tests/unit/runtime/evidence/test_certify_cli.py`, `test_trace_cli.py`, `test_evidence_cli.py`.
5. Boundary and qualification neighbours (one run): `tests/unit/architecture/test_ebh_3_dependency_ownership_gate.py`, `test_ebh_2a_public_contract_boundary_gate.py`, `test_ebh_2i_final_rescan_gate.py`, `tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py`, the EBH-7 F10 modules from step 1, `tests/qualification/ebh_7/test_ebh_7_w8_ecap_l3_gates.py`.
6. Lint: `uv run --extra dev-ci ruff check <changed .py files>` then `ruff format --check <same>`.
7. Matrix: `uv run python scripts/qualification/render_ebh7_f10_normative_matrix.py` (writes the file; confirm the only change is the C1-TIER-0 row), then re-run the EBH-7 F10 module once.
8. Diff guards: `git diff --name-only origin/development..HEAD` equals sections 3.1 + 3.2 plus the ticket and plan; `git diff --name-only origin/development..HEAD -- scripts/governance/verify_iiac_policy_activation.py scripts/governance/iiac_activation_gate.py scripts/governance/iiac_common.py scripts/governance/verify_iiac_audit_manifest.py scripts/governance/verify_iiac_role_separation.py docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` prints nothing.
9. Informational, EXPECTED RED, do not fix: `uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force` (ACT-N11 lists the product paths). Record the lines verbatim.
10. `uv run python -m scripts.governance.igx_gov ticket` if feasible.
11. Independent replay (full QUAL-X P5 BND manifest, EBH replay) belongs to the auditor on a clean pinned checkout and needs a separate Operator authorization (C10/C15).

**Negative probe (recorded once, never committed):** after C1, in a scratch worktree add `from applications.governed_contractor_application.host.stores import FilesystemHostStore` inside a function in `intergrax/cli/main.py` -> step 2 gate test 1 must fail naming that file; revert with `git checkout -- intergrax/cli/main.py`. Repeat with `importlib.import_module("external_contractor_adapter.external_work_adapter")` -> gate test 2 must fail. The permanent equivalents are the synthetic controls in 3 above.

**Phase-2 pre-checks that may change the plan (STOP and report, do not improvise):**
(a) the IIAC G0 inventory / arch-freeze closed-world inventories (`scripts/governance/iiac_g0_inventory.py`, `verify_iiac_g0_inventory_completeness.py`) must not fail on NEW files under `intergrax/cli/` and `applications/governed_contractor_application/` (the planner did not execute them);
(b) `test_ebh5_rpl_g_...` is a text search: after the move no file in `intergrax/cli/` may contain the text `from applications` even in a comment or docstring;
(c) whether `tests/qualification/ebh_7/test_ebh_7_w8_ecap_l3_gates.py` or any test pins the prose row `F10-C1-TIER-EXC-01` in the audit report (read of that test was limited to a grep hit at line 17);
(d) the launcher resolves `external_contractor_adapter` and `governed_contractor_application` on a fresh `uv run` from the repo root with no `PYTHONPATH` set (not executed by the planner);
(e) the regenerated matrix differs from the committed one in the C1-TIER-0 row only.

### 5. Rollback

`git revert` C2 then C1: restores the previous CLI, the ebh5 skip, the exception entry and the matrix in one step; no data or state migration (the CLI persists nothing new; store formats are untouched). Revert is also the response to any phase-2 STOP. After a revert, the F-001 exception is again unrecorded (status quo).

### 6. ACT-N11 and sequencing

- Lifecycle `plan_pending`, HELD, no approval line. Product-path change: it turns `verify_iiac_policy_activation --require-in-force` red (ACT-N11 "product path changed after the audited SHA", `scripts/governance/iiac_activation_gate.py:576`) until the single re-activation. It must therefore land inside the product batch before that re-activation (DR-12=B; W1-PLAN step 3).
- Exact files under `intergrax/`, `agents/`, `applications/` (7, all in section 3.1): `intergrax/cli/command_set.py`, `intergrax/cli/main.py`, `intergrax/cli/external_work.py`, `applications/governed_contractor_application/__main__.py`, `applications/governed_contractor_application/host/cli_commands.py`, `applications/governed_contractor_application/host/offline_demo.py`, `applications/governed_contractor_application/tests/host/test_cli_verification_hardening.py` (a test, but under `applications/`, so it counts). `agents/`: none.
- None of the six `MATERIAL_GOVERNANCE_PATHS` (`scripts/governance/iiac_activation_gate.py:40-49`) is touched; guard in 4.8.
- File-disjoint from HELD `T-IGX-030-B` (contracts `runtime_error_code`, nexus `error_codes`, `tool_operation_recovery`) and `T-IGX-030-C` (`invoker.py`); one IIAC cycle can cover all. Not checked against `T-IGX-032-A` (another planner, not on disk at this HEAD).

### 7. Roles

Planner: this session (architect-planner, read-only). Plan approver: a separate session, or the Operator; never the planner or implementer. Implementer: fresh `implementer` session after approval and the single re-activation sequencing. Tester: runs section 4 under the heavy lock and records the checkpoint. Reviewer: code review handoff. Independent auditor (G6): a fresh `independent-quality-auditor` session, not the implementer. Certifier / signing: separate dedicated session.

### 8. Risks

| Risk | Mitigation |
|---|---|
| Documented invocation of two command groups changes (b2) | Q1; docs commit C2; printed `recovery_cmd` updated; optional Tier-0 stub (Q2) |
| A behaviour drift hidden in the move | verbatim move, checked with `git diff --color-moved=dimmed-zebra`; same tests drive the new entry point |
| Module identity (`applications.` vs bare prefix) changes by accident | keep the exact imports; pre-check (d) |
| `sys.path` bootstrap now lives in the launcher; plain `python -m applications.governed_contractor_application` outside the repo root fails | documented: run from the repo root, as before for `intergrax` demo commands (the old code also used `Path.cwd()` / repo root) |
| Editing a qualification input (`TIER_IMPORT_EXCEPTIONS`) and regenerating an EBH-7 record | removal only strengthens; the regenerated diff is confined to one row (pre-check e); Operator Q3 |
| A reviewer reads "static = 0" as "BND-03 closed" | explicit in section 2 and the ticket: FRZ-BND-03 stays OPEN; the `product_manifest_registry.py` dynamic edge and the `sys.path` conveniences remain recorded |

### 9. Residuals and what the planner could not verify

Not executed (brief forbids tests): that the new gate is red at HEAD and green after (derived from the static facts in 1.1); the launcher import resolution; the G0 inventory and arch-freeze inventories on new files; the render script's determinism; the exact ACT-N11 trigger set; whether any of the HIST-02 candidate gates (`ebh_3`, `ebh_2a`, `ebh_2i`) would also flag the edge. Read only partially: `test_ebh_7_w8_ecap_l3_gates.py` (grep line), the AC1-AC6 audit document (grep lines 118, 178), `scripts/gates/check_mvp_promotion_gates.py` (matched a broad regex, not examined). The `Grep` tool timed out once over the whole repo, so repo-wide claims rely on `git grep` (tracked files only). Residual after this ticket: dynamic edge `product_manifest_registry.py`; `apps.py` `sys.path` convenience; the dual module identity; FRZ-BND-03 and the whole family stay OPEN and need the independent audit and Operator decisions of their own.

### 10. Questions for the Operator

- **Q1 (decides the shape).** Accept that `intergrax external-work ...` and `intergrax demo governed-contractor ...` become `python -m applications.governed_contractor_application external-work|demo ...` (Tier-0 `intergrax` keeps `receipt verify` and every other command)? If NO, the surface-preserving fallback is (b1) entry points, which costs a new platform plugin group, catalog work and install-metadata risk; this plan would be revised, not silently widened.
- **Q2.** Tier-0 behaviour for the old names after the move: plain argparse "invalid choice" (default of this plan) or a 3-line stub that exits 2 and prints the new command?
- **Q3.** Is regenerating `EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md` (one row) acceptable inside this batch? It is unavoidable if the exception entry is removed.
- **Q4.** Open a separate finding and ticket for the `product_manifest_registry.py` Tier-0 -> Tier-3 dynamic edge, and decide whether the dynamic-edge gate should later cover all of `intergrax/` (it covers only `intergrax/cli/` here, because widening it would turn red on code this ticket does not own, and an allow-list is forbidden)?
- **Q5.** Confirm this ticket joins the single IIAC re-activation with 030-B and 030-C (DR-12=B).
- **Q6.** Only if the Operator rejects Q1 and prefers (c): the Operator's own decision id for an ACCEPTED_LIMITATION (not available from the planner).

### 11. Out of scope

Editing, weakening, skipping or allow-listing any gate other than the two named tolerance removals; the `intergrax/integrations/providers/` skip; the `product_manifest_registry.py` edge; the dual module identity; `AGENTS.md`, roadmap, FRZ checklist and IIAC record edits; any promotion of FRZ-BND-03/BND-04 or any PASS/closure claim; the other replay-red gates; Docker, vendor or production configuration (C10/C15); push, merge, rebase, PR.
