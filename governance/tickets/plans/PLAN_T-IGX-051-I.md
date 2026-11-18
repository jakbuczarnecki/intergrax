# PLAN T-IGX-051-I — RW train: thread the A4-007 import allowlist through the product callers (seams b1 and b2)

Draft, 2026-11-17. Nothing was changed outside `.tmp`; no pytest process was started, no worktree or branch was created. No PASS, closure, FRZ promotion, stage certification or independent verification is claimed anywhere in this plan. Evidence notation `path@sha:line`; `T` = `6b94d202fb79a0611404f8d83d14505c6d7cb0c5` (`origin/arch-freeze/rw-train`, PT-WA and PT-WB merged), `D` = `44d1e5d105beb9b62d0a12e693066b0b26c125eb` (`origin/development`). The facts are in `FACT_CHECK.md` (ids F-xx).

## 0. Revision 2 (2026-11-17): changes against the draft of digest `2add59746348470889b69598bda2498f8e022cd7e78fe97b1f8b099c7ad5a244`

Cause: the step-0 dry run `dry-run/DRY_RUN_T-IGX-051-I.md` at the train head `881428fb231f34215a67364aa5788a4f2f986e83` (DRY_RUN_FINDINGS, I-01..I-06). The previous plan text is kept as `dry-run/PLAN_T-IGX-051-I.v2add597.md`. **The write set is unchanged** (10 firm, 2 conditional). Added to the regression scope only (no edit): the two AC3 files named by I-02. Added tests: CTL-A7C-05 and CTL-A7C-06.

| Finding | Disposition | Where |
|---|---|---|
| I-01 PT-FC2 not merged, `wiring_posture.py` absent | already the plan's step-0 stop; made a stated start condition of the ticket | sections 1, 10 |
| I-02 registry-derived set misses installed distribution agents; two AC3 files likely red | **re-derived by reading, the conclusion differs**: no second registered source is needed. Revision-bound installed agents are authorised by digest through the factory resolver and carry no `import_path`, so they never reach seam b1. The two named files use LAB manifests and a context without an environment, so the policy derived from the context is unrestricted. Both files join V2 and two control tests pin the shapes. Residual RES-5 and question Q-A7C-09 for the rehydration context. Still not confirmed by a run | sections 5a, 8, 11, 16 |
| I-03 F-31 wrong: `AgentBinding` fills `factory_path` from a typed factory | accepted; facts, section 2 and the factory-set description corrected; `registered_factory_prefixes()` is not empty | sections 2, 4; FACT_CHECK F-31 |
| I-04 `check_application_production_gates.py` is the entrypoint of the fast-mandatory check `ENT-A-PRD-01` | accepted; stated in Q-A7C-06 and the ACT-N11 section | sections 12, 16 |
| I-05 ruff I001 and format hunks on touched files | accepted; one mechanical lint commit before the behavioural edit | section 10 |
| I-06 stale "v15 has no block" | accepted; RW-PLAN v18 carries the block with an equal write set | sections 9, 16 |
| source-text pins in V2 files (dry run, traps) | accepted as design constraints | section 5a |

## 1. Purpose and findings

Train node `T-AF2-A4007-CALLERS` (ticket `T-IGX-051-I`; a node added after RW-PLAN v15; RW-PLAN v18 carries its block, see section 9). PT-WB (`T-IGX-043-H`, merged into the train as `6b94d202f`) landed the loader capability for the importlib seams: the helper `import_allowlist.py`, and the keyword-only parameters `allowed_prefixes` and `fail_closed` on `resolve_agent_type`, `load_callable`, `_load_manifest_for_package`, `check_application_migrations` and `load_mcp_server_builder` (F-01..F-05). No product caller passes either parameter, so with the defaults (`None`, `False`) the check is a no-op and the seams b1 and b2 still import on every production path (F-06, F-07). The PT-WB ticket records this as a residual ("Implementer checkpoint", F-08) and the independent review of PR #654 recorded finding H-1 with the orchestrator ruling "recorded residual" (F-09). The closure condition recorded there: "threading callers edited after `wiring_posture.py` is on the train, plus a product derivation of the permitted sets from the registered source". The orchestrator ruled (option b) that this is a separate small ticket, not an amendment of `T-IGX-043-B` (F-10).

| Item | Statement (labels quoted with attribution) | What this ticket does | What it cannot close alone |
|---|---|---|---|
| EQOA-A4-007 seam b1 (`resolve_agent_type`, `load_callable`) | PT-WB ticket checkpoint: "b1 ... still import without a prefix check on production paths"; PR #654 review H-1 | Part B: the product callers of the census pass the permitted set and `fail_closed` | the exact-SHA wave audit of the RW train (flag from RW-PLAN `audit_flags`); this ticket claims no audit outcome |
| EQOA-A4-007 seam b2 (`_load_manifest_for_package`) | the same | Part C: the two product callers of `check_application_migrations` (both are scripts) pass the permitted set and `fail_closed` | same |
| The product derivation of the permitted sets | PT-WB plan Q-WB-06 option A and rule of section 4.E: "derived from one existing registered source, never a second hand-typed list" | Part A: one new module that derives the sets from the product manifest registry | which source is the registered one is open question Q-A7C-01 (default: the registry) |

Dependencies: PT-FC2 (`T-IGX-043-B`) merged into the train (it creates `wiring_posture.py` with `wiring_fails_closed(env)`; absent at `T` and at the train head `881428fb2`, and PT-FC2 is not on `origin`; **the ticket cannot start before it merges**, F-11); PT-WA and PT-WB merged (done). Overlaps with other train tickets: none (section 9).

## 2. Facts at the train head: the caller census (re-derived, not copied)

Method: `git grep` at `T` over `intergrax agents applications scripts platform_proofs` for each loader and wrapper name; every row read in context (F-12..F-30). The PT-WB census (F-08) is confirmed line by line; three items it did not list are added (rows X-1..X-3).

| Row | Caller (path:line at `T`) | Reaches an import? | Posture or manifest available at the call | Action in this plan |
|---|---|---|---|---|
| C-1 | `wiring.py:109` `resolve_builder` calls `resolve_agent_type_from_binding` (only when `builders` is given and neither `factory` nor `builder_key` hit) | yes (`import_path` of a serialized binding) | none inside; the only caller `build_agent_from_binding` (`wiring.py:209`) holds `ctx` | thread a policy from `ctx` |
| C-2 | `wiring.py:127` `_validate_factory_result` calls it (when `agent_type` or `import_path` is set) | yes | none inside; callers `invoke_canonical_agent_factory` (`:150`) and `invoke_legacy_compatible_agent_factory` (`:182`) hold `ctx` | thread a policy from `ctx` |
| C-3 | `wiring.py:214` `load_callable(binding.factory_path)` | yes (`factory_path`) | `ctx` in `build_agent_from_binding` | pass `factory_modules` and `fail_closed` |
| C-4 | `wiring.py:219` `resolve_agent_type_from_binding` | yes | `ctx` | thread a policy from `ctx` |
| C-5 | `environment_conformance.py:65` in `validate_binding(binding, env)` | yes (after the early returns for factory, `factory_path`, contract-reference bindings) | `env` is a parameter | policy from `env` |
| C-6 | `environment_diff_wiring.py:108` in `_binding_key` | **no**: reached only when `contract_id` and `import_path` are both falsy (`:101-105`); `import_path` is `None` or a validated dotted path (`manifest.py:252`), so the call resolves a typed `agent_type` or raises `ValueError` before any import (`agent_resolution.py:26-48`) | none | not edited; a control test proves it (CTL-A7C-03); Q-A7C-04 |
| C-7 | `roster_agent_contract_authority.py:117-120` in `resolve_roster_agent_contract(..., allow_compatibility_resolver=True)` | yes | none (`binding` only) | forward `allowed_prefixes`, `fail_closed` from the callers that hold a posture (C-9) |
| C-8 | `roster_agent_contract_authority.py:131-137` in `materialize_manifest_contract_authority_lab_compat(manifest)` | yes | the manifest: `manifest.resolved_environment()` (`manifest.py:422`) | policy from the manifest's resolved environment |
| C-9 | `package_wiring.py:114-117` `collect_application_dependencies(manifest, env, ...)` is the only caller of C-7 that holds an `env`; its compat flag is `not requires_revision_authority`, and `requires_revision_authority` is PRODUCT **and** STRICT (`:108-112`), while `wiring_fails_closed` is PRODUCT **or** STRICT (F-11), so PRODUCT-BALANCED and LAB+STRICT hosts reach C-7 inside the guard postures | yes | `env` | conditional path, Q-A7C-02 |
| C-10 | `check_application_migrations` callers: `scripts/maintenance/check_application_migrations.py:24` and `scripts/gates/check_application_production_gates.py:229-232` | yes (`{package}.manifest` for every `*_application` directory with migration JSON, `migration_wiring.py:56-60,327-349`) | none: scripts; no `env` | derive the guarded policy (Part C) |
| X-1 | `wiring.py:604-608` `load_agent_class(import_path)` calls `resolve_agent_type(agent_type=None, ...)` directly | yes | none | **not a caller of the census, and no product caller exists** (only re-exports in `_shared/__init__.py` and `conformance.py`); left unedited, residual RES-1 |
| X-2 | `agent_certification_wiring.py:47-50` calls `resolve_roster_agent_contract(... allow_compatibility_resolver=True)` | yes | none (reference-roster staging helper, no `env`) | not edited, residual RES-2 |
| X-3 | `agent_resolution.py:111` imports `{package}.contract` of an already resolved class | yes, but the package comes from a class that has passed the check, not from caller data | n/a | unchanged (PT-WB plan 4.E left it unchanged), residual RES-3 |
| n/a | `load_mcp_server_builder` | no product caller (F-08) | n/a | nothing to thread |

Other facts that shape the design: the shipped manifests build their rosters with `AgentBinding.mount(<class>)`, so `import_path` is auto-filled and `agent_type` is set (`manifest.py:178`); `resolve_agent_type` returns early for a typed binding (`agent_resolution.py:36-40`), so typed rosters are never refused. `AgentBinding` fills `factory_path` from a typed `factory` (`manifest.py:299-305`), so three of the four shipped manifests carry a `factory_path` and the registry-derived factory set is `{dispute_sim_application, legal_application, local_workspace_application}` at `881428fb2` (dry run I-03, F-31); the `load_callable` branch of `build_agent_from_binding` runs only when `factory` is `None` (`wiring.py:213`), so no shipped roster takes it. The product manifest registry is `_REGISTERED` filled by `register_product_manifest` from the Tier-3 catalog (F-32); the gate scripts import the catalog, the maintenance script does not (F-33). No application migration JSON exists under `applications/` at `T` (F-34), so the b2 gate scans nothing at the head; the exposure is the callable shape, which is what the closure condition names.

## 3. Design rules

1. **Callers only.** The write set is the census callers, one new derivation module and the tests. The loaders, `import_allowlist.py`, `wiring_posture.py`, `migration_wiring.py` and `mcp_import_guard.py` are not edited (PT-WB and PT-FC2 files; shared paths are serialization, not conflict).
2. **Posture through the one helper.** Wherever a caller holds an environment profile, the fail-closed decision is `wiring_fails_closed(env)` imported from `intergrax.applications._shared.wiring_posture`; this ticket defines no second predicate. LAB (and any non-guard posture) stays unrestricted as today (Q-WB-06 option A).
3. **No hand-typed list.** The permitted sets are computed from the registered product manifests at call time; no literal package name appears in the new module (UT-A7C-01 proves that the sets move with the registry).
4. **Fail closed on an empty source.** In a guard posture an empty registry yields an empty allowlist (permits nothing), never `None`.
5. **Typed rosters are unaffected.** Only serialized `import_path` / `factory_path` values are checked; a typed `AgentBinding.mount(...)` never reaches an import.
6. **No new `Any`, `object`, `getattr` or `type: ignore`**; nothing under `intergrax/contracts/` is edited (OD-AF-11 ratchet).

## 4. Part A — the derivation module (firm, new)

New `intergrax/applications/_shared/import_policy.py` (parent exists; classified `COMPOSITION_CONFIGURATION` by the longest-prefix rule `intergrax/applications/`, F-37).

```python
@dataclass(frozen=True)
class ImportPolicy:
    agent_modules: ImportPrefixAllowlist | None      # for resolve_agent_type
    factory_modules: ImportPrefixAllowlist | None    # for load_callable
    application_packages: ImportPrefixAllowlist | None  # for _load_manifest_for_package
    fail_closed: bool

UNRESTRICTED: ImportPolicy   # (None, None, None, False): today's behaviour

def registered_agent_prefixes() -> ImportPrefixAllowlist
def registered_factory_prefixes() -> ImportPrefixAllowlist
def registered_application_prefixes() -> ImportPrefixAllowlist
def guarded_import_policy() -> ImportPolicy                       # fail_closed=True, three sets from the registry
def import_policy_for_environment(env: ApplicationEnvironmentProfile) -> ImportPolicy
def import_policy_for_manifest(manifest: ApplicationManifest) -> ImportPolicy
```

- `registered_agent_prefixes()`: for every `(product_id, manifest)` of `iter_product_manifests()` and every binding of `manifest.agents`, the top-level package of `binding.import_path` (typed bindings carry the qualified name, `manifest.py:178`). `registered_factory_prefixes()`: the same over `binding.factory_path`, which `AgentBinding` fills from a typed factory (three application packages at `881428fb2`, F-31). `registered_application_prefixes()`: `f"{product_id}_application"` for each registered product id that is a valid identifier (convention evidence: the glob `*_application` of `migration_wiring.py:56` and the product ids of the catalog, F-32; Q-A7C-05).
- `import_policy_for_environment(env)`: `guarded_import_policy()` when `wiring_fails_closed(env)`, else `UNRESTRICTED`. `import_policy_for_manifest(manifest)`: the same over `manifest.resolved_environment()`.
- `guarded_import_policy()` is also what the two scripts use: they run no composition and hold no `env`; a gate is not a LAB composition (Q-A7C-03 asks whether the orchestrator wants the literal `fail_closed=True` instead).
- Dependencies: `import_allowlist`, `product_manifest_registry`, `wiring_posture`, the manifest contract. No import of top-level `applications/` (Tier-0 boundary, `AGENTS.md`).
- The registry is read on every call (no cache): the registry is mutable by `register_product_manifest`, and the cost is a loop over four manifests (R-4).

## 5. Part B — the agent-import callers (b1)

- **`agent_resolution.py` (firm).** `resolve_agent_type_from_binding(binding, *, allowed_prefixes: ImportPrefixAllowlist | None = None, fail_closed: bool = False)` forwards both to `resolve_agent_type`; `resolve_agent_contract_from_binding` takes and forwards the same two. Defaults keep today's behaviour for any caller that is not threaded (LAB). The loader `resolve_agent_type` is unchanged.
- **`wiring.py` (firm).** A private helper `_import_policy_for_context(ctx)` returns `import_policy_for_environment(ctx.environment)` when `ctx.environment` is set, else `import_policy_for_manifest(ctx.manifest)`. `build_agent_from_binding` computes the policy once and passes it to `resolve_builder` (new keyword-only `import_policy: ImportPolicy = UNRESTRICTED`), to `load_callable(binding.factory_path, allowed_prefixes=policy.factory_modules, fail_closed=policy.fail_closed)` and to `resolve_agent_type_from_binding(binding, allowed_prefixes=policy.agent_modules, fail_closed=policy.fail_closed)`. `_validate_factory_result` gains the same keyword; `invoke_canonical_agent_factory` and `invoke_legacy_compatible_agent_factory` compute it from their `ctx`. `load_callable` and `load_agent_class` are not edited. A synthesized lab context (`wiring.py:593-601`) stays unrestricted by construction.
- **`environment_conformance.py` (firm).** `validate_binding` passes `import_policy_for_environment(env)` to `resolve_agent_type_from_binding`.
- **`roster_agent_contract_authority.py` (firm).** `resolve_roster_agent_contract` gains keyword-only `allowed_prefixes` and `fail_closed` forwarded to `resolve_agent_contract_from_binding` (C-7); `materialize_manifest_contract_authority_lab_compat(manifest)` computes `import_policy_for_manifest(manifest)` and forwards it at `:137` (C-8). Its callers (`capability_graph_wiring.py:110`, `strict_product_manifest_ci_gates.py:43,72,89`, `harness_host_runtime.py:388`) need no edit; the harness caller runs only when `production_mode` is false (`harness_host_runtime.py:383`).
- **`package_wiring.py` (conditional, Q-A7C-02).** `collect_application_dependencies` passes `import_policy_for_environment(env)` into `resolve_roster_agent_contract`. Without this edit C-7 has no passer for the guard postures it can reach (C-9), and the new keywords on `resolve_roster_agent_contract` would be dead code — the failure mode of PT-WB's H-1. Default: included as conditional; it is the single widening of the census and needs the orchestrator's ruling before approval.
- **`environment_diff_wiring.py` (conditional, default no edit).** C-6 cannot import; CTL-A7C-03 pins that. Edited only if the orchestrator rules that every census caller must be threaded uniformly (Q-A7C-04).

## 5a. Revision-bound rosters and installed distribution agents (dry run I-02)

How a revision-bound roster reaches the seam (read at `881428fb2`): `build_application_registry(manifest, ctx, effective_roster=..., runtime_revision=..., factory_resolver=...)` resolves each entry's factory with the resolver, which refuses a package digest that is not in `runtime_revision.installed_agent_package_digests` (`runtime_agent_factory_resolver.py:85-95`), and calls `invoke_canonical_agent_factory(resolved_factory, ctx, binding)` (`wiring.py:410-411`). The binding is built by `_roster_entry_binding_overrides`: for a manifest-origin entry it is a copy of the manifest binding (it keeps the manifest's `import_path`); for an operator-added entry (no manifest base) it is `AgentBinding(contract_id=..., factory_path=... or builder_key=...)` with **no `import_path`** (`wiring.py:300-345`). `_validate_factory_result` resolves the class only when `agent_type` or `import_path` is set (`wiring.py:124-127`). Consequences: (1) an agent installed from a distribution is identified by digest and never reaches seam b1 through an importlib string, so the registry-derived set need not cover it and no second source is needed; (2) the seam is reached only for a manifest-origin binding with a serialized `import_path`, which in a guard-posture context must belong to a registered manifest; (3) the context handed to `build_application_registry` decides the posture: the projection bundle builds `ApplicationBuildContext.for_manifest(manifest)` (posture of the manifest, `registry_projection_input_bundle.py:253`) or the descriptor builds a context with a default LAB environment (`registry_projection_descriptor.py:80-95`, RES-5). The two AC3 files use LAB manifests and `for_manifest(manifest)` contexts (`test_production_registry_startup_ac3_final.py:195-230,421`; `test_ac3_phase5_production_lifecycle_e2e.py:288-300,534`), so by reading the derived policy is unrestricted there; the `product_defaults` environment those tests create goes to the launcher and governance wiring, not to the build context. CTL-A7C-05 and CTL-A7C-06 pin both shapes. Not confirmed by a run.

Source-text pins in the V2 files (dry run; design constraints, not write-set changes): `resolve_builder` must keep a signature without a parenthesis in the new keyword default (the regex `def resolve_builder\([^)]*\)` in `test_ebh_2d_c_r2_canonical_agent_factory_contract.py:66-69`), hence `= UNRESTRICTED`; the `if binding.factory_path is not None` block must end at the lazy `agent_resolution` import and contain only the legacy invoker (`:136-145`), so the policy is computed before that block; `agent_resolution.py` must not contain the text `agent_type()` (`test_ebh_2e_r4_r1_structural_replaceability_and_metadata_resolution.py:86-88`).

## 6. Part C — the migration-gate callers (b2)

- `scripts/maintenance/check_application_migrations.py:24`: add `import shipped_product_manifest_catalog  # noqa: E402, F401` after the `sys.path` setup (the gate script at `scripts/gates/check_application_production_gates.py:20` already does; without the registry every package would be refused and the gate would go falsely red), then `policy = guarded_import_policy()` and `check_application_migrations(APPLICATIONS_ROOT, allowed_prefixes=policy.application_packages, fail_closed=policy.fail_closed)`.
- `scripts/gates/check_application_production_gates.py:229-232`: the same call with the same policy; the module-level catalog import stays.
- Behaviour change to record: an application package that has migration JSON but is not a registered product would now be reported as "migrations present but manifest could not be loaded" (`migration_wiring.py:413-414`). None exists at `T` (F-34). Whether that is wanted is Q-A7C-05; whether the two script edits are a gate-meaning change is Q-A7C-06.

## 7. Write set

Firm (10): product 4 edited + 1 new, scripts 2, tests 3 new.

| Path | Kind | Class | Verified at `T` |
|---|---|---|---|
| `intergrax/applications/_shared/import_policy.py` | new | firm | NEW_PARENT_EXISTS |
| `intergrax/applications/_shared/agent_resolution.py` | file | firm | EXISTS (PT-WB firm path) |
| `intergrax/applications/_shared/wiring.py` | file | firm | EXISTS (PT-WB firm path) |
| `intergrax/applications/_shared/environment_conformance.py` | file | firm | EXISTS (PT-WB cond path, not edited by PT-WB) |
| `intergrax/applications/_shared/roster_agent_contract_authority.py` | file | firm | EXISTS (PT-WB cond path, not edited by PT-WB) |
| `scripts/maintenance/check_application_migrations.py` | file | firm | EXISTS |
| `scripts/gates/check_application_production_gates.py` | file | firm | EXISTS |
| `tests/unit/applications/test_a4007_caller_import_refusal.py` | new | firm | NEW_PARENT_EXISTS |
| `tests/unit/applications/test_a4007_import_policy_derivation.py` | new | firm | NEW_PARENT_EXISTS |
| `tests/unit/scripts/test_a4007_migration_gate_allowlist.py` | new | firm | NEW_PARENT_EXISTS |

Conditional (2): `intergrax/applications/_shared/package_wiring.py` (condition: Q-A7C-02 ruled in, which makes NT-A7C-08 firm) and `intergrax/applications/_shared/environment_diff_wiring.py` (condition: Q-A7C-04 ruled for uniform threading; a PT-WB cond path, not edited by PT-WB).

Counts: firm product paths under `intergrax/`, `agents/`, `applications/` = 5 (1 new, 4 edited); firm scripts = 2 (not product prefixes, `iiac_common.py:22`); firm tests = 3; **firm total 10, conditional 2**. Sensitivity PRODUCT.

Forbidden (not edited): `wiring_posture.py` and every PT-FC2 path; `import_allowlist.py`, `migration_wiring.py`, `mcp_import_guard.py`, `scoreboard.py` and every other PT-WB path not listed above; `product_manifest_registry.py`, the manifest contract and `applications/shipped_product_manifest_catalog.py` (read only); `intergrax/contracts/**`; `tests/unit/applications/test_wb_import_allowlist.py` and all existing tests (regression only); `.github/`, `pyproject.toml`, `uv.lock`, docs.

## 8. Negative tests (red-first by assertion at the red-first base of section 10, step 1)

Pattern: a recording wrapper around `importlib.import_module` that delegates for real names and records the synthetic ones; each negative test asserts (a) the refusal and (b) that no import of the synthetic module was attempted. At the pre-change SHA the callers import the module (`ModuleNotFoundError` becomes the same `AgentImportError`), so the assertion on the recorded calls is what fails. The behavioural file imports only existing modules, so each test is red by assertion and not by a collection error; the derivation file imports the new module and is red by absence (not counted as negative). Environments are built with existing constructors (`ApplicationEnvironmentProfile.product_defaults()`; a LAB profile with `ExecutionMode.STRICT` for the union case), confirmed at step 0.

| ID | Subject | Expected after the change | Seam |
|---|---|---|---|
| NT-A7C-01 | `build_agent_from_binding`, PRODUCT context, serialized binding `zz_synth_t2.agent.Agent` | `AgentImportError`; `zz_synth_t2.agent` never imported | b1, C-4 |
| NT-A7C-02 | the same with a STRICT-only (LAB profile, STRICT mode) context | same (union posture) | b1, C-4 |
| NT-A7C-03 | `build_agent_from_binding`, PRODUCT, `factory_path="zz_synth_t2.factory.build"` | `ImportPathNotAllowedError`; no import | b1, C-3 |
| NT-A7C-04 | `build_agent_from_binding` with an empty `builders` map, PRODUCT, serialized binding outside the registered set | refused; no import | b1, C-1 |
| NT-A7C-05 | typed `factory=` callable returning an agent, binding with a serialized `import_path` outside the registered set, PRODUCT | refused in the result validation; no import | b1, C-2 |
| NT-A7C-06 | `EnvironmentSkillToolConsistencyCheck.validate_binding(binding, product_env)` with an unregistered serialized `import_path` | `AgentImportError`; no import | b1, C-5 |
| NT-A7C-07 | `materialize_manifest_contract_authority_lab_compat` over a PRODUCT manifest holding an unregistered serialized binding | `AgentImportError`; no import | b1, C-8 |
| NT-A7C-08 (cond) | `collect_application_dependencies(manifest, product_balanced_env)` with an unregistered serialized binding | refused; no import | b1, C-9 / C-7 |
| NT-A7C-09 | maintenance gate `main()` with `APPLICATIONS_ROOT` pointed at a temporary tree holding `zz_synth_app_application/migrations/<valid>.json` and a `manifest.py` with a side effect | the side effect did not run; exit 1 with "migrations present but manifest could not be loaded" | b2, C-10 |
| NT-A7C-10 | `scripts.gates.check_application_production_gates.check_application_migrations()` with the same temporary root | same violation text; no import | b2, C-10 |

Controls (must pass before and after): CTL-A7C-01 LAB context loads serialized `echo.echo_agent.EchoAgent` (unrestricted as today); CTL-A7C-02 PRODUCT context resolves a serialized path of a registered package (`legal.legal_agent.LegalAgent`; the catalog is registered by `tests/unit/applications/conftest.py:28`); CTL-A7C-03 `diff_roster` over typed, serialized and reference bindings calls `import_module` zero times (the proof behind C-6); CTL-A7C-04 both script entry points return clean over the real tree; CTL-A7C-05 `invoke_canonical_agent_factory(resolved_factory, product_ctx, binding)` for an operator-added binding (no `import_path`, `factory_path` only: the shape `_roster_entry_binding_overrides` builds without a manifest base) under a PRODUCT context returns the agent and calls `import_module` zero times (installed distribution agents are not refused); CTL-A7C-06 a LAB-profile manifest with `ApplicationBuildContext.for_manifest(manifest)` and a serialized binding of an unregistered package (`echo.echo_agent.EchoAgent`) through `invoke_canonical_agent_factory` stays unrestricted (the shape of the two AC3 files).

Unit tests of the new module (derivation file, 5): UT-A7C-01 the three sets equal an independent recomputation from the registered manifests, grow when a synthetic manifest is registered and shrink when it is removed; UT-A7C-02 an empty registry in a guard posture yields empty allowlists and `fail_closed=True`; UT-A7C-03 `import_policy_for_environment` returns `UNRESTRICTED` for a non-guard profile (and `import_policy_for_manifest` for a LAB-profile manifest) and the guarded policy for PRODUCT and for STRICT-only, and follows `wiring_fails_closed` when that name is patched (proves delegation to the one helper); UT-A7C-04 the new keywords of `resolve_agent_type_from_binding`, `resolve_agent_contract_from_binding` and `resolve_roster_agent_contract` forward to the loader and refuse; UT-A7C-05 every registered product id maps to an existing `<product_id>_application` directory (guards the convention of Q-A7C-05).

Counts: negative tests 10 (9 firm, 1 conditional), controls 6, unit tests 5, 21 tests in all.

## 9. Overlaps, lane, serialization

- **Overlap computation.** Every path of the write set was compared with every `scope` entry (globs expanded) of all 22 tickets of `rw_plan.v1.json` (v15) by a script (F-38). Shared paths: only PT-WB (merged into the train): `wiring.py` and `agent_resolution.py` (firm there), `environment_conformance.py`, `environment_diff_wiring.py` and `roster_agent_contract_authority.py` (its three conditional Part E callers, unedited by PT-WB). PT-FC2 lists `tests/unit/applications/test_package_wiring.py` as a conditional test; this ticket does not edit it (regression only), and `package_wiring.py` itself is in no other ticket. No other ticket lists a write-set path. The sibling drafts present in `w3-drafts/` (deps-batch and the seven `rw-*` directories) were searched for the same path names: no hit (the one textual match in deps-batch is `check_environment_conformance.py`, a different file).
- **Earliest lane.** Hard precedence: PT-FC2 (lane 2) -> this ticket, because it imports `wiring_posture.wiring_fails_closed`. Dependency-earliest lane is therefore **lane 3**. RW-PLAN v15 lane 3 holds five tickets (`T-AF2-RW-RUNTIME`, `PT-WD`, `PT-FC1`, `PT-FC2-E`, `T-AF2-RW-RECOVERY`), which is the cap of the start rule (at most 5 in flight), so the capacity-placed lane is **lane 4** (`PT-FC3`, `T-AF2-RW-SUPPLY`, `T-AF2-RW-GATES`, `PT-WE`; `PT-WE` is conditional), which has one free slot. It may start earlier if a lane-3 slot frees and PT-FC2 has merged (start rule: predecessors merged and a slot free).
- **Serialization reasons.** (1) after PT-FC2: import of `wiring_posture.py`. (2) after PT-WA and PT-WB: plan G steps 1-2, and PT-WB owns the loaders and two of the callers (shared path, not a conflict; PT-WB is merged). (3) No file overlap with any lane-3, lane-4 or later ticket (item above). (4) `T-AF2-RW-SUPPLY` is "serialized with PT-FC2/PT-WB on applications/_shared wiring" (plan Z); its `_shared` paths are disjoint from this write set, so no order between the two is needed. (5) `T-AF2-RW-TYP` waits for the last ticket editing `intergrax/contracts/**`; this ticket edits none.
- **RW-PLAN consequence.** RW-PLAN v15 had no block for this node and listed A4-007 as owned by PT-WB (T02 "Closes" cell, which contradicts the PT-WB checkpoint). RW-PLAN v18 already carries the block `T-AF2-A4007-CALLERS`, lane 4, serialization after PT-FC2 and PT-WB, with a scope equal to section 7 path by path (dry run, baseline facts; F-43). The ownership move of A4-007 b1/b2 is the orchestrator's (`FOLLOW_UPS.md` 2026-11-17 correction). This plan does not edit RW-PLAN (Q-A7C-07).

## 10. Order of work

0. Preflight: restate read scope, edit scope and tests (token-budget rule). Confirm that PT-FC2 is merged (**start condition**: it is not on `origin` at `881428fb2`) and that `intergrax/applications/_shared/wiring_posture.py` defines `wiring_fails_closed(env)` with `env.application_profile is PRODUCT or env.execution_mode is STRICT`; if the signature or meaning differs, **stop and ask**. Re-run the caller census of section 2 by `git grep` at the pre-change head; any new caller outside the write set is a stop. Static census of existing tests whose **manifest or build context** is PRODUCT-profile or carries a guard-posture `environment`, and whose roster has a **serialized** `import_path` or `factory_path` outside the registered packages (none expected: typed `mount` bindings do not import; the census keys on the manifest and the context, not on a `product_defaults` environment used elsewhere in the test, which is what made the two AC3 files look red in the dry run); a hit in a non-test file or in more than 3 test files is a stop (an amendment, not a silent edit).
1a. Mechanical lint commit first (dry run I-05): `ruff check --fix` for the two I001 findings (`environment_conformance.py`, and `package_wiring.py` if included) and `ruff format` for the files that would reformat (`environment_conformance.py`, `roster_agent_contract_authority.py`, and, if included, `package_wiring.py` and `environment_diff_wiring.py`), one commit, no behaviour change, so the later diff shows only the threading.
1. Red-first: write the three test files; at the red-first base, which is the train head at the start of this ticket with PT-WB and PT-FC2 merged (a detached worktree under `D:/Projekty/_wt/<repo>/<slug>`, one pytest process at a time under `D:/Projekty/_wt/_locks/pytest.lock`), record that NT-A7C-01..07 (and 08 if included), 09 and 10 fail and that the controls pass. The base cannot be the audit baseline `a7f023615`: the tests need PT-WB code (`ImportPathNotAllowedError`, the allowlist parameters), which does not exist there. The b1/b2 defect itself is unchanged from `a7f023615`: no caller passes an allowlist at either SHA (section 3 of the delta audit; the loaders gained the parameters, the callers did not change). The implementer records the SHA of that base in the ticket checkpoint when the ticket starts. Keep the log under `.tmp/session/arch-freeze/T-IGX-051-I/red-first/` (the path PT-Q02 re-runs from).
2. Part A, with the derivation tests.
3. Part B, then the behavioural negatives.
4. Part C, then the script negatives.
5. V1..V5 (section 11), `ruff` and `pyright` on the changed files, then V6.
6. PR body lists the red-first log, the exit codes and the residuals RES-1..RES-5; no AI-tool attribution; the commit trailer is `Ref: governance/tickets/T-IGX-051-I.md`.

## 11. Regression scope and test commands

- V1 new tests (one command): `uv run pytest tests/unit/applications/test_a4007_caller_import_refusal.py tests/unit/applications/test_a4007_import_policy_derivation.py tests/unit/scripts/test_a4007_migration_gate_allowlist.py -p no:cacheprovider`.
- V2 regression (one command), 25 existing files (199 test definitions by definition count, parametrization not expanded): `tests/unit/applications/contracts/test_agent_resolution.py`, `test_wb_import_allowlist.py`, `test_migration_wiring.py`, `test_environment_diff_wiring.py`, `test_package_wiring.py`, `test_agent_binding_conformance.py`, `test_agent_certification_gate.py`, `test_agent_runtime_governance_wiring.py`, `test_application_capability_projection.py`, `test_capability_graph_deploy_gate.py`, `test_ac5_phase3_factory_e2e.py`, `test_production_factory_invocation_ac5.py`, `test_production_registry_authority_guards_ac3.py`, `test_prod_q_r1_r7_q_p2_c_package_closure_authority.py`, `governed_contractor/test_prod_q_r1_pis_02_provider_invocation_lifecycle_binding.py` (all under `tests/unit/applications/`), `tests/unit/architecture/test_ebh_2d_c_r2_canonical_agent_factory_contract.py`, `test_ebh_2d_c_r3_tier2_agent_factory_result_contract.py`, `test_ebh_2e_ar1_a_r1_r1_install_authority_closure.py`, `test_ebh_2e_ar1_a_r1_r1_r1_install_to_rollback_contract_authority.py`, `test_ebh_2e_ar1_a_r1_revision_bound_agent_contract_authority.py`, `test_ebh_2e_r4_r1_r1_canonical_agent_contract_ownership.py`, `test_ebh_2e_r4_r1_structural_replaceability_and_metadata_resolution.py`, `tests/unit/runtime/architecture/test_platform_execution_unification_u3_agent_plugin_execution_closure.py`, and, added by revision 2 (dry run I-02; regression only, not edited), `tests/unit/applications/test_production_registry_startup_ac3_final.py` (11) and `tests/unit/applications/test_ac3_phase5_production_lifecycle_e2e.py` (8): they drive the revision-bound production path with serialized `echo` bindings and, by reading, stay green because their manifests are `ApplicationManifest.lab` and their build contexts carry no environment (section 5a). If either is red at step 0, **stop and ask**: an expected-red alternative would need an owner and a ruling, and none is proposed. The existing `tests/unit/scripts/test_check_application_migrations.py` is not in the list: it runs `scripts/check_application_migrations.py`, a path that does not exist at `T` (F-35; the real script is under `scripts/maintenance/`), so it cannot serve as a regression oracle for this ticket; it is marked `gate` and `no_ci`, and this ticket does not edit it. Node-level expansion and a definition search of each node is step 0 work (not done by this draft). The qualification file `tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_package_closure_regression.py` mentions the same symbols; PROD-Q node runs need a separate authorisation (freeze C10/C15) and are not run.
- V3 `uv run python scripts/maintenance/check_application_migrations.py` (exit 0 expected, exit code to the PR body). The full `check_application_production_gates` script is not part of the contract (large); its migrations function is covered by NT-A7C-10 and CTL-A7C-04.
- V4 step-0 and final census by `git grep` (no pytest), repeated by the auditor: no product caller of the three wrappers or of `load_callable` or `check_application_migrations` outside the write set passes less than the policy; the rows X-1 and X-2 are still the only unthreaded ones.
- V5 `uv run ruff check` and `uv run pyright` on the changed Python files.
- V6 `uv run python -m scripts.governance.igx_gov ticket` before push.
- Ratchet (OD-AF-11): this ticket does not edit `intergrax/contracts/`; if an amendment does, the wave-packet contract check (`rw-plan/work/typ05/measure_typ05.py` plus `typ_measure.py`) must report nothing new.
- No broad suite, no diagnostic replay, no Docker, no vendor integration, no production configuration.

## 12. ACT-N11 and the IIAC consequence

Five (seven with the conditional paths) files start with `intergrax/`, which is a product path prefix (`scripts/governance/iiac_common.py:22`); every merge of them breaks the IIAC activation until the single wave re-pin, and a revert does not restore it (PT-WB ticket text, F-08). The two scripts are not product-prefix paths. The dry run found the only red at the train head for the planned product edits to be KR-08 (15 ACT-N11 nodes, the registered set, no new class by reading); the new file does not affect the IIAC matrix completeness test. `check_application_production_gates.py` is the entrypoint of the fast-mandatory enterprise-audit check `ENT-A-PRD-01` (green at the head, unchanged by the planned edit): the wave packet should say so (Q-A7C-06). This ticket proposes no re-pin. EBH-7/B6-F10 classification: the new file and the four edited files fall under the rule `intergrax/applications/` (`COMPOSITION_CONFIGURATION`, F-37); rules for `scripts/maintenance/` and `scripts/gates/` exist (`_ebh7_f10_canonical_inputs.py@T:417,441`), their classes were not read (UNVERIFIED). The new tests are outside the identity allowlist and must be in place before the final pin.

## 13. What closes what

Delivers (if implemented and independently audited): the callers of the census pass the permitted set and `fail_closed` in the guard postures, with negative tests that fail before the change. It does not claim A4-007 closed: closure needs the exact-SHA wave audit. Not closed here (residuals): RES-1 `load_agent_class` (no caller; X-1); RES-2 the compat roster resolver for the reference-roster staging helper (X-2); RES-3 the `{package}.contract` second import (X-3, derived from a checked class); RES-4 PRODUCT hosts that load a serialized roster from a source that is not a registered manifest are now refused (intended; see risk R-1); RES-5 the rehydration build context `registry_projection_descriptor.py:80-95` carries a default LAB environment, so the context-derived posture is unrestricted there (section 5a, Q-A7C-09); the other importlib loaders outside A4-007 (`context_engine_resolver.py` and five more) stay with PT-G01 as in PT-WB Q-WB-07.

## 14. Risks

| ID | Risk | Mitigation |
|---|---|---|
| R-1 | A real PRODUCT or STRICT host loads a serialized roster whose packages are not in a registered manifest and is now refused at composition | typed `mount` rosters are unaffected; step-0 static census; Q-A7C-01 settles the source; LAB unchanged; the refusal is the typed error naming the path |
| R-2 | The registry is empty when a gate script runs (catalog not imported), so everything is refused | the maintenance script imports the catalog (Part C); UT-A7C-02 and NT-A7C-09 pin both directions |
| R-3 | `wiring_posture.py` differs from the draft (name, signature, meaning) at the time of implementation | step 0 stop; no local predicate is ever defined here |
| R-4 | Per-call derivation cost in `build_agent_from_binding` | loop over the registered manifests only; measured at step 0 if a test suite slows; no cache because the registry is mutable |
| R-5 | The conventional application package name `<product_id>_application` drifts from the directory name | UT-A7C-05; Q-A7C-05 |
| R-6 | The widening to `package_wiring.py` is refused and the keywords on `resolve_roster_agent_contract` become dead code | if Q-A7C-02 is ruled out, the keywords are not added and C-7 stays an explicit residual |
| R-7 | IIAC activation broken by the product merges | known train-wide consequence (section 12) |

## 15. Rollback

Revert the product commit(s) on `arch-freeze/rw-train`; the loaders and `import_allowlist.py` (PT-WB) and `wiring_posture.py` (PT-FC2) stay. A revert does not restore the IIAC activation (section 12). The new tests are removed with the same revert.

## 16. Open questions with draft defaults

| ID | To | Question | Draft default |
|---|---|---|---|
| Q-A7C-01 | orchestrator (Operator if it widens authority) | Which registered source supplies the Tier-2 agent set: the product manifest registry, the fleet inventory `build/agent_fleet_inventory.json` (a build artefact, used by the scoreboard), the repository `agents/` tree (the only derivation that exists, in test code, `test_wb_import_allowlist.py:33-37`), or the runtime revision package digests | the product manifest registry (already named in PT-WB plan 4.E; present at composition time; independent of the string being loaded when the manifest is registered code). **No second source is added**: revision data names package digests and `distribution_package_id` (`agent_distribution/roster.py:112-130`), not importable module names, so it cannot feed a module-prefix allowlist, and installed agents do not need one (section 5a) |
| Q-A7C-02 | orchestrator | Add `package_wiring.py` (the only env-holding caller of the compat roster resolver, reachable in PRODUCT-BALANCED hosts) as a conditional path although it is not a census caller | yes, conditional; otherwise C-7 stays a recorded residual and `resolve_roster_agent_contract` is not edited |
| Q-A7C-03 | orchestrator | The gate scripts hold no `env`: `guarded_import_policy()` (`fail_closed=True`, sets from the registry) or a posture computed through `wiring_fails_closed` from a registered strict manifest | `guarded_import_policy()`; a gate is not a LAB composition, and the brief's "through `wiring_fails_closed`" applies where an `env` exists |
| Q-A7C-04 | orchestrator | `environment_diff_wiring.py:108` cannot import (C-6): edit for uniformity or leave with the control test | leave, with CTL-A7C-03 |
| Q-A7C-05 | Operator or orchestrator | b2 permitted set = `<product_id>_application` of registered products: unregistered application packages with migrations would fail the gate; alternatives: add a package field to the registry (contract change, widening) or find the defining module by reflection | the naming convention, guarded by UT-A7C-05; no unregistered package has migrations at `T` |
| Q-A7C-06 | Operator | (`check_application_production_gates.py` is the entrypoint of the fast-mandatory `ENT-A-PRD-01`.) Are the two script edits a gate-meaning change (Operator personal route in OD-AF-22) or an input-hardening of an unchanged check | not gate-meaning: the checks, messages and exit codes are unchanged and the result at `T` is identical (nothing is scanned); if ruled gate-meaning, Part C moves to the wave packet and Parts A and B stay |
| Q-A7C-07 | orchestrator | the A1 block and the lane entry exist in RW-PLAN v18; remaining: move the A4-007 b1/b2 ownership from PT-WB to this node and update the register text in `T-IGX-050-M` | as in `FOLLOW_UPS.md` 2026-11-17; not done here |
| Q-A7C-09 | orchestrator | the rehydration build context (`registry_projection_descriptor.py:80-95`) builds a default LAB environment, so the context-derived policy is unrestricted for rehydrated revision-bound projections: add that file (and its tests) as a conditional path, or accept RES-5 | accepted as RES-5: the file is not in this write set, and a guard policy there needs the real posture of the host, which the descriptor does not carry |
| Q-A7C-08 | orchestrator | Should the residual rows RES-1 (`load_agent_class`) and RES-2 (reference-roster compat) be given a node or accepted as residuals | accepted as recorded residuals (no product caller, no env) |

## 17. Authorization requests

1. Wave-packet approval of this plan's digest under OD-AF-22 (Operator), then the delegate approval block and the delegation record; the drafter writes neither.
2. Rulings on Q-A7C-01..08 before the digest is approved; any change of the plan text voids the digest and needs a recompute.
3. No PROD-Q node runs, no Docker, no vendor integration, no production configuration (C10/C15 freeze).
