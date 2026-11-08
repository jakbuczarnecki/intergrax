## PLAN `T-IGX-031-A` — TESTS-ONLY batch: restore `test_sec_ent.py` (HIST03-F-006) and isolate the integration catalog for `test_sec_ent.py` and W5 b2/c/d (HIST03-F-016)

Campaign: Integrax Historical Certification Closure, W1-PLAN step 1 (accepted as a SEQUENCE only; it approves no ticket plan). Authority for the scope: DR-12 = B (tests now, product edits batched with the next IIAC re-activation). Planner != implementer != auditor != plan approver. This plan claims no PASS, no closure, no FRZ promotion and changes no FRZ status.

Base: `origin/development` `6734971cf11297e9ac9edb0b01b8bbe8bb5799f4` (planner worktree `plan-tests`, branch `hist-cert/plan-tests-batch`).

### 1. Summary of findings (read this first)

Two corrections to the worker record, both measured at HEAD `6734971cf` (section 3):

1. F-006 is real and still red at HEAD: `test_sec_ent.py` 7 failed / 6 passed (exit 1). The worker thesis "ineffective patch target" is TRUE as a static fact but is NOT the cause of any of the 7 failures: the two patch-target tests die earlier, at a bare `register_default_integrations()` call. Four independent defects produce the 7 reds (section 4).
2. F-016 is mis-attributed. The W5 b2/c/d files are VICTIMS of registry pollution, not polluters. In the minimal reproduction the polluter is `test_sec_ent.py` itself: its failed bootstrap leaves a half-filled global integration catalog. Run in one process, `test_sec_ent.py` then W5 b2/c/d gives exactly the 8 W5 failures seen on the pin (b2 x5, c x2, d x1). So F-006 and F-016 share one root cause and are fixed together.

No product-code edit is needed for the proposed fix. One product-level observation (two providers share the slug `ollama`) is recorded as a residual and is NOT fixed here (section 9).

### 2. Exact edit scope (phase 2 may change ONLY these)

| # | File | Change |
|---|---|---|
| 1 | `tests/unit/runtime/security/test_sec_ent.py` | fixes for 7 red tests + 1 new test + autouse catalog fixture (ledger, section 5) |
| 2 | `tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py` | autouse catalog-isolation fixture only; zero assertion change |
| 3 | `tests/unit/runtime/events/test_enterprise_scale_resilience_w5_c_event_export.py` | same |
| 4 | `tests/unit/runtime/events/test_enterprise_scale_resilience_w5_d_exporter_composition.py` | same |
| 5 | `tests/support/integration_catalog_isolation.py` (NEW) | one helper function used by the four autouse fixtures |

Forbidden, absolute: anything under `intergrax/`, `agents/`, `applications/` (IIAC ACT-N11: `PRODUCT_PATH_PREFIXES` in `scripts/governance/iiac_common.py`, reported at `iiac_activation_gate.py` ~L567-576); the six `MATERIAL_GOVERNANCE_PATHS` (`iiac_activation_gate.py` L40-49); `scripts/`; `conftest.py` files; pytest markers (`no_ci` stays on `test_sec_ent.py`); `pyproject.toml`.

Why item 5 is a new file and not four inline copies: one cleanup list, one place to audit. Fallback if the ACT pre-check (section 7, step 8) goes red because of the new file: inline the helper into each of the four files (no other change) and say so in the checkpoint. `scripts/` is NOT needed.

### 3. Evidence gathered by the planner (what was run, what was read)

Two pytest runs, both in the planner worktree, each under the heavy lock (mkdir acquire, `holder.txt`, released; no foreign lock touched). The venv did not exist; the Operator approved creating `.venv` in the worktree. Logs: `D:/Projekty/intergrax/.tmp/session/hist-cert-campaign/plan-tests/` (`run1_sec_ent.log/.exit`, `run2_sec_ent_plus_w5.log/.exit`).

| Run | Command (cwd = worktree) | Result at HEAD `6734971cf` |
|---|---|---|
| 1 | `uv run --extra dev-ci pytest tests/unit/runtime/security/test_sec_ent.py -q` | exit 1, 7 failed, 6 passed (83.8 s) |
| 2 | `uv run --extra dev-ci pytest <test_sec_ent.py> <w5_b2> <w5_c> <w5_d> -q` (that order) | exit 1, 15 failed, 18 passed (41.6 s); the 7 above + 8 W5 |

Run 1 failure classes (from the log, VERIFIED): `TypeError: Pre-built integration for category 'secrets_store' is _FakeSecretsStore, expected SecretsStoreIntegrationContract` (1), same message with `object` (1), `ValueError: Integration slug 'ollama' is already registered` raised from a bare `register_default_integrations()` (1), `ValueError: Integration slug 'redis' is already registered` from the next bare call (3), `RuntimeError: active execution identity required` (1).

Run 2: the W5 failures are `ValueError: Integration slug 'redis' is already registered` in `build_harness_host_runtime` / `wire_application_environment` -> `bootstrap_catalogs` -> `register_default_integrations` (11 `redis` hits in total, 1 `ollama`). They pass when run alone (worker's `iso_1..3.log`, pin `ba07eba`: 5 / 6 / 9 passed; NOT re-run at HEAD by the planner, budget).

Not run (budget: two pytest runs): any W5 file alone at HEAD; any proposed fix; the 27-path combined selection.

### 4. Root cause per file (verified vs thesis)

**Shared mechanism (verified by trace + static reading).** `register_default_integrations(preset="full", override=False)` (`intergrax/integrations/registry/bootstrap.py`) registers the core set, then the extended set. The extended set registers slug `ollama` twice: `model_serving_runtime/ollama` (via `bootstrap_m6_p4.py`) and `embedding_provider/ollama` (via `register_all.py`, called last from `bootstrap_extended.py:223`). With `override=False` the second registration raises `ValueError` ("already registered", `catalog.py:99/104`). The exception is raised AFTER the core slugs (including `redis`) were registered and BEFORE `_BOOTSTRAPPED = True`. Result: a half-filled global `_CATALOG` and `_BOOTSTRAPPED == False`. Every later bootstrap in the process starts with `override=not integration_catalog_snapshot()` (`intergrax/core/catalog_bootstrap.py:92-96`), which is `False` for a non-empty catalog, and collides on the first core slug, `redis`. The production path avoids this only because it starts from an empty catalog, where `override=True` makes the last `ollama` registration win. THESIS (not verified): with `override=True` the `model_serving_runtime` `ollama` row is silently replaced by the `embedding_provider` row; this is a product-level question (section 9, R1).

**`test_sec_ent.py`** (7 reds, 4 independent causes):

| Tests | Cause | Status |
|---|---|---|
| `test_resolve_encryptor_fails_closed_on_resolution_error`, `test_harness_envelope_encryptor_not_selected_by_resolver` | (a) bare `register_default_integrations()` fails on a clean catalog (first call: `ollama`; second and later: `redis`); (b) the `patch("intergrax.integrations.registry.factory.resolve_from_profile")` target is ineffective: `security_runtime_bridge.py:14` does `from ...factory import resolve_from_profile`, so the name the code looks up lives in `intergrax.applications._shared.security_runtime_bridge`; patching the defining module does not rebind it | (a) VERIFIED (run 1 traces, the failure is on the `register_default_integrations()` line, before the `with patch`); (b) VERIFIED by reading, effect not observed because (a) hides it |
| `test_resolve_encryptor_uses_valid_secrets_store`, `test_resolve_encryptor_fails_closed_on_conformance_error` | the pre-built-instance type guard (`intergrax/integrations/contracts/prebuilt_category_guard.py`, reached from `IntegrationProfile.instance_for_category`, called by `_secrets_store_configured`, `security_runtime_bridge.py:47`) requires a `SecretsStoreIntegrationContract` instance; the test passes a plain `_FakeSecretsStore` and a bare `object()`. The guard raises `TypeError` outside the `try` blocks of the bridge, so neither the success path nor the "does not satisfy SecretsStore contract" message is reachable | VERIFIED (run 1 messages) |
| `test_resolve_encryptor_uses_secrets_store_when_resolvable`, `test_register_security_hooks_wires_spine_subscriber` | bare `register_default_integrations()` fails on the polluted catalog (`redis`) | VERIFIED as the failure; that nothing else is wrong behind it is THESIS |
| `test_security_spine_counters_increment` | `build_platform_signal_event` calls `require_active_execution_identity()` and `require_active_execution_id()` (`spine_consolidation.py:187-188`); the test binds no identity, and passes non-canonical `task_id="t1"`, `run_id="r1"` which `validate_task_id` / `validate_run_id` and the `run_id` conflict check would reject next | identity failure VERIFIED; the follow-on validation is VERIFIED by reading only |

**W5 b2 / c / d** (8 reds when run after `test_sec_ent.py`): victims of the shared mechanism. They contain no registry code of their own; all go through `build_harness_host_runtime` / `wire_application_environment` -> `bootstrap_catalogs`. VERIFIED: they pass alone on the pin and fail after `test_sec_ent.py` at HEAD. THESIS: `test_sec_ent.py` is the only polluter in the worker's 27-path selection; other files that bootstrap the catalog or reset one flag without the other could pollute too (many tests call `clear_catalog()` / `reset_default_integrations_state()`). The W5 fixture below is therefore victim-side hardening: it makes the three files order-independent whoever the polluter is.

### 5. Assertion / patch ledger (every change; "kept" means byte-identical)

Authority for the semantics: the tests state the contract "RESTRICTED payload encryptor: no secrets_store declared -> None; declared and healthy -> `SecretsStorePayloadEncryptor`; declared but unresolvable or non-conforming -> `RestrictedPayloadEncryptorResolutionError` (fail closed)". All of it is kept.

`tests/unit/runtime/security/test_sec_ent.py`

| # | Test / object | Before | After | Why this does not weaken the gate |
|---|---|---|---|---|
| S1 | NEW autouse fixture `_clean_catalogs` | none | calls `reset_integration_tool_skill_catalogs()` before and after each test (also clears the new class-level store of S2) | adds isolation only; no assertion touched |
| S2 | NEW class `_PrebuiltFakeSecretsStore(SecretsStoreIntegrationContract)` | n/a | subclass with `stored: ClassVar[dict[str, str]]` and the same `get_secret` / `put_secret` / `delete_secret` bodies as `_FakeSecretsStore`; built with `SecretsStoreIntegrationContract.for_provider` semantics (`category_for_provider` builds `contract_cls(...)`, so a subclass works; mirrors `InfisicalSecretsStoreIntegration`) | test double only. `_FakeSecretsStore` is kept unchanged for the middleware tests that pass it straight to `SecretsStorePayloadEncryptor` |
| S3 | `test_resolve_encryptor_uses_valid_secrets_store` | `store = _FakeSecretsStore()` | `store = _PrebuiltFakeSecretsStore.for_provider(provider_id="sec-ent-fake")` | all assertions kept (isinstance `SecretsStorePayloadEncryptor`, `secret` removed, `__secret_ref`, `encryption_envelope == "secrets_store.v1"`, stored value). The input now satisfies the stricter pre-built guard; the encryption effect and the vaulted value are still asserted |
| S4 | `test_resolve_encryptor_fails_closed_on_conformance_error` | `_env_with_secrets_store(object())` | `_env_with_secrets_store(SecretsStoreIntegrationContract.for_provider(provider_id="sec-ent-nonconforming"))` | same negative probe, same expected exception type and same message `"does not satisfy SecretsStore contract"`: a category-contract object WITHOUT `get_secret`/`put_secret`/`delete_secret` must fail closed at the conformance gate. `object()` can no longer reach that gate because the guard rejects it earlier (S5 keeps that input covered) |
| S5 | NEW `test_prebuilt_non_contract_object_is_rejected_by_type_guard` | n/a | `_env_with_secrets_store(object())` -> `pytest.raises(TypeError, match="expected SecretsStoreIntegrationContract")` | pure addition; keeps the old `object()` input under test; documents that the guard is the earlier, stricter gate |
| S6 | `test_resolve_encryptor_fails_closed_on_resolution_error` | `register_default_integrations()`; `patch("intergrax.integrations.registry.factory.resolve_from_profile", ...)` | `_bootstrap_catalog()` (helper = `bootstrap_application_integration_catalog()` from `intergrax/applications/_shared/integration_wiring.py`, the sanctioned Tier-3 path whose docstring says to prefer it over bare `register_default_integrations()`); `patch("intergrax.applications._shared.security_runtime_bridge.resolve_from_profile", ...)` | patch now binds where the code looks the name up, so the injected `RuntimeError("secrets backend unavailable")` is actually raised. Kept: `pytest.raises(RestrictedPayloadEncryptorResolutionError)`, the message `"failed to resolve configured secrets_store"`, and `__cause__` is `RuntimeError`. The registry call stays (a slug string profile may be validated against the catalog), so the test is not reduced to a mock-only check |
| S7 | `test_harness_envelope_encryptor_not_selected_by_resolver` | same two lines as S6 | same two replacements as S6 | kept: `pytest.raises(RestrictedPayloadEncryptorResolutionError)`; resolution failure still must NOT fall back to an encryptor. Residual (not changed, not asked): the test asserts only the exception type |
| S8 | `test_resolve_encryptor_uses_secrets_store_when_resolvable` | bare `register_default_integrations()` | `_bootstrap_catalog()` | assertions kept (`encryptor is not None`, isinstance `SecretsStorePayloadEncryptor`). The real catalog path (harness_defense_stack -> `infisical` factory) is still exercised, nothing is stubbed |
| S9 | `test_register_security_hooks_wires_spine_subscriber` | bare `register_default_integrations()` | `_bootstrap_catalog()` | assertion kept (`counters.subscription_id`) |
| S10 | `test_security_spine_counters_increment` | no identity; `task_id="t1"`, `run_id="r1"` | `ids = runtime_event_test_identity()` (`testing_support.runtime_events`); `bind_active_execution_identity(run_id=ids["run_id"], attempt_id=ids["attempt_id"], execution_id=ids["execution_id"], task_id=ids["task_id"])` in `try` / `finally reset_active_execution_identity(token)`; the event is built with `task_id=ids["task_id"]`, `run_id=ids["run_id"]` | kept: the payload, `await bus.publish(event)` and `counters.defense_blocked == 1`. Only the execution-identity precondition that the production API now demands is supplied (same pattern as `tests/integration/autonomous_work/test_aw_7b_ephemeral_capability_execution.py:559`); the identity is reset in `finally` |
| S11 | helper `_bootstrap_catalog()` (module-level function) | n/a | one line calling `bootstrap_application_integration_catalog()` | single replacement point for the four bare calls |

W5 files (`..._w5_b2_...`, `..._w5_c_...`, `..._w5_d_...`): add `@pytest.fixture(autouse=True)` calling `reset_integration_tool_skill_catalogs()` before and after each test. No test body, import used by a test, parametrization, marker or assertion changes.

`tests/support/integration_catalog_isolation.py` (NEW): `reset_integration_tool_skill_catalogs()` calls, in this order, the existing public test helpers only: `clear_catalog`, `clear_tool_catalog`, `clear_skill_catalog` (integrations, tools, skills catalogs), `reset_default_integrations_state`, `reset_default_tools_bootstrap`, `reset_default_skills_for_tests`, `reset_tier0_catalog_bootstrap_for_tests`. It touches no private name. Convention: clean before and clean after, the same pattern as `tests/unit/applications/test_p6_integration_tool_wiring.py` (`_clean_state`) and `tests/unit/core/plugins/test_plug_02_integration_catalog_admission.py` (`_reset`).

Gate-weakening checklist (the auditor re-checks each line): no `skip` / `xfail` / marker added or removed; `no_ci` kept; no assertion deleted, loosened, widened to a set or turned into `in`; no `try/except` swallowing around an assertion; every ledger "kept" item byte-identical; no effect / ledger / replay check dropped (S3 still checks the vaulted value, S6/S7 still check fail-closed); negative probes kept (S4 same message, S5 adds one, S6/S7 now actually inject their fault); only added code is isolation and test doubles.

### 6. Roles

Implementer, independent auditor and plan approver are three different fresh sessions. The planner (this plan's author) is none of them. `Approved:` is written only by the Operator or by a commissioned `operator-delegate` session (README forms); never by the implementer. The auditor must: (1) re-run section 7 on the exact final SHA; (2) do the gate-weakening review against the checklist in section 5, hunk by hunk; (3) verify the negative probes (section 7, steps 6-7) themselves, restoring the broken patch target and showing the test goes red again; (4) check the diff guard (nothing outside the five files plus ticket/plan/checkpoint).

### 7. Test contract (phase 2)

Common rules: ONE pytest / ruff / uv process at a time; each under the heavy lock `D:/Projekty/intergrax/.tmp/session/hist-cert-campaign/locks/heavy.lock` (acquire with `mkdir`, write `holder.txt`, release with `rm holder.txt; rmdir`; never remove a lock you do not hold; on BUSY wait, do not break it); pytest's own exit code captured with `; echo EXIT=$?` into the log, not through a pipe; logs under `.tmp/session/t-igx-031-a/`; no `--basetemp` outside the repo; no `-p no:cacheprovider`; no `-n` / xdist.

Paths: `SE=tests/unit/runtime/security/test_sec_ent.py`, `B2=tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py`, `C=tests/unit/runtime/events/test_enterprise_scale_resilience_w5_c_event_export.py`, `D=tests/unit/runtime/events/test_enterprise_scale_resilience_w5_d_exporter_composition.py`.

| Step | Command | Expected before (HEAD `6734971cf`) | Expected after |
|---|---|---|---|
| 1 | `uv run --extra dev-ci pytest $SE -q` | MEASURED: 7 failed, 6 passed, exit 1 | 14 passed, exit 0 (13 existing + S5) |
| 2 | `uv run --extra dev-ci pytest $B2 $C $D -q` | not measured at HEAD; pin: 20 passed (5+6+9) | 20 passed, exit 0 |
| 3 | isolation proof A, ONE process: `uv run --extra dev-ci pytest $SE $B2 $C $D -q` | MEASURED: 15 failed, 18 passed, exit 1 | 34 passed, exit 0 |
| 4 | isolation proof B, ONE process, reversed order: `uv run --extra dev-ci pytest $D $C $B2 $SE -q` | not measured (expected: only `$SE` failures) | 34 passed, exit 0 (order independent) |
| 5 | single node, registration-order check: `uv run --extra dev-ci pytest "$SE::test_security_spine_counters_increment" -q` | n/a | 1 passed. If red with an unknown event kind, add inside S10 `if get_event_kind_entry(KIND_DEFENSE_BLOCKED) is None: register_security_payload_schemas()` (no idempotence assumption) and re-run steps 1, 3, 4 |
| 6 | NEGATIVE PROBE A (uncommitted scratch edit on the final tree, then restored): in the final `$SE` revert ONLY (i) the two patch targets in S6/S7 to `intergrax.integrations.registry.factory.resolve_from_profile`, (ii) S4 input to `object()`, (iii) S10 identity binding removed; keep the bootstrap helper and the fixture. `uv run --extra dev-ci pytest $SE -q` | n/a | expected red for exactly: S6, S7 (DID NOT RAISE or wrong behaviour, because the unpatched real resolution succeeds), S4 (`TypeError`), S10 (`RuntimeError`); all other tests green. Record the actual set; any other result is reported, not interpreted away. Restore, `git diff` must be empty afterwards |
| 7 | NEGATIVE PROBE B (uncommitted scratch edit, restored): remove the autouse catalog fixtures from `$SE`, `$B2`, `$C`, `$D` and revert S6-S9 to the bare `register_default_integrations()`; `uv run --extra dev-ci pytest $SE $B2 $C $D -q` | n/a | expected red, matching the HEAD measurement (15 failed, 18 passed, plus the S3/S4/S5 effects as applicable). Proves the fixture, not luck, makes step 3 green. Restore; `git diff` empty |
| 8 | ACT-N11 pre-check after the commit, separate holds, expected green: `uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force`; then `uv run --extra dev-ci pytest tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_evol_x_real_repo_fresh_clone.py tests/unit/governance/test_iiac_g0_inventory.py -q` | n/a | all green. If red: STOP, do not fix, do not touch the gate or IIAC records, report to MASTER (the tests-only premise or the new-file choice is wrong; try the inline fallback of section 2) |
| 9 | `uv run --extra dev-ci ruff check <5 files>` then `uv run --extra dev-ci ruff format --check <5 files>` | n/a | clean |
| 10 | diff guard: `git diff --name-only 6734971cf..HEAD` | n/a | only the five files plus `governance/tickets/T-IGX-031-A.md`, `governance/tickets/plans/PLAN_T-IGX-031-A.md`; nothing under `intergrax/`, `agents/`, `applications/`, `scripts/`; no CRLF in new content |
| 11 | OPTIONAL, needs a MASTER-granted heavy slot (about 8 minutes on the pin): the worker's 27-path selection in ONE process (`run1.sh` in `.tmp/session/hist-cert-campaign/hist03/`) | pin: 21 failed, 331 passed | no `already registered` error anywhere. The 6 `p2_c_http_execution_admission` failures (F-015, a separate P2-C corrective) are expected to remain and surface with their own error; they are out of scope. Also shows whether any later file relied on a populated catalog left behind by the W5 files |

`igx_gov ticket` wraps the broad regression `fast` tier; run it only with Operator budget under C10/C15.

### 8. Escape clause (anything needing product code leaves this batch)

If, while making steps 1-5 green, a failure remains whose only fix is a change under `intergrax/`, `agents/` or `applications/`, that item is OUT of this batch: the implementer stops, records the exact test, the error and the product line, does NOT edit product code, and the item moves to the product batch (W1-PLAN step 3, one IIAC re-activation). Specific candidates the planner could not rule out statically:

- S10: after the identity is supplied, `bus.publish` / the payload schema or `wire_security_spine_subscriber` could reject the event (not observable before the identity fix).
- S8 / S9: after bootstrap, `harness_defense_stack()` resolving the `infisical` factory, or `register_application_security_hooks`, could fail for a reason unrelated to the catalog.
- S3: whether the subclass instance satisfies `SecretsStore` as a `runtime_checkable` Protocol and the contract model accepts the extra methods and `ClassVar` (reading says yes; `InfisicalSecretsStoreIntegration` is the precedent).

A partial result is acceptable and honest: each remaining red test is listed with its cause in the checkpoint; no test is skipped or xfailed to get green.

### 9. Risks, rollback, residuals, not verified

Risks. (1) Clean-before/clean-after leaves the three catalogs EMPTY after these four files; a later file in the same process that silently relied on a populated catalog would now fail. Mitigation: step 11 (optional) and the repo convention above; the alternative (snapshot and restore) needs private catalog API and was rejected. (2) The new file under `tests/support/` could interact with the IIAC G0 inventory verifier; the matrix enumerates paths at a recorded `inventory_source_sha`, so a new file is not expected to matter, but this is NOT verified; step 8 covers it, with the inline fallback. (3) `test_sec_ent.py` is `no_ci` (excluded from GitHub CI, `pyproject.toml` ~L432, `unit-tests.yml` ~L287): even green, CI will not run it; removing the marker is a separate Operator decision and is NOT done here. (4) Other tests may have the same latent defect (bare `register_default_integrations()` and `object()`/plain fakes for pre-built secrets stores). A grep shows about 75 other bare call sites under `tests/`; none was run or edited by the planner.

Rollback. `git revert` of the implementation commit(s); no data, no product, no gate change. The ticket and plan files stay as the record.

Residuals for the Operator / product batch (not fixed here):
- R1 (product, W1-PLAN step 3): two providers register the slug `ollama` (`model_serving_runtime` and `embedding_provider`). `register_default_integrations(preset="full", override=False)` therefore cannot complete on an empty catalog; production works only through `override=not snapshot()`, where the last registration silently replaces the first. Needs an owner decision: distinct slugs, or a merged multi-category entry.
- R2: whether the other bare `register_default_integrations()` test callers are red or green in a clean process is unmeasured.
- R3: F-016's "run in separate processes" workaround in selection manifests can only be lifted after this batch is audited green; this batch does not edit any manifest.
- R4: `test_p0_safety_8_retry_redelivery_authorization.py` (the other file in F-006) already passed on the pin and is untouched.

Could not verify: every "after" column (nothing was run after a change, by design: the planner writes no test); the effect of `override=True` on the `ollama` rows; W5 b2/c/d alone at HEAD; the pin-versus-HEAD identity of the worker's 27-path result; the D-3 delegated signatures and anything about the ENT-AUDIT-X stream; whether one IIAC re-activation covers both streams (W1-PLAN step 2).

### 10. Out of scope

Everything under `intergrax/`, `agents/`, `applications/`, `scripts/`; the six IIAC material governance paths; `T-IGX-030-B`, `T-IGX-030-C`, the P2-C (F-015) corrective, BND-03, `T-IGX-021-B/C`; `conftest.py` and marker changes; removing `no_ci`; selection manifests; ENT-AUDIT-X artefacts; IIAC records and activation; roadmap / FRZ edits and promotions; PASS / closure claims; Docker, vendor and production configuration (C10/C15); push, merge, rebase, PR.
