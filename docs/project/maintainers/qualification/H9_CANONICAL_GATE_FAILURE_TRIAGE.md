# H9 — Canonical Gate Failure Triage

**Task:** `HARDENING_9_CANONICAL_GATE_FAILURE_TRIAGE`  
**Gate execution SHA:** `b176598a0bb4d127059ff4d16cac4bfd4c8cecb7` (at run time `HEAD == origin/development`; working tree clean).  
**Artifact commit SHA:** record via `git rev-parse HEAD` at commit time (local `development` may advance during long gate).  
**Gate host:** Windows 10, Python 3.12.11, pytest 8.4.2, uv 0.8.15  
**Sync:** `uv sync --extra dev --extra dev-ci --extra dev-unit-cert --frozen` (pyarrow RECORD warning; hardlink fallback)

## Canonical H9 gate (repo command)

```bash
uv run pytest tests/unit -m "gate and not no_ci" -q --tb=line -p no:xdist
```

| Metric | Prior cert (stale) | Current HEAD |
|--------|-------------------:|-------------:|
| Passed | 9733 | **9817** |
| Failed | 418 | **403** |
| Errors | 35 | **35** |
| Skipped | 4 | **4** |
| Deselected | 24289 | **24370** |
| Duration | — | **3390 s (~56.5 min)** |

**Collection:** `34629` collected, `0` collection errors.

**Repository quality gate:** `tests/unit/runtime/architecture/test_repository_quality_gate.py` — **2 passed**.

## Mandatory revalidation

| Check | Result |
|-------|--------|
| `test_npsc5f_final_mandatory_regression_matrix_passes` | **FAIL** — embedded matrix: 4 drift sentinels (see RC-01) |
| `test_hardening_6_execution_authority_gate.py` | **PASS** (4/4 in dedicated run) |
| H3 layer boundary | PASS |
| Execution identity single authority | PASS |
| H5 plugin architecture | PASS |
| GR-3 canonical inner enforcement | PASS |
| GR-5-R2-R2 progress boundary | PASS |
| Root admission (`test_root_execution_authority_admission.py`) | PASS |
| Evidence persistence boundary | PASS |
| UE-8B1 budget architecture gate | PASS |

## Root-cause family matrix

| ID | Category | Count (fail+err) | H9 blocker | Suggested task |
|----|----------|------------------:|:----------:|----------------|
| RC-01 | POST_H9 debt / re-freeze gap | 4 (+5 gate echo) | **YES** | `HARDENING_9_NPSC5F_OBS_ASOF_DRIFT_REFREEZE` |
| RC-02 | STALE_FIXTURE (execution identity spine) | ~130 | **YES** | `HARDENING_9_EXECUTION_IDENTITY_FIXTURE_MIGRATION` |
| RC-03 | STALE_FIXTURE (`Cfg.production_mode`) | ~70 | **YES** | `HARDENING_9_TOOL_INVOKER_TEST_CFG_PARITY` |
| RC-04 | TEST_DRIFT (reference production lifecycle) | 44 | NO | `HARDENING_9_REFERENCE_PRODUCTION_TEST_HARNESS` |
| RC-05 | STALE_FIXTURE (profile durability pinning) | 23 | NO | `HARDENING_9_EFFECTIVE_PROFILE_TEST_STORE` |
| RC-06 | STALE_FIXTURE (Task / RuntimeEvent / TaskResult DTO) | ~30 | NO | `HARDENING_9_APPLICATION_DTO_FIXTURE_REFRESH` |
| RC-07 | STALE_GUARD (public docs contracts) | 25 | NO | `HARDENING_9_PUBLIC_DOCS_GUARD_REFRESH` |
| RC-08 | OPTIONAL_INTEGRATION_ENVIRONMENT | 35E + 11F | NO | CI/local extras: `llm-langchain-ollama`, `llm-ollama` |
| RC-09 | PRE_EXISTING_BASELINE (testing_support→tests) | 1 | **YES** | `HARDENING_9_TESTING_SUPPORT_IMPORT_DECOUPLING` |
| RC-10 | PRODUCTION_DEFECT (syntax in Tier-3 test) | 1 | **YES** | `HARDENING_9_LKW_SENTRY_PROOF_SYNTAX_REPAIR` |
| RC-11 | STALE_FIXTURE / host wiring | ~40 | NO | `HARDENING_9_APPLICATIONS_HOST_COMPOSITION_FIXTURES` |
| RC-12 | STALE_GUARD (scaffold / ADR) | 6 | NO | `HARDENING_9_SCAFFOLD_GUARD_REFRESH` |
| RC-13 | ENVIRONMENT | 2 | NO | `ANTHROPIC_API_KEY` for Claude adapter tests |
| RC-14 | UNRELATED / misc assertions | ~15 | NO | Per-file narrow tasks |

**H9 independent blockers:** **6** (RC-01, RC-02, RC-03, RC-09, RC-10; RC-01 also invalidates “NPSC5F CLOSED” at this SHA).

## RC-01 — NPSC-5F mandatory matrix (proven)

**Root cause:** Protected drift sentinels flag post-baseline edits to:

- `intergrax/runtime/observability/historical_reconstruction.py`
- `intergrax/runtime/observability/reconstruction/execution_reconstruction.py`

**Introduction window:** `52b9dc41e` (`OBS-ASOF-REBASE-R1`) after baseline `743a38651`.

**Repro:**

```bash
uv run pytest tests/unit/testing_support/test_npsc5f_final_protected_drift.py -q --tb=line
```

**Remediation:** Qualified re-freeze / baseline update per EE-FINAL-02 process — not silent sentinel weakening.

## RC-02 — Execution identity fixture drift (proven)

Dominant signatures: `RuntimeRequest.__init__` missing `task_id`/`run_id`, `EmitContext.__init__` missing `attempt_id`/`execution_id`, `active execution identity required`.

**Subsystems:** nexus (~57), token_optimization (~43), kernel (~19), task (~18).

**Repro:**

```bash
uv run pytest tests/unit/runtime/nexus/ -k "RuntimeRequest" -q --tb=line --maxfail=1
```

**Correlation:** Canonical identity helpers landed in `b176598a0` (`intergrax/dev_support/execution_identity_scope.py`); gate tests not migrated.

## RC-03 — Tool invoker `production_mode` (proven)

**Root cause:** `RuntimeToolInvoker._require_agent_runtime_governance` reads `state.context.config.production_mode`; unit stubs use bare `Cfg` without attribute.

**Introduction:** `e4f2113f1` (`fix(execution): close agent plugin governance gaps`).

**Repro:**

```bash
uv run pytest "tests/unit/runtime/tools/test_fresh_side_effect_authorization.py::test_security_matrix_platform_se_fail_closed_1[A-read_only_without_policy_allowed]" -q --tb=short
```

## RC-09 — testing_support imports tests (proven)

Violations (stable): `testing_support/npsc5f_r1_legacy_targets.py`, several `testing_support/runtime/*_harness.py` → `tests.unit.*`.

**Repro:**

```bash
uv run pytest tests/unit/agent_distribution/test_ac6_architecture_gates.py::test_testing_support_does_not_import_tests -q --tb=short
```

## RC-10 — Tier boundary syntax scan (proven)

`tests/unit/architecture/test_tier_dependency_boundaries.py` reports syntax error in `applications/local_workspace_application/tests/test_lkw_sentry_proof_endpoint.py:167`.

## Remediation order

1. RC-01 (unblocks NPSC-5F matrix + evidence certification narrative)  
2. RC-02 (largest fail reduction; unblocks nexus/kernel/token paths)  
3. RC-03 (side-effect + token optimization clusters)  
4. RC-10 (tier scan / product tree hygiene)  
5. RC-09 (harness boundary — parallelizable with RC-02)  
6. P2/P3 buckets (RC-04–08, RC-11–14)

## Expected failure reduction (estimate)

| Task | RC | Δ failures |
|------|-----|----------:|
| NPSC5F re-freeze | RC-01 | ~9 |
| Execution identity fixtures | RC-02 | ~130 |
| Cfg production_mode parity | RC-03 | ~70 |
| Reference production harness | RC-04 | ~44 |
| Profile store fixtures | RC-05 | ~23 |
| DTO fixtures | RC-06 | ~30 |
| Docs guards | RC-07 | ~25 |

## Authority revalidation (static + guards)

At triage SHA, dedicated architecture guards confirm **single** execution authority (H6 PASS), execution identity authority PASS, root admission PASS, GR-5 progress boundary PASS. Failures above are **not** alternate executors; they are fixture/test drift and evidence-plane baseline debt.

## Verdict

**PASS_TRIAGE_COMPLETE** — 438 gate fail/error nodes classified; **UNKNOWN H9 blockers = 0**.

**Recommended next task (P0):** `HARDENING_9_NPSC5F_OBS_ASOF_DRIFT_REFREEZE`

---

> **GitHub audit:** Wprowadzone zmiany oraz wynik triage muszą zostać zaudytowane na podstawie kodu znajdującego się aktualnie na GitHub. **Gate metrics bind to `b176598a0`**; po merge lokalnych commitów ahead-of-origin wykonaj krótki gate delta na `origin/development`.
