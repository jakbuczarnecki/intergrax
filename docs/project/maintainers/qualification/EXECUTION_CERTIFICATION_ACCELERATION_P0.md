# Execution Certification Acceleration — P0 Inventory

**Status:** `INVENTORY QUALIFIED` (P0 — no runner implementation)

**Task:** Execution Certification Acceleration/P0

**Branch:** `development`

**Scope:** `tests/**`, `docs/**` only · production code unchanged

---

## Purpose

Evidence-backed inventory of existing Execution Engine certification suites, isolation hazards, and documentation gaps to prepare **R1 — Isolated Bounded Parallel Qualification Runner**.

**Principle (unchanged):** parallel execution may reduce wall-clock time; it must not reduce test scope, proof quality, process isolation, or deterministic PASS/FAIL semantics.

---

## Qualification runners found (OBSERVED)

| Mechanism | Location | Role |
| --- | --- | --- |
| **`_run_pytest` subprocess gate** | `tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py`, `test_npsc5e_r3_child_fanout_partial_recovery.py`, `test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py`, `test_npsc5e_r2_h2_q1_frozen_regression_closure.py`; reused from `test_npsc5e_final_recovery_plane_qualification_and_freeze.py` | Runs `uv run pytest <targets> -q --tb=no` in a **child process** per mandatory suite label |
| **NPSC-5E Final aggregator** | `tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py` | Composes R3 Final gate + in-process E2E/static proofs + selective subprocess spot checks |
| **R1 Final in-process regression import** | `tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py` | `importlib.import_module` over `_FROZEN_REGRESSION_MODULES` (same interpreter as parent gate) |
| **NPSC-5D Final** | `tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py` | In-process gate tests; `subprocess` only for `git cat-file` provenance |
| **DG-001 R1 Final** | `tests/unit/runtime/diagnostics/test_dg001_lineage_read_integration_r1_final_qualification.py` | Large in-process matrix (no `_run_pytest` composition) |
| **Platform proof suite** | `scripts/proof/intergrax_proof_runner.py::run_suite` | Sequential platform proof manifest execution; **`SuiteReceipt` / `ProofRunResult`** (`intergrax.proof_suite_receipt.v1`) — different product surface, reusable **pattern** for R1 aggregation |
| **Enterprise verification record** | `docs/project/maintainers/qualification/EXECUTION_ENGINE_ENTERPRISE_VERIFICATION.md` | Documents NPSC-3C–3G + NPSC-4 gate commands (not a runner) |

**Reuse-first verdict:** Execution certification already uses **subprocess pytest composition** via `_run_pytest`. There is **no** shared execution-certification coordinator abstraction — **`MISSING REUSABLE ABSTRACTION`** (R1 decision).

---

## Pytest temp isolation (PROVEN — repo infrastructure)

Root `conftest.py` calls `apply_invocation_pytest_basetemp` (`testing_support/pytest_temp_root.py`):

- Namespace: `build/pytest/<pid>-<random>/`
- Each **pytest invocation** (including each `_run_pytest` child) gets a **distinct** basetemp unless `--basetemp` is passed explicitly.

Verified by: `tests/unit/testing_support/test_pytest_temp_root.py`.

This supports **process-level** isolation for parallel leaf invocations; it does **not** override tests that write **fixed repo paths** (see risks below).

---

## Top-level certification gates (inventory)

| Suite ID | Test path | Launch |
| --- | --- | --- |
| **NPSC-5E Final** | `tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py` | `uv run pytest <file> -q` |
| **NPSC-5E R3 Final** | `tests/unit/runtime/architecture/test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py` | same |
| **NPSC-5E R2 Final** | `tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py` | same |
| **NPSC-5E R1 Final** | `tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py` | same |
| **NPSC-5D Final** | `tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py` | same |
| **DG-001 R1 Final** | `tests/unit/runtime/diagnostics/test_dg001_lineage_read_integration_r1_final_qualification.py` | same |
| **NPSC-5E P0A** | `tests/unit/runtime/architecture/test_npsc5e_p0a_execution_lineage_baseline_qualification.py` | same |
| **Execution Engine enterprise verification** | Multiple gates listed in `EXECUTION_ENGINE_ENTERPRISE_VERIFICATION.md` | per gate file |

**Wall-clock:** **NOT AVAILABLE** for full NPSC-5E Final / R3 Final composition in maintainer docs (no authoritative duration table found in qualification records reviewed for P0).

---

## R3 Final mandatory subprocess suites (OBSERVED)

Source: `_MANDATORY_SUITES` in `test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py`.

**ISOLATION SAFETY** = safe as one **isolated** `uv run pytest` child relative to other isolated children (distinct process + invocation basetemp). **COMPOSITION / PARITY** = scope duplication or parent-gate coupling — separate from isolation.

| Label | Target(s) | ISOLATION SAFETY | COMPOSITION / PARITY |
| --- | --- | --- | --- |
| R1 Final | `test_npsc5e_r1_final_retry_attempt_qualification.py` | **PARALLEL_SAFE** | **SERIAL_ONLY** if a parent gate (R3 Final, NPSC-5E Final) already subprocess-invokes this file — redundant scope |
| R2 Final | `test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py` | **PARALLEL_SAFE** | **SERIAL_ONLY** if parent gate already invokes this file; internal nested `_run_pytest` is serial **within** the child only |
| R3 implementation gate | `test_npsc5e_r3_child_fanout_partial_recovery.py` | **REQUIRES_EXCLUSIVE_RESOURCE** | Mutex on `.tmp/session/npsc5e-r3/cross.db`; also duplicated if parent runs same file |
| P0A | `test_npsc5e_p0a_execution_lineage_baseline_qualification.py` | **PARALLEL_SAFE** | None beyond normal mandatory-matrix single invocation |
| DG_001 | `test_execution_lineage_contracts.py` + `tests/unit/runtime/execution/lineage/` | **PARALLEL_SAFE** | Directory target is one pytest invocation — no nested gate |
| NPSC-5A | `test_npsc5a_coordination_delegation_e2e.py` | **PARALLEL_SAFE** | None |
| NPSC-5B Final | `test_npsc5b_final_production_fanout_fanin_qualification.py` | **PARALLEL_SAFE** | None |
| NPSC-5C | `test_npsc5c_decision_execution_e2e.py` | **PARALLEL_SAFE** | None |
| NPSC-5D Final | `test_npsc5d_final_multi_agent_governance_qualification.py` | **PARALLEL_SAFE** | **SERIAL_ONLY** if R3 Final parent already subprocess-invokes this file |
| HITL R3 | `test_npsc5d_r3_governed_continuation.py` | **PARALLEL_SAFE** | None |
| Attempt lifecycle | `test_attempt_lifecycle.py`, `test_attempt_lifecycle_durability_gate.py`, `conformance/.../test_attempt_lifecycle.py` | **PARALLEL_SAFE** | None |
| Child execution | `test_child_execution.py`, `test_child_execution_authority_policy.py` | **PARALLEL_SAFE** | None |
| Terminal | `test_p0c6_terminal_outcome_convergence.py` | **PARALLEL_SAFE** | None |
| Cancellation | `test_p0c5_cancellation_continuity.py` (`-k not survives_process_restart`), `test_p0c5a_explicit_terminal_wiring.py`, `test_task_control_governed_resume.py` | **PARALLEL_SAFE** | `survives_process_restart` excluded from this matrix label but may run under other gates — parity only, not isolation |
| Checkpoint store | `test_checkpoint_store.py` | **PARALLEL_SAFE** | None |
| Long-running | six files under `tests/unit/runtime/long_running/` (see gate source) | **PARALLEL_SAFE** | None |
| Fan-out | `test_bounded_multi_agent_fanout.py` | **PARALLEL_SAFE** | None |

### Per-suite isolation facts (frozen matrix — static read)

| Label | Filesystem | SQLite / DB | Environment | Ports | Network | Process-global / nested |
| --- | --- | --- | --- | --- | --- | --- |
| R1 Final | `build/pytest/*` basetemp | In-memory / no shared repo DB | No `os.environ` / `monkeypatch.setenv` in file | **NONE OBSERVED** | **NONE OBSERVED** | `importlib` frozen regression in child interpreter only |
| R2 Final | basetemp; E2E uses `tmp_path`/`cross.db` per test | Per-test `tmp_path` SQLite | No env mutation in gate file | **NONE OBSERVED** | **NONE OBSERVED** | Nested `_run_pytest` inside child only |
| R3 implementation | **Fixed** `.tmp/session/npsc5e-r3/cross.db` | Shared file path | No env mutation observed | **NONE OBSERVED** | **NONE OBSERVED** | No nested pytest composition |
| P0A | Read-only repo scans (`rglob`); no repo writes | In-memory lineage stores in tests | No env mutation | **NONE OBSERVED** | `git rev-parse` subprocess (local VCS, not live service) | `importlib` + `contextvars` thread tests — scoped to child process |
| DG_001 | basetemp | In-memory; lineage dir uses fixture-local `monkeypatch` | Fixture-local patches only | **NONE OBSERVED** | **NONE OBSERVED** | No subprocess gate |
| NPSC-5A | basetemp | In-memory harness | No env mutation | **NONE OBSERVED** | **NONE OBSERVED** | In-process only |
| NPSC-5B Final | basetemp | In-memory | `bind_root_execution_budget` / `reset_active_execution_budget` per test | **NONE OBSERVED** | **NONE OBSERVED** | In-process only |
| NPSC-5C | basetemp | In-memory qualification fixtures | No env mutation | **NONE OBSERVED** | **NONE OBSERVED** | In-process only |
| NPSC-5D Final | basetemp; AST scans skip `build`/`.tmp` | In-memory proofs | No env mutation | **NONE OBSERVED** | `git cat-file` subprocess (local VCS) | In-process only |
| HITL R3 | basetemp | In-memory | Contextvar budget bind/reset in tests | **NONE OBSERVED** | **NONE OBSERVED** | In-process only |
| Attempt lifecycle | basetemp | `InMemory*` stores; conformance uses per-test durable backing fixture | No env mutation | **NONE OBSERVED** | **NONE OBSERVED** | In-process only |
| Child execution | basetemp | No SQLite observed | No env mutation | **NONE OBSERVED** | **NONE OBSERVED** | In-process only |
| Terminal | basetemp | `tmp_path` `*.db` | No env mutation | **NONE OBSERVED** | **NONE OBSERVED** | In-process only |
| Cancellation | basetemp | `tmp_path` `*.db` in filtered files | No env mutation | **NONE OBSERVED** | **NONE OBSERVED** | Restart test excluded by `-k` for this label |
| Checkpoint store | basetemp | `tmp_path` `ckpt.db` | No env mutation | **NONE OBSERVED** | **NONE OBSERVED** | In-process only |
| Long-running | basetemp | `tmp_path` per test across files | `monkeypatch` on loop in `test_runtime_checkpoint.py` only (fixture-local) | **NONE OBSERVED** (checkpoint *port* tests are API contracts, not TCP) | **NONE OBSERVED** | Optional `pytest.importorskip("celery")` — skip, not shared resource |
| Fan-out | basetemp | In-memory coordination harness | No env mutation | **NONE OBSERVED** | **NONE OBSERVED** | In-process only |

### Matrix-wide risk summary (P0 complete for R3 Final mandatory targets)

| Concern | Finding |
| --- | --- |
| **Filesystem** | Default: invocation `build/pytest/*` (PROVEN). **Exclusive:** `.tmp/session/npsc5e-r3/cross.db` (R3 implementation gate only). |
| **SQLite** | Shared repo path above; otherwise `tmp_path` or in-memory (OBSERVED). |
| **Environment** | No durable `os.environ` mutation in mandatory targets; fixture `monkeypatch` only. R1 runner must still snapshot/restore env **between** children (inheritance hazard), not observed inside leaf files. |
| **Ports** | **NONE OBSERVED** for live bind/listen across mandatory matrix (static read). |
| **Global state** | Contextvars / in-process registries reset per test within a child; **cross-suite** hazard is **same-process** parent gates (NPSC-5E Final, R3 Final file) — **SERIAL_ONLY** as composed gates, not leaf isolation. |
| **Network** | **NONE OBSERVED** (HTTP/TCP); local `git` subprocess only on P0A / NPSC-5D Final provenance checks. |

---

## Safe parallelization groups (PROPOSED — not approved for production cert yet)

### SERIAL ONLY

- **NPSC-5E Final** (full file)
- **NPSC-5E R3 Final** (full file — composes nested finals)
- **NPSC-5E R2 Final** (full file)
- **R1 Final** when any other gate already runs R1 or descendants
- **R3 implementation gate** alone or parallel with anything touching `.tmp/session/npsc5e-r3/cross.db`
- Any gate using **in-process** `importlib` frozen regression (R1 Final pattern)

### GROUP A — **leaf subprocess invocations** (R1 scheduling hypothesis)

All R3 Final mandatory labels except **R3 implementation gate** are **PARALLEL_SAFE** under isolated-child rules (see table). R1 should still enforce **exclusive lock** on `.tmp/session/npsc5e-r3/cross.db` for the R3 implementation label.

**Not in R3 mandatory matrix (separate classification):**

| Suite | ISOLATION SAFETY | Notes |
| --- | --- | --- |
| **DG-001 R1 Final** (`test_dg001_lineage_read_integration_r1_final_qualification.py`) | **PARALLEL_SAFE** as isolated child | Large in-process matrix; no fixed repo DB path observed; no nested `_run_pytest` |
| **NPSC-5E Final** (full file) | **SERIAL_ONLY** | Composes R3 Final + in-process proofs + spot subprocess — parent gate |
| **Platform `run_suite`** | **UNRESOLVED_ARCHITECTURAL_DECISION** | Separate certification plane; out of Execution frozen matrix — AD-R1-1 / manifest scope |

---

## Shared risk register

| Risk | Severity | Detail |
| --- | --- | --- |
| **SHARED TEMP PATH** | High | `.tmp/session/npsc5e-r3/cross.db` in R3 implementation gate |
| **SHARED DB** | High | Same path — concurrent writers corrupt cross-process qualification |
| **GLOBAL STATE** | High | R1 Final `importlib` regression in parent process |
| **ENVIRONMENT MUTATION** | Medium | Subprocess inherits parent `os.environ`; R1 must snapshot/restore |
| **PORT** | Low (matrix) | **NONE OBSERVED** for mandatory R3 Final targets |
| **Nested duplication** | High | **Composition / parity** — parallelizing leaf files while a parent gate already invokes them → redundant load; not an isolation defect for leaves |

**SAFE MAX CONCURRENCY RECOMMENDATION (P0):** **1** for current composed gates; R1 should start with **bounded leaf-only** scheduling after audit, default cap **2–4** until evidence supports more.

---

## Performance baseline

| Metric | Value |
| --- | --- |
| **CURRENT SERIAL WALL-CLOCK** | **NOT AVAILABLE** (no maintainer-recorded end-to-end timing for NPSC-5E Final composition) |
| **ESTIMATED BOUNDED PARALLEL WALL-CLOCK** | **NOT AVAILABLE** (requires R1 runner + measured leaf durations) |

Historical gate log reference (partial scope): `.tmp/session/EXECUTION-ENGINE-ENTERPRISE-VERIFICATION/gates-all.log` (NPSC-3/4 gates only — OBSERVED path in enterprise verification doc).

---

## Failure semantics (PROPOSED for R1)

| Mode | Rule |
| --- | --- |
| **Default** | **COLLECT ALL** mandatory suite results; final gate **FAIL** if any mandatory suite **FAIL** |
| **FAIL FAST** | Only when coordinator cannot spawn isolated processes reliably (manifest missing, basetemp allocation failure, corrupt result contract) — results from unstarted suites marked **SKIP** with explicit catastrophic reason |
| **Skips** | Certification policy **UNEXPECTED SKIPS = 0** remains unless explicitly classified pre-existing in suite contract |
| **Pre-existing failures** | Must be labeled in suite result (`pre_existing_failure` classification) — no silent weakening |

Boundary documented in: `docs/project/maintainers/architecture/EXECUTION_QUALIFICATION_ACCELERATION_ARCHITECTURE.md`.

---

## Deterministic aggregation (PROPOSED)

Final report order = **stable suite definition order** (e.g. manifest declaration order), **not** process completion order.

---

## Result contract (PROPOSED — reuse pattern)

**Existing partial reuse:** `SuiteReceipt` / `ProofRunResult` in `scripts/proof/intergrax_proof_contracts.py` (platform proofs).

**Execution certification R1 contract** (extend, do not duplicate loosely):

```text
suite_id
command
exit_code
duration_seconds
stdout_stderr_log_ref
status: PASS | FAIL | SKIP
pre_existing_failure: bool + classification id (optional)
```

---

## P0 validation

| Check | Command / artifact |
| --- | --- |
| P0 static gate | `tests/unit/scripts/proof/test_execution_qualification_parallelization_p0.py` |
| Basetemp isolation | `tests/unit/testing_support/test_pytest_temp_root.py` |
| Representative in-process Final check | `test_npsc5e_final_recovery_plane_qualification_and_freeze.py::test_section_94_regression_labels_composed_in_r3_final_gate` |

Full **NPSC-5E Final** subprocess matrix **not** re-run in P0 (cost vs existing freeze record).

---

## Related documents

- Architecture (R1 target): [`EXECUTION_QUALIFICATION_ACCELERATION_ARCHITECTURE.md`](../architecture/EXECUTION_QUALIFICATION_ACCELERATION_ARCHITECTURE.md)
- Documentation inventory: [`EXECUTION_ENGINE_DOCUMENTATION_INVENTORY.md`](../architecture/EXECUTION_ENGINE_DOCUMENTATION_INVENTORY.md)
- Recovery plane freeze: [`NPSC_5E_FINAL_RECOVERY_PLANE_QUALIFICATION_AND_FREEZE.md`](NPSC_5E_FINAL_RECOVERY_PLANE_QUALIFICATION_AND_FREEZE.md)

---

## Architectural decisions required

| ID | Decision |
| --- | --- |
| AD-R1-1 | Canonical suite manifest owner (tests-only module vs maintainer YAML) |
| AD-R1-2 | Whether R1 re-runs leaf files only vs composed finals (scope parity proof) |
| AD-R1-3 | Reuse `SuiteReceipt` schema vs execution-specific receipt version |

**NPSC-5F** qualification artifacts (untracked parallel session) — **not modified** by P0; refer only.
