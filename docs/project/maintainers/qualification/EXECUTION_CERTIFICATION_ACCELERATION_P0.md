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

| Label | Target(s) | Parallel-safe |
| --- | --- | --- |
| R1 Final | `test_npsc5e_r1_final_retry_attempt_qualification.py` | **NO** (nested subprocess + in-process imports; duplicates work if run beside decomposed R1 targets) |
| R2 Final | `test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py` | **NO** (nested `_run_pytest` tree) |
| R3 implementation gate | `test_npsc5e_r3_child_fanout_partial_recovery.py` | **NO** (fixed `.tmp/session/npsc5e-r3/cross.db` — PROVEN) |
| P0A | `test_npsc5e_p0a_execution_lineage_baseline_qualification.py` | **UNKNOWN** |
| DG_001 | contracts + `tests/unit/runtime/execution/lineage/` | **UNKNOWN** (directory scope) |
| NPSC-5A | `test_npsc5a_coordination_delegation_e2e.py` | **UNKNOWN** |
| NPSC-5B Final | `test_npsc5b_final_production_fanout_fanin_qualification.py` | **UNKNOWN** |
| NPSC-5C | `test_npsc5c_decision_execution_e2e.py` | **UNKNOWN** |
| NPSC-5D Final | `test_npsc5d_final_multi_agent_governance_qualification.py` | **NO** if parallel with R3 Final (R3 Final already invokes this subprocess) |
| HITL R3 | `test_npsc5d_r3_governed_continuation.py` | **UNKNOWN** |
| Attempt lifecycle | multiple paths | **UNKNOWN** |
| Child execution | multiple paths | **UNKNOWN** |
| Terminal | `test_p0c6_terminal_outcome_convergence.py` | **UNKNOWN** |
| Cancellation | filtered pytest args (`-k not survives_process_restart`) | **UNKNOWN** (explicit restart test excluded from matrix but invoked elsewhere in Final gate) |
| Checkpoint store | `test_checkpoint_store.py` | **UNKNOWN** |
| Long-running | multiple paths | **UNKNOWN** |
| Fan-out | `test_bounded_multi_agent_fanout.py` | **UNKNOWN** |

Per-suite filesystem / DB / env (summary):

| Concern | Finding |
| --- | --- |
| **Filesystem** | Default: invocation `build/pytest/*` (PROVEN). Exception: **R3 implementation** uses repo `.tmp/session/npsc5e-r3/cross.db` (PROVEN). |
| **SQLite** | R3 cross-process test uses shared file path above; most unit tests use `tmp_path` (OBSERVED in R2 Final E2E helpers). |
| **Environment** | Gate tests set `INTERGRAX_HARNESS_API_KEY` via fixtures in root conftest when used; child subprocess inherits parent env unless cleared in R1. |
| **Ports** | Not systematically audited in P0 — **UNKNOWN** for live-service suites (out of Execution frozen matrix). |
| **Global state** | R1 Final imports regression modules in **parent interpreter** — **SERIAL ONLY** relative to other tests in same process. |
| **Network** | Frozen architecture gates: predominantly unit/in-process — **UNKNOWN** for integration proofs not in this matrix. |

---

## Safe parallelization groups (PROPOSED — not approved for production cert yet)

### SERIAL ONLY

- **NPSC-5E Final** (full file)
- **NPSC-5E R3 Final** (full file — composes nested finals)
- **NPSC-5E R2 Final** (full file)
- **R1 Final** when any other gate already runs R1 or descendants
- **R3 implementation gate** alone or parallel with anything touching `.tmp/session/npsc5e-r3/cross.db`
- Any gate using **in-process** `importlib` frozen regression (R1 Final pattern)

### GROUP A — candidate **leaf subprocess invocations** (R1 only, after per-file audit)

Hypothesis: single pytest file targets with **no** `_run_pytest` composition and **no** fixed repo temp paths, each launched as **one** isolated `uv run pytest` with R1-provided env/temp overrides.

**Evidence required before YES:** per-file static audit (fixed paths, env mutation, ports) — **not completed in P0**.

Example candidates for **future** audit (UNKNOWN today):

- `tests/unit/runtime/architecture/test_npsc5a_coordination_delegation_e2e.py`
- `tests/unit/runtime/architecture/test_npsc5c_decision_execution_e2e.py`

### UNRESOLVED / UNKNOWN

- All other R3 Final labels until R1 file-level isolation audit
- **DG-001 R1 Final** (monolithic in-process matrix)
- **NPSC-5D Final**
- Platform **`run_suite`** proofs (separate certification plane)

---

## Shared risk register

| Risk | Severity | Detail |
| --- | --- | --- |
| **SHARED TEMP PATH** | High | `.tmp/session/npsc5e-r3/cross.db` in R3 implementation gate |
| **SHARED DB** | High | Same path — concurrent writers corrupt cross-process qualification |
| **GLOBAL STATE** | High | R1 Final `importlib` regression in parent process |
| **ENVIRONMENT MUTATION** | Medium | Subprocess inherits parent `os.environ`; R1 must snapshot/restore |
| **PORT** | Unknown | Not inventoried for execution frozen gates |
| **Nested duplication** | High | Parallelizing R2 + R1 files while R3 Final also runs both → redundant load + race on any shared resource |

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
