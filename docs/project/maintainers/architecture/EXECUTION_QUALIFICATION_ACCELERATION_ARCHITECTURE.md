# Execution Qualification Acceleration — Architecture (R1 Target)

**Status:** `R1 IMPLEMENTED` (bounded coordinator + subprocess executor in `testing_support/execution_qualification/`; gate integration remains R2)

**Scope:** Qualification / certification orchestration for Execution Engine frozen gates.

---

## Problem

Execution certification today composes large matrices via **nested subprocess pytest** (`_run_pytest`) and **in-process** imports (R1 Final). Wall-clock is serial; some leaf targets may be safe as **isolated processes**, but P0 found **fixed temp paths** and **nested duplication** hazards.

---

## Target shape (R1)

```text
Qualification Coordinator
        |
        +--> bounded isolated subprocess (suite A)
        +--> bounded isolated subprocess (suite B)
        +--> bounded isolated subprocess (suite C)
                |
                +--> isolated env snapshot/restore
                +--> isolated temp (build/pytest + optional suite prefix)
                +--> deterministic SuiteResult
```

**Non-goal:** shared-thread isolation as primary mechanism.

---

## Enterprise requirements

| Requirement | R1 obligation |
| --- | --- |
| Modularity | Coordinator · scheduler · subprocess launcher · result collector · reporter |
| Reusable components | Single launcher wrapping `uv run pytest` (generalize `_run_pytest` pattern) |
| Abstraction | Suite manifest independent of NPSC naming |
| Pluginable strategy | Manifest-driven suite list + execution policy (serial group vs parallel group) |
| Hard contracts | Typed `SuiteResult` / `QualificationRunReport` — no `dict[str, Any]` interfaces |
| Bounded concurrency | Explicit `max_parallel` |
| Deterministic aggregation | Sort by manifest `suite_id` / declaration index |
| Isolation by construction | Per-child env + basetemp; deny list for fixed repo paths unless suite declares exclusive lock |
| Collect-all | Run all mandatory suites unless catastrophic init failure |
| Fail-fast | Only catastrophic coordinator failure |
| PASS / FAIL / SKIP / pre-existing | Explicit enums |
| Zero semantic weakening | Same pytest targets and labels as serial certification |

---

## Suite manifest (`IMPLEMENTED` — typed Python, R1)

```text
suite_id: stable string
targets: list[str]  # pytest paths/args
parallel_group: serial | group_id
requires_exclusive_paths: list[str]  # e.g. .tmp/session/npsc5e-r3/cross.db
estimated_cost_class: optional enum
```

P0 inventory seeds mandatory labels from R3 Final `_MANDATORY_SUITES`.

---

## Isolation policy

P0 distinguishes **ISOLATION SAFETY** (parallel isolated children) from **COMPOSITION / PARITY** (parent gates that subprocess-invoke the same leaf file). A leaf may be `PARALLEL_SAFE` while still `SERIAL_ONLY` under a composed parent — schedule mutex is a parity rule, not proof of shared temp.

| Layer | Mechanism |
| --- | --- |
| Process | `subprocess.run` one pytest invocation per suite |
| Temp | Inherit repo `apply_invocation_pytest_basetemp`; optional R1 `TMPDIR` under `build/qualification/<run_id>/` |
| Env | Copy parent env; inject harness keys; restore after child |
| DB | Treat any suite declaring `requires_exclusive_paths` as **mutex serial group** (P0: `.tmp/session/npsc5e-r3/cross.db` for R3 implementation gate) |
| Ports | P0 mandatory matrix: **no live port bind observed**; R1 manifest should still declare `live_service: false` or port ranges when adding new suites |

### P0 isolation outcomes (R3 Final mandatory labels)

| Class | Labels |
| --- | --- |
| **PARALLEL_SAFE** (isolated child) | All mandatory labels except R3 implementation gate |
| **REQUIRES_EXCLUSIVE_RESOURCE** | R3 implementation gate (`cross.db` path) |
| **SERIAL_ONLY** (composition) | NPSC-5E Final, R3 Final full file, R2 Final full file when used as parent gates; leaf labels when parent already invoked same target |
| **UNRESOLVED_ARCHITECTURAL_DECISION** | Platform `run_suite` plane (manifest owner — AD-R1-1) |

---

## Result contract

Per suite (align with platform `ProofRunResult` fields where sensible):

| Field | Type |
| --- | --- |
| `suite_id` | str |
| `command` | list[str] |
| `exit_code` | int |
| `duration_seconds` | float |
| `stdout_stderr_log_ref` | path relative to run artifact dir |
| `status` | PASS \| FAIL \| SKIP |
| `pre_existing_failure` | optional structured classification |

**R1:** `ExecutionQualificationSuiteResult` / `ExecutionQualificationRunResult` in `testing_support.execution_qualification.contracts` (platform `SuiteReceipt` unchanged — AD-R1-3).

---

## Deterministic aggregation

1. Load manifest order `O = [s1, s2, …, sn]`.
2. Execute with bounded parallelism.
3. Collect results map `id → SuiteResult`.
4. Emit report rows **strictly in O**, filling missing entries as SKIP (catastrophic) only when launch never attempted.

Completion order must not reorder output.

---

## Failure semantics

```text
COLLECT ALL mandatory suite results
→ aggregate FAIL if any mandatory status == FAIL
→ aggregate SKIP policy per certification rules (unexpected skip = FAIL)
```

**FAIL FAST** when:

- manifest invalid
- cannot allocate isolation directories for any mandatory suite
- result schema validation fails for a completed child (treat as coordinator FAIL)

Not fail-fast on single suite test failure.

---

## Integration (R2+)

- Wire coordinator into CI / maintainer qualification entrypoint
- Preserve existing gate files as **suite targets** until parity proof replaces composed finals

---

## References

- P0 inventory: [`EXECUTION_CERTIFICATION_ACCELERATION_P0.md`](../qualification/EXECUTION_CERTIFICATION_ACCELERATION_P0.md)
- Doc inventory: [`EXECUTION_ENGINE_DOCUMENTATION_INVENTORY.md`](EXECUTION_ENGINE_DOCUMENTATION_INVENTORY.md)
