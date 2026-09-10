# Execution Qualification Acceleration — Architecture (R1 Target)

**Status:** `PROPOSED` (P0 defines inventory; **no implementation** in P0)

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

## Suite manifest (PROPOSED)

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

| Layer | Mechanism |
| --- | --- |
| Process | `subprocess.run` one pytest invocation per suite |
| Temp | Inherit repo `apply_invocation_pytest_basetemp`; optional R1 `TMPDIR` under `build/qualification/<run_id>/` |
| Env | Copy parent env; inject harness keys; restore after child |
| DB | Treat any suite declaring `requires_exclusive_paths` as **mutex serial group** |
| Ports | R1 manifest must declare port ranges or `live_service: false` |

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

**Reuse:** extend `intergrax.proof_suite_receipt.v1` pattern or define `intergrax.execution_qualification_run.v1` with shared field names — AD-R1-3.

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
