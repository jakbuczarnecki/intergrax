# PLAN T-IGX-043-J — PT-WF scaffold (ARCH-FREEZE development tor)

Status: **plan §7 filled (autonomy fala 3, 2026-11-21)**. Product slices after G-09 may proceed per **OD-AF-AUTO-03**. Canonical wave: `WP12_TICKETS_DRAFT.md` row PT-WF; **AR-44**.

## 1. Purpose

Close PT-WF prerequisites before RW milestone: G-09 tenant-scoped evaluation registry on STRICT/PRODUCT harness hosts, experiment workflow governance identity (see `PLAN_T-IGX-041-E` finding F-D5-07 as related prior art), and conditional E2–E4 repairs per Campaign B-1.

## 2. Prerequisites

- `SCOPE` formally valid on train (blocked until `T-AF2-RW-PLAN-AUDIT` PASS).
- **PT-B01** diagnostics when required by DAG.
- No concurrent edit of `harness_host_runtime.py` with TENANT batch.

## 3. Write set (initial, to be detailed before `plan_pending`)

| Area | Paths (indicative) |
|------|---------------------|
| G-09 / harness | `applications/_shared/harness_host_runtime.py`, `evaluation_wiring.py` |
| E2 | `runtime/execution/runtime.py` |
| E3 | `compat/langchain/documents.py` |
| E4 | `experiments/store.py`, `debug/router.py`, `experiments/workflow.py` |

## 4. Test contract (placeholder)

- Red-first tests to be named in plan revision 2 (NT-WF-*).
- Regression: subset of ENT-A and evaluation qualification nodes (TBD from WP-4 `rows.jsonl`).
- `uv run python -m scripts.governance.igx_gov ticket` before push.

## 5. Out of scope

Same as ticket `T-IGX-043-J.md`.

## 6. Next steps (operator)

1. Expand sections 3–4 with file-level diffs and node ids (delegate or architect session).
2. Record plan digest in the ticket; move lifecycle to `plan_pending`.
3. Operator or delegate approval line; then `plan_approved` and product work in `af-dev-pt-wf`.

## 7. Test contract (fala 3 — verified node ids)

Notation: paths are pytest node ids on `development` @ `2c521ccd5` (verified with `uv run pytest --collect-only -p no:cacheprovider`, 2026-11-21). Logs: `.tmp/session/arch-freeze/REGRESSION-78-fala3/`.

### 7.1 Negative tests (red-first, `NT-WF-*`)

| Node id | Assertion | Pytest node id | Slice |
|---------|-----------|----------------|-------|
| NT-WF-G09-01 | STRICT + online registry without `tenant_id` → validation error contains `G-09` | `tests/unit/applications/test_harness_evaluation_wiring.py::test_validate_evaluation_wiring_strict_online_registry_requires_tenant` | G-09 (**green** @ `aac4b083b`) |
| NT-WF-G09-02 | `assert_evaluation_assembly_valid` raises `EvaluationAssemblyError` for same wiring | `tests/unit/applications/test_harness_evaluation_wiring.py::test_validate_evaluation_wiring_strict_online_registry_requires_tenant` (second assert in test) | G-09 (**green**) |
| NT-WF-G09-03 | baseline release requires trend comparison when configured | `tests/unit/applications/test_harness_evaluation_wiring.py::test_validate_evaluation_wiring_rejects_baseline_without_trend` | G-09 (**green**) |
| NT-WF-NEG-E2-01 | Campaign B-1 runtime execution path (MEP §F E2) | `tests/unit/runtime/execution/` — **TBD** after B-1 | E2 (**BLOCKED**) |
| NT-WF-NEG-E3-01 | LangChain `Document` compat seam (MEP §F E3) | `tests/unit/compat/langchain/` — **TBD** after B-1 | E3 (**BLOCKED**) |
| NT-WF-NEG-E4-01 | experiments workflow governance id (F-D5-07) | `tests/unit/experiments/` — **TBD** after B-1 | E4 (**BLOCKED**) |

New negative file (owner-repair batch, not started): `tests/unit/applications/test_wf_fail_closed.py` — holds `NT-WF-OWN-*` / `NT-WF-PLG-*` when FRZ row owners are scheduled.

### 7.2 FC-3 regression anchor (batch 78 file bundle; PLAN_T-IGX-043-C §6)

When `regression_nodes.txt` is absent from the plan package, use the file bundle (same as fala 1–2 session record):

`governance/tickets/plans/regression_scope_fc3_bundle.txt` (copied from `.tmp/session/arch-freeze/REGRESSION-78-fala1/regression_scope.txt` with path fixes).

Command 3 (regression, one process; lock `D:/Projekty/_wt/_locks/pytest.lock`):

```text
uv run pytest $(grep -v '^#' governance/tickets/plans/regression_scope_fc3_bundle.txt | tr '\n' ' ') -p no:cacheprovider
```

(PowerShell: read non-comment lines from `regression_scope_fc3_bundle.txt` into one `pytest` invocation.)

ENV (fala 3): `uv sync --extra dev-ci` plus `langchain-core` / modality extras as in `REGRESSION-78-fala2/env_collect_blocked.md` when collect errors cite missing modules.

### 7.3 WF evaluation regression nodes (PT-WF harness scope)

| Node id | Pytest node id |
|---------|----------------|
| NT-WF-REG-01 | `tests/unit/applications/test_harness_evaluation_wiring.py::test_resolve_evaluation_wiring_options_maps_profile_fields` |
| NT-WF-REG-02 | `tests/unit/applications/test_harness_evaluation_wiring.py::test_wire_application_evaluation_builds_registry_and_bridge` |
| NT-WF-REG-03 | `tests/unit/applications/test_harness_evaluation_wiring.py::test_build_harness_host_runtime_wires_evaluation_policy_bundle` |

### 7.4 Commands (this ticket)

1. G-09 / harness regression: `uv run pytest tests/unit/applications/test_harness_evaluation_wiring.py -p no:cacheprovider`
2. FC-3 batch bundle: command in §7.2
3. Before push: `uv run python -m scripts.governance.igx_gov ticket` with `Ref: governance/tickets/T-IGX-043-J.md`