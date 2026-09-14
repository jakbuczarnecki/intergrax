# INTEGRAx Qualification — Global Semantic Parity Certification

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-QUALIFICATION-GLOBAL-SEMANTIC-PARITY-CERTIFICATION` |
| Base revision | `1dc93afaba9c6573e2d26979d3bbe02a7073af7e` |
| Scope | Canonical NPSC-5E/5F qualification profiles — semantic coverage and verdict parity vs legacy reference |
| Performance certification | Out of scope |

## Scope

Global semantic certification covering exact legacy vs canonical pytest argument vectors, root reachability, all-PASS, per-leaf FAIL/SKIP injection, collect-all receipts, determinism, architecture guards, and minor hardening (builder immutability, expansion cycle diagnostics).

Not in scope: wall-time benchmarks, cross-run caching, production runtime changes, Execution Engine redesign.

## Certified Profiles

| Profile ID |
| --- |
| `npsc5e-r1-final` |
| `npsc5e-r2-final` |
| `npsc5e-r3-final` |
| `npsc5e-final` |
| `npsc5f-r1-final` |
| `npsc5f-r2-final` |
| `npsc5f-r3-final` |
| `npsc5f-r4-final` |
| `npsc5f-final` |

## Certification Architecture

- `QualificationSemanticParityCase` — typed matrix row (`profile_id`, `legacy_semantic_source`, `canonical_profile_id`, `expected_root_ids`).
- `QualificationSemanticParityCertifier` — compiles via `QualificationCatalog`, runs plans through `QualificationPlanRunner` + `FakeQualificationSuiteExecutor` (DI), compares invariants.
- `QualificationSemanticParityReport` / `QualificationProfileParityResult` — hard PASS/FAIL per invariant (no scores).

Code: `testing_support/execution_qualification/semantic_parity/`.

## Coverage Parity Model

Legacy reference leaf vectors from `mandatory_sources` expanded via `unique_required_leaf_targets` (orchestrator expansion only as reference evidence).

Canonical leaf vectors from compiled plan suite `pytest_arguments`.

Comparison: `frozenset[tuple[str, ...]]` exact equality.

## Legacy Reference Model

Legacy semantics come from catalog `mandatory_sources` frozen tuples — not private legacy `_MANDATORY_SUITES` modules and not composition input for the canonical compiler.

## Root Obligation Model

Each matrix row declares `expected_root_ids`. Certifier verifies plan roots, gate kinds, and that every mandatory leaf is in the transitive dependency closure of profile roots.

## All-PASS Certification

Fake executor with no overrides → `QualificationRunStatus.PASS` and all root gate receipts `PASS`.

## FAIL Injection Matrix

For each physical leaf `suite_id`: inject `QualificationSuiteStatus.FAIL` only for that suite; every root transitively depending on the leaf must receipt `FAIL`.

## SKIP Injection Matrix

Same as FAIL injection; run status must not be `PASS` when any mandatory path receives `SKIP`.

## Receipt Consistency

- `suite_receipts` align with `plan.leaf_suite_ids` and `receipt.suite_id`.
- `root_gate_receipts` align with `plan.root_gate_ids`.
- Gate receipts: `consumed_node_ids == node.dependencies`; PASS iff no `failure_dependencies`.

## Reachability

`set(plan.leaf_suite_ids) == reachable leaves from roots` (no dead leaf definitions).

## Determinism

Double compile → identical `QualificationExecutionPlan`. Repeated injected run → identical statuses and root receipts.

## Suite Identity

Global registry: normalized pytest args map to at most one canonical `suite_id` (test T16).

## Physical Dedup

At most one physical execution per `suite_id` per plan run (`FakeQualificationSuiteExecutor.invocation_counts`).

## Architecture Guards

- No orchestrator paths as leaves (`CANONICAL_ORCHESTRATOR_PATHS`).
- Catalog does not import legacy regression matrix SSOT modules or `tests.*`.
- Generic core (`compiler`, `coordinator`, `plan_runner`, `aggregate`) has no profile-specific branches.
- Aggregate layer does not import `subprocess`.

## Minor Hardening Closure

- **MINOR-01:** `PROFILE_BUILDERS` is a `MappingProxyType` over an inline dict literal (no exported mutable `_PROFILE_BUILDER_MAP`).
- **MINOR-02:** Expansion cycle diagnostics use ordered `tuple[str, ...]` stack + membership set; cycle `A→B→C→A` reports `A, B, C, A`.

## Certification Matrix

| Profile | Legacy leafs | Canonical leafs | Coverage | All PASS | FAIL Matrix | SKIP Matrix | Determinism |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `npsc5e-r1-final` | 1 | 1 | PASS | PASS | 1/1 | 1/1 | PASS |
| `npsc5e-r2-final` | 18 | 18 | PASS | PASS | 18/18 | 18/18 | PASS |
| `npsc5e-r3-final` | 20 | 20 | PASS | PASS | 20/20 | 20/20 | PASS |
| `npsc5e-final` | 20 | 20 | PASS | PASS | 20/20 | 20/20 | PASS |
| `npsc5f-r1-final` | 24 | 24 | PASS | PASS | 24/24 | 24/24 | PASS |
| `npsc5f-r2-final` | 29 | 29 | PASS | PASS | 29/29 | 29/29 | PASS |
| `npsc5f-r3-final` | 33 | 33 | PASS | PASS | 33/33 | 33/33 | PASS |
| `npsc5f-r4-final` | 38 | 38 | PASS | PASS | 38/38 | 38/38 | PASS |
| `npsc5f-final` | 43 | 43 | PASS | PASS | 43/43 | 43/43 | PASS |

## Tests

- `tests/unit/testing_support/execution_qualification/test_global_semantic_parity_certification.py` — T1–T18 orchestration.
- `tests/unit/testing_support/execution_qualification/catalog/test_expansion_cycle_diagnostics.py` — T19 cycle path.
- Full suite: `uv run pytest tests/unit/testing_support/execution_qualification/ -q`

## Static Quality

```bash
uv run pytest tests/unit/testing_support/execution_qualification/ -q
uv run ruff check <changed scope>
uv run ruff format --check <changed scope>
uv run pyright testing_support/execution_qualification/semantic_parity testing_support/execution_qualification/fake_executor.py
```

## Production Changes

**NONE** — testing support and certification tests only.

## Findings

No semantic mismatch detected at certification time. Minor hardening items MINOR-01 and MINOR-02 closed in catalog/expansion.

## Certification Decision

`GLOBAL SEMANTIC PARITY CERTIFICATION = PASS`

## Final Verdict

All nine canonical profiles satisfy exact semantic target-set parity, verdict propagation, receipt contracts, determinism, and architecture guards under fake deterministic execution.
