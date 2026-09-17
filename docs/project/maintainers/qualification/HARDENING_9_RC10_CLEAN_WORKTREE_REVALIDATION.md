# HARDENING-9 RC-10 Clean Worktree Revalidation

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `HARDENING_9_RC10_CLEAN_WORKTREE_REVALIDATION` |
| Branch target | `development` |
| Qualification date | 2026-09-17 |
| Isolated clean environment | `.tmp/session/rc10-clean-revalidation/worktree` |
| `START_HEAD` (`origin/development` @ run start) | `fdf552f70dee1fd2212ec0873543c5e10527d22b` |
| `QUALIFICATION_HEAD` | `fdf552f70dee1fd2212ec0873543c5e10527d22b` |
| `END_REMOTE_HEAD` | `96740bd47a95532f17aadd1886dae84f9a7961b5` |
| Main worktree | **not used for gates** (parallel local WIP present) |
| Environment sync | `uv sync --extra dev --extra dev-unit-cert` |

## Previous RC-10 failure (`HARDENING_9_RC10_FINAL_CLOSURE`)

| Field | Value |
| --- | --- |
| Reported error | `TypeError: non-default argument 'assembly_scope' follows default argument` (`ContextAssemblyRequest`) |
| Reproduced on clean `START_HEAD` | **NO** |
| Classification | **LOCAL_WIP_CONTAMINATION** |

Evidence (clean worktree @ `START_HEAD`):

```bash
uv run python -c "from intergrax.context.contracts import ContextAssemblyRequest"
uv run pytest tests/unit/context --collect-only -q
```

Result: import OK; 260 tests collected (no dataclass `TypeError`).

## Freeze state (authorities unchanged)

| Constant | SHA |
| --- | --- |
| `R1_POST_R2_QUALIFIED_BASELINE_SHA` | `33576b80521dda7dfc0e5895c943f91dfebffa94` |
| `R2_POST_QUALIFIED_BASELINE_SHA` | `22c4793da4ba751fff6c93f780a7f6848650a5d9` |
| `R3_POST_QUALIFIED_BASELINE_SHA` | `48a33db23fafab89b5fdb4ff217dfcb113dd6cc5` |
| `NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA` | `33576b80521dda7dfc0e5895c943f91dfebffa94` |
| `NPSC5F_R3_H1_QUALIFIED_BASELINE_SHA` | `62fdceac2122738751a8a1caeffe16c986dfe47d` |
| `NPSC5F_R3_H1_QUALIFICATION_RECORD_SHA` | `c94ba8ffe16e80466637a842e8ce71af854ccb67` |

`baselines modified`: **NO**

## Gate matrix (@ clean `START_HEAD`)

| Gate | Command (pytest node) | Result |
| --- | --- | --- |
| R1 protected drift | `test_npsc5f_p0_r1_protected_evidence_surfaces_have_no_unqualified_post_r1_drift` | **PASS** (1.18s) |
| R2 protected drift | `test_r2_final_no_unqualified_protected_drift_since_qualified_baseline` | **PASS** (0.72s) |
| R3 protected drift | `test_r3_final_no_unqualified_protected_drift_since_qualified_baseline` | **PASS** (1.26s) |
| Final Evidence Plane | `test_npsc5f_final_predecessor_drift_sentinels_empty`, `test_npsc5f_final_qualification_gate` | **PASS** (1.81s) |
| H1 integrated | `tests/unit/runtime/architecture/test_npsc5f_r3_final_h1_upstream_runtime_event_drift_reconciliation.py` | **PASS** (9 tests, 3.05s) |
| Docker runtime-context | `tests/unit/applications/test_hardening_9_docker_runtime_context_syntax.py` | **PASS** (3 tests, 10.81s) |
| Mandatory regression matrix | `test_npsc5f_final_mandatory_regression_matrix_passes` | **PASS** (87.09s) |

## Head movement follow-up

Commits `fdf552f70..96740bd47` (context MEM-XINT, INSPECT-01, MP-4, GR-7) — **outside** NPSC-5F / mandatory-matrix / Docker-syntax closure surface (no `testing_support/npsc5f*` or matrix definition changes).

Confirmatory mandatory matrix @ `END_REMOTE_HEAD` (`.tmp/session/rc10-clean-revalidation/worktree-end`): **PASS** (150.75s).

## Architecture invariants (gate scope)

| Invariant | Status |
| --- | --- |
| Contract-first preserved | **YES** (drift + matrix gates) |
| Pluginability preserved | **YES** |
| External strategy injection preserved | **YES** |
| Layer boundaries preserved | **YES** |
| Bypass detected | **NO** |
| Duplicate authority detected | **NO** |
| Private cross-component access detected | **NO** |
| Vendor-specific core coupling detected | **NO** |

## Unresolved blockers

**NONE** (revalidation scope)

## Verdict

**PASS** — mandatory HARDENING_9 regression matrix and RC-10 canonical gates pass on clean isolated trees; previous `ContextAssemblyRequest` failure was **LOCAL_WIP_CONTAMINATION**.
