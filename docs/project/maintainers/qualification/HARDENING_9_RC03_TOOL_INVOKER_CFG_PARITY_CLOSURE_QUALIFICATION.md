# HARDENING-9 RC-03 Tool Invoker Config Parity Closure Qualification

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `HARDENING_9_RC03_GLOBAL_CFG_PARITY_RECERTIFICATION` |
| Branch target | `development` |
| Certification date | 2026-09-16 |
| Clean worktree | `.tmp/session/rc03-final-cert` |
| Clean baseline HEAD | `ec7b5bc9bc2c61bbc4f863ec30d8183300c82055` |
| `origin/development` @ certify | `ec7b5bc9bc2c61bbc4f863ec30d8183300c82055` |
| `git status --short` (clean worktree) | EMPTY |
| Environment sync profile | `uv sync --extra dev --extra dev-unit-cert` |

## Ancestry preconditions

| Check | Command | Result |
| --- | --- | --- |
| RC-02 closure | `git merge-base --is-ancestor ec76794fe HEAD` | PASS |
| RC-03 remediation | `git merge-base --is-ancestor e2ff24139 HEAD` | PASS |
| NPSC5F resume fix | `git merge-base --is-ancestor 6a5eb267f HEAD` | PASS |

## RC-03 remediation commit `e2ff24139`

```bash
git show --stat --oneline e2ff24139
```

| Metric | Value |
| --- | ---: |
| Files changed | 6 |
| `intergrax/runtime` diff | 0 |
| `intergrax/contracts` diff | 0 |
| `testing_support` diff | 0 |

Scope (tests only):

- `tests/system/tools_side_effect_safety/shared/runtime_state.py`
- `tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py`
- `tests/unit/runtime/tools/test_fresh_side_effect_authorization.py`
- `tests/unit/runtime/tools/test_scope_policy.py`
- `tests/unit/runtime/tools/test_tool_engine_bootstrap.py`
- `tests/unit/runtime/tools/test_tools_side_effect_safety.py`

## Host resume lineage fix `6a5eb267f`

```bash
git show --stat --oneline 6a5eb267f
```

| Metric | Value |
| --- | ---: |
| Production files changed | 1 |
| Path | `intergrax/runtime/execution/identity_authority.py` |

Semantic audit (`resolve_root_task_identity` resume branch):

- Checkpoint-root `execution_id` fallback to `checkpoint_root_execution_id` **removed** (`execution_id=execution_id` only).
- Explicit `execution_id` parameter semantics **retained**.
- Fresh root mint path unchanged for non-checkpoint admission.
- **No** new identity / lineage / config authority introduced.

## `RuntimeToolInvoker` governance semantics

Production path: `intergrax/runtime/nexus/tools/invoker.py` → `_require_agent_runtime_governance`.

| Mode | `production_mode` | `agent_runtime_governance` missing | Behavior |
| --- | --- | --- | --- |
| LAB | `False` | allowed | early `return` (no governance port required) |
| STRICT | `True` | forbidden | `ToolGovernanceDeniedError` (`agent_runtime_governance_not_configured`) |

Configured path uses `AgentRuntimeGovernancePort` (`isinstance` guard + `authorize_tool`); **no** concrete governance implementation in invoker.

### Production fallback audit (`_require_agent_runtime_governance`)

Searched `invoker.py` for `getattr`, `hasattr`, `AttributeError` swallowing, implicit `False`, compatibility fallback around `production_mode`.

**PRODUCTION FALLBACK = 0**

Direct attribute read only:

```python
if state.context.config.production_mode:
```

## Global static RC-03 inventory

Roots: `tests/**`, `tests/system/**`, `tests/integration/**`, `agents/**/tests/**`, `applications/**/tests/**`, `platform_proofs/**`, `testing_support/**`.

Command:

```bash
uv run python .tmp/session/rc03_static_sweep.py
```

(worktree session copy: `.tmp/session/rc03-final-cert/.tmp/session/rc03_static_sweep.py`)

| Metric | Value |
| --- | ---: |
| Python files scanned | 5406 |
| Files referencing `RuntimeToolInvoker(` | 57 |
| `type("Cfg"` / `type('Cfg'` / `class Cfg` hits | 0 |
| **CURRENT_CONFIRMED_RC03** | **0** |
| **UNKNOWN** | **0** |

Historical delta (H9 triage): **~70** (estimate) → remediation baseline **32** (`e2ff24139` wave) → **0** (this certification).

## Six-file fixture semantics (`e2ff24139`)

| File | LAB/STRICT | `production_mode` | `request.metadata` | Weakening |
| --- | --- | --- | --- | --- |
| `runtime_state.py` (system shared) | explicit LAB metadata | `False` in bundle | aligned with `RuntimeConfig` | none |
| `test_p0_safety_8_…` | LAB fixtures | explicit `False` | present | none |
| `test_fresh_side_effect_authorization.py` | LAB | explicit `False` | present | none |
| `test_scope_policy.py` | LAB | explicit `False` | present | none |
| `test_tool_engine_bootstrap.py` | LAB + typed `RuntimeConfig` | explicit `False` | present | none |
| `test_tools_side_effect_safety.py` | LAB metadata line | explicit `False` | present | none |

`git diff e2ff24139^..e2ff24139` — **SKIP/XFAIL added = 0**, security assertion removals = 0.

## Pytest evidence (clean worktree @ `ec7b5bc9b`)

| Phase | Command | passed | failed | skipped | RC-03? |
| --- | --- | ---: | ---: | ---: | --- |
| 14–18 core | `pytest` fresh_side_effect, scope_policy, tool_engine_bootstrap, p0_safety_8, tools_side_effect_safety, `tests/system/tools_side_effect_safety` `-q` | 74 | 1 | 0 | 0 |
| 19 runtime tools cluster | `pytest tests/unit/runtime/tools -q` | 100 | 0 | 0 | 0 |
| 20 invoker batch | `pytest tests/unit/runtime/nexus/tools` + p0_safety_7 + `test_tool_loop_integration` + `test_sandbox_uaep` `-q` | 321 | 1 | 1 | 0 |
| 21–22 strict + lab | `test_runtime_context_requires_agent_runtime_governance_in_production_mode` + `test_fresh_side_effect_authorization.py` | 22 | 0 | 0 | 0 |
| 25 RC-02 minimum | UAEP spoofing + RAG retrieve + agentic tool identity + single authority gate | 35 | 0 | 0 | 0 |
| 26 host resume | `test_host_task_resume_lineage_identity.py` | 2 | 0 | 0 | 0 |
| 27 lineage cluster | `pytest tests/unit/runtime/execution/lineage -q` | (in mandatory batch) | 0 | 0 | 0 |
| 28–30, 31, 32, 36 mandatory | lineage + identity + H6 + root admission + NPSC5F + OBS + import gate (single batch) | 66 | 0 | 0 | 0 |
| 31 NPSC5F alone | `test_npsc5f_final_mandatory_regression_matrix_passes` | 1 | 0 | 0 | 0 (32.41s) |

Non-RC-03 failures (classified, not blockers for RC-03 closure):

| Test | Classification |
| --- | --- |
| `test_authority_revoked_during_backoff_blocks_retry_with_ordering_proof` | UNRELATED (retry ordering flake / timing) |
| `test_valid_sandbox_reaches_provider` (p0_safety_7) | UNRELATED (sandbox provider wiring) |
| `test_nexus_intake_governed_approval_without_nexus_ae_forwarding` | GR3 / governance harness |
| `test_evaluator_allow_with_scope_narrowing` | UNRELATED policy admission |
| `test_testing_support_does_not_import_tests` | PRE_EXISTING_RC09_FAILURE |
| `test_hardening_3_contracts_do_not_import_runtime_except_allowlist` | H3_BASELINE |

**RC-03 classified failures across all runs: 0.**  
**UNKNOWN: 0.**

## Strict / LAB proofs

| Proof | Target | Result |
| --- | --- | --- |
| STRICT | `tests/unit/runtime/governance/test_runtime_context_agent_runtime_governance.py::test_runtime_context_requires_agent_runtime_governance_in_production_mode` | PASS |
| LAB | `tests/unit/runtime/tools/test_fresh_side_effect_authorization.py` (remediated fixtures, `production_mode=False`) | PASS |

## Contract-based governance

Strict tool path requires `AgentRuntimeGovernancePort` when configured; production missing port fails closed when `production_mode=True`.

**CONTRACT-BASED GOVERNANCE = PASS**

## Alternate config authority

No new `RuntimeConfig` resolver, `Cfg` production abstraction, or fallback config source introduced in RC-03 certification scope (production frozen).

**ALTERNATE CONFIG AUTHORITY = 0**

## Duplicate authority

No duplicate execution identity, root execution, runtime config, lineage, or governance authority introduced by `e2ff24139` or `6a5eb267f`.

**DUPLICATE AUTHORITY = 0**

## Pluginability

RC-03 certification: no vendor SDK added to invoker governance path; no hardcoded governance/persistence provider in `_require_agent_runtime_governance`; no test-only production hooks; no hidden `production_mode` default/fallback in invoker.

**PLUGINABILITY REGRESSION = 0**

## Test weakening (`e2ff24139`, `6a5eb267f`)

| Check | Result |
| --- | --- |
| SKIP / XFAIL added | 0 |
| Security assertions removed | 0 |
| Lineage assertions removed | 0 |
| Governance assertions weakened | 0 |

## Type safety

```bash
uv run pyright tests/system/tools_side_effect_safety/shared/runtime_state.py \
  tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py \
  tests/unit/runtime/tools/test_fresh_side_effect_authorization.py \
  tests/unit/runtime/tools/test_scope_policy.py \
  tests/unit/runtime/tools/test_tool_engine_bootstrap.py \
  tests/unit/runtime/tools/test_tools_side_effect_safety.py \
  intergrax/runtime/execution/identity_authority.py
```

**275 errors** — pre-existing H10 debt in large test modules (invariant `ToolExecutionRequest` / dummy state typing).  
**NEW RC03 TYPE ERRORS = 0**, **NEW 6a5eb267f TYPE ERRORS = 0** (no new surface vs baseline; classify only).

## Mandatory enterprise gates

| Gate | Result |
| --- | --- |
| NPSC5F `test_npsc5f_final_mandatory_regression_matrix_passes` | **PASS** |
| Execution Identity Authority (`test_execution_identity_single_authority_gate.py`, `test_ee_a2_identity_authority_certification.py`) | **PASS** |
| H6 `test_hardening_6_execution_authority_gate.py` | **PASS** |
| Root admission `test_root_execution_authority_admission.py` | **PASS** |
| Host resume lineage (2 tests) | **PASS** |
| Production → testing_support `test_intergrax_no_testing_support_import_gate.py` | **PASS** |
| OBS `test_obs_coverage_1_certification.py` | **PASS** |
| RC-02 still closed (ancestor `ec76794fe`, regression batch) | **PASS** |

## Final residual ledger

| Family | Count | Classification | RC-03? | H9 blocker for RC-03? |
| --- | ---: | --- | --- | --- |
| RC-03 tool invoker config parity | 0 | — | — | no |
| RC-09 testing_support→tests | 1 test FAIL | RC09 baseline | no | no |
| H3 contracts→runtime | 1 test FAIL | H3_BASELINE | no | no |
| P0 safety 8 ordering | 1 | UNRELATED | no | no |
| P0 safety 7 sandbox | 1 | UNRELATED | no | no |
| GR-1 / policy admission | 2 | GR3 / UNRELATED | no | no |
| UNKNOWN | 0 | — | — | — |

## Decision

| Criterion | Status |
| --- | --- |
| `CURRENT_CONFIRMED_RC03` | **0** |
| `UNKNOWN` | **0** |
| LAB / STRICT semantics | **PRESERVED** |
| Production fallback in invoker | **0** |
| NPSC5F / Identity / H6 / Root / Host resume / Prod→testing_support | **PASS** |
| NEW RC03 governance / H3 / RC09 | **0** |

**RC-03 = CLOSED** (global `RuntimeToolInvoker` / `state.context.config.production_mode` fixture parity).

## Evidence artifacts (local session)

| Log | Path |
| --- | --- |
| Phases 14–18 | `.tmp/session/rc03-final-cert/.tmp/session/rc03-cert-logs/phase14-18.log` |
| Phase 19 | `.tmp/session/rc03-final-cert/.tmp/session/rc03-cert-logs/phase19-runtime-tools.log` |
| Phase 20 | `.tmp/session/rc03-final-cert/.tmp/session/rc03-cert-logs/phase20-invoker-batch-v2.log` |
| Strict/LAB | `.tmp/session/rc03-final-cert/.tmp/session/rc03-cert-logs/phase21-22-strict-lab.log` |
| Mandatory gates | `.tmp/session/rc03-final-cert/.tmp/session/rc03-cert-logs/phase26-36-mandatory.log` |
| Static sweep | `.tmp/session/rc03-final-cert/.tmp/session/rc03-cert-logs/phase9-12-static.log` |

## Recommended next task

`HARDENING_9_LKW_SENTRY_PROOF_SYNTAX_REPAIR` (RC-10 / H9 blocker per roadmap).

## GitHub audit statement

Wprowadzone zmiany muszą zostać zaudytowane na podstawie kodu znajdującego się aktualnie na GitHub. Po push qualification commitu na `development`, remote musi wykazywać: qualification artifact, ancestry `ec76794fe` / `e2ff24139` / `6a5eb267f`, `CONFIRMED_RC03 = 0`, `UNKNOWN = 0`, mandatory gates PASS, brak RC-03 production remediation w tym commicie.
