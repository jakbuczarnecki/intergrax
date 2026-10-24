# TOOL-LIFE-X Batch B — Retry, Idempotency & Unknown-Outcome Evidence

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-BATCH-B` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5) — **not CLOSED** |
| **Predecessor** | Batch A — **READY FOR INDEPENDENT AUDIT** @ `e5db3b29346000b2f1f2e8e2716aec7ba53297f2` (65/65 L2) |
| **Remediation START_HEAD** | `60a3ad53b8caf530027500803f3285d55ebda2c2` |
| **Prior Batch B L2 (historical)** | **57/57 passed** @ `60a3ad53b8caf530027500803f3285d55ebda2c2` — log `.tmp/session/TOOL-LIFE-X-BATCH-B/l2-batch-b.log` |
| **Branch** | `development` |
| **Disposition** | **`TOOL-LIFE-X BATCH B — READY FOR INDEPENDENT AUDIT`** (publication SHA recorded below) |

**Non-closure:** Does not close `TOOL-LIFE-X`, does not promote global `FRZ-*` PASS, does not rerun Batch A 65-test suite (no shared admission semantics changed).

## Independent audit findings (remediation scope)

| ID | Finding | Root cause | Remediation |
| --- | --- | --- | --- |
| **B1** | `EXPLICITLY_RETRY_SAFE` + idempotency key classified as `EXTERNALLY_DEDUPLICATED` | `classify_tool_operation_retry_class` treated key presence as proof of remote-provider deduplication | Classification derived only from `side_effects` + `SideEffectRetrySafety`; key affects **retry budget** only for `IDEMPOTENT_WRITE` |
| **B5** | Batch B/P0-SAFETY-8 fixtures patched `require_meaningful_side_effect_authorization` | Autouse governance scope bypassed the real declarative gate under test | New module `test_tool_life_x_batch_b_real_governance_authorization.py` exercises the real gate (deny before executor, non-enforcing mode, per-attempt gate calls, scope revoke on retry); mocked autouse retained only where other invariants are under test |

## Canonical owner inventory (closed-world)

| Concern | Owner | Contract |
| --- | --- | --- |
| Tool invocation execution | `RuntimeToolInvoker` | Tool execution request/result |
| Retry policy / attempt budget | `RuntimeToolInvoker` + `effective_max_retry_attempts` | `ToolRetryPolicy`, `ToolOperationRetryClass` |
| Operation safety classification | `classify_tool_operation_retry_class` | `SideEffectRetrySafety`, `side_effects` |
| Idempotency identity / ledger | `IdempotencyPreEffectCoordinator` + `IdempotencyStore` | Operation identity, claim ledger |
| External-effect authorization | `require_meaningful_side_effect_authorization` | `DeclarativePolicyEnforcer` ENFORCE |
| Provider invocation | `ToolExecutor` | Handler registry |
| Outcome certainty | `ToolEffectCertainty` / idempotency outcome | UNKNOWN blocks blind replay |
| Cancellation / backoff | Cooperative delay + `_require_current_attempt_authorization` | Scope / sandbox re-check |

**Not introduced:** retry coordinator, idempotency store owner change, alternate invocation path, provider dedup inference.

## Classification truth table (post-remediation)

| `side_effects` | `side_effect_retry_safety` | `idempotency_key` | `ToolOperationRetryClass` | `effective_max_retry_attempts` (policy `max_attempts=N`) |
| --- | --- | --- | --- | --- |
| `False` | * | * | `READ_ONLY` | `N` |
| `True` | `NOT_RETRY_SAFE` | absent/present | `NON_IDEMPOTENT_WRITE` | `1` |
| `True` | `IDEMPOTENT` | absent | `IDEMPOTENT_WRITE` | `1` |
| `True` | `IDEMPOTENT` | present | `IDEMPOTENT_WRITE` | `N` |
| `True` | `EXPLICITLY_RETRY_SAFE` | absent/present | `EXPLICITLY_RETRY_SAFE_WRITE` | `N` |
| * | * | * | `EXTERNALLY_DEDUPLICATED` | **Not emitted** unless a future ratified contract proves external dedup |

**Separation preserved:** (1) operation class, (2) key presence, (3) retry authorization, (4) outcome certainty, (5) external dedup guarantee.

## Governance enforcement matrix (real gate vs mocked)

| Scenario | Real `require_meaningful_side_effect_authorization` | Evidence |
| --- | --- | --- |
| Side effect, no policy bundle | Rejects; executor 0 calls | `test_batch_b_governance_unauthorized_side_effect_zero_executor_calls` |
| Side effect, `audit_only` enforcer | `NON_ENFORCING_MODE`; executor 0 calls | `test_batch_b_governance_non_enforcing_mode_blocks_before_executor` |
| Retry-safe side effect, ENFORCE allow | Gate invoked each physical attempt (wraps real function) | `test_batch_b_governance_real_gate_invoked_on_each_retry_attempt` |
| Scope revoked during backoff | Second attempt blocked; executor 1 call | `test_batch_b_governance_scope_revoked_during_backoff_blocks_retry` |
| Retry/idempotency/UNKNOWN invariants | Autouse patch on gate (documented) | `test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py`, P0-SAFETY-8, side-effect safety |

Reused unchanged: `test_ee_b3_c_mutating_tool_without_side_effect_authorization_zero_calls`, `test_fresh_side_effect_authorization` matrix (P0-SAFETY-4).

## Adverse-case coverage (A–V, remediation delta)

| Case | Status | Evidence |
| --- | --- | --- |
| A–F | Classification + L1 | Batch B tests + new explicit/key + NOT_RETRY_SAFE+key |
| G | **Closed (B1)** | `test_batch_b_classify_explicit_retry_safe_with_key_not_external_dedup` |
| H | No ratified external-dedup contract | Enum retained; runtime never emits |
| I–V | Unchanged valid L2 | Prior 57 tests + P0-SAFETY-8 + side-effect safety |

## TL1–TL15 (Batch B reconciliation)

| ID | Classification | Owner | Existing evidence SHA | Missing action | Min validation | FRZ |
| --- | --- | --- | --- | --- | --- | --- |
| TL1 | ALREADY PROVEN | Batch A boundary | `e5db3b29…` | None | Batch A L2 (not rerun) | — |
| TL2 | ALREADY PROVEN | Invoker deadline | Batch A | None | — | — |
| TL3 | ALREADY PROVEN (scoped) | Cancel/backoff | A+B L2 | Parent real-infra | L2 revoke tests | FRZ-REL-04 scoped |
| TL4 | **ALREADY PROVEN** | Classification | `60a3ad53…` + remediation L2 | B1 fix | L1 classification + L2 | FRZ-REL-04, FRZ-REL-07 scoped |
| TL5 | **ALREADY PROVEN** | UNKNOWN outcome | Batch B L1 | None | timeout/ledger tests | FRZ-REL-03 scoped |
| TL6 | **ALREADY PROVEN** | Idempotency coordinator | Batch B L2 | Key ≠ external dedup (B1) | tenant/payload tests | FRZ-REL-06 scoped |
| TL7 | OPERATOR DECISION | Batch C parent | — | Batch C | — | — |
| TL8 | ALREADY PROVEN | Batch A | `e5db3b29…` | None | — | — |
| TL9 | ALREADY PROVEN (scoped) | Stale guard | Batch A | Provider matrix parent | — | — |
| TL10–TL13 | BLOCKED | Batches D–E | — | Parent | — | — |
| TL14 | ALREADY PROVEN | Single invoker path | Prior arch gates | None | — | FRZ-BND-03 |
| TL15 | BLOCKED | Parent E | — | Real-infra | — | — |

## Implementation (remediation package)

| Module | Change |
| --- | --- |
| `intergrax/runtime/tools/tool_operation_retry_classification.py` | Remove key→`EXTERNALLY_DEDUPLICATED` inference; `EXPLICITLY_RETRY_SAFE_WRITE` from contract only |
| `tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py` | Classification negatives (G, unsafe+key) |
| `tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_real_governance_authorization.py` | **NEW** — real governance gate negatives (B5) |

`invoker.py` unchanged in remediation (classification-only fix).

## Validation

### L1 (critical safety, post-remediation)

```text
uv run pytest -p no:xdist -k "batch_b_l1 or batch_b_classify_explicit or batch_b_non_retry_safe or batch_b_governance" \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_real_governance_authorization.py -q
```

| Metric | Value |
| --- | --- |
| **Passed** | **10** |
| **Failed** | **0** |
| **Execution SHA** | same commit as publication (see below) |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-B/l1-batch-b-remediation.log` |

### L2 (consolidated qualification)

```text
uv run pytest -p no:xdist \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_real_governance_authorization.py \
  tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py \
  tests/unit/runtime/tools/test_tools_side_effect_safety.py -q
```

| Metric | Value |
| --- | --- |
| **Passed** | **63** (57 historical + 6 new) |
| **Failed** | **0** |
| **Log** | `.tmp/session/TOOL-LIFE-X-BATCH-B/l2-batch-b-remediation.log` |

## Scoped FRZ / ECAP (no global promotion)

| ID | Finding |
| --- | --- |
| FRZ-REL-03 | UNKNOWN / side-effect safety — scoped L2 only |
| FRZ-REL-04 | Bounded retry + per-attempt auth — B1 truth table + L2 |
| FRZ-REL-05 | Idempotency key scope — unchanged Batch B tests |
| FRZ-REL-06 | Coordinator path — unchanged |
| FRZ-GOV-02 | Real gate evidence added (B5); mocked tests labeled |
| FRZ-BND-03 | No second retry/governance owner |

## Remaining parent obligations

- Batch C: STALLED/ABANDONED, worker takeover, store-unavailable recovery.
- Batches D–E: provider real-infra + TL15.
- Batch A 65-test L2 **not** re-run.

## Publication

| Field | Value |
| --- | --- |
| **Publication SHA** | `development` commit `fix(runtime): Batch B audit remediation for retry class and governance evidence` (verify `git rev-parse HEAD` after push) |
| **Historical 57/57** | Preserved at `60a3ad53b8caf530027500803f3285d55ebda2c2` |
