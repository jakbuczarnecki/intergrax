# Collaborative Work — Enterprise E2E Qualification (COLLAB-WORK-E2E-1)

## 1. Purpose

First enterprise-grade end-to-end qualification of Collaborative Work / Multiplayer MP-1.
Proves that **current** platform capabilities compose correctly through public production
boundaries — without WorkItem / Artifact / Decision / Activity expansion.

Baseline ancestor: `b9de0278411d97e3f935528f2e19da02be335644`.

## 2. User / business flows

| Flow | What the user/agent experiences |
|------|----------------------------------|
| Direct allow | Active member with authority executes a governed mutation once |
| Authority deny | Missing/inactive authority blocks side effect — no fallback |
| Delegation | Delegate acts within delegator-bounded scope |
| Isolation | Tenant/workspace/resource mismatch denied |
| Policy composition | Resource DENY wins over authority/workspace/runtime ALLOW |
| HITL | REQUIRE_HUMAN pauses; canonical approve → scoped grant → execute once |
| Security negatives | Grant replay, stale policy bundle, amplification blocked |
| Durability | Close bundle → independent reopen → same enforcement outcome |

## 3. Canonical architecture used

```text
Actor / Principal
  → Workspace membership (repository, CANONICAL_PRINCIPAL)
  → Authority grant / delegation (repository)
  → Operation policy profile (repository)
  → Collaborative policies (repository)
  → CollaborativeWorkEnforcementGate
  → Runtime meaningful-side-effect policy
  → Policy composition
  → MeaningfulSideEffectAuthorizationBoundary.authorize_and_execute
  → ALLOW | DENY | REQUIRE_HUMAN
  → HumanPauseCoordinator + GovernedContinuationGrantCoordinator (HITL)
  → side-effect execute callback (test probe only)
```

## 4. Platform reuse matrix

| Component | Reused | Notes |
|-----------|--------|-------|
| `resolve_collaborative_work_repositories` | YES | Provider-neutral materialization |
| `CollaborativeWorkEnforcementGate` | YES | Full policy stack |
| `MeaningfulSideEffectAuthorizationBoundary` | YES | Pre-side-effect boundary |
| `HumanPauseCoordinator` | YES | Canonical HITL pause |
| `GovernedContinuationGrantCoordinator` | YES | Scoped grant create/consume |
| `IntegrationProfile` | YES | SQLite / PostgreSQL differ by config only |
| New orchestration engine | NO | — |
| New HITL system | NO | — |
| New provider registry | NO | — |

**E2E composition classification:** `EXTEND_EXISTING_PRODUCTION_COMPOSITION_NARROWLY`

## 5. Provider abstraction

- Core harness: `tests/e2e/collaborative_work/harness/composition.py` — zero vendor branches.
- Vendor setup: `harness/profile_factory.py` + `conftest.py` fixtures only.
- Architecture gate: `test_architecture_gates.py`.

## 6. E2E scenario matrix

See §18 Acceptance matrix.

## 7. PostgreSQL proof

Real backend: `infra/docker/postgresql` (`intergrax-postgresql`, port 5434).

```powershell
$env:INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN="postgresql://intergrax:intergrax@localhost:5434/intergrax"
uv run pytest tests/e2e/collaborative_work/ -m "e2e and integration and network" -q
```

Result: **11 passed** (2026-09-02).

## 8. SQLite proof

```powershell
uv run pytest tests/e2e/collaborative_work/ -m "e2e and not network" -q
```

Result: **24 passed** (2026-09-02).

## 9. HITL proof

Scenarios 7–10 in `test_hitl_e2e.py`:

- REQUIRE_HUMAN pauses before side effect
- `HumanPauseCoordinator.resolve_human_response(APPROVE)` → `create_grant_from_approval`
- Matching grant consumed → execute once
- Replay (wrong run / side_effect_scope_id) blocked
- Policy bundle change invalidates stale grant

## 10. Security negative paths

- Delegation amplification (delegate scope ⊄ delegator authority) → DENY
- Grant replay → DENY, probe count 0
- Unauthorized paths: probe count asserted 0

## 11. CAS / concurrency

- Stale revision → `WorkspaceMembershipRevisionConflict`
- Concurrent PG updates → one winner, one conflict
- Idempotency replay + semantic conflict → typed exceptions

## 12. Restart / recovery

Independent bundle reopen via `reopen_multiplayer_e2e_context` — not same-object reuse.
SQLite + PostgreSQL persist/reopen enforcement tests pass.

## 13. Observability / diagnostics boundary

- CW library path: `PolicyDecision.audit_payload`, enforcement composition layers.
- Host runtime events: not required for library-level E2E composition (below host DIAG boundary).
- No `CollaborativeWorkProblemStore` introduced.

## 14. Evidence map

Recoverable from allowed/denied/HITL paths:

- tenant, workspace, principal, delegation (when used)
- authority scope, resource scope, operation
- policy decision + determining layer
- task_id, run_id, side_effect_scope_id, side_effect_scope_digest
- governed continuation / grant correlation on HITL path

## 15. side_effect_scope_id conclusion

**Classification: `SAFE_BY_EXISTING_DOWNSTREAM_MATCHING`**

E2E proves `resource_scope` and `side_effect_scope_id` are matched independently by
`matches_current_requirement`. Changing scope_id with same resource denies replay.
No bypass observed; no local scope derivation added.

## 16. Limitations

- Does not prove WorkItem / Artifact / Decision / Activity.
- Does not prove multi-process consensus (independent connections only).
- HITL task state is in-memory runtime (CW state is durable).
- LKW adoption out of scope.

## 17. Commands / results

| Command | Result |
|---------|--------|
| `uv run pytest tests/e2e/collaborative_work/ -m "e2e and not network" -q` | 24 passed |
| `uv run pytest tests/e2e/collaborative_work/ -m "e2e and integration and network" -q` | 11 passed |
| `uv run pytest tests/unit/collaborative_work/ -q` | 218 passed |
| `uv run pytest tests/integration/collaborative_work/test_postgresql_repository.py -m "integration and network" -q` | 15 passed |
| `uv run ruff check tests/e2e/collaborative_work/` | clean |

## 18. Acceptance matrix

| Scenario | Backend | Expected | Actual | Side FX | Persist | Evidence | Result |
|----------|---------|----------|--------|---------|---------|----------|--------|
| 1 Direct ALLOW | SQLite+PG | ALLOW | ALLOW | 1 | YES | audit_payload | PASS |
| 2 Authority DENY | SQLite+PG | DENY | DENY | 0 | YES | denial reason | PASS |
| 3 Valid delegation | SQLite+PG | ALLOW | ALLOW | 1 | YES | delegation in audit | PASS |
| 4 Amplification | SQLite+PG | DENY | DENY | 0 | YES | authority deny | PASS |
| 5 Resource scope | SQLite | DENY | DENY | 0 | YES | resource layer | PASS |
| 6 Tenant isolation | SQLite+PG | DENY | DENY | 0 | YES | membership deny | PASS |
| 7 REQUIRE_HUMAN | SQLite | pause | pause | 0 | YES | continuation req | PASS |
| 8 Approval continuation | SQLite+PG | execute once | once | 1 | YES | grant consumed | PASS |
| 9 Grant replay | SQLite | DENY | DENY | 0 | N/A | no match | PASS |
| 10 Stale policy | SQLite | new HITL | new HITL | 0 | N/A | grant cleared | PASS |
| 11 Stale CAS | SQLite+PG | conflict | conflict | 0 | YES | typed exc | PASS |
| 12 Concurrent CAS | PG | 1 win | 1 win | 0 | YES | typed exc | PASS |
| 13 Idempotency | SQLite+PG | replay/conflict | typed | 0 | YES | idem key | PASS |
| 14 Provider failure | N/A | fail closed | IntegrationConfigurationError | 0 | N/A | no fallback | PASS |
| 15 Persist/reopen | SQLite+PG | same outcome | same | 1 | YES | reload | PASS |
| 39 Policy composition | SQLite | DENY | DENY | 0 | YES | determining layer | PASS |

## 19. R1 — canonical HITL regression closure (execution identity lifecycle)

After lifecycle-owned `ExecutionId` hardening (`a63867f396df70b3c8053f15e7d7ce4b2de46ba8`),
`runtime_event_from_task_state` requires active `ExecutionId` via `require_active_execution_id()`.
The governed-continuation grant regression (`test_g5c2b1_governed_continuation_grant.py`) and shared
HITL intake helper (`test_g5b_hitl_resolution.py`) still bound only `run_id`/`attempt_id`.

**Root cause:** `SHARED_TEST_HELPER_DRIFT` — test fixtures invoked intake/HITL runtime emission without
binding canonical `ExecutionId` in the active execution identity ContextVar.

**Repair:** reusable `bound_hitl_test_execution_identity()` helper (public `bind_active_execution_identity`
+ `reset_active_execution_identity`); no production changes; fail-closed invariant preserved.

**Regression (2026-09-02, R1 `bfd084e4`+):**

| Command | Result |
|---------|--------|
| `uv run pytest tests/unit/runtime/human/test_g5c2b1_governed_continuation_grant.py -q` | 15 passed |
| `uv run pytest tests/unit/runtime/human/ -q` | 107 passed |
| `uv run pytest tests/unit/runtime/execution/test_execution_boundary.py tests/unit/runtime/long_running/test_ue_9c_execution_tree_checkpoint.py -q` | 55 passed |
| `uv run pytest tests/e2e/collaborative_work/ -m e2e -q` | 35 passed (24 SQLite + 11 PG when DSN set) |
| `uv run pytest tests/e2e/collaborative_work/ -m "e2e and integration and network" -q` | 11 passed |
| `uv run pytest tests/unit/collaborative_work/ -q` | 218 passed |
| `uv run pytest tests/unit/core/qualification/ -q` | 208 passed |

## 20. Next phase

Independent COLLAB-WORK-E2E-1 SHA audit on committed revision. Do not start real-vendor-wide qualification.
