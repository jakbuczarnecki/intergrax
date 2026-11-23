# SFX R2 — Autonomous Mission Specification (PROPOSED)

**Status:** PROPOSED · binds only after operator R2 adoption and ADR/COMPAT review  
**Ticket:** T-IGX-070-A · **Schema:** [`design/schemas/proposed/sfx.mission_record.v1.schema.json`](../design/schemas/proposed/sfx.mission_record.v1.schema.json)

## 1. Execution modes

| Mode | `execution_mode` | Driver | L2 sandbox ops | Engineering submit |
| --- | --- | --- | --- | --- |
| Manual | `MANUAL` | Human per step | Per separate approval | Human each intent (HO-2) |
| Assisted | `ASSISTED` | Agent proposes; human per milestone | Per approval | Human each intent (HO-2) |
| Bounded autonomous | `BOUNDED_AUTONOMOUS` | Agent within `AutonomyEnvelope` | Pre-approved grant | **PROPOSED:** grant-bound derived intents (see SEF amendment) |

**Invariant SFX-INV-R2-01:** Mode never elevates authority level; L3/production effects remain forbidden in all modes.

## 2. Core records (typed contracts)

| Record | Schema (PROPOSED) | Owner | Persistence |
| --- | --- | --- | --- |
| `MissionRecord` | `sfx.mission_record.v1` | SFX Tier-2 | CW `WorkArtifactVersion` |
| `BusinessGoal` | extends CON-01/03 fields | SFX | CW |
| `OutcomeAcceptance` | embedded in spec / mission | SFX | CW |
| `AutonomyEnvelope` | `sfx.autonomy_envelope.v1` | SFX policy view; **grant authority** = existing DecisionBinding | CW + `approval_ref` |
| `DelegationGrantRef` | pointer to signed grant digest | Platform decision binding; verified by SEF on submit | Decision store |
| `CandidateIteration` | `sfx.candidate_iteration.v1` | SFX view | CW append-only |
| `ScenarioEvaluationVerdict` | `sfx.scenario_evaluation_verdict.v1` | SFX view (data from EvalRunner/evidence) | CW |
| `EscalationRequest` | reuse lifecycle / HITL patterns | Platform HITL | CW |

No new execution FSM authority: product-facing states (`DISCOVERING`, `IMPROVING`, …) are **projections** of artifact versions + SEF job refs.

## 3. Mission lifecycle (normative)

### 3.1 Creation

- **Input:** `tenant_id`, `workspace_id`, `scenario_id`, `execution_mode`, optional `goal_draft_ref`.
- **Output:** `mission_id`, `mission_digest`, `state_projection=CREATED`.
- **Gate:** L0/L1 only until envelope approved.

### 3.2 Goal & acceptance lock

- **Input:** discovery + spec artifacts (CON-01..03).
- **Output:** pinned `goal_digest`, `acceptance_suite_digest`, `forbidden_outcomes[]`.
- **Fail-closed:** missing measurable acceptance → `NEEDS_INPUT`, no L2.

### 3.3 Envelope approval (human)

- **Input:** `AutonomyEnvelope` draft bound to `goal_digest`, `blueprint_digest`, budgets.
- **Output:** `approval_ref`, `DelegationGrantRef` (PROPOSED cross-track), `expires_at`.
- **Invariants:** `max_iterations`, `max_parallel_candidates`, `cost_budget`, `deadline`, `allowed_asset_refs[]`, `deny_egress=true` (L2 default).

### 3.4 Iteration loop (bounded)

For each `candidate_revision`:

1. **Plan** — SFX writes `ImprovementHypothesis` (typed sub-record) or baseline plan; no repo side effects.
2. **Build** — `EngineeringHandoffRequest` with `mode`; SEF job under grant validation (when authorized).
3. **Qualify** — SAST/deps/tenant/network gates (SFX-P9).
4. **Execute** — canonical runtime only; capture `run_id`, evidence refs.
5. **Evaluate** — `ScenarioEvaluationVerdict` per criterion: `PASS|FAIL|INSUFFICIENT_EVIDENCE|BLOCKED_POLICY|BLOCKED_ENVIRONMENT`.
6. **Select** — update champion only if non-regressing on security + regression suite.
7. **Stop or continue** — see §4.

### 3.5 Revocation

- Grant revocation → immediate `PAUSED`; in-flight jobs: SEF cancel + DLQ per HO-5; no new submits.
- **Evidence:** immutable `REVOKED` lifecycle event with actor + digest.

## 4. Stopping rules

| Code | Condition | Action |
| --- | --- | --- |
| STOP-OK | All acceptance criteria `PASS` | `ACCEPTANCE_MET` → champion freeze |
| STOP-BUDGET | Cost/iteration/time exceeded | `BOUND_REACHED`, escalate |
| STOP-SAFE | Safety invariant violation | `ESCALATED`, preserve champion |
| STOP-STAGNATE | No metric delta N rounds | `BOUND_REACHED` |
| STOP-REVOKE | Grant revoked/expired | `PAUSED` |
| STOP-SCOPE | New external system/side effect | `ESCALATED` — new grant required |

**Invariant SFX-INV-R2-02:** No silent budget or scope extension.

## 5. Budgets & parallelism

| Budget | Enforced by | On exceed |
| --- | --- | --- |
| `cost_budget` | SFX mission driver + SEF lease accounting | STOP-BUDGET |
| `max_iterations` | SFX | STOP-BUDGET |
| `deadline` | wall clock | STOP-BUDGET |
| `max_parallel_candidates` | SFX scheduler (references only) | queue or deny |

Parallel candidates are **logical**; engineering parallelism is SEF-owned within lease limits.

## 6. Failure recovery

| Failure | Idempotency key | Recovery |
| --- | --- | --- |
| Crash mid-iteration | `mission_id+candidate_revision+phase` | Resume from last CW version + SEF job ref |
| Duplicate handoff | `handoff_request_digest` | Reject duplicate (HO-4 pattern) |
| Partial eval | `run_id` | Re-run eval only; no re-build unless verdict requires |

## 7. Human escalation

`EscalationRequest` classes: `MISSING_BUSINESS_DECISION`, `SCOPE_INCREASE`, `POLICY_CHANGE`, `STAGNATION`, `BUDGET_EXHAUSTED`, `SAFETY_VIOLATION`.

Studio/MCP/CLI parity: same op catalog (`sfx.mission.pause`, `sfx.mission.resume`, `sfx.mission.revoke_grant` — PROPOSED op IDs).

## 8. Evidence (immutable)

Each iteration appends: `goal_digest`, `blueprint_digest`, `engineering_job_id`, `evaluation_suite_digest`, `evidence_refs[]`, `policy_or_grant_ref`, `artifact_digest`, `code_sha`, `run_id`, cost counters.

**Invariant SFX-INV-R2-03:** Business pass never inferred from code-build pass alone.