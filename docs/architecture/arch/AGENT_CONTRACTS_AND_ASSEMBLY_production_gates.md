# AGENT_CONTRACTS_AND_ASSEMBLY — production gates (§40+)

**Parent hub:** [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](../AGENT_CONTRACTS_AND_ASSEMBLY.md)

# 40. Production Reliability, Safety, Persistence, and Release Gates

**Purpose:** Close the gap between **canonical architecture** (§13–§39) and **safe production coding**. New Tier-2 agents for mutating workloads MUST satisfy **ACP-PROD-*** platform modules **and** **ACP-CLOSE-PROD-*** host depth before `production_mode` promotion — both tracks **Done** at platform level (2026-06-13).

**Cross-domain:** [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) · [`OBSERVABILITY.md`](OBSERVABILITY.md) · [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.12 tools · [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) eval gates · §20 lifecycle governance

**Status:** Normative spec — **platform implemented** (ACP-PROD-1..12 **Done**); **host depth + prod evidence** (**ACP-CLOSE-PROD-1..8 Done**); CI aggregate **`check_agent_acp_close_ci.py` green**; per-agent promotion via §40.15 scoreboard thresholds.

---

## 40.1 Checkpoint, resume, and replay

Builds on §37.2 `state_delta` and Nexus task checkpoints.

### 40.1.1 Checkpoint scopes

| Scope | Store | Contents | Owner |
|-------|-------|----------|-------|
| **Step checkpoint** | Agent run store | `{run_id, step_index, acp.state.v1, side_effect_ledger[], trace_cursor}` | Harness after successful kernel cycle |
| **Task checkpoint** | Nexus checkpoint DB | Graph cursor, `SharedContextView` snapshot ref, node run_ids | NexusLoop on pause/HITL/node complete |
| **Session checkpoint** | Optional host store | User thread metadata | Tier-3 |

### 40.1.2 When checkpoint is written

```text
After HarnessKernel.execute_step completes successfully AND:
  - state_delta applied + _version bumped
  - all tool/RAG/LLM calls for step recorded in side_effect_ledger
  - policy post-check passed (or step marked failed with no partial commit — see 40.1.3)

Default: checkpoint_every_step = true (§29.2.1 AgentExecutionOptions)
Override: long steps may set checkpoint_every_step=false only for read-only steps declared on tool contract
```

### 40.1.3 Transaction boundary (step vs side effects)

**Normative rule — step checkpoint is transactional with respect to agent state, not always with external systems:**

| Phase | On failure | State | External side effect |
|-------|------------|-------|----------------------|
| Pre-tool policy deny | Roll back step intent | No `_version` bump | None executed |
| Tool in flight | See §40.2 idempotency | Step not checkpointed | At-least-once + idempotency key |
| Tool succeeded, state merge fails | **Critical** — mark step `INTERNAL_ERROR`; do not advance `step_index` | Replay from last checkpoint | Rely on tool idempotency §40.2 |
| Tool succeeded, checkpoint write fails | Retry checkpoint write; if exhausted → HITL + alert | Same | Side effect may exist — ledger records `committed_externally=true` |

**Anti-pattern:** advancing `step_index` without durable checkpoint when `checkpoint_every_step=true`.

### 40.1.4 Resume after crash

```text
1. Load last step checkpoint for run_id (or request.state if client-supplied and version valid)
2. If request.state._version < checkpoint._version → VALIDATION_FAILED unless force_resume governance flag
3. Rebuild EffectiveAgentRunEnvironment from host profile (not from stale in-memory)
4. Replay side_effect_ledger: skip tools with status=committed matching idempotency_key
5. Continue Agent.run() loop from step_index (not from zero)
```

### 40.1.5 Replay (debug / eval)

| Mode | Behavior |
|------|----------|
| **Trace replay** | Read-only reconstruction from `AgentRunTrace` — no tool re-invoke |
| **Deterministic replay** | Lab only; mock gateways; same inputs → compare StepOutcome |
| **Production replay** | **Forbidden** for mutating tools without explicit `dry_run` + new run_id |

**Plan:** ACP-PROD-1 + ACP-CLOSE-PROD-1..2 — `checkpoint_store.py` + `acp_checkpoint_host_wiring.py` + harness task enricher (**Done** on all Tier-3 harness hosts).

---

## 40.2 Idempotency for side effects

Required for **mutating** tools in both immediate and declarative modes (§32.8).

### 40.2.1 Identifiers

```text
SideEffectRecord:
    side_effect_id: str              # uuid — unique per attempted effect
    idempotency_key: str             # stable business key — dedupe scope
    run_id: str
    step_index: int
    kind: tool | rag_write | llm_cache_write | artifact_publish
    target: str                      # tool_id or resource
    status: pending | committed | failed | compensated
    committed_at: datetime | null
    external_ref: str | null         # provider message id, ticket id, etc.

StepActionRequest (declarative — §32.8):
    ... existing fields ...
    idempotency_key: str             # REQUIRED for mutating kind
    side_effect_id: str | null        # assigned by harness if omitted
```

**Key generation (normative default):**

```text
idempotency_key = hash(run_id, step_index, kind, target, canonical_args)
```

Authors MAY supply explicit keys for business-level dedupe (e.g. `email:{case_id}:{template_id}`).

### 40.2.2 Delivery semantics

| Tool class | Semantics | Harness behavior |
|------------|-----------|------------------|
| **Read-only** | At-most-once (retry safe) | Retry on transient failure |
| **Mutating idempotent** | **Effective exactly-once** via key + store | Dedupe on retry/resume |
| **Mutating non-idempotent** | **Blocked in STRICT prod** unless tool declares idempotency support | Register gate ACP-PROD-2 |

**Dedupe policy:** `ReliabilityProfile.idempotency_store` (see [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md)) — TTL ≥ max task duration.

### 40.2.3 Retry policy (side effects)

```text
SideEffectRetryPolicy:
    max_attempts: int
    backoff_ms: list[int]
    retriable_codes: list[AgentRunErrorCode]   # TOOL_FAILED, LLM_FAILED
    non_retriable: POLICY_DENIED, VALIDATION_FAILED
```

Retries MUST reuse same `idempotency_key`. **Plan:** ACP-PROD-2 (**Done** — `SideEffectLedger`).

---

## 40.3 Tool transaction and compensation model

Extends tool allowlists §30 with **tool capability metadata** on `ToolRegistry` entries.

### 40.3.1 Tool classification (required metadata)

```text
ToolExecutionProfile:
    tool_id: str
    mutability: read_only | mutating
    reversibility: none | compensatable | manual
    requires_approval: bool              # HITL pre-invoke when true
    supports_dry_run: bool
    requires_idempotency_key: bool       # mandatory for mutating in STRICT
    compensation_tool_id: str | null     # e.g. email.send → email.recall (if exists)
    max_retry: int
    timeout_ms: int
```

### 40.3.2 Execution phases

```text
1. classify tool via ToolExecutionProfile
2. if requires_approval → pause_hitl before invoke
3. if supports_dry_run and execution_options.dry_run → simulate, no commit
4. invoke with idempotency_key
5. on step failure after commit → compensation policy:
     - compensatable + handler registered → enqueue compensation_tool
     - manual → HITL ticket + SideEffectRecord.status=failed
     - read_only → no compensation
```

### 40.3.3 Compensation

```text
CompensationRequest:
    original_side_effect_id: str
    compensation_tool_id: str
    args: dict
    idempotency_key: str                # distinct key derived from original
```

Compensation runs through same gateways; recorded in trace. **Plan:** ACP-PROD-3 + ACP-CLOSE-PROD-5 (**Done** — enqueue, durable `CompensationQueueStore`, `drain_pending_compensation_jobs`).

---

## 40.4 Retry, timeout, and circuit breaker policy

Agent session inherits **`ReliabilityProfile`** from host (circuit breaker, timeouts).

| Layer | Policy source | Applies to |
|-------|---------------|------------|
| Step loop | `AgentExecutionOptions.max_steps`, budgets §32.6 | Whole `run()` |
| Tool invoke | `ToolExecutionProfile.timeout_ms`, retry | Per tool call |
| LLM | `LLMProfile` adapter timeouts | Per `llm_calls` record |
| Circuit breaker | `ReliabilityProfile` | Integration slugs |
| Nexus task | `OrchestrationProfile.max_run_retries` | Task-level only |

**Rule:** agent MUST NOT implement private retry loops for tools — use harness retry + idempotency.

**Plan:** ACP-PROD-4 — `AgentSessionReliability` in `HarnessKernel.execute_step` (**Done**).

---

## 40.5 Concurrency model for shared context

Extends §34 `SharedContextView`.

### 40.5.1 Parallelism rules

| Context | Parallel agents allowed? | Rule |
|---------|-------------------------|------|
| Same `run()` session | **No** | Single-threaded step loop |
| Same Task, different graph nodes | **Yes** if graph spec declares parallel edges | Nexus scheduler |
| Same shared key | **Controlled** | optimistic locking §40.5.2 |

### 40.5.2 SharedContextView concurrency

```text
SharedContextEntry:
    key: str
    value: JSONValue
    version: int                       # monotonic per key
    updated_by: str                   # run_id or node_id
    visibility: node | subgraph | task

SharedContextView:
    get(key) -> (value, version)
    publish(key, value, *, expected_version: int | null) -> PublishResult
        # expected_version match → atomic write, version++
        # mismatch → CONFLICT — caller replan or HITL
    compare_and_swap(key, expected_version, new_value) -> bool
```

**Defaults:**

- `publish` without `expected_version` allowed only in **BALANCED/EXPLORATORY**; **STRICT** requires CAS for mutating keys.
- Artifact keys use content-addressed ids to reduce collision.

### 40.5.3 Conflict resolution

| Strategy | When |
|----------|------|
| **Last-write-wins** | EXPLORATORY lab only |
| **Optimistic lock + replan** | Default BALANCED |
| **HITL on conflict** | STRICT prod shared mutable keys |

**Plan:** ACP-PROD-5 (**Done** — per-key `publish` / `compare_and_swap` on `SharedContextView`).

---

## 40.6 Artifact contract

Replace loose `artifacts: list[str]` on `AgentRunResult` with typed refs.

```text
ArtifactRef:
    schema_version: str = "artifact_ref.v1"
    artifact_id: str
    type: str                           # report, attachment, structured_json, ...
    uri: str                            # s3, file, memory blob ref — no secrets in uri query
    mime_type: str | null
    provenance: ArtifactProvenance
    retention_class: str                # maps to host retention policy §40.8
    sensitivity: public | internal | confidential | pii
    checksum: str | null                 # sha256
    size_bytes: int | null
    created_at: datetime
    trace_id: str
    run_id: str
    step_index: int | null

ArtifactProvenance:
    created_by_agent_id: str
    created_by_tool_id: str | null
    source_side_effect_id: str | null
```

**Rules:**

- Harness registers artifacts when tools return artifact payloads or `StepOutcome.artifacts` lists ids.
- Two agents publishing same logical artifact → distinct `artifact_id`; dedupe via `checksum` optional at app layer.
- **Plan:** ACP-PROD-6 (**Done** — `intergrax/contracts/artifact_ref.py`, `AgentRunResult.artifact_refs`).

---

## 40.7 Threat model (agent layer)

Formal requirements — enforcement via policy, gateways, CI (§40.10).

| Threat | Vector | Mitigation | Verify |
|--------|--------|------------|--------|
| **Prompt injection** | User/metadata in `input` | `prompt_security`, guardrails §39, org envelope | pre-LLM scan |
| **Tool injection** | Adversarial tool args / skill payloads | `tool_injection_defense`, schema validation | TOOL_FAILED + policy |
| **RAG poisoning** | Malicious corpus docs | retrieval poisoning defense, collection ACL | RAG trust tier |
| **Memory poisoning** | Cross-session write | namespace isolation §30.3, tenant scope | memory namespace test |
| **Cross-tenant leakage** | Wrong namespace / shared key | tenant_id on all stores, STRICT isolation | integration test |
| **Secret exfiltration** | Prompt/trace/tool args | redaction §40.8, no secrets in state | lint + trace audit |
| **Unsafe tool chaining** | agent chains mutating tools without review | policy rules, HITL on risky profiles | policy test |
| **Malicious document content** | RAG/intake files | sandbox parse, modality scanners | ingest pipeline |
| **Agent-to-agent data leak** | Over-broad `shared_context.publish` | visibility + CAS §40.5 | graph test |
| **SDK bypass** | Direct vendor import in Tier-2 | tier boundary, `check_agents_vendor_imports.py` | CI |
| **Org rule bypass** | configure_run widen in STRICT | §39.4 STRICT deny | **Closed** — ACP-CLOSE-ORG-1 |

**Plan:** ACP-PROD-7 (**Done** — `scripts/check_agent_threat_model.py`).

---

## 40.8 Privacy, retention, and redaction

Data governance for memory, RAG, trace, prompts.

### 40.8.1 Classification

```text
DataClassification:
    level: public | internal | confidential | pii | secret
    fields: list[str]                   # optional path patterns in metadata/state
```

Host **`ObservabilityProfile`** + **`MemoryProfile`** declare default classification per tenant.

### 40.8.2 Rules (normative)

| Data plane | Requirement |
|------------|-------------|
| **Trace (Plane B)** | PII fields hashed or truncated; raw prompts optional per `store_raw_prompts` flag (default **false** prod) |
| **Trace (Plane A)** | Summaries only; join via `trace_id` |
| **Memory** | Tenant namespace; retention TTL; right-to-delete API on host |
| **RAG** | Collection ACL; no cross-tenant retrieval |
| **Artifacts** | `sensitivity` on `ArtifactRef`; retention_class enforced at store |
| **Export/audit** | Sanitized export bundle; secrets never included |

### 40.8.3 Redaction

- Intake redaction before `AgentRunRequest.metadata` persisted.
- `AgentStepRecord.state_snapshot` — redacted view of `acp.state.v1`.
- `PolicyVerdictRecord.message` — no raw user content.

**Plan:** ACP-PROD-8 (**Done** — `privacy_redaction.py` on policy verdict reasons).

---

## 40.9 Evaluation and release gates

Mandatory before **production_mode** promotion (extends §20 lifecycle).

### 40.9.1 Required suites per agent

| Suite | Purpose | Gate |
|-------|---------|------|
| **Golden** | Expected output on fixed inputs | Block promotion on diff |
| **Regression** | Prior release corpus | No capability regression |
| **Scenario** | Org playbook / UC-* flows §35 | Scenario pass rate |
| **Tool failure** | Injected TOOL_FAILED / timeout | Graceful terminal_reason |
| **Policy violation** | Org envelope STRICT §39 | Expected POLICY_DENIED paths |
| **Cost regression** | Token/$ budget vs baseline | Block if > threshold |
| **Latency regression** | p95 step duration | Warn/block per profile |
| **Trace completeness** | All steps have tool/LLM records when used | §40.10 |
| **Evidence / hallucination** | RAG citations required when configured | CVL hooks §CRITIC |

Register in **Evaluation registry**; wired via Tier-3 host before roster `production_mode`.

### 40.9.2 Release gate workflow

```text
dev → eval suites green → staging shadow → certification §20 → production_mode
```

**Plan:** ACP-PROD-9 (**Done** — `scripts/check_agent_release_gates.py`).

---

## 40.10 CI conformance matrix

Normative CI checks before merge to agent roster (extends §45).

| ID | Check | Script / test |
|----|-------|---------------|
| CI-01 | Agent contract fields complete | `check_agents_lifecycle_metadata.py` |
| CI-02 | No vendor SDK in Tier-2 | `check_agents_vendor_imports.py` |
| CI-03 | No `os.environ` in agents | lint / dedicated script |
| CI-04 | UAEP / run path — post-LEG fleet migration | `check_agent_fleet_migration.py` · `check_agent_acp_close_ci.py` |
| CI-05 | Capability declared matches class | `check_agent_pattern_conformance.py` (ACP-13) |
| CI-06 | Capability routing integration | ACP-CON-6 test |
| CI-07 | state_delta merge unit | ACP-CON-2 test |
| CI-08 | Policy denial paths | org fixture test |
| CI-09 | Side-effect mode exclusivity | ACP-CON-3 test |
| CI-10 | LLM routing within profile | ACP-LLM-1 test |
| CI-11 | Memory namespace isolation | MEM + agent integration |
| CI-12 | Trace completeness schema | ACP-OBS-1 test |
| CI-13 | Idempotency key on mutating tool fixtures | ACP-PROD-2 test |
| CI-14 | Checkpoint resume smoke | ACP-PROD-1 test |
| CI-15 | Release eval suites | ACP-PROD-9 |
| CI-16 | Production readiness scoreboard blockers | `check_agent_acp_close_ci.py` |
| CI-17 | ACP-AP-02 — no tool loops in graph orchestration | `check_agent_acp_ap02_tool_loop_boundary.py` |
| CI-18 | Token budget contract — kernel metering + no agent budget `state_delta` | `check_agent_token_budget_contract.py` |

**Rule:** new agent PR MUST declare which CI rows apply; all applicable rows green.

**Plan:** ACP-PROD-10 (**Done** — `scripts/check_acp_ci_conformance_matrix.py`).

---

## 40.11 Versioning and migration policy

All runtime contracts carry **`schema_version`**. Breaking changes require ADR + migration window.

| Contract | Current | Compatibility rule |
|----------|---------|-------------------|
| `AgentRunRequest` | `agent_run.v1` | Readers accept v1; writers emit latest |
| `AgentRunResult` | `agent_run.v1` | Same |
| `AgentRunTrace` | `agent_run_trace.v1` | Trace consumers ignore unknown step fields |
| `acp.state.v1` | embedded in state | `_version` int; merge patch only §37.2 |
| `StateDelta` | merge patch | No schema field — keys only |
| `ArtifactRef` | `artifact_ref.v1` | — |
| `SideEffectRecord` | `side_effect.v1` | Required for resume |
| `OrganizationalPolicyEnvelope` | `org_policy_envelope.v1` | Host reload on change |

**Migration strategy:**

1. Additive fields — minor bump, old readers ignore.
2. Semantic change — new schema_version; adapter layer for one release (`intergrax/contracts/migrations/agent_run_v1_to_v2.py`).
3. Deprecation — `DeprecationWarning` in harness one release; remove with ADR.

**Plan:** ACP-PROD-11 (**Done** — `intergrax/contracts/migrations/registry.py` + `check_contract_schema_versions.py`).

---

## 40.12 Production readiness checklist

Before **`production_mode`** on roster entry — all MUST be true:

```text
□ §29–§32 run/on_next_step/advance_step/kernel path used — not legacy AgentEngine-only
□ §37 enums for errors and terminal_reason
□ §40.1 checkpoint + resume tested for mutating agent
□ §40.2 idempotency keys on all mutating tools in agent tests
□ §40.3 ToolExecutionProfile declared for each used mutating tool
□ §40.6 ArtifactRef populated — not raw paths only
□ §40.7 threat mitigations verified for agent's data classes
□ §40.8 retention/redaction profile wired on host
□ §40.9 eval suites registered and green on staging
□ §40.10 applicable CI rows green
□ §40.11 schema_version declared on contract payloads
□ §39 org envelope tested if UC-11 applies
□ §20 lifecycle certification recorded
```

Waivers require ADR + operator sign-off — not silent skip.

**Aggregated view:** use **Agent Production Readiness Scoreboard** §40.15 (`ACP-PROD-12`) — single report per agent instead of manual checklist hunting.

---

## 40.15 Agent Production Readiness Scoreboard

**Purpose:** One typed **`AgentProductionReadinessReport`** per agent — 10 dimensions scored **0–100%**, rolling up to `overall_pct` and `production_eligible_recommendation`. Replaces scattered gate knowledge for roster promotion decisions.

**Plan:** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) §6.1az · implementation **ACP-PROD-12**.

| Dimension | Canon |
|-----------|--------|
| Contract | §12 · ACP-CON-4 |
| Runtime | §13 · §32 · §32.0 · fleet migration Wave 8 |
| Policy | §37.7 · §39 |
| Observability | §31 |
| Checkpointing | §40.1 |
| Idempotency | §40.2 |
| Security | §40.7 |
| Evaluation | §40.9 |
| Lifecycle | §20 |
| Capability routing | §37.6 |

**Production roster (mutating / customer-facing):** `overall_pct ≥ 90` and no scored dimension below **80%** (unless `not_applicable`) — plus ACP-PROD-1..3 and ACP-PROD-9..10 **Done** in code. Thresholds are **not negotiable** without ADR.

**Fleet migration:** Wave **8** (`ACP-MIG-*`) must reach Runtime **100%** roster-wide before declaring ACP-LEG-2 Done — see plan fleet tracker.

---

## 40.13 Maturity gate summary

| Milestone | Spec completeness | Allowed work |
|-----------|-------------------|--------------|
| **Architecture canon** | §13–§36 | Design, scaffold, read-only agents |
| **Pre-implementation contracts** | + §37–§39, §32.0 | Typed contracts + READ/UPDATE/DECIDE author kit, lab agents |
| **Production coding** | + §40 implemented (ACP-PROD) | Mutating prod agents, org simulation prod |
| **Roster production_mode** | + §40.9 gates green | Customer-facing deployment |

**Audit scores (2026-06-13 — post ACP-FINISH):**

| Dimension | Score |
|-----------|--------|
| Conceptual architecture | **10/10** ✓ |
| Platform implementation (ACP waves 0–8 + ACP-FINISH) | **9.5/10** ✓ |
| Architecture ↔ code doc sync | **10/10** ✓ (ACP-FINISH-DOC-1) |
| Mutating agents production-ready | **Done** — §40.12 + ACP-CLOSE-PROD-* + ACP-TOK-* + STRICT configure_run deny + compensation queue + CI-1/3 + UC-11 product golden **Done** |

### 40.13.1 Audit acceptance (2026-06)

**Accepted as target canon** — architecture §13–§40 and plan register ACP-* are **decision-complete and implementation-complete** for token budget depth (§25.4–§25.5). Further architecture iteration MUST be driven by **implementation gaps** (ADR + plan row), not open-ended doc expansion.

| Decision | Verdict |
|----------|---------|
| Adopt §13–§39 execution model (`run` / `on_next_step` / `advance_step` / `HarnessKernel` / `NexusLoop`) | **Yes** — **delivered** |
| Adopt §40 production gate for mutating / customer-facing agents | **Yes** — **delivered** |
| Update implementation plan from this canon | **Yes** — Phase ACP · ACP-CLOSE · **ACP-FINISH Done** (2026-06-13) |
| Platform ACP modules (ACP-DX through ACP-PROD-12) | **Done** (2026-06-11) |
| Token metering + limits + reactions (§25.4–§25.5) | **Yes** — **delivered** (ACP-TOK-1..3 · ACP-TOK-CI) |
| Declare mutating agents **production-ready** | **Yes** — when host passes ACP-PROD + ACP-CLOSE-PROD + APP-PROD gates |

**Clarification (2026-06-13):** §25.4–§25.5 closed via ACP-FINISH; AUDIT-IDEAL-19.1/20.1/31.1 **Done** — §12–§20 layer complete. Nexus `RunBudget` graph env cap remains COST-1 **Partial** (does not block per-agent enforcement).

**Next work (non-blocking):** COST-1 graph cap · per-roster `production_mode` promotion (§40.15) · gate maintenance §6.1.

---

## 40.14 Plan register (ACP-PROD)

| ID | Deliverable |
|----|-------------|
| ACP-PROD-1 | Checkpoint store + resume/replay semantics |
| ACP-PROD-2 | Side-effect idempotency ledger + dedupe |
| ACP-PROD-3 | ToolExecutionProfile + compensation |
| ACP-PROD-4 | ReliabilityProfile in kernel (retry/CB/timeout) |
| ACP-PROD-5 | SharedContextView CAS + conflict policy |
| ACP-PROD-6 | `ArtifactRef` contract |
| ACP-PROD-7 | Threat model CI + doc cross-refs |
| ACP-PROD-8 | Privacy/redaction on trace/memory |
| ACP-PROD-9 | Release eval gates + certification script |
| ACP-PROD-10 | CI conformance matrix automation |
| ACP-PROD-11 | Schema version registry + migration adapters |
| ACP-PROD-12 | Agent Production Readiness Scoreboard — §40.15 |
| ACP-MIG-1..7 | Fleet migration program — plan Wave 8 |
| ACP-DOC.9 | This section §40 |
| ACP-DOC.13 | Wave 8 + scoreboard plan — §6.1az |

---
