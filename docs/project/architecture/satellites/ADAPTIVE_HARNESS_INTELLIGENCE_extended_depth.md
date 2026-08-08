# ADAPTIVE_HARNESS_INTELLIGENCE — §8+ extended architecture

**Parent hub:** [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](../ADAPTIVE_HARNESS_INTELLIGENCE.md)

## 8. Target architecture overview

### 8.1 Logical placement in four-tier model

```text
Tier-0  Platform catalogs (tools, skills, LLM, integrations)
           ↑ resolved by profile versions
Tier-1  Nexus Runtime + Adaptive Control Plane (NEW)
           ↑ consumes signals from runs
Tier-2  Agents (bounded local loops; optional bandit hints via profile)
           ↑
Tier-3  Applications (AdaptiveProfile weights, business outcome hooks)
```

### 8.2 Adaptive Control Plane — box diagram

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                     ADAPTIVE CONTROL PLANE (Tier-1)                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐               │
│  │ Signal      │   │ Process      │   │ Adaptation      │               │
│  │ Collector   │   │ Pattern      │   │ Engine          │               │
│  │             │   │ Miner        │   │ (bandit/rules)  │               │
│  └──────┬──────┘   └──────┬───────┘   └────────┬────────┘               │
│         │                 │                     │                         │
│         └─────────────────┴─────────────────────┘                         │
│                               │                                           │
│                               ▼                                           │
│                    ┌──────────────────────┐                               │
│                    │ Proposal Builder     │                               │
│                    │ → AdaptiveLoopProposal│                              │
│                    └──────────┬───────────┘                               │
│                               │                                           │
│                               ▼                                           │
│                    ┌──────────────────────┐                               │
│                    │ Governance Gate        │◄── adaptive_governance.py   │
│                    │ + Capability Graph     │◄── capability_graph_*       │
│                    │ + Human approval (opt) │◄── HITL / ops workflow      │
│                    └──────────┬───────────┘                               │
│                               │                                           │
│                               ▼                                           │
│                    ┌──────────────────────┐                               │
│                    │ Adaptation Executor    │                               │
│                    │ shadow → canary → apply│                              │
│                    └──────────┬───────────┘                               │
│                               │                                           │
│                               ▼                                           │
│                    ┌──────────────────────┐                               │
│                    │ Verification Loop      │◄── eval registry trends   │
│                    │ + auto-rollback          │◄── regression suites      │
│                    └──────────────────────┘                               │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
         ▲                                      │
         │ trace, metrics, eval, cost, HITL     │ mutates (versioned)
         │                                      ▼
    Nexus Runtime                         Profile Version Store
    AgentEngine                           ApplicationEnvironmentProfile
    ToolRuntime                           RagProfile / OrchestrationProfile
    PolicyEngine                          RuntimePolicyBundle fragments
```

### 8.3 Dual-loop integration (canon §9)

| Loop | AHI touchpoint | Sync/async |
|------|----------------|------------|
| **Global Nexus Loop** | Post-task signal emit; periodic proposal batch | Async scheduler |
| **Local Agent Loop** | Read bandit weights for RAG tier / tool order | Sync read-only |

**Rule:** `AdaptationEngine.propose()` MUST NOT run inside `NexusLoop` iteration hot path.

---

## 9. Adaptive Control Plane — component specification

### 9.1 SignalCollector

**Responsibility:** After each completed run (or batch window), assemble `HarnessOutcomeSignal`.

**Inputs:**

| Source | Fields extracted |
|--------|------------------|
| `RunTraceWriter` / persisted run | duration, step count, tool/LLM counts |
| `export_run_metrics()` | behavioral ratios, modality counters |
| `ExecutionGuard.evaluate_run()` | regression flags |
| Online/shadow evaluation | quality score, pass/fail |
| Cost subsystem | normalized cost vs budget |
| HITL subsystem | intervention count, pause duration |
| Tier-3 webhook (optional) | `business_outcome` float |

**Outputs:** `HarnessOutcomeSignal` persisted to `SignalStore` (SQLite or file-backed v1).

**Trigger:** Hook on task completion + optional cron aggregation.

**Non-goals:** Real-time streaming ML feature store (future enhancement).

---

### 9.2 AdaptationEngine

**Responsibility:** Transform signal history into ranked `AdaptiveLoopProposal` list.

**Sub-engines:**

| Sub-engine | Loop kind | Method |
|------------|-----------|--------|
| `RoutingTuningEngine` | `ROUTING_TUNING` | Contextual bandit over model/RAG tier arms |
| `ExecutionStrategyEngine` | `EXECUTION_STRATEGY_TUNING` | Rule + SPC on step/retry/parallel metrics |
| `PolicyLearningEngine` | `POLICY_LEARNING` | Eval adversarial + tool abuse signals → deny list deltas |
| `EvaluationFeedbackEngine` | `EVALUATION_FEEDBACK` | Benchmark regression → re-eval triggers (observe only) |

**State:** `BanditStateStore` per `(tenant_id, task_class, arm_id)`.

**Constraints:**

- Respect `AdaptiveLoopEnvelope.max_delta_percent`.
- Respect `cooldown_seconds` between proposals for same `loop_id`.
- Never propose changes exceeding registry compatibility (pre-check via capability graph).

---

### 9.3 ProposalBuilder

**Responsibility:** Wrap engine output in existing `AdaptiveLoopProposal` contract:

```python
# Existing contract — intergrax/runtime/architecture/adaptive_governance.py
AdaptiveLoopProposal(
    envelope=AdaptiveLoopEnvelope(...),
    proposed_change_summary="...",
    human_approver_id="...",       # required for POLICY_LEARNING
    evaluation_signal_id="...",    # links to HarnessOutcomeSignal
)
```

**Additional metadata (new):** `ProfileVersionDraft` attached as opaque payload validated by executor.

---

### 9.4 Governance Gate (existing + extensions)

**Stage 1 — Envelope validation:** `evaluate_bounded_adaptive_loop(proposal)` (existing).

**Stage 2 — Capability graph impact:** `evaluate_capability_graph_compatibility()` for affected nodes.

**Stage 3 — Authority routing:**

| `AdaptiveAuthorityLevel` | Behavior |
|--------------------------|----------|
| `OBSERVE_ONLY` | Log only; no executor invocation |
| `RECOMMEND` | Ops report + optional auto-shadow if tenant enables |
| `AUTO_WITH_HUMAN_GATE` | Block until `human_approver_id` confirms via HITL/ops API |

**Stage 4 — Regression pre-check:** Golden scenario smoke before shadow allocation.

---

### 9.5 AdaptationExecutor

**Responsibility:** Materialize approved proposals as new profile versions and shift traffic pointers.

**Stages:**

```text
SHADOW    → run candidate profile on shadow eval metadata (existing hook)
CANARY    → percentage or tenant-allowlist via ApplicationEnvironmentProfile
APPLY     → atomic pointer swap in ProfileVersionStore
ROLLBACK  → restore previous pointer on verification failure
```

**Mutatable artifacts (v1):**

| Artifact | Example change |
|----------|----------------|
| `OrchestrationProfile` | `max_parallel_nodes`, retry policy name |
| `RagProfile` | `route_mode`, `deep_query_min_words`, retriever weights |
| `LLMProfile` routing table | model selection per task class |
| `RuntimePolicyBundle` fragment | tool deny list tightening (policy learning) |

**Immutable:** Agent source code, Tier-0 catalog entries (only references change).

---

### 9.6 VerificationLoop

**Responsibility:** Post-apply monitoring over SLO window (default: 7 days or N runs).

**Checks:**

1. Evaluation registry trend — candidate utility ≥ baseline + `min_improvement_delta`.
2. No increase in `ExecutionGuard` regression rate beyond threshold.
3. Cost within budget envelope.
4. Security adversarial suite still green.

**Failure:** Auto-rollback + incident event + block further auto-apply for loop kind.

---

### 9.7 ProcessPatternMiner

**Responsibility:** Offline job on trace event sequences.

**Algorithm (v1):** PrefixSpan or simple n-gram frequency on:

```text
(task_class, agent_id, tool_id, hitl_pause, outcome=success)
```

**Output:** `ProcessPatternProposal`:

| Field | Description |
|-------|-------------|
| `pattern_id` | Stable hash of sequence |
| `support_count` | Occurrences in window |
| `avg_utility` | Mean U for runs matching pattern |
| `suggested_action` | `CREATE_SKILL_DRAFT`, `TUNE_ROUTING`, `DOCUMENT_RUNBOOK` |
| `evidence_run_ids` | Sample for human review |

**Tier handoff:** Skill creation uses `python -m intergrax.scaffold new-skill` — human/agent author completes Tier-2 work.

---

### 9.8 ProfileVersionStore

**Responsibility:** Git-like versioning for harness profiles.

```text
ProfileVersionRecord:
  version_id: str          # semver or ulid
  artifact_type: enum      # orchestration | rag | llm_routing | policy_fragment
  artifact_payload: dict   # validated Pydantic model dump
  parent_version_id: str | null
  created_by: str          # proposal_id or human operator
  rollback_of: str | null
  status: draft | shadow | canary | active | retired
```

**Storage v1:** SQLite under `build/adaptive_harness/` (gitignored) + export to ops artifacts.

---

### 9.9 AdaptationScheduler

**Responsibility:** Cron/worker triggering:

| Job | Cadence |
|-----|---------|
| `collect_signals_batch` | Every 5 min |
| `run_adaptation_engine` | Hourly (configurable) |
| `run_pattern_miner` | Daily |
| `run_verification_loop` | Continuous on active canaries |

**Integration:** Celery task via existing `wire_modality_extras()` message bus pattern OR in-process scheduler for lab.

---

## 10. Signal model and utility function

### 10.1 HarnessOutcomeSignal contract

```python
class HarnessOutcomeSignal(BaseModel):
    schema_version: str = "1.0.0"
    signal_id: str
    run_id: str
    tenant_id: str
    application_id: str
    agent_id: str
    task_class: str                    # from Nexus classifier
    timestamp: datetime

    # Quality
    quality_score: float               # 0.0–1.0 from eval registry
    validation_passed: bool
    eval_mode: str                     # offline | online | shadow | human

    # Efficiency
    cost_normalized: float             # actual / budget (1.0 = at budget)
    latency_ms: int
    total_tokens: int
    step_count: int
    tool_calls: int
    llm_calls: int

    # Governance
    hitl_interventions: int
    regression_flags: list[str]        # from ExecutionGuard

    # Optional business (Tier-3)
    business_outcome: float | None     # app-defined; nullable

    # Composite (computed)
    utility: float | None = None
```

### 10.2 Utility function U

Configured per `ApplicationEnvironmentProfile.adaptive_profile.weights`:

```text
U = w_q * quality_score
  - w_c * max(0, cost_normalized - 1.0)
  - w_l * normalize(latency_ms, latency_slo_ms)
  - w_h * min(1.0, hitl_interventions / max_hitl)
  - w_r * regression_penalty(regression_flags)
  + w_b * (business_outcome or 0)        # optional; default w_b = 0
```

**Default weights (conservative):**

| Weight | Default | Notes |
|--------|---------|-------|
| `w_q` | 0.50 | Quality dominates |
| `w_c` | 0.25 | Cost awareness |
| `w_l` | 0.10 | Latency |
| `w_h` | 0.10 | Human burden penalty |
| `w_r` | 0.05 | Regression penalty multiplier |
| `w_b` | 0.00 | Opt-in per Tier-3 app |

### 10.3 Bandit arm definition

For routing tuning, arms are **profile version candidates**:

```text
context = (tenant_id, task_class, time_of_day_bucket)
arm     = (llm_model_id, rag_tier, orchestration_profile_version)
reward  = U (delayed — attributed after run completes)
```

Use ** Thompson sampling** with Beta distribution per arm for v1 (simple, auditable).

---

## 11. Adaptation loops — four canonical kinds

Maps 1:1 to existing `AdaptiveLoopKind` enum.

### 11.1 ROUTING_TUNING

| Attribute | Value |
|-----------|-------|
| **Observes** | U by model × RAG tier × task_class |
| **Proposes** | Shift routing weights; RAG `route_mode` thresholds |
| **Default authority** | `RECOMMEND` → tenant opt-in `AUTO_WITH_HUMAN_GATE` |
| **Max delta** | 10% traffic shift per proposal |
| **Existing hook** | `LLMRoutingEvaluator` + `ModelRouter` + `FailoverLLMAdapter` — see [`LLM_ADAPTERS.md`](LLM_ADAPTERS.md) § LLM routing rules · [ADR-LLM-003](../adr/entries/2026-06-19/ADR-LLM-003.md). Persistent profile versions → **AHI-MAINT-06** / **M-LLM-X.10**. |

### 11.2 EXECUTION_STRATEGY_TUNING

| Attribute | Value |
|-----------|-------|
| **Observes** | step explosion, retry rate, parallel efficiency |
| **Proposes** | `RetryPolicy`, `max_parallel_nodes`, planner strategy name |
| **Default authority** | `RECOMMEND` |
| **Max delta** | 15% change in max steps / retries |
| **Existing hook** | `NexusLoop` construction via `build_nexus_loop_from_environment` |

### 11.3 POLICY_LEARNING

| Attribute | Value |
|-----------|-------|
| **Observes** | tool injection near-miss, adversarial eval failures |
| **Proposes** | `RuntimePolicyBundle` tool deny/allow adjustments |
| **Default authority** | `AUTO_WITH_HUMAN_GATE` (mandatory) |
| **Max delta** | 25% envelope (existing gate rule) |
| **Existing hook** | `PolicyEngine`, `tool_security.py` |

### 11.4 EVALUATION_FEEDBACK

| Attribute | Value |
|-----------|-------|
| **Observes** | benchmark regression deltas |
| **Proposes** | Trigger re-eval; block promotion — no config auto-apply |
| **Default authority** | `OBSERVE_ONLY` |
| **Max iterations** | 20 (existing gate allows higher for this kind) |
| **Existing hook** | `prompt_regression_suite.py`, `evaluation_registry_trends.py` |

---

## Governance Boundary

Adaptive Harness Intelligence (AHI) is a **controlled mechanism for observation, proposal, and evaluation** of harness changes — not an autonomous self-modifying runtime.

**Normative rule:** Adaptive Harness Intelligence may observe, analyze, recommend and evaluate changes. It **MUST NOT** silently mutate production prompts, routing, policies, profiles, retrievers, critic thresholds or tool-selection behavior without explicit governance approval.

AHI extends the laboratory evidence discipline (`ExperimentSession` → KEEP/DISCARD) into production learning. Every recommendation or applied change must remain **versioned, gated, rollback-ready, and traceable** through the observability spine.

**Cross-refs:** [`SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md) §9 · [`MATURITY_TAXONOMY.md`](../guides/MATURITY_TAXONOMY.md) · [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §12.2 (S8) · [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md#production-boundary) · [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) · [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md#verification-safety-boundaries) · [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md#attempt-ledger) · [`OBSERVABILITY.md`](OBSERVABILITY.md#observability-event-spine) · [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md)

---

## Allowed AHI actions

AHI **MAY**:

- observe runtime outcomes,
- analyze traces, events, failures, costs, latencies and quality signals,
- detect recurring execution patterns,
- propose bounded configuration changes,
- propose prompt/profile/routing adjustments,
- propose context ranking or retriever-selection changes,
- propose critic threshold changes,
- simulate or evaluate candidate changes offline,
- run shadow evaluation where explicitly enabled,
- generate governance-ready change proposals,
- recommend canary rollout,
- recommend rollback,
- produce evidence reports.

---

## Disallowed AHI actions

AHI **MUST NOT**:

- silently mutate production prompts,
- silently mutate production routing,
- silently mutate `RuntimePolicyBundle` or equivalent policy profiles,
- silently change critic thresholds,
- silently change retriever selection,
- silently change tool permissions or `ToolProfiles`,
- silently change Tier-3 application rosters,
- bypass maturity/evidence requirements,
- bypass HITL/governance approval,
- bypass `RuntimeEvent` / observability spine,
- treat correlation as causation without evidence,
- optimize for cost or latency at the expense of safety/policy,
- self-promote target architecture to production-ready implementation,
- auto-apply high-risk changes without explicit product/governance decision.

---

## AHI change lifecycle

Every AHI-driven or AHI-recommended change follows this lifecycle:

1. **Observe** — collect signals from runs, traces, eval, cost, HITL.
2. **Detect** — identify recurring patterns, regressions, or optimization opportunities.
3. **Propose** — emit bounded `AdaptiveLoopProposal` / profile version draft.
4. **Evaluate** — offline simulation, shadow eval, or regression pre-check.
5. **Classify risk** — assign low / medium / high / critical (see [Change risk classes](#change-risk-classes)).
6. **Collect evidence** — link to `HarnessOutcomeSignal`, eval registry, capability graph impact.
7. **Request governance approval** — human gate, ops workflow, or explicit product decision.
8. **Shadow / canary if approved** — traffic shift within envelope limits only.
9. **Apply only through approved configuration/profile mechanisms** — `ProfileVersionStore` pointer swap; no ad-hoc runtime mutation.
10. **Monitor** — `VerificationLoop` over SLO window.
11. **Roll back if needed** — restore previous profile version on failure.
12. **Record outcome** — persist proposal ID, version lineage, utility delta.

**Traceability rule:** Every AHI-applied or AHI-recommended change must be traceable through the observability spine and must preserve enough evidence to explain why the change was proposed.

---

## Change risk classes

### Low risk

**Examples:**

- documentation recommendation,
- dashboard suggestion,
- non-production evaluation,
- lab-only profile recommendation.

May be proposed freely. Still requires trace/evidence if recorded as AHI output.

### Medium risk

**Examples:**

- prompt/profile recommendation for controlled environment,
- retriever ranking proposal,
- non-critical cost/latency tuning,
- canary candidate.

Requires owner review before production use.

### High risk

**Examples:**

- production policy change,
- tool permission change,
- HITL boundary change,
- critic threshold relaxation,
- routing change affecting high-risk workflows,
- memory/RAG source trust change.

Requires explicit governance approval, evidence, rollback plan and production readiness statement ([`MATURITY_TAXONOMY.md`](../guides/MATURITY_TAXONOMY.md)).

### Critical risk

**Examples:**

- automatic side-effect authorization changes,
- high-risk irreversible workflow changes,
- compliance/legal approval bypass,
- production auto-apply of safety-related behavior.

Must not be auto-applied. Requires human/authoritative approval and policy-level authorization.

---

## Production auto-apply rule

**Production auto-apply is disabled by default.**

It may be enabled only when **all** conditions are met:

- explicit product/governance decision,
- bounded change type,
- maturity statement using [`MATURITY_TAXONOMY.md`](../guides/MATURITY_TAXONOMY.md),
- evidence threshold,
- rollback plan,
- observability coverage,
- policy approval,
- canary or shadow validation where applicable,
- owner assigned.

If any condition is missing, AHI may only **propose**, not **apply**.

---

## Cursor review checklist

Before adding or modifying AHI behavior, Cursor must verify:

- [ ] Is this observe/propose/evaluate, or does it apply changes?
- [ ] If it applies changes, is auto-apply explicitly approved?
- [ ] What risk class is the change?
- [ ] Is there evidence for the recommendation?
- [ ] Is the maturity level stated using [`MATURITY_TAXONOMY.md`](../guides/MATURITY_TAXONOMY.md)?
- [ ] Is there a rollback path?
- [ ] Are `RuntimeEvent` / observability records preserved?
- [ ] Does this affect prompts, routing, policy, tools, retrievers, memory, critic thresholds or HITL boundaries?
- [ ] Could this weaken safety, evidence, policy or human review?
- [ ] Is the change applied only through approved profile/config mechanisms?
- [ ] Does this avoid self-modifying runtime behavior?

---

## 12. Lifecycle modes — Observe through Verify

### 12.1 Mode definitions

| Mode | Code | Mutates runtime | User visibility |
|------|------|-----------------|-----------------|
| **Observe** | L4-O | No | Internal dashboards |
| **Recommend** | L4-R | No | Ops report + API |
| **Shadow** | L4-S | Shadow eval only | Trace tagged `shadow_profile_version` |
| **Canary** | L4-C | Partial traffic | Tenant allowlist |
| **Apply** | L4-A | Active pointer swap | Registry version bump |
| **Verify** | L4-V | Rollback if fail | Trend reports |

### 12.2 Promotion flow (profile versions)

```text
draft ──► shadow ──► canary ──► active ──► retired
                  └─ rollback ◄─┘
```

Aligns with `agent_promotion.py` evidence pattern — reuse promotion checklist adapted for profiles.

### 12.3 Relationship to laboratory workflow (§35)

| Lab phase | AHI production equivalent |
|-----------|---------------------------|
| Hypothesis | `AdaptiveLoopProposal.proposed_change_summary` |
| Run via Nexus | Shadow/canary runs |
| Validation criteria | Utility U + regression suites |
| KEEP / DISCARD | Apply / reject proposal |
| Delete | Rollback + retire version |

---

## 13. Process pattern intelligence

### 13.1 Business intent

Surface **hidden operational paths** — recurring sequences of tools, agents, and human gates that correlate with high or low utility — without claiming full business process management (BPM).

### 13.2 Example patterns

| Pattern | Interpretation | Suggested action |
|---------|----------------|------------------|
| `research → websearch.read_url → confluence.search` × 50/week, high U | Effective research workflow | Promote to SkillManifest draft |
| `legal_agent → hitl × 3` × 20/week, low U | Unclear escalation policy | Recommend policy review |
| `vendor_discovery → jira.create` × 5/week, high business_outcome | Valuable automation path | Tier-3 dashboard highlight |

### 13.3 Outputs never auto-execute in v1

`ProcessPatternProposal` creates **tickets/recommendations** only:

- Scaffold skill stub (human completes).
- Ops runbook entry.
- Adaptive routing hint (if mapped to ROUTING_TUNING).

---

## 14. Integration with existing Intergrax subsystems

### 14.1 Nexus Runtime

| Integration point | Change |
|-------------------|--------|
| Task completion hook | Call `SignalCollector.emit()` |
| `Agent.run()` | Read active profile version IDs from context |
| Metadata `harness_shadow_eval` | Extend with `candidate_profile_version_id` |

### 14.2 PolicyEngine

- Executor submits policy fragments as **new registry versions**.
- Runtime loads active version from `ProfileVersionStore` pointer.
- Deny-path tests mandatory before apply.

### 14.3 Evaluation subsystem

| Component | Role in AHI |
|-----------|-------------|
| `FileOnlineEvaluationRegistry` | Shadow run scores |
| `evaluation_registry_trends.py` | VerificationLoop baseline compare |
| `evaluation_automation.py` | Rule + LLM judge inputs to quality_score |
| `NexusEvalRunner` (V-REM-A.1) | Golden scenario execution |

### 14.4 Capability graph

Before any proposal affecting skills/tools/policy:

```text
impact = compute_blast_radius(proposal.target_nodes)
if impact.incompatible_edges: REJECT proposal
```

### 14.5 ApplicationEnvironmentProfile (Tier-3)

New section `AdaptiveProfile`:

```python
class AdaptiveProfile(BaseModel):
    enabled: bool = False
    mode: Literal["observe", "recommend", "shadow", "canary", "apply"] = "observe"
    enabled_loops: list[AdaptiveLoopKind] = Field(default_factory=list)
    utility_weights: UtilityWeights = Field(default_factory=UtilityWeights)
    canary_tenant_allowlist: list[str] = Field(default_factory=list)
    canary_traffic_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    human_approver_group: str | None = None
```

Default for all apps: `enabled=False`, `mode=observe`.

### 14.6 ExperimentSession

Reuse patterns:

- `register()` → proposal registration.
- `evaluate_against_criteria()` → verification checks.
- `decide(KEEP|DISCARD)` → apply/rollback.

---

## 15. Tier placement and dependency rules

### 15.1 Strict dependencies

```text
Tier-3 AdaptiveProfile (config) ──► Tier-1 ACP (engine) ──► Tier-0 catalogs (read-only)
                                         │
                                         ▼
                                   Tier-2 agents (consume profiles; no adaptation logic)
```

### 15.2 Forbidden patterns

| Anti-pattern | Why forbidden |
|--------------|---------------|
| Agent imports `AdaptationEngine` | Agents execute; harness adapts |
| Direct SQLite writes to profile store from Tier-2 | Bypasses governance |
| Second trace system for AHI | Violates §5.2 reuse |
| Auto prompt string mutation without Prompt Registry | Violates §53.5 |
| Training PyTorch models in Nexus hot path | Latency + audit failure |

### 15.3 New module location

```
intergrax/runtime/adaptive/          # NEW package (Tier-1)
├── signal_collector.py
├── signal_store.py
├── adaptation_engine.py
├── proposal_builder.py
├── adaptation_executor.py
├── verification_loop.py
├── profile_version_store.py
├── bandit_state.py
├── process_pattern_miner.py
├── scheduler.py
└── contracts.py                     # HarnessOutcomeSignal, etc.
```

Extend (don't fork):

```
intergrax/runtime/architecture/adaptive_governance.py   # existing
intergrax/runtime/architecture/runtime_governance_bridge.py
```

---

## 16. Security, governance, and human-in-the-loop

### 16.1 Threat model

| Threat | Mitigation |
|--------|------------|
| Reward hacking (low cost, garbage output) | Multi-signal U; quality weight ≥ 0.5 default |
| Policy drift opening unsafe tools | POLICY_LEARNING human gate; max 25% delta |
| Cross-tenant leakage in bandit state | Partition stores by `tenant_id` |
| Malicious Tier-3 business_outcome injection | Validate webhook signatures; cap w_b |
| Denial of service via proposal flood | Cooldown + rate limits per loop_id |
| Rollback failure | Pre-apply snapshot mandatory |

### 16.2 Audit trail requirements

Every adaptation event emits `RuntimeEvent`:

| Event type | Payload |
|------------|---------|
| `ADAPTIVE_SIGNAL_RECORDED` | signal_id, run_id, U |
| `ADAPTIVE_PROPOSAL_CREATED` | proposal_id, loop_kind, summary |
| `ADAPTIVE_PROPOSAL_GATED` | passed, reasons |
| `ADAPTIVE_PROFILE_SHADOW` | version_id, scenario_ids |
| `ADAPTIVE_PROFILE_APPLIED` | version_id, previous_version_id |
| `ADAPTIVE_PROFILE_ROLLBACK` | reason, verification_failures |

### 16.3 Human approval workflow

For `AUTO_WITH_HUMAN_GATE`:

```text
Proposal created → Notification (Slack/Teams adapter) → Human approves via ops API
  → Executor proceeds to shadow/canary → Verify → Apply
```

Reuse existing `notification_adapter` and HITL pause infrastructure.

---

## 17. Data contracts (Pydantic reference)

### 17.1 New contracts summary

| Model | Package |
|-------|---------|
| `HarnessOutcomeSignal` | `intergrax/runtime/adaptive/contracts.py` |
| `UtilityWeights` | same |
| `ProfileVersionRecord` | same |
| `ProfileVersionDraft` | same |
| `ProcessPatternProposal` | same |
| `AdaptationExecutionResult` | same |
| `VerificationReport` | same |
| `AdaptiveProfile` | `intergrax/applications/contracts/environment_profile.py` |

### 17.2 Existing contracts (unchanged)

| Model | Location |
|-------|----------|
| `AdaptiveLoopEnvelope` | `adaptive_governance.py` |
| `AdaptiveLoopProposal` | `adaptive_governance.py` |
| `AdaptiveLoopKind` | `adaptive_governance.py` |
| `OnlineEvaluationObservation` | `online_evaluation_models.py` |

---

## 18. End-to-end flow diagrams

### 18.1 Run-time signal path (synchronous tail)

```mermaid
sequenceDiagram
    participant User
    participant Nexus as NexusLoop
    participant Agent as AgentEngine
    participant Trace as RunTraceWriter
    participant Guard as ExecutionGuard
    participant SC as SignalCollector
    participant Store as SignalStore

    User->>Nexus: Submit task
    Nexus->>Agent: Execute agent
    Agent->>Trace: Emit trace events
    Nexus->>Guard: evaluate_run (post-run)
    Guard-->>Nexus: RegressionSignals
    Nexus->>SC: emit(run_id, metrics, eval, guard)
    SC->>Store: persist HarnessOutcomeSignal
```

### 18.2 Adaptation batch path (async)

```mermaid
flowchart TD
    A[AdaptationScheduler tick] --> B[Load signals window]
    B --> C[AdaptationEngine.analyze]
    C --> D{Proposals generated?}
    D -- No --> Z[End]
    D -- Yes --> E[ProposalBuilder]
    E --> F[Governance Gate]
    F --> G{Passed?}
    G -- No --> H[Log ADAPTIVE_PROPOSAL_GATED fail]
    G -- Yes --> I{Authority level?}
    I -- OBSERVE --> J[Ops report only]
    I -- RECOMMEND --> K[Ops report + optional shadow]
    I -- AUTO+HITL --> L[Await human approval]
    L --> M[AdaptationExecutor]
    K --> M
    M --> N[Shadow runs]
    N --> O[Canary traffic]
    O --> P[Apply active pointer]
    P --> Q[VerificationLoop]
    Q --> R{Improvement verified?}
    R -- Yes --> S[Mark version active stable]
    R -- No --> T[Rollback + incident]
```

### 18.3 Profile version promotion

```mermaid
stateDiagram-v2
    [*] --> draft: Engine proposes
    draft --> shadow: Gate passed
    shadow --> canary: Shadow U >= baseline
    canary --> active: Canary verification OK
    active --> retired: Superseded
    shadow --> draft: Shadow fail
    canary --> draft: Canary fail
    active --> draft: Rollback
```

### 18.4 Process pattern mining

```mermaid
flowchart LR
    T[Trace DB] --> M[ProcessPatternMiner]
    M --> P[ProcessPatternProposal]
    P --> R{Human review}
    R --> S[scaffold new-skill]
    R --> D[Document runbook]
    R --> X[Dismiss]
```

---

## 19. Phased implementation roadmap — Phase W-ADAPT

### 19.1 Phase overview

| Wave | ID prefix | Goal | Duration estimate |
|------|-----------|------|-------------------|
| W0 | W-ADAPT-0.* | RFC acceptance, plan sync, ADR | 1 week |
| W1 | W-ADAPT-1.* | SignalCollector + SignalStore + utility | 2–3 weeks |
| W2 | W-ADAPT-2.* | AdaptationEngine (recommend only) + ops report | 2–3 weeks |
| W3 | W-ADAPT-3.* | ProfileVersionStore + shadow executor | 3 weeks |
| W4 | W-ADAPT-4.* | Canary + apply + rollback | 3 weeks |
| W5 | W-ADAPT-5.* | VerificationLoop + L4 evidence | 2 weeks |
| W6 | W-ADAPT-6.* | ProcessPatternMiner | 2–3 weeks |
| W7 | W-ADAPT-7.* | Tier-3 AdaptiveProfile wiring + docs | 1–2 weeks |

**Total estimate:** 16–20 weeks with gate green after each wave.

### 19.2 Wave W-ADAPT-1 — Observe (L4-O)

| Task | Deliverable | Acceptance |
|------|-------------|------------|
| W-ADAPT-1.1 | `HarnessOutcomeSignal` contract + tests | Schema validated |
| W-ADAPT-1.2 | `SignalCollector` hooked to task completion | Signals in store per run |
| W-ADAPT-1.3 | Utility computation | U populated on signal |
| W-ADAPT-1.4 | `scripts/release/phase_w_adapt_report.py` | Report lists signals + U trends |

### 19.3 Wave W-ADAPT-2 — Recommend (L4-R)

| Task | Deliverable | Acceptance |
|------|-------------|------------|
| W-ADAPT-2.1 | `RoutingTuningEngine` (bandit skeleton) | Proposals for ROUTING_TUNING |
| W-ADAPT-2.2 | `ExecutionStrategyEngine` | Proposals from step/retry metrics |
| W-ADAPT-2.3 | Integration with `cost_optimization.py` | Cost anomalies → proposals |
| W-ADAPT-2.4 | Ops report shows gated proposals | No runtime mutation |

### 19.4 Wave W-ADAPT-3 — Shadow (L4-S)

| Task | Deliverable | Acceptance |
|------|-------------|------------|
| W-ADAPT-3.1 | `ProfileVersionStore` | CRUD + rollback pointers |
| W-ADAPT-3.2 | `AdaptationExecutor.shadow()` | Shadow runs tagged in trace |
| W-ADAPT-3.3 | Extend shadow eval metadata | Candidate version in observation |
| W-ADAPT-3.4 | Unit + integration tests | Gate green |

### 19.5 Wave W-ADAPT-4 — Apply (L4-A)

| Task | Deliverable | Acceptance |
|------|-------------|------------|
| W-ADAPT-4.1 | Canary traffic switch in Tier-3 wiring | Allowlist respected |
| W-ADAPT-4.2 | `AdaptationExecutor.apply()` | Atomic pointer swap |
| W-ADAPT-4.3 | HITL approval for POLICY_LEARNING | Cannot apply without approver |
| W-ADAPT-4.4 | ADAPTIVE_* runtime events | Events in trace export |

### 19.6 Wave W-ADAPT-5 — Verify (L4-V) — **Done**

| Task | Deliverable | Acceptance |
|------|-------------|------------|
| W-ADAPT-5.1 | `VerificationLoop` | Compares eval registry trends + utility/regression/cost/security |
| W-ADAPT-5.2 | Auto-rollback | Failed verification restores pointer + blocks loop kind |
| W-ADAPT-5.6 | `phase_w_adapt_closeout_gate.py` | `--enforce-l4-runtime` CI gate |
| W-ADAPT-5.11 | `l4_runtime_evidence.json` | 30-day golden scenario utility artifact |

### 19.7 Wave W-ADAPT-6 — Pattern intelligence — **Done**

| Task | Deliverable | Acceptance |
|------|-------------|------------|
| W-ADAPT-6.1 | `ProcessPatternMiner` | N-gram frequency over trace sequences |
| W-ADAPT-6.2 | `PersistedTraceSequenceReader` | Reuses `RunTraceReader.list_runs` |
| W-ADAPT-6.3 | Pattern report in `phase_w_adapt_report.py` | `process_patterns.json` export |
| W-ADAPT-6.5 | `AdaptationScheduler.run_pattern_miner` | Daily job entry point |

### 19.8 Wave W-ADAPT-7 — Tier-3 wiring — **Done**

| Task | Deliverable | Acceptance |
|------|-------------|------------|
| W-ADAPT-7.1 | Default `AdaptiveProfile` on lab/reference apps | `enabled=False` initially |
| W-ADAPT-7.3 | AGENT_CREATION_GUIDE Appendix V | Control plane map |
| W-ADAPT-7.6 | Acceptance E2E observe→recommend | No apply in test path |

### 19.9 Dependencies

```text
W-ADAPT-0 → W-ADAPT-1 → W-ADAPT-2 → W-ADAPT-3 → W-ADAPT-4 → W-ADAPT-5
W-ADAPT-1 → W-ADAPT-6 (parallel after W1)
Phase V Done + V-REM Done → prerequisite
```

---

## 20. KPIs, acceptance gates, and L4 evidence

### 20.1 Quantitative KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| Signal coverage | ≥ 95% completed runs emit signal | SignalStore / completed runs |
| Proposal gate pass rate | Tracked; no target | Governance reports |
| Shadow improvement rate | ≥ 60% shadow candidates beat baseline U | Eval registry |
| Apply rollback rate | < 10% of applies | VerificationLoop |
| Mean time to rollback | < 5 minutes | Ops metrics |
| Utility improvement (golden) | ≥ 10% vs static baseline | Benchmark suite |
| Policy learning without approver | **0** | Security audit |
| Pattern proposals reviewed | ≥ 80% within 14 days | Ops queue |

### 20.2 L4 runtime readiness gate (extends Phase V)

All must pass:

1. L3 criteria stable (existing Phase V gate).
2. W-ADAPT-5 complete with CI closeout green.
3. Documented 30-day window showing `mean(U_candidate) > mean(U_baseline)` on ≥ 3 golden scenarios.
4. Zero critical incidents from auto-apply during window.
5. Rollback drill executed successfully in ops runbook.

### 20.3 Evidence artifacts

| Artifact | Path |
|----------|------|
| Signal trend report | `build/adaptive_harness/signal_trends.json` |
| Proposal log | `build/adaptive_harness/proposals.json` |
| Verification report | `build/adaptive_harness/verification_report.json` |
| L4 runtime evidence | `build/adaptive_harness/l4_runtime_evidence.json` |

---

## 21. Operational model

### 21.1 Roles

| Role | Responsibility |
|------|----------------|
| Harness architect | Owns AHI design, envelope policies |
| Platform engineer | Implements W-ADAPT waves |
| Ops / SRE | Reviews recommendations, approves canary |
| Security | Approves POLICY_LEARNING proposals |
| Agent author | Consumes recommended profiles; implements skill drafts from patterns |

### 21.2 Cadence

| Activity | Frequency |
|----------|-----------|
| Signal health review | Weekly |
| Proposal review (RECOMMEND mode) | Weekly |
| Verification report | Per apply + weekly summary |
| Pattern proposal review | Biweekly |
| L4 evidence audit | Per release candidate |

### 21.3 Runbooks (W-ADAPT-5 — Done)

- `runbook/adaptive/rollback_profile.md`
- `runbook/adaptive/approve_policy_learning.md`
- `runbook/adaptive/shadow_failure_triage.md`

---

## 22. Risks, anti-patterns, and mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| False L4 declaration | High | Separate governance L4 vs runtime L4 gates |
| Cold start (no signals) | Medium | Heuristic defaults; min run threshold before bandit |
| Overfitting to golden sets | Medium | Online eval + shadow on diverse tasks |
| Tenant config explosion | Medium | Limit active profile versions per artifact type |
| Engineer bypass via manual config | Medium | Registry pointers as source of truth in strict mode |
| Marketing as "RL" misleading buyers | Medium | Use AHI terminology consistently |

### 22.1 Anti-patterns (forbidden)

1. **Autonomous agent that edits its own prompts in production.**
2. **Second PolicyEngine for experiments.**
3. **Applying adaptations without ProfileVersionStore lineage.**
4. **Skipping capability graph check for skill/policy changes.**
5. **Embedding domain business rules in AdaptationEngine.**

---

## 23. Competitive differentiation summary

| Capability | Typical harness | Intergrax + AHI |
|------------|-----------------|-----------------|
| Trace | ✅ | ✅ |
| Eval benchmarks | Partial | ✅ First-class registry |
| Cost budgets | Rare | ✅ Enforced |
| Policy engine | Partial | ✅ Unified |
| Closed-loop tuning | ❌ Manual | ✅ Governed ACP |
| Rollback of config changes | Manual git revert | ✅ ProfileVersionStore |
| Process pattern mining | ❌ | ✅ Trace-native |
| Human-gated policy learning | ❌ | ✅ Envelope enforced |
| Capability graph impact | ❌ | ✅ Pre-apply validation |

**Positioning statement:**

> Intergrax is the Harness AI platform that **operationalizes improvement** — every run makes the runtime smarter within auditable bounds, not the agent autonomously rewriting itself.

---

## 24. Conclusions and recommendations

### 24.1 Conclusions

1. **Adaptive Harness Intelligence is strategically aligned** with Intergrax's harness-first mission and L4 maturity vision.
2. **Implementation is complete** — Phase W-ADAPT **70/70 Done** (Wave 0–7); runtime package `intergrax/runtime/adaptive/`.
3. **Classical RL is the wrong implementation model**; contextual bandits + governed proposals + verification loops are the right fit.
4. **L4 runtime readiness is achieved in code** — governance L4 (Phase V) + runtime L4 (W-ADAPT-5 closeout gate); production utility evidence accumulates when lab observe mode is active.
5. **Process pattern discovery belongs in Tier-1 mining + Tier-2 authoring**, keeping Nexus domain-agnostic.
6. **Differentiation is real** — policy gates, rollback, VerificationLoop, and measurable utility improvement are shipped.

### 24.2 Recommendations

| # | Recommendation | Priority | Status |
|---|----------------|----------|--------|
| R1 | Accept this RFC and add **Phase W-ADAPT** to implementation plan | P0 | **Done** (2026-06-05) |
| R2 | Default reference apps to safe posture; **lab** enables observe (`enabled=True`, `mode=observe`) | P0 | **Done** — `LAB_ADAPTIVE_OBSERVE`; product hosts remain `enabled=False` |
| R3 | Implement W-ADAPT-1 before any auto-apply code | P0 | **Done** |
| R4 | Rename outward-facing term: **Adaptive Harness Intelligence**, not "RL" | P1 | **Done** |
| R5 | Extend `phase_v_closeout_gate.py` to distinguish governance-L4 vs runtime-L4 | P1 | **Done** (W-ADAPT-5.8) |
| R6 | Author ADR-ADAPT-001 from Appendix C | P1 | **Done** — [`docs/project/technical/adr/entries/2026-06-05/ADR-ADAPT-001.md`](adr/entries/2026-06-05/ADR-ADAPT-001.md) |
| R7 | Defer ProcessPatternMiner until W-ADAPT-5 verifies core loop | P2 | **Done** (W-ADAPT-6 after W-ADAPT-5) |
| R8 | Enforce `--enforce-l4-runtime` in CI and release pipeline | P1 | **Done** — `unit-tests.yml` + `harness-release.yml` |

### 24.3 Decision requested — **Closed**

Phase W-ADAPT Wave 0–7 **Done** (2026-06-02). Ongoing work: §6.1 harness maintenance, lab signal collection, production 30-day L4 evidence when sufficient run volume exists.

---

## Appendix A — Mapping to existing code

| AHIA component | Existing module | Action |
|----------------|-----------------|--------|
| Governance gate | `adaptive_governance.py` | Reuse |
| Shadow eval | `runtime_governance_bridge.py` | Extend |
| Regression signals | `history_evaluator.py` | Feed SignalCollector |
| Post-run governance | `execution_guard.py` | Feed SignalCollector |
| Metrics export | `metrics/export.py` | Feed SignalCollector |
| Cost recommendations | `cost_optimization.py` | Feed AdaptationEngine |
| Online eval | `online_evaluation_registry.py` | VerificationLoop |
| Trends | `evaluation_registry_trends.py` | VerificationLoop |
| Promotion pattern | `agent_promotion.py` | Mirror for profiles |
| Graph impact | `capability_graph_compatibility.py` | Pre-apply gate |
| Lab workflow | `experiments/workflow.py` | Pattern reference |
| RAG tuning target | `rag/routing/query_router.py` | Accept profile overrides |
| Nexus wiring | `applications/_shared/nexus_factory.py` | Load profile versions |
| Maturity evidence | `maturity_gate_evidence.py` | Add runtime L4 inputs |

---

## Appendix B — Proposed implementation plan task IDs

Insert into [`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md) as **Phase W-ADAPT** — **synced 2026-06-05** (70 tasks, Band 2y):

```text
Phase W-ADAPT — Adaptive Harness Intelligence (AHI)
Prerequisite: Phase V Done, Phase V-REM Done, W-OPS Done, EVAL/COST/CG closeouts Done
Band: 2y (§4.0) — default implementation queue after §6.1 maintenance
Scope: Tier-1 intergrax/runtime/adaptive/ + Tier-3 AdaptiveProfile + scripts + tests
Out of scope: K.1/K.2, deep RL, foundation model training

See plan: Phase W-ADAPT master register (W-ADAPT-0.1 … W-ADAPT-7.7) · Appendix K · §6.1t · §6.2ac
```

---

## Appendix C — ADR decision record

**Canonical ADR:** [`docs/project/technical/adr/entries/2026-06-05/ADR-ADAPT-001.md`](adr/entries/2026-06-05/ADR-ADAPT-001.md)

**ADR-ADAPT-001: Adaptive Harness Intelligence over classical RL**

| Field | Value |
|-------|-------|
| Status | Accepted (via this RFC) |
| Context | Need L4 differentiated harness capability |
| Decision | Implement governed Adaptive Control Plane with bandit/rule engines, not deep RL |
| Consequences | (+) Auditability, reuse of Phase V; (−) No neural policy optimality claims |
| Alternatives rejected | End-to-end RL fine-tuning; per-agent self-modifying code; external AutoML SaaS |

---

*End of document — Intergrax Adaptive Harness Intelligence Architecture v1.0.0*
