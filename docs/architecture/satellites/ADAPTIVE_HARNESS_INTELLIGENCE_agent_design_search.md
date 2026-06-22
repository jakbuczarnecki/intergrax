# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

# ADAPTIVE HARNESS INTELLIGENCE — ADAS / Agent Design Search Architecture

**Status:** Proposed architecture  
**Layer:** `ADAPTIVE_HARNESS_INTELLIGENCE`  
**Sub-capability:** ADAS / Agent Design Search  
**Canonical placement:** Tier-1 Adaptive Control Plane  
**Implementation mode:** Architecture-first; no runtime implementation implied by this document  

---

## 1. Executive summary

ADAS — **Automated Design of Agentic Systems** — is an enterprise-grade extension of Intergrax `ADAPTIVE_HARNESS_INTELLIGENCE`.

Its purpose is to let Intergrax systematically design, evaluate, archive, compare, and promote better agent candidates over time.

ADAS does **not** replace the existing adaptive runtime. It extends it.

ADAS does **not** create a new detached architecture layer. It lives inside AHI as a specialized adaptive loop for agent design.

ADAS does **not** directly mutate production agents. It creates candidate agent designs, evaluates them against baselines, stores their lineage and evidence, and promotes them only through governed lifecycle gates.

Strategic statement:

```text
Intergrax does not merely run agents.
Intergrax can discover, measure, and promote better agent designs over time.
```

The core enterprise value is that agent improvement becomes:

```text
observable
measurable
versioned
auditable
policy-bound
rollback-ready
human-governed
enterprise-safe
```

---

## 2. Canonical architectural decision

The canonical decision is:

```text
ADAPTIVE_HARNESS_INTELLIGENCE
  └── ADAS / Agent Design Search
        └── MAS / Meta Agent Search
```

Meaning:

```text
AHI      = existing adaptive control plane.
ADAS     = specialized adaptive sub-capability for agent design.
MAS      = one replaceable strategy for generating agent candidates.
Scaffold = candidate materialization mechanism.
Eval     = candidate fitness measurement.
Archive  = evolutionary memory.
Promotion = governed path from candidate to usable agent.
Tier-3 ADAS Lab = optional operator application only.
```

This decision is intentionally conservative. It prevents ADAS from becoming a second, parallel harness.

ADAS must be designed as:

```text
a governed adaptive design loop inside AHI
```

not as:

```text
a free-form agent that writes other agents
```

---

## 3. Mandatory integration rule

ADAS must be built on existing Intergrax layers and components.

It must not be a functionally detached subsystem.

ADAS must reuse or extend:

```text
ADAPTIVE_HARNESS_INTELLIGENCE
intergrax/runtime/adaptive/
HarnessOutcomeSignal
existing utility scoring model
existing signal collection concepts
existing signal store concepts
existing AdaptationEngine / proposal concepts
existing governance envelopes
existing policy-first architecture
existing AdaptationExecutor lifecycle semantics
existing shadow → canary → apply → verify flow
existing VerificationLoop concepts
existing ProcessPatternMiner
existing agent scaffold mechanism
existing agent contract model
existing Agent Creation Guide workflow
existing observability and evaluation mechanisms
existing Tier-0 / Tier-1 / Tier-2 / Tier-3 boundaries
```

ADAS must not duplicate:

```text
governance stack
evaluation registry
tracing system
policy engine
agent registry
scaffold system
promotion lifecycle
signal collection
utility computation
observability events
storage conventions
```

If an ADAS component appears to duplicate an existing AHI, runtime, scaffold, evaluation, governance, or observability component, the design must be refactored to reuse the existing component or explicitly justify a narrow adapter.

---

## 4. What ADAS is

ADAS is a governed agent design search capability.

It provides:

```text
1. Objective definition
2. Baseline selection
3. Candidate generation
4. Candidate materialization through scaffold
5. Static safety gate
6. Offline evaluation
7. Utility scoring
8. Candidate archive and lineage
9. Ranking and selection
10. Shadow evaluation
11. Canary promotion
12. Human approval
13. Verification
14. Rollback or retirement
```

ADAS is an adaptive loop whose target artifact is an **agent candidate** rather than a runtime profile.

The closest internal analogy is:

```text
existing AHI profile adaptation
  observes signals
  proposes profile version drafts
  gates proposals
  shadows/canaries/applies
  verifies utility improvement

ADAS agent design search
  observes signals and objectives
  proposes agent candidate drafts
  gates candidates
  evaluates/scaffolds/archives
  shadows/canaries/promotes
  verifies utility improvement
```

---

## 5. What ADAS is not

ADAS is not:

```text
1. A new top-level runtime layer.
2. A parallel governance system.
3. A parallel evaluation system.
4. A parallel scaffold system.
5. A Tier-3-only application.
6. An autonomous code mutator.
7. A mechanism for editing production agents directly.
8. A mechanism for modifying Nexus core.
9. A replacement for AgentContract.
10. A replacement for PolicyEngine.
11. A replacement for existing AHI lifecycle.
12. A hidden self-improvement loop.
13. A black-box RL system.
14. A benchmark-only optimizer.
15. A system that can self-approve its own production promotion.
```

The enterprise-safe framing is:

```text
ADAS is not "an agent that writes agents".
ADAS is a governed adaptive design control plane that uses agents, scaffold, evaluation, archive, policy, observability, and promotion gates to discover better agent candidates safely.
```

---

## 6. Placement in Intergrax tiers

### 6.1 Tier-0 — platform foundation

Tier-0 remains the platform foundation.

ADAS reuses Tier-0:

```text
LLM adapters
model catalogs
tool catalogs
skill catalogs
storage primitives
evaluation primitives
policy primitives
tracing primitives
cost primitives
```

ADAS must not duplicate Tier-0 catalogs.

### 6.2 Tier-1 — adaptive control plane

Tier-1 owns the ADAS control plane.

Proposed location:

```text
intergrax/runtime/adaptive/agent_design_search/
```

Tier-1 responsibilities:

```text
search orchestration
candidate archive
candidate lifecycle state
static gates
evaluation routing
utility scoring
policy/gate enforcement
promotion bridge
verification
hooks/events
enterprise audit reports
```

Tier-1 must remain domain-agnostic.

### 6.3 Tier-2 — design agents and strategy agents

Tier-2 contains optional agents that participate in design search.

Examples:

```text
agents/meta_agent_designer/
agents/candidate_critic/
agents/benchmark_runner/
agents/agent_refiner/
```

These agents may propose candidates, critique candidates, or run specialized evaluation workflows.

They must not own the ADAS control plane.

They must not directly write runtime code.

They must not directly promote candidates.

### 6.4 Tier-3 — ADAS Lab / operator environment

Tier-3 may contain an optional ADAS Lab application.

Example:

```text
applications/adas_lab/
```

Tier-3 responsibilities:

```text
operator UI
objective creation
approval/rejection workflow
dashboard
experiment workspace
reporting
manual override screens
```

Tier-3 must not own:

```text
archive logic
evaluation logic
governance logic
promotion lifecycle
runtime mutation logic
candidate lifecycle semantics
```

Tier-3 wires and visualizes ADAS. It does not define ADAS.

---

## 7. Relationship to existing AHI

ADAS must reuse the existing AHI pattern:

```text
observe
  → propose
  → validate
  → shadow
  → canary
  → apply/promote
  → verify
  → rollback or keep
```

Existing AHI capabilities reused:

```text
HarnessOutcomeSignal
SignalCollector
SignalStore
AdaptationEngine concepts
ProposalBuilder concepts
Governance Gate
AdaptiveLoopEnvelope
PolicyEngine boundary
ProfileVersionStore concepts
ProposalStore concepts
AdaptationExecutor lifecycle semantics
VerificationLoop checks
ProcessPatternMiner
Cost governance
Evaluation registry / evaluation trends
Observability events
```

ADAS extends AHI by adding a new design-search loop:

```text
AgentDesignSearchLoop
```

This loop targets agent candidates rather than runtime profiles.

The architectural boundary is:

```text
AHI may tune profiles and propose bounded runtime configuration changes.
ADAS may propose new agent candidates, but never directly mutate existing production agents.
```

---

## 8. Relationship to MAS

MAS — Meta Agent Search — is not the architecture.

MAS is one implementation of `AgentDesignStrategy`.

```text
AgentDesignStrategy
  ├── MetaAgentSearchStrategy
  ├── RuleBasedVariantStrategy
  ├── PromptMutationStrategy
  ├── ToolPolicyMutationStrategy
  ├── CognitivePatternSearchStrategy
  └── HumanSeededStrategy
```

MAS may use LLM reasoning to generate candidates, but it must operate inside ADAS gates.

MAS must not:

```text
write directly into production agent folders
modify runtime code
bypass static gates
bypass evaluation
bypass human approval
bypass archive
self-approve promotion
```

MAS output is always a structured `AgentCandidateDraft`.

---

## 9. Relationship to scaffold

Scaffold is the only approved materialization path for new agent candidates.

ADAS does not invent a second generator.

Flow:

```text
AgentCandidateDraft
  → AgentScaffoldBridge
  → existing scaffold
  → sandboxed candidate artifact
  → static gate
  → evaluation
```

The scaffold bridge must use existing scaffold concepts:

```text
new-agent
capabilities
AgentContract
cognitive patterns
generated tests
prompts
schemas
notebooks
```

Candidate materialization must happen in a controlled sandbox or archive location first.

It must not directly overwrite existing production agents.

Recommended candidate materialization location:

```text
build/adaptive_harness/agent_design_search/candidates/<candidate_id>/
```

Only after promotion should a candidate be eligible for controlled movement into a canonical `agents/` package or registry binding.

---

## 10. Relationship to ProcessPatternMiner

`ProcessPatternMiner` becomes one source of ADAS objectives.

Flow:

```text
runtime traces
  → ProcessPatternMiner
  → ProcessPatternProposal
  → AgentDesignObjective
  → AgentDesignSearchRun
```

Example:

```text
ProcessPatternMiner detects repeated high-utility sequence:
  task_class=vendor_discovery
  tools=web_search, summarize
  agent=research_agent

ADAS creates objective:
  "Design a specialized vendor discovery agent that reduces latency
   while preserving quality and lowering tool fanout."
```

This makes ADAS naturally grounded in production or laboratory evidence.

It avoids random agent generation.

---

## 11. Target component model

Proposed Tier-1 module:

```text
intergrax/runtime/adaptive/agent_design_search/
  __init__.py
  contracts.py
  search_controller.py
  search_context.py
  search_policy.py
  strategy.py
  mas_strategy.py
  candidate_archive.py
  candidate_evaluator.py
  static_gate.py
  scaffold_bridge.py
  design_hooks.py
  design_events.py
  promotion_bridge.py
  verification.py
  report.py
```

This module is an AHI submodule.

It is not a separate top-level architecture domain.

---

## 12. Component responsibilities

### 12.1 `AgentDesignSearchController`

The orchestrator of an ADAS search run.

Responsibilities:

```text
load objective
select baseline
load archive history
invoke strategy
call scaffold bridge
run static gate
run evaluation
compute utility
archive candidate
rank candidates
enforce budgets
enforce stop conditions
emit events
call hooks
prepare promotion request
```

It must not generate code directly.

It must not own agent domain logic.

It must not bypass evaluation or archive.

### 12.2 `AgentDesignStrategy`

Protocol for candidate generation.

Responsibilities:

```text
inspect objective
inspect baseline
inspect archive
propose candidate draft
explain rationale
```

Example implementations:

```text
MetaAgentSearchStrategy
RuleBasedVariantStrategy
PromptMutationStrategy
ToolPolicyMutationStrategy
PatternSearchStrategy
HumanSeededStrategy
```

### 12.3 `MetaAgentSearchStrategy`

MAS implementation.

Responsibilities:

```text
generate candidate design hypotheses
use archive as evolutionary memory
avoid repeated failed variants
use previous winners as parents
explain design rationale
propose measurable changes
```

MAS output is `AgentCandidateDraft`.

### 12.4 `AgentScaffoldBridge`

Adapter from ADAS candidate draft to existing scaffold.

Responsibilities:

```text
translate candidate draft into scaffold parameters
materialize candidate in sandbox/archive
capture scaffold output reference
prevent writes to production paths
prevent runtime edits
```

### 12.5 `AgentStaticGate`

Static safety validator.

Responsibilities:

```text
validate AgentContract
validate capability IDs
validate tier boundaries
detect forbidden imports
detect runtime edits
detect application imports inside agent package
validate generated tests exist
validate prompt/schema assets
validate policy/tool declarations
```

A candidate that fails static gate cannot be evaluated or promoted.

### 12.6 `AgentCandidateEvaluator`

Offline evaluator.

Responsibilities:

```text
run baseline scenarios
run candidate scenarios
collect quality/cost/latency/token/step/tool metrics
run golden scenarios
run negative scenarios
run adversarial scenarios
compute utility
produce evaluation result
```

### 12.7 `AgentDesignArchive`

Append-only candidate archive.

Responsibilities:

```text
store candidate records
preserve lineage
store parent-child relationships
store evaluation result
store utility delta
store rejection reasons
support search by objective/status/lineage
support enterprise audit
```

Storage style should mirror existing AHI stores:

```text
protocol
in-memory implementation
SQLite-backed implementation
```

Recommended SQLite path:

```text
build/adaptive_harness/agent_design_search/candidates.db
```

### 12.8 `AgentCandidatePromotionBridge`

Promotion adapter.

Responsibilities:

```text
create shadow registration request
create canary registration request
require human approval
prevent direct production apply
link promoted candidate to active agent registry
record rollback pointer
```

### 12.9 `AgentCandidateVerificationLoop`

ADAS-specific verification loop.

Responsibilities:

```text
compare promoted candidate against baseline over verification window
check utility trend
check regression rate
check cost budget
check security/adversarial pass rate
trigger rollback request when verification fails
```

It should reuse existing verification checks where possible.

---

## 13. Data contracts

### 13.1 `AgentDesignObjective`

```python
class AgentDesignObjective(BaseModel):
    objective_id: str
    tenant_id: str
    application_id: str | None = None
    task_class: str
    target_capability: str

    baseline_agent_id: str | None = None
    baseline_candidate_id: str | None = None

    goal_summary: str
    success_metric: str

    max_iterations: int = 5
    max_candidates: int = 20
    max_generation_cost: float | None = None
    max_eval_cost: float | None = None

    min_utility_delta: float = 0.05
    require_human_approval: bool = True

    allowed_patterns: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)

    created_by: str
    created_at: datetime
```

### 13.2 `AgentDesignSearchRun`

```python
class AgentDesignSearchRun(BaseModel):
    search_run_id: str
    objective_id: str
    tenant_id: str
    task_class: str

    strategy_name: str
    status: str

    started_at: datetime
    completed_at: datetime | None = None

    generated_candidate_ids: list[str] = Field(default_factory=list)
    promoted_candidate_id: str | None = None

    total_generation_cost: float = 0.0
    total_evaluation_cost: float = 0.0

    stop_reason: str | None = None
```

### 13.3 `AgentCandidateDraft`

```python
class AgentCandidateDraft(BaseModel):
    candidate_id: str
    objective_id: str
    parent_candidate_id: str | None = None
    lineage_id: str

    generated_by: str
    generation_strategy: str
    design_rationale: str

    scaffold_pattern: str
    capabilities: list[str]

    contract_draft: dict[str, Any]
    prompt_assets: dict[str, Any]
    tool_policy: dict[str, Any]
    state_schema: dict[str, Any] | None = None

    source_artifact_ref: str | None = None
    scaffold_output_ref: str | None = None

    created_at: datetime
```

### 13.4 `AgentCandidateStatus`

```python
class AgentCandidateStatus(str, Enum):
    DRAFT = "draft"
    SCAFFOLDED = "scaffolded"
    STATIC_GATE_FAILED = "static_gate_failed"
    STATIC_GATE_PASSED = "static_gate_passed"
    EVALUATED = "evaluated"
    REJECTED = "rejected"
    SHADOW = "shadow"
    CANARY = "canary"
    PROMOTED = "promoted"
    RETIRED = "retired"
    ARCHIVED = "archived"
```

### 13.5 `AgentCandidateRecord`

```python
class AgentCandidateRecord(BaseModel):
    candidate_id: str
    objective_id: str
    parent_candidate_id: str | None = None
    lineage_id: str

    status: AgentCandidateStatus

    scaffold_pattern: str
    agent_contract_snapshot: dict[str, Any]
    source_artifact_ref: str
    scaffold_output_ref: str | None = None
    eval_suite_id: str | None = None

    static_gate_result: dict[str, Any] | None = None
    evaluation_result: dict[str, Any] | None = None

    utility_score: float | None = None
    baseline_utility_score: float | None = None
    utility_delta: float | None = None

    novelty_score: float | None = None
    complexity_score: float | None = None
    generation_cost: float | None = None
    evaluation_cost: float | None = None

    rejection_reason: str | None = None
    promotion_request_id: str | None = None

    created_by: str
    created_at: datetime
    archived_at: datetime | None = None
```

### 13.6 `AgentCandidateEvaluationResult`

```python
class AgentCandidateEvaluationResult(BaseModel):
    candidate_id: str
    baseline_agent_id: str | None = None

    passed: bool

    quality_score: float
    cost_normalized: float
    latency_ms: int
    total_tokens: int
    step_count: int
    tool_calls: int
    llm_calls: int
    hitl_interventions: int

    regression_flags: list[str] = Field(default_factory=list)

    utility_score: float
    baseline_utility_score: float | None = None
    utility_delta: float | None = None

    golden_pass_rate: float
    negative_pass_rate: float
    adversarial_pass_rate: float

    failure_reasons: list[str] = Field(default_factory=list)
```

---

## 14. Candidate lifecycle

Happy path:

```text
DRAFT
  → SCAFFOLDED
  → STATIC_GATE_PASSED
  → EVALUATED
  → SHADOW
  → CANARY
  → PROMOTED
  → VERIFIED
```

Failure paths:

```text
DRAFT
  → STATIC_GATE_FAILED
  → ARCHIVED

EVALUATED
  → REJECTED
  → ARCHIVED

SHADOW / CANARY / PROMOTED
  → RETIRED
  → ARCHIVED
```

A candidate cannot skip:

```text
static gate
offline evaluation
archive write
human approval
verification
```

---

## 15. Search lifecycle

```text
1. Create objective
2. Select baseline
3. Load archive history
4. Start search run
5. Generate candidate draft
6. Run scaffold bridge
7. Run static gate
8. Run offline evaluation
9. Compute utility
10. Archive candidate
11. Rank candidate
12. Decide:
    - reject
    - generate next candidate
    - shadow
13. Shadow evaluation
14. Canary promotion
15. Human-approved promotion
16. Verification
17. Keep or rollback
```

---

## 16. Enterprise hook model

ADAS must support controlled extension points.

### 16.1 Hook protocol

```python
class AgentDesignHook(Protocol):
    def on_search_started(self, context: AgentDesignContext) -> None: ...
    def on_candidate_generated(self, draft: AgentCandidateDraft) -> AgentCandidateDraft: ...
    def on_candidate_scaffolded(self, draft: AgentCandidateDraft) -> None: ...
    def on_static_gate_completed(self, result: AgentStaticGateResult) -> None: ...
    def on_evaluation_started(self, candidate_id: str) -> None: ...
    def on_evaluation_completed(self, result: AgentCandidateEvaluationResult) -> None: ...
    def on_candidate_archived(self, record: AgentCandidateRecord) -> None: ...
    def on_candidate_ranked(self, record: AgentCandidateRecord) -> None: ...
    def on_promotion_requested(self, request: AgentCandidatePromotionRequest) -> None: ...
    def should_stop(self, context: AgentDesignContext) -> bool: ...
```

### 16.2 Hook rules

Hooks may:

```text
enrich candidates
reject candidates
add custom evaluation
add custom ranking
stop search
request human review
```

Hooks must not:

```text
bypass static gate
bypass evaluation
bypass archive
bypass policy
mutate production agents
edit runtime
write to forbidden paths
```

---

## 17. Event model

ADAS emits structured events.

```text
agent_design.search.started
agent_design.search.completed
agent_design.search.stopped

agent_design.candidate.generated
agent_design.candidate.scaffolded
agent_design.candidate.static_gate.passed
agent_design.candidate.static_gate.failed
agent_design.candidate.evaluation.started
agent_design.candidate.evaluation.completed
agent_design.candidate.archived
agent_design.candidate.rejected
agent_design.candidate.ranked

agent_design.promotion.shadow.requested
agent_design.promotion.shadow.started
agent_design.promotion.canary.requested
agent_design.promotion.canary.started
agent_design.promotion.production.requested
agent_design.promotion.production.approved
agent_design.promotion.production.rejected

agent_design.verification.started
agent_design.verification.passed
agent_design.verification.failed
agent_design.rollback.requested
agent_design.rollback.completed
```

Every event must include:

```text
event_id
timestamp
tenant_id
objective_id
search_run_id
candidate_id
baseline_agent_id
task_class
correlation_id
trace_id
actor
```

---

## 18. Utility model

ADAS uses an agent-specific extension of AHI utility.

Base utility:

```text
U = quality - cost - latency - hitl - regression + business_outcome
```

Agent design utility:

```text
U_agent =
  w_quality     * quality_score
- w_cost        * cost_penalty
- w_latency     * latency_penalty
- w_hitl        * hitl_penalty
- w_regression  * regression_penalty
- w_complexity  * complexity_penalty
- w_instability * variance_penalty
+ w_novelty     * useful_novelty
+ w_transfer    * cross_task_transfer_score
+ w_business    * business_outcome
```

Promotion requires:

```text
utility_delta >= objective.min_utility_delta
golden_pass_rate >= threshold
negative_pass_rate >= threshold
adversarial_pass_rate >= threshold
cost regression within budget
no critical regression flags
static gate passed
human approval present
```

Novelty must be useful novelty, not randomness.

A candidate is better only if it improves measured utility without violating cost, policy, security, or regression thresholds.

---

## 19. Archive and lineage model

ADAS archive is append-only.

It must preserve:

```text
candidate_id
parent_candidate_id
lineage_id
objective_id
strategy_name
design rationale
scaffold pattern
contract snapshot
source artifact reference
evaluation results
utility score
baseline score
utility delta
rejection reason
promotion status
created_at
archived_at
```

Lineage example:

```text
candidate_a
  ├── candidate_b
  │     └── candidate_d
  └── candidate_c
```

Archive enables:

```text
avoiding repeated failed ideas
selecting parent candidates
analyzing improvement over generations
explaining why a candidate won
explaining why a candidate failed
enterprise audit
```

---

## 20. Static gate

Static gate is mandatory.

Checks:

```text
1. AgentContract exists.
2. Capability IDs are valid.
3. Agent does not import applications.*.
4. Agent does not import forbidden runtime internals.
5. Agent does not edit intergrax/runtime.
6. Agent does not bypass PolicyEngine.
7. Agent does not define unmanaged tool access.
8. Agent has smoke test.
9. Agent has evaluation scenario mapping.
10. Prompt assets are present.
11. Schemas are valid.
12. Risk level is declared.
13. Max steps are bounded.
14. Tool policy is declared.
15. Tier boundary is preserved.
```

Failure blocks evaluation and promotion.

---

## 21. Evaluation model

Evaluation stages:

```text
1. Smoke evaluation
2. Golden scenario evaluation
3. Negative scenario evaluation
4. Adversarial scenario evaluation
5. Cost evaluation
6. Latency evaluation
7. Regression evaluation
8. Optional human review
9. Utility computation
10. Baseline comparison
```

A candidate is not compared in isolation.

It is always evaluated against:

```text
baseline agent
baseline candidate
or current production profile
```

---

## 22. Promotion model

Promotion stages:

```text
EVALUATED
  → shadow request
  → human approval
  → shadow registration
  → shadow evaluation
  → canary request
  → canary approval
  → canary allocation
  → production promotion request
  → production approval
  → active registration
  → verification
```

Default:

```text
require_human_approval = true
```

Auto-promotion is allowed only for future low-risk lab profiles and only after explicit product decision.

---

## 23. Rollback model

Rollback must restore the previous active agent registration or routing pointer.

Rollback triggers:

```text
utility trend below threshold
regression rate above threshold
cost above budget
adversarial failure
operator rejection
policy violation
production incident
```

Rollback result must be archived.

---

## 24. Security model

ADAS security boundaries:

```text
1. No direct runtime edits.
2. No direct production source overwrite.
3. No unmanaged tool access.
4. No policy bypass.
5. No hidden network access in generated candidates.
6. No secret exposure in prompts or archive.
7. No cross-tenant archive access.
8. No promotion without approval.
9. No candidate execution outside sandbox/evaluation host before gate.
10. No self-approval by MAS.
```

---

## 25. Multi-tenant model

Every ADAS object must be tenant-scoped.

Required fields:

```text
tenant_id
application_id
objective_id
search_run_id
candidate_id
task_class
```

Archive queries must be scoped by tenant.

Cross-tenant learning is forbidden by default.

Future cross-tenant learning requires:

```text
explicit anonymization
explicit product decision
explicit governance approval
```

---

## 26. Cost and budget control

ADAS can be expensive. Therefore every objective must define limits:

```text
max_iterations
max_candidates
max_generation_cost
max_eval_cost
max_runtime_cost_regression
stop_if_no_improvement_after_n
min_utility_delta
max_candidate_complexity
```

Stop conditions:

```text
budget_exhausted
max_iterations_reached
max_candidates_reached
no_improvement
static_gate_repeated_failures
evaluation_regression
operator_cancelled
policy_blocked
```

---

## 27. Observability

ADAS must emit observable traces for:

```text
objective creation
search run start/stop
candidate generation
scaffold materialization
static gate
evaluation
archive write
ranking
promotion request
approval
shadow
canary
verification
rollback
```

Every ADAS run must produce an auditable report:

```text
objective
baseline
candidates generated
candidates rejected
best candidate
utility trend
cost summary
gate failures
promotion decisions
rollback decisions
```

---

## 28. Reliability and failure modes

Failure modes:

```text
candidate generation failure
scaffold failure
static gate failure
evaluation failure
archive write failure
budget exhaustion
policy block
approval timeout
shadow regression
canary regression
verification failure
rollback failure
```

Each failure must have:

```text
structured error
event emission
archive record
operator-visible reason
safe default behavior
```

Safe default:

```text
reject candidate and archive reason
```

---

## 29. Enterprise anti-patterns

Forbidden:

```text
1. ADAS as a separate top-level architecture layer.
2. ADAS as Tier-3-only application.
3. MAS directly writes production code.
4. MAS edits existing production agent.
5. MAS edits Nexus runtime.
6. MAS bypasses scaffold.
7. MAS bypasses static gate.
8. MAS bypasses archive.
9. MAS bypasses evaluation.
10. MAS bypasses human approval.
11. ADAS duplicates evaluation registry.
12. ADAS duplicates observability.
13. ADAS duplicates governance.
14. ADAS duplicates promotion lifecycle.
15. ADAS stores candidates without lineage.
16. ADAS scores candidates without baseline comparison.
17. ADAS optimizes quality while ignoring cost.
18. ADAS optimizes benchmark only and ignores adversarial scenarios.
19. ADAS allows cross-tenant candidate leakage.
20. ADAS declares success without verification window.
```

---

## 30. Implementation plan

### Phase AHI-ADAS-00 — Documentation and ADR

Purpose:

```text
Document ADAS as AHI sub-capability.
```

Tasks:

```text
AHI-ADAS-00.1 Update AHI architecture with ADAS section.
AHI-ADAS-00.2 Add ADAS satellite architecture.
AHI-ADAS-00.3 Update AHI implementation plan.
AHI-ADAS-00.4 Add ADR explaining why ADAS lives inside AHI.
```

Acceptance:

```text
ADAS is not separate top-level layer.
ADAS is not Tier-3-only.
Existing AHI reuse is explicit.
No code implemented.
```

### Phase AHI-ADAS-10 — Contracts and Archive

Purpose:

```text
Define core data contracts and candidate archive.
```

Tasks:

```text
AHI-ADAS-10.1 Add AgentDesignObjective.
AHI-ADAS-10.2 Add AgentDesignSearchRun.
AHI-ADAS-10.3 Add AgentCandidateDraft.
AHI-ADAS-10.4 Add AgentCandidateRecord.
AHI-ADAS-10.5 Add AgentCandidateEvaluationResult.
AHI-ADAS-10.6 Add AgentDesignArchive protocol.
AHI-ADAS-10.7 Add in-memory archive.
AHI-ADAS-10.8 Add SQLite archive.
```

Acceptance:

```text
Candidate lineage persisted.
Candidate status persisted.
Evaluation result persisted.
Rejection reason persisted.
Tenant scoping enforced.
```

### Phase AHI-ADAS-20 — Scaffold Bridge and Static Gate

Purpose:

```text
Materialize candidate agents safely through existing scaffold.
```

Tasks:

```text
AHI-ADAS-20.1 Add AgentScaffoldBridge.
AHI-ADAS-20.2 Add sandbox output location.
AHI-ADAS-20.3 Add AgentStaticGate.
AHI-ADAS-20.4 Add forbidden import checks.
AHI-ADAS-20.5 Add AgentContract validation.
AHI-ADAS-20.6 Add capability validation.
AHI-ADAS-20.7 Add smoke test presence validation.
```

Acceptance:

```text
Candidates are scaffolded, not manually written.
Runtime is not edited.
Production agents are not overwritten.
Static gate blocks unsafe candidates.
```

### Phase AHI-ADAS-30 — Candidate Evaluation

Purpose:

```text
Evaluate candidate agents against baseline.
```

Tasks:

```text
AHI-ADAS-30.1 Add AgentCandidateEvaluator.
AHI-ADAS-30.2 Add baseline runner.
AHI-ADAS-30.3 Add candidate runner.
AHI-ADAS-30.4 Add golden scenario adapter.
AHI-ADAS-30.5 Add negative scenario adapter.
AHI-ADAS-30.6 Add adversarial scenario adapter.
AHI-ADAS-30.7 Add utility scoring.
AHI-ADAS-30.8 Add evaluation report.
```

Acceptance:

```text
Candidate is compared to baseline.
Utility delta is computed.
Cost and latency are measured.
Regression flags are captured.
Failed candidates are archived.
```

### Phase AHI-ADAS-40 — Search Controller and Strategies

Purpose:

```text
Orchestrate candidate generation and search lifecycle.
```

Tasks:

```text
AHI-ADAS-40.1 Add AgentDesignSearchController.
AHI-ADAS-40.2 Add AgentDesignContext.
AHI-ADAS-40.3 Add AgentDesignStrategy protocol.
AHI-ADAS-40.4 Add RuleBasedVariantStrategy.
AHI-ADAS-40.5 Add MetaAgentSearchStrategy stub.
AHI-ADAS-40.6 Add search budget policy.
AHI-ADAS-40.7 Add stop conditions.
```

Acceptance:

```text
Search run can generate candidates.
Each candidate is gated, evaluated, archived.
Search stops on budget/no improvement.
Best candidate is reported.
```

### Phase AHI-ADAS-50 — Hooks and Events

Purpose:

```text
Make ADAS enterprise-extensible without breaking core.
```

Tasks:

```text
AHI-ADAS-50.1 Add AgentDesignEvent.
AHI-ADAS-50.2 Add AgentDesignHook protocol.
AHI-ADAS-50.3 Add hook dispatcher.
AHI-ADAS-50.4 Add lifecycle events.
AHI-ADAS-50.5 Add operator override hook.
AHI-ADAS-50.6 Add custom evaluator hook.
AHI-ADAS-50.7 Add custom ranker hook.
```

Acceptance:

```text
Custom code can participate safely.
Hooks cannot bypass gates.
Events are emitted at every lifecycle step.
```

### Phase AHI-ADAS-60 — MAS Tier-2 Agents

Purpose:

```text
Add optional MAS agents that use ADAS.
```

Tasks:

```text
AHI-ADAS-60.1 Scaffold meta_agent_designer.
AHI-ADAS-60.2 Scaffold candidate_critic.
AHI-ADAS-60.3 Scaffold benchmark_runner.
AHI-ADAS-60.4 Add MAS prompt assets.
AHI-ADAS-60.5 Add MAS evaluation tests.
```

Acceptance:

```text
MAS is Tier-2.
MAS uses ADAS contracts.
MAS does not own control plane.
MAS does not write runtime.
```

### Phase AHI-ADAS-70 — Shadow / Canary / Promotion Bridge

Purpose:

```text
Safely promote candidate agents.
```

Tasks:

```text
AHI-ADAS-70.1 Add AgentCandidatePromotionRequest.
AHI-ADAS-70.2 Add shadow registration bridge.
AHI-ADAS-70.3 Add canary allocation bridge.
AHI-ADAS-70.4 Add human approval requirement.
AHI-ADAS-70.5 Add rollback pointer.
AHI-ADAS-70.6 Add candidate verification target.
```

Acceptance:

```text
No direct production apply.
Candidate must pass eval before shadow.
Human approval required by default.
Rollback path exists.
```

### Phase AHI-ADAS-80 — ADAS Lab Application

Purpose:

```text
Provide Tier-3 operator interface.
```

Tasks:

```text
AHI-ADAS-80.1 Scaffold adas_lab.
AHI-ADAS-80.2 Add objective creation API.
AHI-ADAS-80.3 Add archive browser API.
AHI-ADAS-80.4 Add approval API.
AHI-ADAS-80.5 Add report dashboard.
```

Acceptance:

```text
Tier-3 only wires ADAS.
No duplicated core logic in application.
Operators can approve/reject candidates.
```

### Phase AHI-ADAS-90 — Enterprise Hardening

Purpose:

```text
Production-readiness hardening.
```

Tasks:

```text
AHI-ADAS-90.1 Add multi-tenant isolation tests.
AHI-ADAS-90.2 Add budget exhaustion tests.
AHI-ADAS-90.3 Add rollback tests.
AHI-ADAS-90.4 Add static gate bypass tests.
AHI-ADAS-90.5 Add observability report.
AHI-ADAS-90.6 Add security review checklist.
```

Acceptance:

```text
Tenant isolation verified.
Unsafe candidates blocked.
Budget limits enforced.
Rollback verified.
Audit report generated.
```

---

## 31. Final enterprise statement

ADAS must be implemented as:

```text
a governed, observable, policy-bound, versioned, measurable,
enterprise-grade adaptive design loop inside AHI.
```

Not as:

```text
a free-form agent that writes other agents.
```

Final architectural rule:

```text
Every ADAS action must be explainable, traceable, reversible, measurable,
tenant-scoped, policy-gated, and archived.
```

---

## 32. Documentation follow-up

This satellite document should be followed by:

```text
1. Update docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md with a short canonical ADAS section.
2. Update docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md with Phase AHI-ADAS.
3. Add ADR explaining why ADAS belongs inside AHI.
4. Run an architecture audit before implementation.
```
