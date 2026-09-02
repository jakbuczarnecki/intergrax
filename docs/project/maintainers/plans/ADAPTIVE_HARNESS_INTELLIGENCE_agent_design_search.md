# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

# ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search - Implementation Plan

**Architecture (1:1):** [`architecture/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](../../architecture/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md)
**Parent architecture hub:** [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md)
**Parent plan hub:** [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md)  
**Detailed plan satellite:** [`satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md)  
**ADR:** [`ADR-ADAPT-002`](../../technical/adr/entries/2026-06-22/ADR-ADAPT-002.md)
**Status:** Planned - Phase **AHI-ADAS-00** complete; implementation begins at **AHI-ADAS-10**  
**Last updated:** 2026-06-22

---

## Cursor read scope

Use this file as the **top-level implementation plan entrypoint** for ADAS / Agent Design Search.

For detailed task tables, read the plan satellite on demand:

```text
satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md
```

Do not read the full AHI plan hub and full ADAS satellite in the same session unless the current task explicitly requires cross-document validation.

Recommended read sequence for implementation:

```text
1. docs/project/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md read-scope block
2. docs/project/architecture/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md relevant sections only
3. this file
4. docs/project/maintainers/plans/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md current phase only
```

---

## 1. Pre-implementation architecture audit

**Audit mode:** Mode I / slice audit before runtime implementation.  
**Audit scope:** ADAS architecture satellite, AHI hub canon, AHI plan hub, ADR-ADAPT-002.  
**Audit result:** **Passed with no blocking architecture issues.**

### 1.1 Evidence reviewed

| Artifact | Result |
|----------|--------|
| AHI hub canonical ADAS section | Present; states ADAS is an AHI Tier-1 sub-capability, not a separate layer |
| ADAS architecture satellite | Present; canonical; contains enterprise contracts, evidence bundle, retention, active registration semantics |
| AHI plan hub | Present; contains Phase AHI-ADAS register |
| ADR-ADAPT-002 | Present; accepted; records ADAS-inside-AHI decision |
| AHI existing components | ADAS explicitly reuses signal, utility, governance, verification, process mining, scaffold, observability, and promotion semantics |

### 1.2 Audit verdict

ADAS is ready for implementation planning because the documentation enforces these invariants:

```text
ADAS lives inside AHI, not beside it.
ADAS is not Tier-3-only.
MAS is a strategy, not the architecture.
Scaffold is the only candidate materialization path.
Candidates are gated, evaluated, archived, and promoted through governed lifecycle stages.
Production mutation requires explicit promotion and human approval by default.
```

### 1.3 Remaining implementation risks

| Risk | Mitigation |
|------|------------|
| Recreating AHI governance in ADAS | Reuse existing AHI governance / lifecycle semantics |
| MAS writing arbitrary code | Scaffold bridge and static gate are mandatory before evaluation |
| Candidate archive becoming unauditable | Evidence bundle, operational envelope, retention, tenant scoping |
| Promotion semantics drifting | Active registration modes A–E are explicit |
| Cost explosion | Budget policy and stop conditions before MAS agents |
| Tier leakage into applications | ADAS Lab delayed until Tier-1 control plane is complete |

---

## 2. Canonical implementation decision

ADAS must be implemented as:

```text
ADAPTIVE_HARNESS_INTELLIGENCE
  └── ADAS / Agent Design Search
        └── MAS / Meta Agent Search as one replaceable strategy
```

Implementation location:

```text
intergrax/runtime/adaptive/agent_design_search/
```

The implementation must reuse existing Intergrax capabilities:

```text
HarnessOutcomeSignal
utility scoring model
SignalCollector / SignalStore concepts
AdaptationEngine / proposal concepts
Adaptive governance envelopes
PolicyEngine boundary
AdaptationExecutor lifecycle semantics
VerificationLoop checks
ProcessPatternMiner
agent scaffold
AgentContract
observability events
evaluation registry / evaluation trends
cost governance
```

ADAS must not create parallel stacks for:

```text
governance
evaluation
tracing
scaffold
policy
promotion
signal collection
utility computation
```

---

## 3. Target implementation loop

```text
objective
  → baseline
  → candidate draft
  → scaffold bridge
  → static gate
  → offline evaluation
  → utility scoring
  → archive + evidence bundle
  → shadow/canary/promotion request
  → verification
  → keep or rollback
```

Implementation should stay conservative. The first MVP stops before promotion.

Recommended MVP:

```text
Generate 3 candidate drafts from deterministic strategy,
materialize through scaffold sandbox,
run static gate,
run offline evaluation,
archive every result,
produce ranked report.
```

---

## 4. Implementation phases

| Phase | Purpose | Status | Detailed plan |
|-------|---------|--------|---------------|
| **AHI-ADAS-00** | Documentation canon + ADR + implementation plan | **Done** (2026-06-22) | This file + satellite |
| **AHI-ADAS-10** | Core contracts + candidate archive | Planned | Satellite §6 / AHI-ADAS-10 |
| **AHI-ADAS-20** | Scaffold bridge + static gate | Planned | Satellite §6 / AHI-ADAS-20 |
| **AHI-ADAS-30** | Candidate evaluation + utility scoring | Planned | Satellite §6 / AHI-ADAS-30 |
| **AHI-ADAS-40** | Search controller + strategies | Planned | Satellite §6 / AHI-ADAS-40 |
| **AHI-ADAS-50** | Hooks and lifecycle events | Planned | Satellite §6 / AHI-ADAS-50 |
| **AHI-ADAS-60** | Optional Tier-2 MAS agents | Planned | Satellite §6 / AHI-ADAS-60 |
| **AHI-ADAS-70** | Shadow / canary / promotion bridge | Planned | Satellite §6 / AHI-ADAS-70 |
| **AHI-ADAS-80** | Optional Tier-3 ADAS Lab application | Planned | Satellite §6 / AHI-ADAS-80 |
| **AHI-ADAS-90** | Enterprise hardening | Planned | Satellite §6 / AHI-ADAS-90 |

Delivery rule:

```text
One AHI-ADAS-* phase or narrowly scoped sub-phase per PR.
Update this plan and the detailed satellite after each PR.
Link evidence bundle when evaluation or promotion gates apply.
```

---

## 5. Normative implementation principles

1. **Extend AHI, do not fork it.** ADAS lives under `intergrax/runtime/adaptive/agent_design_search` and reuses AHI patterns.
2. **No parallel governance stack.** Add adapters only where agent candidates differ from profile versions.
3. **No parallel scaffold.** Candidate materialization goes through `AgentScaffoldBridge` and existing scaffold.
4. **No direct production mutation.** Candidate source starts in sandbox/archive; production routing requires promotion.
5. **Evidence over declaration.** Promotion and verification require `AgentCandidateEvidenceBundle`.
6. **Human approval by default.** Production promotion requires approval unless a later explicit product gate permits low-risk lab auto-promotion.
7. **Tenant isolation by default.** Objectives, search runs, candidates, archive, and bundles are tenant-scoped.
8. **Budget before autonomy.** Search policy must exist before MAS can generate multiple candidates.
9. **Static gate before evaluation.** Unsafe candidates are rejected and archived before evaluation.
10. **Verification before success.** Promoted candidate is successful only after verification window.

---

## 6. Phase summaries

### AHI-ADAS-00 - Documentation canon and ADR

Status: **Done**.

Deliverables:

```text
ADAS architecture satellite
AHI hub canonical ADAS section
AHI plan hub Phase AHI-ADAS
ADR-ADAPT-002
enterprise details: operational envelope, evidence bundle, retention, active registration
implementation plan top-level file and satellite
```

### AHI-ADAS-10 - Contracts and candidate archive

Purpose: add minimal typed substrate without executing search.

Deliverables:

```text
intergrax/runtime/adaptive/agent_design_search/ package skeleton
AdasOperationalEnvelope
AgentDesignObjective
AgentDesignSearchRun
AgentCandidateDraft
AgentCandidateRecord + AgentCandidateStatus
AgentCandidateEvaluationResult
AgentCandidateEvidenceBundle
AgentDesignArchive protocol
in-memory archive
SQLite archive
lineage queries
package exports
unit tests
```

Gates:

```text
pytest tests/unit/runtime/adaptive/agent_design_search/
python -m compileall intergrax/runtime/adaptive/agent_design_search
```

### AHI-ADAS-20 - Scaffold bridge and static gate

Purpose: materialize candidates safely through existing scaffold and block unsafe candidates.

Deliverables:

```text
AgentScaffoldBridge
sandbox materialization path
scaffold pattern validation
scaffold output manifest
AgentStaticGate
AgentContract validation
capability validation
forbidden import/path checks
tool policy validation
smoke test presence check
static gate result model
unit tests
```

### AHI-ADAS-30 - Candidate evaluation and utility scoring

Purpose: compare candidate agents against baselines using existing evaluation, signal, cost, and regression concepts.

Deliverables:

```text
AgentCandidateEvaluator
baseline runner adapter
candidate runner adapter
golden scenario adapter
negative/adversarial scenario adapter
cost/latency/token collection
compute_agent_candidate_utility()
evidence bundle assembly for evaluation
archive write after evaluation
evaluation report helper
unit/integration tests
```

### AHI-ADAS-40 - Search controller and strategies

Purpose: orchestrate bounded candidate search with pluggable strategies.

Deliverables:

```text
AgentDesignSearchController
AgentDesignContext
AgentDesignSearchPolicy
stop conditions
ranking and selection
search result model
AgentDesignStrategy protocol
RuleBasedVariantStrategy
MetaAgentSearchStrategy stub
controller tests
```

### AHI-ADAS-50 - Hooks and lifecycle events

Purpose: make ADAS extensible without letting plugins bypass safety boundaries.

Deliverables:

```text
AgentDesignHook protocol
hook dispatcher
custom evaluator hook
custom ranker hook
AgentDesignEvent model
event emitter adapter
lifecycle event coverage
hook/event tests
```

### AHI-ADAS-60 - Optional Tier-2 MAS agents

Purpose: add optional strategy agents that consume ADAS contracts but do not own the control plane.

Deliverables:

```text
agents/meta_agent_designer/
meta_agent_designer contract
MAS prompt assets
agents/candidate_critic/
agents/benchmark_runner/
MAS strategy adapter
safety prompt/eval tests
integration tests
```

### AHI-ADAS-70 - Shadow / canary / promotion bridge

Purpose: safely make evaluated candidates routable through governed active-registration semantics.

Deliverables:

```text
AgentCandidatePromotionRequest
promotion gate adapter
shadow registration bridge
canary allocation bridge
human approval store/link
active registration mode A - registry pointer
active registration mode E - tenant/application binding
optional modes B–D declarations
rollback pointer model
candidate verification target
rollback integration
promotion tests
```

### AHI-ADAS-80 - Optional Tier-3 ADAS Lab application

Purpose: provide operator-facing workflow without moving ADAS logic into Tier-3.

Deliverables:

```text
applications/adas_lab/
objective creation API/screen
candidate archive browser
evidence bundle viewer
approval/rejection workflow
search run dashboard
rollback action wiring
tier boundary tests
```

### AHI-ADAS-90 - Enterprise hardening

Purpose: production-readiness controls for regulated or enterprise deployments.

Deliverables:

```text
retention policy enforcement
PII/secret scanning
legal hold support
tenant deletion / purge workflow
cross-tenant isolation tests
budget exhaustion tests
static gate bypass tests
evidence export
security review checklist
closeout report
```

---

## 7. Out of scope for v1

```text
Deep RL / neural policy training
Foundation model fine-tuning
Direct mutation of production agents
Direct writes to intergrax/runtime by MAS
Automatic production promotion for high-risk candidates
Cross-tenant learning without anonymization and governance approval
Standalone ADAS SaaS layer
Parallel evaluation registry
Parallel PolicyEngine
Parallel tracing system
```

---

## 8. Closeout criteria

ADAS implementation can be considered enterprise-ready only when:

```text
1. AHI-ADAS-10 through AHI-ADAS-70 are complete.
2. Candidate archive is tenant-scoped and append-only.
3. Static gate blocks unsafe candidates.
4. Candidate evaluation compares against baseline.
5. Evidence bundles can be sealed and exported.
6. Promotion requires approval by default.
7. Rollback restores prior active registration.
8. Hooks cannot bypass safety gates.
9. MAS is optional and Tier-2 only.
10. ADAS Lab, if present, only wires Tier-1 APIs.
11. Enterprise hardening gates pass.
```

Phase **AHI-ADAS-90** is required before claiming production enterprise readiness.
