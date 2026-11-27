# Adaptive Harness Evolution — Architecture

**Revision:** 1.3
**Document status:** ARCHITECT-DEFINED / OWNER_BOUNDARIES_ACCEPTED_UNDER_OD-AHE-02 / W0_CLOSED
**Parent architecture:** [ADAPTIVE_HARNESS_INTELLIGENCE.md](../ADAPTIVE_HARNESS_INTELLIGENCE.md)
**Implementation roadmap:** [ADAPTIVE_HARNESS_EVOLUTION.md](../../maintainers/plans/satellites/ADAPTIVE_HARNESS_EVOLUTION.md)

Adaptive Harness Evolution (AHE) is a **governed implementation program** within the existing Adaptive Harness Intelligence (AHI) domain. It is not a new architectural layer, runtime, or competing execution engine.

## Architecture acceptance register

Acceptance metadata update only. Normative architecture sections §5.1–§5.31 are unchanged.

| Item | Record |
|---|---|
| Operator decision | **OD-AHE-01** — `APPROVED` (2026-11-27). I approve the full scope of the presented operator decision, including the limited historical waiver **F-AHE-01**. This decision does not authorize product-code implementation, production authority, replacement of external program-owner acceptance, or G6/IIAC certification. |
| Independent documentation qualification | **AHE-W0-IQ-01** — audited SHA `1d0181051fa034b5f45101755002e48f5dc35715`; architecture revision **1.3**; roadmap revision **1.3**. Qualification scope: architecture and roadmap documentation only. |
| Qualified architecture documentation SHA | `1d0181051fa034b5f45101755002e48f5dc35715` |
| Reuse baseline SHA (separate) | `a4ff0130dbaf054bfdaeb8522644c2f8b39d28ac` (23-entry reuse matrix in §5.16 unchanged) |
| Historical publication finding | **F-AHE-01** — `HISTORICAL_NONCOMPLIANCE_ACCEPTED_BY_OPERATOR` under **OD-AHE-01**; historical publication compliance for the three specified commits remains **FAIL**; corrective publication evidence SHA `1d0181051fa034b5f45101755002e48f5dc35715` (corrective commit metadata **PASS**). |
| External program owner acceptance | **PENDING** |
| Executable behavioral qualification | **NOT_EXECUTED** |
| Operator decision (ownership boundaries) | **OD-AHE-02** — `APPROVED` (2026-11-27). As operator and authorized owner of the stated domains, I approve the full presented scope of responsibility boundaries, preparation of exact future canonical ADAS register updates, and formal W0 settlement. Does not approve a product implementation ticket. |
| W0 program closure | **CLOSED** under architect verdict **AHE-G0 ACCEPTED_WITH_HISTORICAL_WAIVER**; settlement operation **AHE-W0-SETTLEMENT-01** reviewed commit `550b258bf8d524d907b7720da41ecac64a6bb1a5`; verification authority ChatGPT architecture review; publication scope two AHE documentation files; normative architecture unchanged relative to `1d0181051fa034b5f45101755002e48f5dc35715`; publication metadata **PASS**; remote reachability **PASS**. |
| Cross-program boundary status | **ARCHITECT_DEFINED / OWNER_BOUNDARIES_ACCEPTED_UNDER_OD-AHE-02** (§5.18) |

**Independent auditor final verdict (preserved):** `CONDITIONAL_ARCHITECTURE_QUALIFIED / PROGRAM_G0_NOT_CLOSABLE`

**Operator disposition (separate; not a re-issued auditor verdict):** `INDEPENDENT_DOCUMENTATION_QUALIFICATION_ACCEPTED_SCOPED_UNDER_OD-AHE-01`

Scoped qualification verdicts recorded under **AHE-W0-IQ-01**: architecture **PASS_SCOPED**; roadmap **PASS_SCOPED**; canonical ADAS compatibility **PASS_ARCHITECTURE_ONLY**; reuse **PASS**; cross-program boundaries **PASS_ARCHITECT_DEFINED**; W1 implementation boundary **PASS_DEFINITION**; enterprise safety **PASS_INVARIANTS_ON_PAPER**; historical publication governance **FAIL_WITH_OPERATOR_WAIVER**; executable behavioral qualification **NOT_EXECUTED**; G6 certification **NOT_CLAIMED**; IIAC certification **NOT_CLAIMED**.

---

## 5.1 Purpose

Adaptive Harness Evolution is a governed implementation program delivering experimental harness configuration improvement and evidence-driven experience reuse within the existing Adaptive Harness Intelligence domain.

Its objective is to improve verified task success, quality, cost efficiency, human intervention requirements and generalization without modifying foundation-model weights or expanding execution authority.

Google Harness² is a research reference, not an architectural dependency. See [5.15 Research traceability](#515-research-traceability).

The program shall use **REUSE**, **WIRING**, **EXTEND** and **NEW** in that order.

---

## 5.2 Architectural ownership

AHE owns **program delivery coordination**, not independent runtime semantic authority. No duplicate execution engine, evaluation engine, memory store, approval authority, resource budget authority or promotion mechanism shall be created.

| Concern | Canonical owner |
|---|---|
| Verified execution outcome collection | Existing Execution and AHI signal mechanisms |
| Adaptation proposals and configuration lifecycle | AHI |
| Candidate design, search and composition | ADAS within AHI |
| Experimental task execution | Experimentation and canonical Execution Runtime |
| Candidate evaluation and comparative evidence | Evaluation and Decision Verification |
| Procedural knowledge and experience retrieval | Memory and Context Engineering |
| Recovery strategy experience | Self-Healing |
| Skills, tools and capability definitions | Existing capability owners |
| Resource budgets and consumption accounting | Existing execution and cost governance owners |
| Promotion, approval and rollback | Existing AHI and governance authorities |
| Goal-level coordination | Integrax Intelligence |
| Scenario-specific business requirements | Scenario Factory and scenario owners |

---

## 5.3 Functional capability inventory

The following eight vectors describe **behavioral capabilities**, not eight new runtime packages.

1. **Harness Design Intelligence:** design alternative authorized harness configurations using task requirements, verified execution outcomes and scoped experience.
2. **Experimental Intelligence:** execute isolated, bounded and comparable trials through canonical runtime mechanisms.
3. **Composition Intelligence:** combine independently evaluated configuration changes and verify interactions.
4. **Selection Intelligence:** compare candidates using deterministic evidence, business criteria, cost, safety and advisory model-based assessment.
5. **Experience Intelligence:** retain verified strategy knowledge with provenance, applicability, limitations and outcome references.
6. **Continual Improvement:** reuse validated experience in subsequent configuration design.
7. **Resource Intelligence:** bound and optimize experiment count, token consumption, execution time and cost.
8. **Governed Evolution:** ensure scoped authority, version identity, independent acceptance, recoverability and safe promotion.

---

## 5.4 Canonical adaptive lifecycle

Target lifecycle:

```text
Baseline resolution
  → Scoped experience retrieval
  → Baseline execution
  → Diagnosis
  → Candidate proposal
  → Candidate validation
  → Experimental execution
  → Evaluation
  → Candidate composition, when applicable
  → Composite re-evaluation, when applicable
  → Selection or abstention
  → Experimental evidence archival
  → Experience qualification
  → Authorized experience publication
  → Optional governed configuration promotion
  → Verification and rollback
```

Rules:

- The accepted objective, task inputs and acceptance criteria remain unchanged across comparable trials.
- A candidate may modify only explicitly admitted configuration surfaces (see [5.5 Configuration surfaces](#55-configuration-surfaces)).
- Every execution must use the canonical execution authority.
- Every proposed change must have traceable lineage.
- Composed candidates must be evaluated again.
- The system must retain the baseline or return no accepted candidate when evidence is insufficient.
- A candidate selection decision does not authorize production promotion.
- A research or LAB outcome does not establish production readiness.

Normative distinctions:

- Experimental evidence archival records successful, unsuccessful and rejected experiments.
- Experimental evidence archival is not experience acceptance.
- Candidate selection is not production promotion authority.
- Experience qualification is required before an experience may influence future proposals as trusted knowledge.
- Rejected and failed candidates may contribute qualified negative lessons but cannot be represented as successful strategies.
- Raw experimental evidence, trusted reusable experience and promoted configuration remain separate lifecycle concepts.
- The minimum W1 scope ends at experimental evidence archival.
- W1 does not perform trusted experience publication or production promotion.

---

## 5.5 Configuration surfaces

**Initial admitted candidate surfaces:**

- Prompt and instruction references.
- Existing skill selection and configuration.
- Existing authorized tool selection and guidance.
- Existing model and routing configuration.
- Existing execution strategy configuration.
- Existing bounded context and retrieval configuration.
- Existing authorized agent composition.

**Excluded from automatic experimental mutation:**

- Security policies and authorization boundaries.
- Active production configuration.
- Platform source code.
- Executable plugin installation.
- Credential configuration.
- Tenant ownership and access boundaries.

New executable skills, scripts, tools or plugins must enter their existing engineering, security and admission lifecycle.

W1 execution boundaries:

- The complete admitted-surface inventory is a target architecture, not blanket authorization for W1.
- W1 shall initially evaluate one bounded change to an already admitted execution configuration.
- Prefer existing prompt/instruction references or an existing execution-strategy choice.
- Keep model, tools, tenant, objective, dataset, permissions and external-effect policy unchanged between baseline and candidate.
- No generated executable code, newly installed skills, new plugins, new capability registration or security-policy mutation is permitted in W1.

---

## 5.6 Propose–Probe–Compose

### Propose

Generate bounded alternatives against an immutable baseline. Distinguish invariant execution improvements from domain-specific knowledge and procedures.

### Probe

Evaluate candidates using comparable datasets, task definitions, model settings, resource envelopes and execution conditions. Prevent real external effects during experimental execution.

### Compose

Combine only compatible, evidence-supported configuration changes. Validate the resulting candidate as a new version and repeat the required evaluation. Do not infer that individually successful changes remain successful when combined.

---

## 5.7 Candidate selection

Selection must consider:

- Deterministic acceptance criteria.
- Business correctness.
- Safety and authorization.
- Output completeness.
- Cost and latency.
- Human intervention.
- Evidence quality.
- Uncertainty.

Support blinded candidate comparison when subjective evaluation is appropriate. Candidate anonymization must hide origin and experimental arm information while preserving legitimate content required for evaluation.

Model-based judgments are advisory and cannot override deterministic failure, mandatory human review or safety gates.

A **baseline fallback** and a **no-acceptable-candidate** outcome must both be supported.

---

## 5.8 Experience ownership and reuse

Experience must retain or reference:

- Tenant and workspace scope.
- Task class and domain.
- Baseline and candidate version identities.
- Strategy and configuration changes.
- Applicable conditions.
- Outcome and evaluation references.
- Failure modes and counterexamples.
- Provenance.
- Freshness.
- Confidence and limitations.
- Retention and deletion requirements.

Reuse existing profile, experiment, evaluation and memory authorities. Do not duplicate complete artifacts when canonical references are sufficient.

- **AHI / ADAS** owns candidate lineage and configuration semantics.
- **Memory** owns authorized knowledge retention and retrieval.
- **Self-Healing** retains ownership of recovery-specific strategy knowledge.

Experience must never confer execution permission. Cross-tenant sharing is prohibited unless a separately governed, explicitly authorized and sanitized sharing mechanism exists.

---

## 5.9 Resource governance

All experiment sessions require finite limits for:

- Number of candidates.
- Number of executions.
- Token consumption.
- Monetary cost.
- Wall-clock duration.
- Parallel work.

Resource reservations and accounting must use existing platform owners. The adaptive budget policy may recommend additional experimentation only within an existing authorized aggregate budget.

Early stopping must support exhausted budget, insufficient expected improvement, invalid candidates, safety failures and achievement of the defined objective.

`RunBudget` is an existing per-execution budget carrier. It must not be represented as proof of aggregate search-budget enforcement.

From W1, every ADAS search must operate under:

- Finite candidate count.
- Finite total execution count.
- Finite aggregate token allowance.
- Finite aggregate cost allowance.
- Finite wall-clock deadline.
- Explicit cancellation and stop conditions.
- Canonical per-execution resource enforcement.

W1 is sequential and uses at most one candidate and two task executions: baseline and candidate.

Any additional model call used to produce, diagnose or evaluate a proposal must be included in the applicable aggregate budget.

Do not dispatch work when required resource admission or accounting is unavailable.

Do not introduce a competing platform-wide budget authority.

Dynamic budget allocation, expected-value experimentation and optimized stopping remain W5 scope.

---

## 5.10 Enterprise safety invariants

AHE must not:

1. Expand permissions.
2. Bypass canonical execution.
3. Bypass mandatory governance.
4. Duplicate external business effects.
5. Access another tenant's private data or experience.
6. Treat selected candidates as automatically promoted.
7. Treat model judgments as authoritative correctness proof.
8. Treat missing mandatory evidence as PASS.
9. Modify active production configuration without canonical authorization.
10. Perform uncontrolled or unbounded recursion.
11. Self-approve or self-certify.
12. Introduce a competing truth owner.

Enforce tenant isolation and authority constraints from the first executable wave.

LAB experiments may not create real business side effects. Production promotion remains disabled until its entire required approval, persistence, consistency, qualification and recovery chain has been accepted.

---

## 5.11 Existing implementation anchors

Reference existing mechanisms rather than creating replacements. Preserve the distinction between implemented code, verified runtime wiring, scoped certification and production qualification.

| Mechanism | Path |
|---|---|
| Adaptation engine | `intergrax/runtime/adaptive/adaptation_engine.py` |
| Adaptation executor | `intergrax/runtime/adaptive/adaptation_executor.py` |
| Governance pipeline | `intergrax/runtime/adaptive/governance_pipeline.py` |
| Adaptation scheduler | `intergrax/runtime/adaptive/adaptation_scheduler.py` |
| Profile version store | `intergrax/runtime/adaptive/profile_version_store.py` |
| Verification loop | `intergrax/runtime/adaptive/verification_loop.py` |
| Process pattern miner | `intergrax/runtime/adaptive/process_pattern_miner.py` |
| Evaluation feedback engine | `intergrax/runtime/adaptive/evaluation_feedback_engine.py` |
| Experiments workflow | `intergrax/experiments/workflow.py` |
| Eval comparison | `intergrax/eval/eval_comparison.py` |
| Procedural memory service | `intergrax/memory/procedural_memory_service.py` |
| Self-healing knowledge evolution | `intergrax/runtime/self_healing/knowledge_evolution/` |

**ADAS** remains a planned capability where executable implementation has not been demonstrated. Do not assert new certification claims.

---

## 5.12 Existing governance limitations

Reuse canonical ownership of existing AHI improvement obligations (see [ADAPTIVE_HARNESS_INTELLIGENCE plan](../../maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md)):

- **AHI-PROMOTION-AUTHORITY-INTEGRITY**
- **AHI-EVIDENCE-QUALIFICATION-INTEGRITY**
- **AHI-ACTIVATION-CONSISTENCY-INTEGRITY**

AHE must not create parallel remediation initiatives.

Record the existing default **in-memory approval** and **procedural-memory persistence** limitations documented for AHI. These limitations do not prohibit authorized isolated LAB research, but they prevent unsupported production durability or promotion claims.

---

## 5.13 Integration with Integrax Intelligence

Integrax Intelligence retains ownership of goal-level coordination, its consumer wiring and program-level acceptance. AHE does not authorize changes to Intelligence's master plan or automatically close its packages.

AHE shall supply reusable capability and qualification evidence. Proposed evidence and dependency mapping:

| Intelligence package | AHE contribution |
|---|---|
| II-W7-01 | Verified adaptive outcome signals |
| II-W7-02 | Validated strategy patterns and composition evidence |
| II-W7-03 | Scoped experience retention and retrieval |
| II-W7-04 | Configuration and resource optimization recommendations |
| II-W7-05 | Governed and durable adaptation evidence |
| II-W7-06 | Controlled experiments, qualification, promotion and rollback evidence |

Consumer register: [INTEGRAX_INTELLIGENCE.md](../../capabilities/plan/INTEGRAX_INTELLIGENCE.md).

---

## 5.14 Scenario Factory adoption

Scenario Factory shall consume accepted AHE capabilities through the existing platform boundaries.

AHE may support scenario design, comparative evaluation and configuration optimization before repetitive execution. Scenario Factory shall not create a private adaptive harness.

AHE completion is not an additional ARCH-FREEZE gate or an automatically mandatory prerequisite for starting Scenario Factory. Production admission remains owned by existing scenario and live-environment authorities.

---

## 5.15 Research traceability

**References (research only, not architectural dependencies):**

- [Harness² paper](https://arxiv.org/abs/2610.14224)
- [Harness² source (pinned ref)](https://github.com/google-research/harness2/tree/6868817518876de8fd170c6a8354daca273b6191)

**Adopted research mechanisms:**

- Propose–Probe–Compose
- Invariant versus domain-specific refinement
- Blinded candidate selection
- Experience retention and reuse
- Bounded experimentation
- Evidence-driven continual improvement

Distinguish implemented Harness² mechanisms from AHE-specific extensions such as dynamic resource allocation. No unsupported superiority claim is permitted.

## 5.16 Exact-SHA ownership and reuse baseline

**Baseline:** `a4ff0130dbaf054bfdaeb8522644c2f8b39d28ac`

Architecture decisions are supplied by ChatGPT.

| ID | Mechanism | Existing source | Disposition | Scope |
|---|---|---|---|---|
| AHE-R01 | Execution outcome collection | `intergrax/runtime/adaptive/signal_emission.py` | REUSE/WIRING | W1+ |
| AHE-R02 | Adaptation proposal engine | `intergrax/runtime/adaptive/adaptation_engine.py` | REUSE | W1+ |
| AHE-R03 | Profile versions | `intergrax/runtime/adaptive/profile_version_store.py` | REUSE | W1+ |
| AHE-R04 | Existing profile contracts | `intergrax/runtime/adaptive/contracts.py` | REUSE | W1+ |
| AHE-R05 | Experiment lifecycle | `intergrax/experiments/workflow.py` | REUSE/WIRING | W1+ |
| AHE-R06 | Experiment identity and persistence | `intergrax/experiments/models.py`, `persistence_contract.py`, `store.py` | REUSE | W1+ |
| AHE-R07 | Canonical task execution | `intergrax/runtime/task/unified_task_runner.py` | REUSE | W1+ |
| AHE-R08 | Runtime evaluation | `intergrax/eval/nexus_eval_runner.py` | REUSE/WIRING | W1+ |
| AHE-R09 | Typed evaluation runs | `intergrax/eval/typed_run.py` | REUSE/WIRING | W1+ |
| AHE-R10 | Comparative evaluation | `intergrax/eval/eval_comparison.py` | REUSE | W1+ |
| AHE-R11 | Existing utility calculation | `intergrax/runtime/adaptive/utility.py` | REUSE | W1+ |
| AHE-R12 | Per-run resource limits | `intergrax/contracts/run_budget.py` | REUSE | W1+ |
| AHE-R13 | Sandbox and isolation | `intergrax/runtime/sandbox/` | REUSE/WIRING | W1+ |
| AHE-R14 | Candidate contracts and archive | AHI-ADAS-10 | IMPLEMENT EXISTING PLANNED SCOPE | W1 |
| AHE-R15 | Candidate static validation | AHI-ADAS-20 | IMPLEMENT/WIRING | W1/W2 |
| AHE-R16 | Candidate evaluation facade | AHI-ADAS-30 | IMPLEMENT/WIRING | W1/W3 |
| AHE-R17 | Search controller and strategies | AHI-ADAS-40 | IMPLEMENT/WIRING | W1/W2/W5 |
| AHE-R18 | Procedural knowledge | `intergrax/memory/procedural_memory_service.py` | REUSE/WIRING | W4 |
| AHE-R19 | Process pattern evidence | `intergrax/runtime/adaptive/process_pattern_miner.py` | REUSE/WIRING | W4 |
| AHE-R20 | Recovery-specific experience | `intergrax/runtime/self_healing/knowledge_evolution/` | REUSE/WIRING | W4 |
| AHE-R21 | Promotion lifecycle | `intergrax/runtime/adaptive/adaptation_executor.py` | REUSE/QUALIFY | W6 |
| AHE-R22 | Adaptation governance | `intergrax/runtime/adaptive/governance_pipeline.py` | REUSE/QUALIFY | W6 |
| AHE-R23 | Verification and rollback | `intergrax/runtime/adaptive/verification_loop.py` | REUSE/QUALIFY | W6 |

Evidence limitations:

- **REUSE** means the existing mechanism is the selected owner and implementation anchor. It does not assert that every required integration path is already qualified.
- **WIRING** means an existing mechanism must be connected and its behavior verified within the AHE composition.
- **IMPLEMENT EXISTING PLANNED SCOPE** means the missing capability belongs to the existing ADAS architecture, rather than a new AHE runtime subsystem.
- **QUALIFY** means the selected existing mechanism must demonstrate the required behavior in the new composition.
- A source-code path is not by itself proof of production readiness.

## 5.17 ADAS canonical contract ownership

The implementation shall retain the existing ADAS contract and component vocabulary:

- AdasOperationalEnvelope
- AgentDesignObjective
- AgentDesignSearchRun
- AgentCandidateDraft
- AgentCandidateRecord
- AgentCandidateStatus
- AgentCandidateEvaluationResult
- AgentCandidateEvidenceBundle
- AgentDesignArchive
- AgentScaffoldBridge
- AgentStaticGate
- AgentCandidateEvaluator
- AgentDesignSearchController
- AgentDesignContext
- AgentDesignSearchPolicy
- AgentDesignStrategy
- RuleBasedVariantStrategy

The canonical runtime package location remains:

`intergrax/runtime/adaptive/agent_design_search/`

AHE is not a new runtime package.

No competing AHE candidate, archive, evaluator or search-controller contract shall be created.

Use existing canonical references and contracts wherever they satisfy the required semantics.

`ProfileVersionStore` remains the owner of existing AHI profile versions.

`AgentDesignArchive` owns ADAS candidate lineage and candidate-specific evidence references as already prescribed by ADAS.

It is not a second profile-version store, general experiment store, general evaluation store or experience-memory system.

The W1 candidate is configuration-only.

Executable agent scaffolding may be deferred until specifically authorized and required. Do not mark the entire AHI-ADAS-20 phase complete from configuration-only validation evidence.

## 5.18 Cross-program implementation boundaries

- ARCH-FREEZE remains unchanged.
- HARNESS-W6 certification is scoped to its accepted baseline and does not automatically certify ADAS or AHE.
- Existing AHI remediation obligations remain under their canonical owners.
- AHE supplies reusable adaptive capability and evidence to Intelligence W7.
- Intelligence retains its consumer and program-level acceptance responsibilities.
- AHE shall not implement competing versions of Intelligence W7 mechanisms.
- Changes to protected or shared mechanisms require explicit authorization from the relevant authority.
- Parallel development is allowed only within the authorized non-conflicting scope.
- Product admission and production promotion remain separately governed.

**Cross-program boundary status:** ARCHITECT_DEFINED / OWNER_BOUNDARIES_ACCEPTED_UNDER_OD-AHE-02

**OD-AHE-02 approved ownership boundaries (2026-11-27):**

| Domain | Acceptance |
|---|---|
| AHI / ADAS | AHE delivers existing canonical ADAS obligations without competing implementation |
| Integrax Intelligence | Retains consumer wiring, roadmap and program-level acceptance |
| Scenario Factory | Consumes accepted capabilities through existing platform interfaces |
| ARCH-FREEZE | No scope expansion and no additional freeze gates |
| Execution | Retains canonical execution authority |
| Evaluation | Retains canonical evaluation authority |
| Memory | Retains trusted experience authority |
| Governance | Retains admission, approval and promotion authority |

**Acceptance authority:** Operator acting within the explicitly declared ownership authority. This record does not represent separately issued signatures from other individuals.

Do not assert that another program has accepted this mapping beyond the recorded operator boundary acceptance under **OD-AHE-02**.

## 5.19 Minimum Adaptive Loop — W1

W1 delivers a deterministic, configuration-only, sequential adaptive search.

The first implementation shall consist of:

1. Objective and scope validation.
2. Immutable baseline resolution.
3. Aggregate resource admission.
4. Canonical baseline execution.
5. Deterministic candidate proposal.
6. Static candidate validation.
7. Canonical candidate execution.
8. Comparative evaluation.
9. Candidate selection, baseline retention or abstention.
10. Experimental evidence archival.

W1 shall not contain:

- Model-driven proposal generation.
- Autonomous source-code generation.
- Multiple candidate search.
- Candidate composition.
- Trusted experience publication.
- Production promotion.
- Automatic policy modification.
- Real external business effects.

The W1 deterministic strategy qualifies the adaptation mechanism and its execution composition. It does not demonstrate general intelligent harness evolution.

## 5.20 Canonical implementation package

All new ADAS runtime components belong to:

`intergrax/runtime/adaptive/agent_design_search/`

Prescribed W1 package structure:

- `__init__.py`
- `contracts.py`
- `archive.py`
- `sqlite_archive.py`
- `static_gate.py`
- `search_policy.py`
- `strategies.py`
- `candidate_evaluator.py`
- `experiment_bridge.py`
- `search_controller.py`

These are planned files, not implemented files.

AHE shall not create a competing top-level runtime package.

Existing owner mechanisms must remain authoritative.

## 5.21 W1 contract inventory

Retain the canonical ADAS names.

### Canonical ADAS contract compatibility

The canonical ADAS architecture, particularly its §13 data contracts, remains authoritative for contract identity and compatibility.

AHE W1 defines a restricted behavioral profile of ADAS. It does not define replacement ADAS contracts.

The semantic field inventories below are additive W1 requirements, not exhaustive replacement schemas.

The W1 implementation shall preserve canonical ADAS field identities, meanings, provenance and compatibility requirements.

Additional W1 fields may strengthen:

- Tenant and workspace binding.
- Immutable configuration references.
- Accepted scope provenance.
- Aggregate resource limits.
- Dataset and configuration digests.
- Mandatory evidence completeness.
- Explicit accounting uncertainty.
- LAB-only execution posture.

Do not remove canonical fields solely because W1 does not exercise their associated behavior.

Where a canonical field requires an artifact or lifecycle capability not yet available in W1, use explicit absence or deferred capability semantics only under an architect-approved executable contract.

Do not fabricate scaffold artifacts, approval references, promotion references or production evidence.

Any required canonical schema amendment must be explicitly authorized before implementation. This documentation task does not authorize amendments to canonical ADAS documents.


**`AdasOperationalEnvelope`**

Required semantic fields:

- Schema version.
- Tenant identity.
- Workspace identity.
- Actor identity.
- Correlation identity.
- Trace identity or reference.
- Accepted scope reference.
- Explicit LAB-only execution posture.
- Retention policy reference.

An envelope records authority provenance. It does not grant authority.

**`AgentDesignObjective`**

Required semantic fields:

- Objective identity.
- Operational envelope.
- Task class.
- Capability reference.
- Immutable baseline configuration reference.
- Dataset identity and content digest.
- Acceptance criteria reference.
- Admitted configuration surface.
- Allowed configuration alternatives.
- Aggregate search limits.
- Explicit external-effect prohibition.

**`AgentDesignSearchRun`**

Required semantic fields:

- Search run identity.
- Objective identity.
- Scope binding.
- Baseline identity.
- Candidate identities.
- Execution references.
- Search lifecycle status.
- Start and terminal timestamps.
- Consumption and accounting references.
- Explicit stop reason.

**`AgentCandidateDraft`**

Required semantic fields:

- Candidate identity.
- Objective and search-run identity.
- Scope binding.
- Baseline identity and digest.
- Single admitted configuration delta.
- Candidate configuration digest.
- Proposal strategy reference.
- Proposal rationale.

Candidate drafts must not contain executable source code, arbitrary callable references, credentials or new permissions.

**`AgentCandidateStatus`**

Preserve the canonical `AgentCandidateStatus` members and serialized values:

- DRAFT = draft
- SCAFFOLDED = scaffolded
- STATIC_GATE_FAILED = static_gate_failed
- STATIC_GATE_PASSED = static_gate_passed
- EVALUATED = evaluated
- REJECTED = rejected
- SHADOW = shadow
- CANARY = canary
- PROMOTED = promoted
- RETIRED = retired
- ARCHIVED = archived

W1 shall exercise only the lifecycle states applicable to configuration-only LAB experiments.

W1 shall not exercise scaffold materialization, shadow, canary, production promotion or production retirement.

The former W1 VALIDATED concept maps to STATIC_GATE_PASSED.

Static validation rejection maps to STATIC_GATE_FAILED and preserves its structured rejection evidence.

Do not introduce SELECTED, NOT_SELECTED or FAILED as replacement candidate lifecycle states.

Selection decisions remain separate semantic outcomes:

- SELECT_CANDIDATE
- RETAIN_BASELINE
- ABSTAIN

Execution failures and cancellation belong to explicit search-run outcomes, stop reasons and evidence.

Candidate archival operations must preserve the actual historical lifecycle state. Persisting a snapshot does not automatically transition the candidate to ARCHIVED.

Selection must never imply activation or promotion.

**`AgentCandidateRecord`**

An immutable candidate snapshot containing identity, lineage, configuration references, status, timestamps and evidence references.

Status changes must produce additional immutable historical records rather than silently overwrite historical evidence.

**`AgentCandidateEvaluationResult`**

Contains:

- Baseline and candidate execution references.
- Canonical EvalRun references.
- Canonical EvalComparison reference or typed result.
- Required evidence completeness verdict.
- Mandatory acceptance verdict.
- Utility result when sufficient evidence exists.
- Failure and rejection reasons.

**`AgentCandidateEvidenceBundle`**

Contains immutable references and digests sufficient to verify:

- Accepted objective and configuration boundaries.
- Baseline identity.
- Candidate identity and lineage.
- Static validation.
- Actual execution identities.
- Evaluation inputs and outcomes.
- Resource evidence.
- Selection outcome.

Evidence hashes must be based on a deterministic canonical serialization.

Do not treat hashes alone as authorization or correctness proof.

## 5.22 Candidate archive

`AgentDesignArchive` is the canonical ADAS archive protocol.

Candidate archival operations must preserve the actual historical lifecycle state. Persisting a snapshot does not automatically transition the candidate to ARCHIVED.

Required operations:

- Append immutable candidate records.
- Retrieve candidate records within an explicit scope.
- Retrieve candidate history.
- List candidates by scoped objective.
- Retrieve candidate lineage.
- Append and retrieve sealed evidence bundles.

Implementations planned for W1:

- In-memory archive.
- SQLite archive.

Every public retrieval operation must require explicit tenant and workspace scope.

Cross-tenant and cross-workspace retrieval must be rejected without exposing foreign record contents.

Candidate identity, objective identity, scope, baseline lineage and configuration digest must remain consistent across lifecycle transitions.

A duplicate identity with different immutable content must fail.

An exact duplicate append may be idempotent.

Both archive implementations must expose equivalent semantic behavior.

SQLite persistence must survive close and reopen.

Use an explicit database path in LAB tests.

No production database migration or default production database initialization.

The archive stores candidate-specific lineage and evidence references. It must not duplicate the general experiment, evaluation, profile-version or memory authorities.

## 5.23 Scope and baseline binding

The controller must obtain its accepted scope from existing authoritative admission context or a separately admitted LAB composition.

A tenant or workspace string supplied by the candidate is not sufficient authority.

When resolving a ProfileVersionStore reference, validate:

- Exact version identity.
- Tenant identity.
- Task class.
- Expected artifact type.
- Configuration digest.
- Accepted scope.

`ProfileVersionStore.get(version_id)` is not itself a tenant-authorized retrieval proof.

Reject foreign or inconsistent records before candidate execution.

Do not modify the existing ProfileVersionStore in W1.

## 5.24 Static candidate gate

`AgentStaticGate` shall validate configuration-only candidates.

The W1 gate must reject:

- Foreign tenant or workspace.
- Incorrect objective or search-run binding.
- Incorrect baseline identity or digest.
- Unknown configuration surface.
- Configuration value outside the explicit allowed set.
- Multiple configuration changes.
- No-op configuration changes.
- Permission expansion.
- Model, tool, capability or policy modification outside accepted scope.
- Generated executable code.
- Changes to external-effect policy.
- Missing mandatory provenance.

The gate must produce a structured rejection reason.

Successful static validation transitions the candidate to STATIC_GATE_PASSED before experimental execution.

Static validation rejection transitions the candidate to STATIC_GATE_FAILED, archives structured rejection evidence and must not proceed to experimental execution.

Rejected candidates must be archived.

No rejected candidate may reach experimental execution.

Full executable agent scaffold materialization remains outside W1.

Do not declare AHI-ADAS-20 complete from configuration-only evidence.

**Configuration-only execution boundary**

A configuration-only variant of an existing authorized execution composition is not a newly materialized executable agent.

W1 may resolve and execute approved configuration alternatives through existing host composition, provided that agent identities, executable artifacts, capabilities, tool permissions and execution authority remain unchanged.

Any creation or materialization of a new executable agent candidate remains subject to the canonical AgentScaffoldBridge and existing scaffold authority.

W1 must not create an alternative scaffold or materialization path.

The LAST_WINS versus CONCAT witness changes only an admitted configuration surface of an existing execution composition.

This scope does not qualify AgentScaffoldBridge or complete AHI-ADAS-20.


## 5.25 Search policy and resource control

`AgentDesignSearchPolicy` owns bounded ADAS search coordination only.

Existing platform runtime remains authoritative for resource enforcement.

W1 limits:

- At most one proposed candidate.
- At most two root task executions.
- Sequential execution.
- Explicit bounded child work inside each root execution.
- Finite aggregate token limit.
- Finite aggregate monetary cost limit.
- Finite deadline.
- No uncontrolled retries.
- Explicit cancellation.

Reuse:

- `intergrax/contracts/run_budget.py`
- `intergrax/contracts/execution_budget_ledger_port.py`
- `intergrax/runtime/execution/budget/ledger.py`
- `intergrax/runtime/architecture/cost_budget.py`

RunBudget and the execution ledger are per-execution mechanisms.

BudgetEnvelope evaluation is not by itself proof of hard monetary enforcement.

ADAS shall coordinate a fixed search envelope and canonical per-run limits without introducing a second platform budget authority.

Before each execution:

- Check remaining search allowance.
- Check remaining deadline.
- Check cancellation.
- Obtain valid canonical resource admission.
- Ensure the maximum admitted consumption fits the remaining allowance.

After each execution:

- Reconcile authoritative resource consumption.
- Record evidence references.
- Do not silently substitute zero for unknown consumption.
- Do not dispatch another execution while previous consumption is unknown or unreconciled.

W1 behavioral qualification shall use a deterministic offline LAB with zero external provider calls and zero external monetary charges.

Do not claim qualification of non-zero-cost provider admission from this offline witness.

## 5.26 Candidate generation strategy

Define the canonical `AgentDesignStrategy` protocol.

Its responsibility is proposal generation only.

It cannot execute candidates, promote configuration, modify runtime state or grant permissions.

Implement the planned `RuleBasedVariantStrategy` for W1.

It receives an accepted objective, immutable baseline reference and admitted alternatives.

It returns at most one candidate that differs on exactly one permitted configuration surface.

The W1 strategy is deterministic.

No LLM proposal call is required.

## 5.27 Experimental execution and evaluation

`experiment_bridge.py` shall implement the ADAS integration seam, not an execution engine.

Reuse existing LAB execution and experiment authorities:

- `ExperimentSession`.
- `ExperimentPersistence`.
- `UnifiedTaskRunner`.
- Existing governed harness root execution port.
- Canonical runtime budget enforcement.

The eventual W1 integration must bind the actual effective configuration to the execution identity.

A configuration digest attached only to metadata is insufficient evidence that the execution consumed the configuration.

Use the existing host composition to materialize the approved configuration.

Use isolated experiment state and equivalent task inputs.

`AgentCandidateEvaluator` shall integrate:

- `EvaluatorRegistry`.
- `EvalDataset`.
- `EvalRun`.
- `EvalSubjectOutput`.
- `execute_eval_run()`.
- `compare_eval_runs()`.
- `compute_utility()`.

The existing typed evaluation executor is synchronous. Do not start nested event loops or repeat task execution merely to construct EvalRun.

Capture actual asynchronous execution outcomes first, then supply those captured outcomes to the existing evaluation composition.

Bind each captured outcome to its exact scenario and execution identity.

Existing EvalRun telemetry and verdict types remain authoritative.

Do not introduce a competing evaluation engine.

The NexusEvalRunner may return zero cost or token values when telemetry is absent. ADAS must distinguish measured zero from unknown data.

If a required metric or mandatory evidence is missing, the candidate cannot be accepted on the basis of that missing value.

**Configuration-only execution boundary**

A configuration-only variant of an existing authorized execution composition is not a newly materialized executable agent.

W1 may resolve and execute approved configuration alternatives through existing host composition, provided that agent identities, executable artifacts, capabilities, tool permissions and execution authority remain unchanged.

Any creation or materialization of a new executable agent candidate remains subject to the canonical AgentScaffoldBridge and existing scaffold authority.

W1 must not create an alternative scaffold or materialization path.

The LAST_WINS versus CONCAT witness changes only an admitted configuration surface of an existing execution composition.

This scope does not qualify AgentScaffoldBridge or complete AHI-ADAS-20.


## 5.28 Selection semantics

W1 supports exactly three semantic outcomes:

- SELECT_CANDIDATE.
- RETAIN_BASELINE.
- ABSTAIN.

Selection rules:

1. Reject unqualified, unsafe or out-of-scope candidates.
2. Reject comparison across different tenant, objective or dataset identities.
3. Reject missing mandatory evidence.
4. A candidate may be selected when it satisfies mandatory acceptance criteria and demonstrates an evidence-backed improvement over the baseline.
5. When both configurations are acceptable and improvement is insufficient, retain baseline.
6. When the candidate is unacceptable but baseline remains acceptable, retain baseline.
7. When no configuration is acceptable, abstain.
8. When mandatory evidence does not support a reliable comparison, abstain.

Utility scoring cannot override mandatory safety, authority, correctness or evidence failures.

Selection must never activate a configuration.

## 5.29 Search controller

`AgentDesignSearchController` coordinates existing ADAS and platform mechanisms.

Prescribed algorithm:

1. Validate admitted scope and objective.
2. Resolve and verify baseline.
3. Admit bounded search.
4. Execute baseline once.
5. Reconcile resource consumption.
6. Generate at most one candidate.
7. Archive candidate draft.
8. Run static candidate gate.
9. On static gate failure: record STATIC_GATE_FAILED, archive structured rejection evidence and terminate safely.
10. Recheck resource allowance.
11. Execute candidate once.
12. Reconcile resource consumption.
13. Construct canonical evaluation evidence from actual outcomes.
14. Compare baseline and candidate.
15. Determine SELECT_CANDIDATE, RETAIN_BASELINE or ABSTAIN (selection outcomes, not lifecycle states).
16. Archive terminal candidate snapshot and evidence bundle while preserving the actual lifecycle state.
17. Return search outcome and stop reason.

The controller must not directly invoke agent execution methods, provider SDKs or tool handlers.

It must not invoke adaptation promotion, configuration activation or trusted memory publication.

Exceptions, cancellation and accounting failures must produce explicit non-success outcomes.

## 5.30 First executable behavioral witness

The initial proposed W1 qualification uses an existing configuration surface:

**`MergeStrategy`**

Existing source:

`intergrax/contracts/orchestration_enums.py`

Existing consumer:

`intergrax/runtime/nexus/nexus_loop.py`

`intergrax/runtime/nexus/response/final_response_composer.py`

Configuration pair:

- Baseline: LAST_WINS.
- Candidate: CONCAT.

The synthetic deterministic task requires the final response to preserve two independently produced result markers.

Both arms must use:

- Identical tenant and workspace.
- Identical task input.
- Identical accepted objective.
- Identical registered agents.
- Identical capability and authority.
- Identical dataset and evaluators.
- Identical external-effect restrictions.
- Identical resource limits.

Only the admitted merge strategy may differ.

Both arms must execute through the existing canonical LAB root execution path, with an actual bounded multi-agent graph.

The existing FinalResponseComposer must consume the effective configuration.

Do not replace the real composition path with mocked final answers.

Test-only deterministic agents are permitted for the behavioral witness.

If the reviewed execution path does not expose the selected configuration behavior, report the exact gap. Do not silently switch to another configuration surface, weaken the witness or modify protected owner components.

**Required positive witness:**

- Baseline and candidate receive distinct canonical execution identities.
- Baseline loses at least one required result marker.
- Candidate preserves both required result markers.
- Independent deterministic evaluation distinguishes the results.
- SELECT_CANDIDATE selection outcome is recorded (not a lifecycle state).
- Evidence is archived.
- No production promotion occurs.
- No real external effects occur.

**Required negative witnesses:**

- Foreign tenant rejected before dispatch.
- Foreign workspace rejected before dispatch.
- Invalid configuration rejected before candidate dispatch.
- Missing mandatory evaluation evidence prevents selection.
- Missing resource accounting prevents additional dispatch.
- Exhausted budget prevents additional dispatch.
- Cancellation prevents additional dispatch.
- Unacceptable baseline and candidate produce ABSTAIN.
- No improvement with acceptable baseline produces RETAIN_BASELINE.
- Archive evidence survives restart.

This witness qualifies deterministic configuration adaptation and canonical composition only.

It is not a general benchmark of intelligent adaptation or a production readiness claim.

## 5.31 Canonical ADAS implementation and acceptance binding

AHE implements mapped canonical ADAS obligations without introducing competing ownership.

Each executable AHE batch must bind its deliverables to exact AHI-ADAS obligations.

For every touched obligation, record one of:

- NOT_STARTED
- IMPLEMENTED_PENDING_QUALIFICATION
- QUALIFIED_SCOPED
- PARTIALLY_SATISFIED
- DEFERRED_WITH_REASON
- ACCEPTED

AHE package publication does not automatically close an ADAS obligation.

An ADAS obligation may become ACCEPTED only after its complete canonical acceptance requirements and applicable independent acceptance requirements are satisfied.

Partial W1 implementations must not be represented as full ADAS phase completion.

Canonical plan updates require separately authorized write scope.

Preserve canonical ADAS delivery rules until an explicit, authorized reconciliation establishes a different applicable publication procedure.
