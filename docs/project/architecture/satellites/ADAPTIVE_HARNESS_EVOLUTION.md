# Adaptive Harness Evolution — Architecture

**Revision:** 1.1
**Document status:** ARCHITECT-DEFINED / PENDING INDEPENDENT ACCEPTANCE
**Parent architecture:** [ADAPTIVE_HARNESS_INTELLIGENCE.md](../ADAPTIVE_HARNESS_INTELLIGENCE.md)
**Implementation roadmap:** [ADAPTIVE_HARNESS_EVOLUTION.md](../../maintainers/plans/satellites/ADAPTIVE_HARNESS_EVOLUTION.md)

Adaptive Harness Evolution (AHE) is a **governed implementation program** within the existing Adaptive Harness Intelligence (AHI) domain. It is not a new architectural layer, runtime, or competing execution engine.

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

**Cross-program boundary status:** ARCHITECT_DEFINED / EXTERNAL_OWNER_ACCEPTANCE_PENDING

Do not assert that another program has accepted this mapping.
