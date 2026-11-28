# II-W0-02 — Capability Contract and Consumer Responsibility Matrix

**Package:** II-W0-02
**Ticket:** T-IGX-079-D
**Correction batch:** II-W0-A-CORRECTIVE-02 (residual settlement); prior II-W0-A-CORRECTIVE-01 preserved in Git ancestry
**Source baseline commit:** `79e978feb280614e35daf2c8f09d83c9634ed348`
**Normative bindings:** Architecture §1, §5, §6, §7.2, II-OPR-01, §11.1, §13.2

## 1. Capability definition and scope

Universal **documentary** consumption semantics for goal realization per II-OPR-01. Behavioral sequence: **Accepted Goal → Plan → Execution → Evidence → Verification → Outcome** (§6.1). Intelligence proposes and plans; Governance permits; Execution admits and owns lifecycle; domain owners retain acceptance criteria.

## 2. Explicit non-goals

- No new runtime API, schema, type, endpoint, or `IntelligenceRunId`.
- No consumer-owned execution lifecycle or implicit permissions.
- No duplicate domain acceptance logic inside a generic Intelligence contract.
- No product production-readiness or adoption authorization for future domain applications through this document.
- No private product harness or competing admission path.

## 3. II-OPR-01 consumption element inventory (12/12)

| # | Normative requirement (II-OPR-01) | Existing documented contract or owner | Proposed logical field (not runtime API) | Authority preserved | Classification | W1 verification | Exact source reference |
|---|---|---|---|---|---|---|---|
| 1 | Objective and accepted goal revision | Host / Autonomous Work goal contracts; Execution admission | Accepted goal revision bundle in consumption envelope | Execution/Cognition; AW for durable goals | EXISTING_DOCUMENTED | Map to existing goal types | `INTEGRAX_INTELLIGENCE.md` II-OPR-01 bullet 1; §7.2 Accepted goal |
| 2 | Product-owned domain acceptance criteria | Product / domain owners | Criteria ID references only | Product retains criteria semantics | EXISTING_DOCUMENTED | Criteria binding tests | II-OPR-01 bullet 2; §7.2 Accepted goal |
| 3 | Tenant and principal from canonical admission | Governance + Execution admission | Principal/tenant from admission record | Governance; Execution applies | EXISTING_DOCUMENTED | Admission identity proofs | II-OPR-01 bullet 3; UEA-INV-022 |
| 4 | Input and context references | Context Engineering / UCL / Memory | Scoped input and context refs | Context Engineering owners | EXISTING_DOCUMENTED | Reference validation | II-OPR-01 bullet 4; §7.2 Accepted goal |
| 5 | Capability restrictions | Governance / capability catalog | Permitted capability envelope | Governance | EXISTING_DOCUMENTED | Restriction enforcement | II-OPR-01 bullet 5 |
| 6 | Autonomy envelope | Governance + Execution bounded autonomy | Autonomy scope and HITL boundaries | Governance permission owner | EXISTING_DOCUMENTED | Negative auth suite | II-OPR-01 bullet 6; II-INV-04 |
| 7 | Cost, time and resource budgets | Budget / RunBudget | Parent and child budget refs | Budget single ledger | EXISTING_DOCUMENTED | Budget propagation tests | II-OPR-01 bullet 7; §7.2 Resource accounting |
| 8 | Canonical execution invocation | Execution Engine entry | Canonical invocation descriptor | Execution sole admission | EXISTING_DOCUMENTED | Admission integration | II-OPR-01 bullet 8; §7.3 sequence |
| 9 | Progress and evidence access | Execution records / Observability | Progress projection refs | Execution truth; Observability records | PROPOSED_LOGICAL | Progress reconstruction | II-OPR-01 bullet 9; §7.2 Progress evidence |
| 10 | Wait, cancel, resume through existing owners | Execution lifecycle | Wait/cancel/resume via Execution ports | Execution lifecycle owner | EXISTING_DOCUMENTED | Lifecycle port tests | II-OPR-01 bullet 10; UEA-INV-012 |
| 11 | Verified outcome | Decision Verification + domain | Verified disposition | Verification separate from executor | PROPOSED_LOGICAL | Verification oracle suite | II-OPR-01 bullet 11; §7.2 Verification result |
| 12 | Explicit partial, failed or blocked when not verified | Execution disposition + domain | Typed non-success outcome | Product criteria + Verification | PROPOSED_LOGICAL | Failure disposition tests | II-OPR-01 bullet 12; §6.3 failure matrix |

**Source commit:** `79e978feb280614e35daf2c8f09d83c9634ed348`
**Architecture blob:** `c89c4ccb9e253108de70b126c62c1c206063ab83`

## 4. Nine logical record categories (architecture §7.2)

Logical semantic roles only — **not** implemented classes or new identifiers. Required information and binding/invariant text match ratified architecture §7.2 at evidence baseline `79e978feb280614e35daf2c8f09d83c9634ed348` (architecture blob `c89c4ccb9e253108de70b126c62c1c206063ab83`).

| Logical record | Required information | Binding and invariant | Classification | W1 obligation |
|---|---|---|---|---|
| Accepted goal | Objective, input/data refs, criteria IDs, constraints, permitted scope, budget ref, version | Existing goal/execution identity; immutable accepted revision; tenant and principal from canonical admission | EXISTING_DOCUMENTED | Map to existing goal types and admission records |
| Plan proposal/revision | Parent revision, task steps, dependencies, capability needs, expected outputs, justification, invalidation map | NexusPlan / existing plan lineage; proposal does not authorize actions; validated graph before admission | EXISTING_DOCUMENTED | NexusPlan lineage and validator mapping |
| Work assignment | Stable node/task ref, role, admitted execution ref, input/artifact refs, budget allocation, acceptance | Use canonical Execution identity for runtime attempt; node identity is not execution identity | EXISTING_DOCUMENTED | Identity separation tests |
| Collaboration envelope | Sender/recipient refs, goal/task/plan version, correlation, message kind, evidence refs, delivery status | Scope checked on send and receive; idempotent and ordered where required; message is data, not permission | PROPOSED_LOGICAL | Message-extension owner mapping (W1) |
| Progress evidence | Criterion status, verified artifact refs, remaining scope, obstacle refs, source version | Projection from authoritative facts; no parallel source of execution truth | PROPOSED_LOGICAL | Progress projection and reconstruction tests |
| Recovery proposal | Failure class, affected work, evidence, uncertainty, candidate strategy, prerequisites, budget | Canonical recovery owner; new effects re-admitted; uncertain effect reconciled | EXISTING_DOCUMENTED | REF-II-04 fault-injection fixtures |
| Verification result | Criterion ID, evaluator identity/strategy, verdict, evidence refs, limitations, evaluated revision | Independent from executor as required; stale evidence invalidated on material revision | EXISTING_DOCUMENTED | Independent evaluation oracle suite |
| Experience record | Task class, strategy/profile revision, validated result, metrics, conditions, provenance, scope and retention | Tenant-bound; no secret replay or grant of permission; promotion belongs to existing owner | EXISTING_DOCUMENTED | Scoped reuse and quarantine gates |
| Resource accounting | Reserved/committed/actual cost, tokens, attempts, capacity, deadline, parent budget | Canonical accounting and admission; no duplicate debit/release across restarts | EXISTING_DOCUMENTED | Single-ledger and restart-idempotence proofs |

## 5. Current versus proposed contract fields (summary)

| Field group | Current (existing owners) | Proposed logical (II-OPR-01 envelope) | Authority unchanged |
|---|---|---|---|
| Goal and criteria | Host / domain product | Bundled consumption contract refs | Product + Execution |
| Permission and principal | Governance admission | Carried on canonical invocation | Governance |
| Plans | NexusPlan / Cognition proposals | Proposal until validated and admitted | Cognition; Orchestration topology |
| Outcome | Decision Verification + domain | Verified vs partial/failed/blocked | Verification + domain |

## 6. Consumer responsibility matrix (architecture §13.2 — complete)

| Consumer | Uses shared Intelligence for | Keeps (no transfer to Intelligence) | Classification |
|---|---|---|---|
| Scenario Factory | Goal interpretation, intelligent discovery, decomposition, capability selection, collaboration, dynamic workflow composition, strategy revision, recovery, and outcome verification | Business scenario semantics, business acceptance, scenario specification, blueprint and package semantics, product lifecycle, Studio and UX, engineering handoff, and product-specific evaluation | EXISTING_DOCUMENTED |
| SEF-X | Shared goal execution, team formation, planning, recovery and evidence-based verification | Engineering-specific authority, engineering workspace, code-change lifecycle, engineering-specific testing, evidence, integration/publication controls; HO-2 and CON-N-01 remain binding until authorized amendment | EXISTING_DOCUMENTED |
| CodeCraft | Shared planning and selection where the engineering goal needs them | Generated code synthesis/execution, qualified engineering tools and sandbox boundaries, artifact lifecycle | EXISTING_DOCUMENTED |
| Autonomous Work | Shared cognition, recovery, and verification inside admitted work | Durable responsibility, Worker goal/cadence, wake-up, dispatch, recovery episode and continuity lifecycle | EXISTING_DOCUMENTED |
| Domain agents and future applications | Shared execution intelligence, tools, context and collaboration | Domain logic, business policies, permissions, UX, acceptance and deployment choices | PROPOSED_LOGICAL |

Future domain applications do **not** obtain adoption or production authorization through this matrix.

## 7. Six reference workload families (§11.1 — canonical names and oracles)

| ID | Family | Required behavior and oracle (documentary obligation) | W1 fixtures |
|---|---|---|---|
| REF-II-01 | Engineering Intelligence | Defect analysis, design, sandboxed implementation, regression discovery, strategy revision, independent correctness proof; no uncontrolled repository or production publication | CodeCraft sandbox manifests |
| REF-II-02 | Research Intelligence | Authorized evidence retrieval, multi-source decomposition, contradiction reconciliation, cited report; evaluate factual coverage, source validity, unsupported claims vs sealed references | Sealed reference sets |
| REF-II-03 | Business Intelligence | Reconcile carrier invoices or equivalent synthetic data; classify discrepancies; verify amounts, linkage, completeness, exception classification | Business oracle data |
| REF-II-04 | Recovery Intelligence | Inject unavailable tools, malformed output, wrong assumptions, failed agents, duplicate delivery, unknown effects; verify recoverable outcomes, preserved authorization, no duplicated effects | Fault injection catalog |
| REF-II-05 | Collaborative Intelligence | Heterogeneous roles, questioning, challenge, artifact handoff, result integration; verify final correctness and causal collaboration evidence | Multi-agent scenarios |
| REF-II-06 | Persistent Intelligence | Interruption and process-loss recovery, canonical state, verified subsequent experience reuse; verify checkpoint, identity, budget, freshness, no replayed effects | Durable resume fixtures |

## 8. Representative deterministic request (documentary)

**Host (Tier-3):** Submit carrier-invoice reconciliation goal (REF-II-03) with sealed oracle data, explicit deterministic rules reference, tenant/principal from admission, zero autonomy expansion. **Expected path:** Execution admits → validated deterministic plan → tool execution under ToolRuntime → evidence → Decision Verification against **product-owned** criteria. Consumer does **not** obtain admission or permission by submission alone.

## 9. Representative intelligent request (documentary)

**Host:** Multi-source research goal (REF-II-02) with citation rubric owned by product. **Expected path:** Cognition proposes plan → Orchestration validates topology → Execution admits children → Governance checkpoints on effects → Verification independent of executor. Strategy may revise; acceptance criteria remain product-owned.

## 10. Oracle evaluation

**Positive oracle:** Both representative requests express the full II-OPR-01 element set without transferring admission, permission, or duplicating domain criteria. **PASS (documentary).**

**Negative oracles (documentary results):**

| Case | Documentary result |
|---|---|
| Implicit permissions | **REJECT** — denied unless Governance explicitly authorizes |
| Private product harness | **REJECT** — non-canonical execution entry |
| Invented runtime API | **REJECT** — until W1 maps to existing ports |
| Consumer-owned lifecycle | **REJECT** — Execution retains lifecycle |
| Duplicated domain acceptance | **REJECT** — generic contract cannot override product criteria |
| Authority expansion via goal submit | **REJECT** — admission is a separate step |

## 11. W1 verification obligations

Map II-OPR-01 fields to existing types; prove negative suite on unauthorized effects; bind six REF-II families to runnable manifests; no competing identifiers.

**Original acceptance criterion:** Definition, scope, non-goals and adoption contract accepted — **met at documentation level.**
