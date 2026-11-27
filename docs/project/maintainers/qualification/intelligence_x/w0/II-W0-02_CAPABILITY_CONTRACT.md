# II-W0-02 — Capability Contract and Consumer Responsibility Matrix

**Package:** II-W0-02  
**Ticket:** T-IGX-079-D  
**Normative bindings:** Architecture §1, §5, §6, §7, II-OPR-01, §13.2, §11.1 reference workloads

## 1. Capability definition and scope

Universal **documentary** consumption semantics for goal realization per II-OPR-01. Behavioral sequence: **Accepted Goal → Plan → Execution → Evidence → Verification → Outcome** (§6.1). Intelligence proposes and plans; Governance permits; Execution admits and owns lifecycle; domain owners retain acceptance criteria.

## 2. Explicit non-goals

- No new runtime API, schema, type, endpoint, or `IntelligenceRunId`.
- No consumer-owned execution lifecycle or implicit permissions.
- No duplicate domain acceptance logic inside a generic Intelligence contract.
- No product production-readiness or adoption authority claim.
- No private product harness or competing admission path.

## 3. Logical contract inventory

| Contract element | Classification | W1 obligation |
|---|---|---|
| Accepted goal revision + objective | EXISTING_DOCUMENTED | W1_VERIFICATION_REQUIRED |
| Domain acceptance criteria (product-owned) | EXISTING_DOCUMENTED | — |
| Canonical execution invocation | EXISTING_DOCUMENTED | W1_VERIFICATION_REQUIRED |
| Plan proposal / revision lineage | EXISTING_DOCUMENTED | W1_VERIFICATION_REQUIRED |
| Autonomy envelope and budgets | EXISTING_DOCUMENTED | W1_VERIFICATION_REQUIRED |
| Progress and evidence access | PROPOSED_LOGICAL | W1_VERIFICATION_REQUIRED |
| Wait / cancel / resume via owners | EXISTING_DOCUMENTED | W1_VERIFICATION_REQUIRED |
| Verified outcome vs explicit partial/fail | PROPOSED_LOGICAL | W1_VERIFICATION_REQUIRED |
| Universal consumption envelope (II-OPR-01 fields) | PROPOSED_LOGICAL | W1_VERIFICATION_REQUIRED |

Documented existence is not runtime qualification.

## 4. Current versus proposed contract fields

| Field | Current (existing owners) | Proposed logical (II-OPR-01) | Authority unchanged |
|---|---|---|---|
| Objective | Host / Autonomous Work goals | Bundled in consumption contract | Product + Execution admission |
| Acceptance criteria | Domain / product | Referenced by ID; not redefined | Product |
| Principal / tenant | Governance + admission | Carried on invocation | Governance |
| Budget ref | RunBudget / Budget | Enforced single ledger | Budget |
| Plan artifacts | NexusPlan / Cognition | Proposal only until validated | Cognition; Execution admits |
| Outcome | Decision Verification + domain | Verified vs failed disposition | Verification + domain |

## 5. Consumer responsibility matrix

| Consumer | Uses shared Intelligence for | Keeps (no transfer to Intelligence) | Classification |
|---|---|---|---|
| Scenario Factory | Interpretation, decomposition, collaboration, recovery, verification support | Business scenario semantics, Studio/UX, product acceptance | EXISTING_DOCUMENTED |
| SEF-X | Shared goal execution, planning, recovery, evidence verification | Engineering authority, workspace, integration/publication controls | EXISTING_DOCUMENTED |
| CodeCraft | Planning/selection for engineering goals | Sandbox, qualified tools, artifact lifecycle | EXISTING_DOCUMENTED |
| Autonomous Work | Cognition, recovery inside admitted work | Durable Worker responsibility, cadence, dispatch | EXISTING_DOCUMENTED |
| Domain agents / future apps | Execution intelligence, tools, context | Domain logic, permissions, deployment | PROPOSED_LOGICAL |

Source: Architecture §13.2 table.

## 6. Six reference workload families (§11.1)

| ID | Family | Covered |
|---|---|---|
| REF-II-01 | Engineering Intelligence | Yes |
| REF-II-02 | Research Intelligence | Yes |
| REF-II-03 | Business Intelligence | Yes |
| REF-II-04 | Recovery Intelligence | Yes |
| REF-II-05 | Collaborative Intelligence | Yes |
| REF-II-06 | Persistent Intelligence | Yes |

W1 defines inventories and commands; W0 accepts normative family semantics only.

## 7. Representative deterministic request (documentary)

**Host (Tier-3):** Submit carrier-invoice reconciliation goal with sealed oracle data, explicit deterministic rules reference, tenant/principal from admission, zero autonomy expansion. **Expected path:** Execution admits → validated deterministic plan → tool execution under ToolRuntime → evidence → Decision Verification against **product-owned** criteria. Consumer does **not** obtain admission or permission by submission alone.

## 8. Representative intelligent request (documentary)

**Host:** Multi-source research goal (REF-II-02) with citation rubric owned by product. **Expected path:** Cognition proposes plan → Orchestration validates topology → Execution admits children → Governance checkpoints on effects → Verification independent of executor. Strategy may revise; acceptance criteria remain product-owned.

## 9. Oracle evaluation

**Positive oracle:** Both representative requests express goal, criteria, and invocation without transferring admission, permission, or duplicating domain criteria. **PASS (documentary).**

**Negative oracles:**

| Case | Expected rejection |
|---|---|
| Implicit permissions | Denied unless Governance explicitly authorizes |
| Private product harness | Reject non-canonical execution entry |
| Invented runtime API | Reject until W1 maps to existing ports |
| Consumer-owned lifecycle | Reject; Execution retains lifecycle |
| Duplicated domain acceptance | Reject generic contract overriding product criteria |
| Authority expansion via goal submit | Reject; admission separate step |

## 10. W1 verification obligations

Map II-OPR-01 fields to existing types; prove negative suite on unauthorized effects; bind six REF-II families to runnable manifests; no competing identifiers.

**Original acceptance criterion:** Definition, scope, non-goals and adoption contract accepted — **met at documentation level.**