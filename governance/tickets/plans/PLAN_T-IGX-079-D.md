# T-IGX-079-D — II-W0-A-ARCHITECTURE-ACCEPTANCE-01 (W0-A packages II-W0-01..04; plan approval required)

## Authority and purpose

Execute documentary acceptance for the first consolidated W0-A batch after R3.1 ratification (OD-INTELLIGENCE-01) and formal corrective G6 closure (T-IGX-079-C). Produce four reviewed matrices and one consolidated closure matrix binding original roadmap acceptance criteria. W0-A is **documentary acceptance only**; no product benchmark, runtime capability certification or W1 executable qualification may be inferred. No package is formally closed merely because a matrix was prepared.

**Normative sources (read-only):**

- Architecture v1.1 R3.1 — blob `c89c4ccb9e253108de70b126c62c1c206063ab83`
- Master Roadmap v1.0 R3 — blob `ce7d248cd43fe2dace182fb6eedf7cdc65b82622`
- Ratification publication SHA `4637c7fa6295a37ad155fbff9d4a9718646c9cf8`

**Preconditions:** Operator approval of this exact plan digest; T-IGX-079-C formal scoped G6 `II-W0-R31-FORMAL-G6-01` recorded with documented non-blocking limitations; ticket lifecycle at least `ready_for_audit` (Operator ticket closure sign-off may remain pending and does not block W0-A plan approval); no gate opening; sequential execution in one approved implementation batch.

## Dependency order and execution contract

1. **II-W0-01** (no package prerequisites)
2. **II-W0-02** (depends on II-W0-01)
3. **II-W0-03** (depends on II-W0-01)
4. **II-W0-04** (depends on II-W0-01 and II-W0-03)

Execute packages **sequentially** in the order above within the approved batch. Do not parallelize matrix preparation that depends on prior package outputs.

## Consolidated W0-A closure matrix (required batch artifact)

After all four packages, produce **one** consolidated closure matrix covering II-W0-01..04. For **every** package row include:

| Field | Requirement |
|---|---|
| Original acceptance criterion | From Master Roadmap package section (unchanged wording) |
| Required output | Package deliverable name |
| Positive oracle | Mandatory positive behavior from roadmap |
| Negative oracle | Mandatory negative behavior from roadmap |
| Exact evidence locator | Repository path or authorized evidence location |
| Candidate SHA | Commit containing the reviewed matrix artifact |
| Publication SHA | Commit integrated to authorized branch |
| Independent review result | Scoped review verdict; no implementer self-certification |
| Residual findings | Open limitations blocking formal closure if any |
| Actual lifecycle | DevOS ticket/batch lifecycle at record time |
| Formal closure eligibility | Explicit yes/no; no closure without authority |

Batch qualification requires: document qualification, source-reference integrity, ownership-boundary checks, invariant traceability, affected-domain checks and canonical governance validation. **Do not** close any package because the matrix row exists.

## PACKAGE A — II-W0-01

**Deliverable:** Architecture Layer & Tier Boundary Matrix and reviewed architecture diagram.

**Logical capability layers (ratified architecture; not additional physical tiers):**

1. Foundation
2. Intelligence
3. Product ecosystem

**Preserve existing Tier-0–Tier-3 classification:**

| Tier | Role |
|---|---|
| Tier-0 | contracts and integrations |
| Tier-1 | canonical runtime |
| Tier-2 | agents and domain behavior |
| Tier-3 | application and host |

**Required matrix fields:** Logical capability layer; Existing physical tier; Concern; Canonical owner; Allowed dependency direction; Prohibited dependency direction; Admission authority; Exact source reference; Source SHA; Acceptance result.

**Positive oracle:** Every relationship has a legal direction and Execution retains sole canonical admission authority.

**Negative oracles:** Physical tier insertion; Reverse import; Second execution root; Private product scheduler; Alternative admission authority.

**Roadmap binding:** Original acceptance — layer diagram and classification; Tier-0–Tier-3 unchanged. Required output — reviewed architecture diagram and tier/authority boundary comparison.

## PACKAGE B — II-W0-02

**Deliverable:** Capability Contract & Consumer Responsibility Matrix.

**Architecture sections:** 1, 5, 6, 7, II-OPR-01, 13.2.

**Cover:** Goal and acceptance semantics; Deterministic execution; Intelligent execution; Proposal authority; Admission authority; Execution authority; Consumer responsibility; Domain acceptance criteria; Evidence and outcome responsibilities; Non-goals; Canonical identity and budget constraints.

**Required contract classification per row:** `EXISTING_DOCUMENTED` | `PROPOSED_LOGICAL` | `W1_VERIFICATION_REQUIRED` (documentary only; does not imply runtime availability).

**Workload families:** Use the six canonical reference workload families from architecture; do not create additional mandatory families.

**Positive oracle:** Representative deterministic and intelligent requests can be expressed without transferring authority or duplicating domain acceptance criteria.

**Negative oracles:** Implicit permissions; Invented runtime API; Private product harness; Consumer-defined execution lifecycle; Duplicate domain criteria.

**Explicit exclusion:** No product types, schemas, API endpoints or runtime code.

**Roadmap binding:** Original acceptance — definition, scope, non-goals and adoption contract accepted. Required output — reviewed contract inventory and consumer responsibility matrix, with current versus proposed fields.

## PACKAGE C — II-W0-03

**Deliverable:** Twelve Architectural Invariants Traceability Matrix.

**Exactly twelve primary rows:**

| ID | Invariant |
|---|---|
| II-INV-01 | Goal First |
| II-INV-02 | Single Execution Authority |
| II-INV-03 | Adaptive Planning |
| II-INV-04 | Governed Autonomy |
| II-INV-05 | Dynamic Collaboration |
| II-INV-06 | Evidence-Based Decisions |
| II-INV-07 | Outcome Verification |
| II-INV-08 | Recovery Before Escalation |
| II-INV-09 | Bounded Intelligence |
| II-INV-10 | Experience Reuse |
| II-INV-11 | Provider Neutrality |
| II-INV-12 | Single Semantic Ownership |

Use normative mappings in architecture section 4. Bind applicable SYS-INV, UEA-INV, FRZ and ECAP sources. Do not manufacture invariant identifiers or claim each source family applies to every row.

**Required matrix fields:** II-INV identifier; Exact normative obligation; Canonical source identifier; Repository path and section; Source commit SHA; Source blob SHA; Authority preservation; Future evidence owner; Positive oracle; Negative oracle; Conflict disposition.

Unresolved normative conflicts must remain explicit and block affected acceptance. Do not resolve semantic conflicts through unilateral document changes.

**Roadmap binding:** Original acceptance — traceability to existing invariants; no conflicting authority. Required output — twelve-row invariant traceability review with exact source revisions and conflict dispositions.

## PACKAGE D — II-W0-04

**Deliverable:** Semantic Ownership & Overlap Matrix.

Preserve architecture section 7.1 as normative ownership source.

**Binding ownership allocations:** Execution (admission, execution identity and lifecycle); Reasoning & Cognition (planning, validation and revision); Orchestration (topology authority); Nexus (scheduling and interpretation, private to Execution); Governance (principal, policy, permissions and approvals); Budget (RunBudget accounting and enforcement); Capacity (infrastructure admission and workload allocation); Autonomous Work (persistent responsibilities and worker envelope); Context Engineering (input-context budget); Execution / ERL / Self-Healing (recovery within existing responsibility boundaries); AHI (durable configuration adaptation); ERI / Diagnostics (read-only evidence interpretation); Memory and existing strategy owners (experience retention and reuse within existing scopes).

Decompose cooperating mechanisms per ratified boundaries. Do not assign multiple competing accountable owners. Explicitly cover overlap with AHI, ERI, Autonomous Work and Self-Healing.

**Required matrix fields:** Concern; Accountable semantic owner; Supporting mechanism; Proposal authority; Decision authority; Execution authority; State/evidence ownership; Prohibited duplication; Exact source reference; Provisional W1 verification; Unresolved ownership decision, if applicable.

**Positive oracle:** Every concern has one accountable owner or an explicit blocking ownership decision before implementation.

**Negative oracles:** Duplicate execution identity; Duplicate authoritative store; Duplicate recovery controller; Duplicate RunBudget ledger; Duplicate policy authority; Second control plane; Unauthorized adaptation.

For unresolved ownership decisions: record the exact concern; identify conflicting canonical sources; preserve existing authority; block affected formal acceptance; do not invent a resolution.

**Roadmap binding:** Original acceptance — single owner per concern; overlap and conflict matrix. Required output — concern-level ownership and overlap matrix with provisional W1 checks.

## Authorized execution write-set (after plan approval only)

Separate authorized implementation batch; **not** part of preparation batches II-W0-A-PREP-01 or II-W0-A-PREP-FINAL-01. These paths are evidence and mutable execution records only; they do not supersede ratified architecture, Master Roadmap or existing semantic owners. **Do not create the five deliverable files during preparation.** If a path conflicts with mandatory canonical documentation convention, stop and report the exact conflict; do not substitute paths autonomously.

Exact authorized write-set for the subsequent approved W0-A implementation batch:

1. `docs/project/maintainers/qualification/intelligence_x/w0/II-W0-01_LAYER_TIER_BOUNDARIES.md`
2. `docs/project/maintainers/qualification/intelligence_x/w0/II-W0-02_CAPABILITY_CONTRACT.md`
3. `docs/project/maintainers/qualification/intelligence_x/w0/II-W0-03_INVARIANT_TRACEABILITY.md`
4. `docs/project/maintainers/qualification/intelligence_x/w0/II-W0-04_SEMANTIC_OWNERSHIP.md`
5. `docs/project/maintainers/qualification/intelligence_x/w0/II-W0-A_CLOSURE_MATRIX.md`
6. `governance/tickets/T-IGX-079-D.md`
7. `docs/project/capabilities/plan/INTEGRAX_INTELLIGENCE.md`

This preparation plan does not authorize product trees, scripts, tests or normative architecture/roadmap blobs.

## Ordered procedure (after Operator `Approved:` only)

Follow the qualification lifecycle sequencing in **Q08** below without reordering canonical DevOS transitions.

## Qualification contract (after plan approval; confirm CLI at execution baseline)

Future W0-A qualification commands only — not instructions to rerun historical R3.1 qualification. Verify canonical CLI compatibility at execution baseline; preserve real command output, exit code, scope and evidence locator; do not weaken validators or substitute simulated results.

### Q01 — Scope integrity

Verify architecture blob `c89c4ccb9e253108de70b126c62c1c206063ab83`, Master Roadmap blob `ce7d248cd43fe2dace182fb6eedf7cdc65b82622`, approved plan digest, exact authorized write-set (seven paths above) and zero product delta outside that write-set.

### Q02 — Layer boundaries

Review every diagram relationship and matrix row against canonical tier and authority rules. Verify that no physical tier insertion, reverse dependency, second execution root or alternative admission authority is introduced.

### Q03 — Consumption contract

Review the complete contract inventory, current versus proposed fields, consumer responsibilities and representative deterministic and intelligent requests. Verify that neither request transfers canonical authority or duplicates domain acceptance criteria. Verify coverage of all six canonical reference workload families.

### Q04 — Invariant traceability

Verify exactly twelve unique II-INV primary rows. Validate every cited source path, identifier, section, source commit and blob. Bind applicable SYS-INV, UEA-INV, FRZ and ECAP sources. Record unavailable or conflicting mandatory sources as explicit blockers. No synthetic invariant identifiers.

### Q05 — Ownership integrity

Verify every concern against ratified architecture section 7.1. Check the Package D positive oracle and all negative oracles. Include AHI, ERI, Autonomous Work and Self-Healing overlap boundaries. Verify provisional W1 obligations without asserting runtime qualification.

### Q06 — Source and document integrity

Verify relative links and anchors. Verify canonical document ownership and applicable documentation-domain checks. Run applicable affected-domain validation under existing EVOL-X and ENT-AUDIT-X rules.

### Q07 — Governance qualification

Bind and execute these canonical commands (profile smoke where noted):

- `uv run python scripts/governance/plan_digest.py governance/tickets/plans/PLAN_T-IGX-079-D.md`
- `uv run python scripts/governance/verify_plan_approval.py governance/tickets/T-IGX-079-D.md --plan governance/tickets/plans/PLAN_T-IGX-079-D.md`
- `uv run python scripts/governance/verify_ticket_lifecycle.py governance/tickets/T-IGX-079-D.md`
- `git diff --check`
- `uv run python -m scripts.governance.igx_gov ticket --profile smoke`

### Q08 — Evidence and lifecycle sequencing

The approved execution batch must follow this order:

1. Actual Operator approval bound to the exact plan digest.
2. Canonical transition to `plan_approved` and `in_progress`.
3. Sequential execution of II-W0-01 through II-W0-04.
4. Production of all five specified evidence artifacts at the authorized paths.
5. Qualification and candidate commit.
6. Independent pre-publication review where required.
7. Authorized publication with verified remote SHA.
8. Canonical `quality_gate` / `ready_for_audit` handoff in the order required by DevOS.
9. Independent formal G6 against the appropriate exact candidate/publication scope.
10. Actual Operator closure authority.
11. Evidence-bound final register reconciliation.

Do not reorder canonical lifecycle transitions. No implementer self-certification. Do not require a commit to contain its own SHA. Record candidate and publication SHA bindings in subsequent authorized evidence updates. At initial candidate creation, independent verdicts, publication SHA and formal closure eligibility remain pending where not yet established. Do not mark packages closed before all required authority and evidence exist.

No W0-A deliverable execution under this ticket before formal plan approval. No independent G6 assertion for 079-D in preparation or implementation without separate authority.

## Rollback

Revert only the isolated W0-A delta on an authorized branch; never force-push or reset shared `development`. Retain ratified D1/D2 blobs and T-IGX-079-C closure record unchanged.