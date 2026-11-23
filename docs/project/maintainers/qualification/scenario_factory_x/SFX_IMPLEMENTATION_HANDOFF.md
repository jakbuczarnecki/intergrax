# SCENARIO-FACTORY-X — Product Vision and Implementation Handoff (R1 + adopted R2)

**Document purpose:** one implementation-facing entry point, not a second master plan, WBS, ADR, requirements registry, or approval authority.
**Prepared:** 2026-11-23; documentation baseline `origin/development@7c16d9b5488234475e80a2e758158decca5e5ac5`.
**Design posture:** R1 is binding except where the operator-adopted R2 architecture overlay in master plan §18 is explicit; proposed R2 contracts, individual ADR changes, delegated engineering grants, and formal entry gates retain their own approvals.
**Delivery posture:** `DOCUMENTATION_HANDOFF_PREPARED`; `IMPLEMENTATION_AUTHORIZED = NO`; `SFX_PRE_IMPLEMENTATION_READY` is not declared by this document.

> Read this document first when planning SFX product implementation. Follow the canonical source for every technical rule. If this handoff and the canonical source conflict, stop and resolve against the master plan, ticket/ADR/OD and owning subsystem. Do not manufacture an approval or silently extend scope.

## 1. Product definition and value

**Product:** a controlled, reproducible *business problem → validated sandbox scenario* factory, with an option to iteratively generate and compare multiple candidates under a bounded mandate. It discovers a business process, frames measurable acceptance, reuses existing Intergrax capabilities, obtains engineering artifacts from SEF-X, tests business outcomes using the canonical runtime, compares results and prepares a delivery dossier. It does **not** turn a successful code build into an accepted business solution or production deployment.

**Business outcomes:** process understanding; measurable objectives and negative cases; reproducible sandbox proof; explicit limitations, costs and ROI assumptions; comparison of variants; a separately governed path to a client MVP or production integration.

**Primary users:** business/process owner, SFX solution engineer, human approver, independent evidence evaluator and commercial owner. Restricted customer self-service is a later initiative, not required for the first managed proof.

**Experiences and boundaries:**

| Experience | Product behavior | Human / authority boundary |
| --- | --- | --- |
| `MANUAL` | Human drives discovery, spec, blueprint and selected engineering steps | L2 only with separately approved operation; HO-2 human intent submit |
| `ASSISTED` | SFX proposes decisions and a candidate; humans approve milestones and submit each EngineeringIntent | Binding first REF-1 POC; HO-2 unchanged |
| `BOUNDED_AUTONOMOUS` (target) | SFX goal/evaluation loop proposes several candidates and next experiments within immutable scope, cost/time/iteration ceilings | **Not executable for delegated multi-submit** until SEF-owned grant/CON-N-01 amendment and OD; no L3 |
| Managed client delivery | Internal operator presents proof, scope, limitations and proposal | Delivery/production admission separate |
| Customer sandbox preview (later) | Restricted tenant-bound demos/quotas and approved templates | Requires separately qualified security, identity, data, abuse and cost controls |

**Source trace:** archived [original strategic R2 blueprint](./r2/research/INTEGRAX_SFX_R2_STRATEGIC_BLUEPRINT_2026-11-23.md) is the historical *proposal* underlying this outcome, **not** an independent binding authority. Approved semantics come from [master plan §18](../../../../../governance/tickets/plans/SFX-MASTER-PLAN.md) and [R2 adoption OD](./operator_decisions/OD-SFX-R2-CANONICAL-ADOPTION.md).

## 2. End-to-end journey: user-visible input → verifiable output

| Step | SFX product responsibility | Expected output / proof | Owning specification |
| --- | --- | --- | --- |
| 1. Intake | Collect business problem, intended effect, roles, source owners and permission boundaries | Problem and source inventory with provenance | R1 §7/§8; [R2 mission](./r2/SFX_R2_AUTONOMOUS_MISSION_SPEC.md) |
| 2. Discovery | Interview only for decision-critical unknowns; do not invent rules or data | Recorded assumptions, missing-info/escalation records | R1 P6, R2-W4-01 |
| 3. Specification | Pin business goal, scope, *measurable* truth set and forbidden outcomes | Validated spec and acceptance suite digests | R1 CON-01/03/07; R2-W3-01, R2-W4-02 |
| 4. Capability resolution | Resolve REUSE / CONFIGURE / EXTEND / MISSING against actual platform inventory | Deterministic capability/architecture diff | R1 P7; R2-W4-03 |
| 5. Blueprint | Choose canonical runtime path, integrations, human approval, exception handling | Versioned blueprint and digest | R1 P7/P8 |
| 6. Scope authorization | Bind L2 allowed assets, data, time/cost/iteration budgets, no external effects | Human-attributed policy/approval reference; otherwise STOP | R1 CON-13, [R2 mission](./r2/SFX_R2_AUTONOMOUS_MISSION_SPEC.md) |
| 7. Generate & engineer | Deterministic scaffold then **SEF-owned** code/tests/repair under an authorized engineering request | Code/worktree SHA, engineering job/evidence references | R1 P8; [SEF amendment](./r2/SFX_R2_SEF_CROSS_PROGRAM_AMENDMENT.md) |
| 8. Qualify & execute | SAST, dependencies, tenant/egress/secrets gates; scenario test via canonical runtime only | Technical verdict, trace, exact SHA/run ID | R1 P9; R2-W5-06 |
| 9. Business evaluation | Compare results against the independently pinned truth set, incl. negatives | `ScenarioEvaluationVerdict`, sourced failure classification | R2-W5-01; [REF/GEN](./r2/SFX_R2_REF_GEN_EVALUATION_PLAN.md) |
| 10. Improve/select | Preserve best safe candidate; produce hypothesis + regression evidence before another request | Candidate comparison / champion evidence | R2-W5-02/03, W6-01; automated resubmit gated |
| 11. Demonstrate/deliver | Prepare isolated demo, source map, risks, cost, ROI assumptions and production roadmap | Reproducible sandbox bundle and commercial dossier | R1 P11; R2-W6-03/04/05 |
| 12. External production decision | **Outside SFX**: per-scenario admission, real integrations, strict rollout | Separate SCENARIO-GATE / PROD-Q decision | R1 §14, [platform roadmap](../../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) |

For steps 7–10, `code PASS != business PASS`. Required business outcomes are `PASS`, `FAIL`, `INSUFFICIENT_EVIDENCE`, `BLOCKED_POLICY` or `BLOCKED_ENVIRONMENT`; no implicit successful outcome on technical execution alone. A scaffold without business logic is not a POC.

## 3. Architecture and exactly-one semantic owner

The **operator-approved Option B** is additive; do not reopen it for this documentation handoff:

| Concern | Single owner | SFX responsibility / prohibition |
| --- | --- | --- |
| Business goals, discovery, candidate strategy, business result and ROI | Tier-2 `agents/scenario_factory/` | Compose product logic and projections, not a new execution platform |
| Product interfaces | Tier-3 Studio/API/MCP/CLI | Same typed contracts, version map, tenant/permission semantics and parity; offline CLI restricted to L0/L1 |
| Engineering tickets, jobs, workspaces, coding backend, code repair, PR | SEF-X engineering execution authority | Emit handoff drafts and consume SEF evidence; **no direct engineering submit under HO-2** |
| Scenario execution, policy and tools | Canonical Intergrax Runtime / NexusLoop | Never replace with an SFX or Cursor coding agent |
| Versioned state, decisions, approvals, evidence, CAS/trace | Existing CW / DecisionBinding / HITL / Evidence owners | New SFX artifact views refer to existing owners; no new registry/store/FSM/approval authority |
| Catalog | Existing manifest and sidecar model | No ApplicationManifest changes by default and no competing semantic registry |
| Production admission | SCENARIO-GATE / PROD-Q programme | Only export admission dossier, never claim `PRODUCTION_ADMITTED` |

Sources: [R1 master plan](../../../../../governance/tickets/plans/SFX-MASTER-PLAN.md) §§3, 7, 14, 18; [R2 target architecture](./r2/SFX_R2_TARGET_PRODUCT_ARCHITECTURE.md); [SEF master plan](../../../../../governance/tickets/plans/ASF-SEF-MASTER-PLAN.md).

## 4. Contract and requirements pack — no new parallel registry

| Source | Role and current status |
| --- | --- |
| [R1 requirements](./SFX_REQUIREMENTS_TRACEABILITY.json) | 54 `REQ-SFX-001..054`, baseline traceability |
| [R2 requirements overlay](./proposals/SFX_REQUIREMENTS_TRACEABILITY_R2_PROPOSAL.json) | 18 `REQ-R2-001..018`; **PROPOSED_NOT_MERGED**; do not silently promote to R1 JSON |
| [R1 schemas](./design/schemas/) | 17 existing design schemas; evaluate ADR/COMPAT acceptance before using as implemented interfaces |
| [R2 contract catalog](./r2/SFX_R2_CONTRACT_CATALOG.md) and [six proposed schemas](./design/schemas/proposed/) | CON-R2-01..06 mission, envelope, source intake, candidate iteration, verdict, candidate comparison; actual versions require contract review |
| [R1 WBS](./SFX_WBS.md) | Original SFX-P0..P13, 9 waves W0..W8 |
| [R2 WBS DAG](./r2/SFX_R2_WBS_DAG_AMENDMENT.md) | 37 R2-W3..W8 overlay tasks — **not** an additional, competing programme |
| [R2 readiness matrix](./r2/SFX_R2_IMPLEMENTATION_READINESS_MATRIX.md) | Requirements/tasks/gates/inputs/outputs/evidence mapping |
| [Lifecycle](./r2/SFX_R2_LIFECYCLE_STAGE_SPEC.md), [canonicalization](./design/SFX_CANONICALIZATION.md) | Versioned projections, evidence digests and idempotency; no second platform FSM |
| [UX R1](./SFX_UX_DESIGN.md), [mission UX](./r2/SFX_R2_UX_MISSION_EXTENSION.md), [metrics](./design/SFX_METRICS.md) | 11-step Studio flow + mission cockpit, evaluation, pause/resume, candidate view and AUTO-01..12 |

**Invariant:** R2 architectural adoption is not blanket acceptance of proposed schemas or permission to override SEF HO-2. Keep requirements status and decision gates visible at each implementation wave.

## 5. Implementation ordering — one executable workstream, distinct release milestones

**Existing roadmap stays authoritative.** Bind work to the existing R1 P5..P13 units, adding the 37 R2 task overlays where mapped; do not create new phase IDs or duplicate WBS.

| Wave | R1 execution scope | R2 additions | Main deliverable | Entry / qualification |
| --- | --- | --- | --- | --- |
| W3 (P5) | Tier-2 package, service, schemas, typed interfaces, adapters, Tier-3 API/CLI | R2-W3-01..05 | Mission/goal/source/envelope/candidate substrate, L0/L1 parity | **After** ARCH-FREEZE CLOSED + GATE-SFX-ENTRY and required ADR/OD |
| W4 (P6–P8) | Discovery, spec, resolver, blueprint, scaffold, engineering handoff | R2-W4-01..08 | One assisted candidate through SEF; lifecycle and proof references | R2-W4-06 requires **GATE-SEF-CONSUME**; R2-W4-07 additionally SEF delegation OD |
| W5 (P9–P10) | Security, test planner, Studio/API/MCP/CLI parity | R2-W5-01..07 | Business truth-set evaluator and safe UI; autonomous driver only on grant path | R2-W5-03 **not** prerequisite for first assisted POC; full certification uses GATE-SFX-FULL |
| W6 (P11) | Reporting, ROI, catalog sidecar, dossier | R2-W6-01..05 | Reproducible POC/delivery bundle, candidate comparison and business value report | Commercial demo ≠ product certification ≠ production admission |
| W7 (P12) | REF-1/2/3, sealed GEN, UX rounds, performance | R2-W7-01..07 | Independent reference evaluation and generalization proof | REF-1 assisted first; GEN reveal only by authorized independent evaluator |
| W8 (P13) | Evidence, audit, IIAC product certification, closure | R2-W8-01..05 | Full factory product qualification and commercial pilot safeguards | Requires SEF + ECAP/IIAC rebind decisions and GATE-SFX-FULL |

**Four distinct milestones:**

1. `DOCUMENTATION_HANDOFF_PREPARED` — this document and R1/R2 designs are located and mapped; **not** implementation approval.
2. `REF1_ASSISTED_SANDBOX_POC` — a demonstrated, measured invoice/PO result with human HO-2 submit; after all relevant gates.
3. `REF1_BOUNDED_AUTONOMOUS_POC` — repeatable mission-level multi-candidate loop, but **only** after SEF owns and approves delegated submits.
4. `SFX_PRODUCT_CERTIFIED` — full W7/W8 evidence, independent qualification, IIAC/operator closure; still **not** client production admission.

[Actual dependency graph](./r2/SFX_R2_WBS_DAG_AMENDMENT.md) takes precedence over the simplified wave order above.

## 6. First vertical slice — REF-1 invoice versus an existing purchase order

**Business goal:** verify incoming invoices against **existing** purchase orders, explain discrepancies, escalate uncertain cases, and produce a traceable business verdict. Optional receipt fixture permits three-way matching; without it report `TWO_WAY_ONLY`. **No actual PO creation, accounting posting, payment or ERP write.**

**Input boundary:** synthetic/sanitized invoice documents and PO fixtures in LAB, explicit source/provenance refs, deterministic policy/tax tables and negative test cases, no production credentials or uncontrolled network egress.

**Minimal sequence:** AF CLOSED → GATE-SFX-ENTRY → W3 mission substrate → W4 CORE (intake/goal/spec/oracle/blueprint/scaffold) → R2-W4-06 (SEF assisted handoff **after GATE-SEF-CONSUME**) → R2-W5-01 (business verdict) → R2-W7-01 (REF-1 replay). R2-W5-03 is **not** a dependency. Certification of the full factory remains later.

| Oracle | Required demonstrable assertion |
| --- | --- |
| REF1-O-01 | Correct invoice, PO, quantity, amounts and tax match is accepted |
| REF1-O-02 | Duplicate invoice rejected idempotently; zero duplicate simulated effects |
| REF1-O-03 | Unknown/missing PO escalated; no silent acceptance |
| REF1-O-04 | Malformed invoice rejected; zero effects |
| REF1-O-05 | Quantity, unit price or totals mismatch detected with reason |
| REF1-O-06 | VAT rule breach detected using explicit fixture rules |
| REF1-O-07 | Prompt-like instructions in an invoice are untrusted data, never commands |
| REF1-O-08 | Every result references source digests and evaluation evidence |

**Definition of done for the assisted sandbox proof:** all eight oracles explicitly reported with PASS/FAIL and source+evidence refs; repeatable fixtures; code and business verdicts kept separate; authorized human HO-2 submit logged; mission/goal/blueprint/code/run SHA or digests pinned; reviewable bundle at `.tmp/proof/sfx/ref-1/<run-id>/`; reproducible independent LAB replay; clear demo label `NOT PRODUCTION`. No fabricated cost, accuracy, time or business ROI.

**STOP:** absent GATE-SFX-ENTRY or GATE-SEF-CONSUME for the required task; unknown acceptance truth; real ERP effects; missing provenance; unauthorized auto-submit; budget/scope change; tenant isolation failure; GEN content exposure; any L3/prod claim.

The exhaustive oracle test plan is [SFX_R2_REF_GEN_EVALUATION_PLAN.md](./r2/SFX_R2_REF_GEN_EVALUATION_PLAN.md). This handoff does not replace its input/verification definitions.

## 7. Bounded autonomy — target product, NOT a precondition for first POC

Loop: `intake → lock measurable goal/oracles → approve immutable envelope → baseline → SEF engineering job → safety checks → canonical runtime evaluation → compare → improvement hypothesis → next authorized candidate → stop/champion/dossier`.

- SFX owns the **business decision to improve**; SEF owns engineering work and validates submitted intents; Runtime owns scenario execution.
- Budget: cost, deadline, max iterations and parallel candidates; stop for revoke/expire, safety, external scope, insufficient evidence, repeated no-progress and exhausted budget.
- On revoke, reject *new* submits immediately. For in-flight work record SEF cancel/unknown-outcome handling; do not claim instantaneous undo.
- Preserve previous champion and all failures; independent evaluator tests regressions and negative oracles.
- Autonomous mode may be described and architected now. **Activation is blocked** by HO-2/CON-N-01 until the SEF-owned amendment and corresponding operator decisions are accepted.
- Grant is human/DecisionBinding authority, never generated by SFX itself. No cross-tenant grants, L3, automatic production merges or unapproved egress.

Owner and unresolved semantics: [SFX↔SEF proposal](./r2/SFX_R2_SEF_CROSS_PROGRAM_AMENDMENT.md), [mission spec](./r2/SFX_R2_AUTONOMOUS_MISSION_SPEC.md), [operator backlog](./r2/SFX_R2_OPERATOR_DECISION_REGISTER.md).

## 8. Security, quality and measurements

**Hard invariants:** zero unauthorized tool actions, cross-tenant reads, leaked secrets, unapproved production effects, self-certification, policy/grant bypass, unlogged source authorization or silent cost extension. Source documents are untrusted; provenance, classification, tenant, redaction, egress and rights must be explicit.

**Qualification:** canonicalized contracts + version parity; idempotency, CAS/replay, cancellation, revocation and recovery; independent business oracles; negative/adversarial inputs; SAST/dependency/sandbox tests; Studio/API/MCP/CLI parity; REF-2/REF-3; six sealed GEN cases without pre-reveal contamination; independent evaluation and full rerun after material post-reveal changes. GEN seal integrity is documented, but confidentiality must be ensured operationally before reveal.

**AUTO-01..12** in [metrics](./design/SFX_METRICS.md) measure time to valid spec/POC, improvements, cost/iteration, human effort, recovery, generalization, reuse and sales conversion. TTFS targets and commercial SLA are proposals until measured. All benchmark/cost/ROI comparisons use pinned fixtures, baselines and run SHAs; absent observation = `INSUFFICIENT_EVIDENCE`.

Sources: [R2 security qualification](./r2/SFX_R2_SECURITY_ADVERSARIAL_QUALIFICATION.md), [R2 REF/GEN plan](./r2/SFX_R2_REF_GEN_EVALUATION_PLAN.md), R1 master plan §§12–14.

## 9. Commercial delivery — staged, no fabricated readiness

| Stage | What can be offered | Evidence / boundary |
| --- | --- | --- |
| Paid process diagnostic (services, now) | Interview, process map, risk, measurable goal draft, scope/estimate | No claim that SFX software already runs |
| Internal managed REF-1 proof (after gates) | Reproducible synthetic invoice/PO sandbox POC | Truth-set and independent review; no production/real customer data |
| Client managed POC / MVP (later) | Bounded discovery, candidate, demonstration and handoff | Contractual data rights, customer scope, validated deployment and support |
| Private self-service (separate programme) | Invite-only limited portal with quotas | Explicit identity, tenant, retention, malware/abuse, billing and privacy gates |
| Real production automation (outside SFX) | Governed client integration/rollout | Separate SCENARIO-GATE, PROD-Q and monitoring/rollback |

Products/packaging: Process Diagnostic; Fixed-Scope POC Sprint; MVP Pilot; Enterprise Deployment & Continuous Improvement. No invented price or SLA. Platform IP and client-specific scenario deliverables must be separated contractually. See [commercial roadmap](./r2/SFX_R2_COMMERCIALIZATION_ROADMAP.md).

## 10. Exact open decisions and implementation STOP/GO

These are **external approvals/implementation prerequisites**, not reasons to write a third architecture or repeat an R2 design audit.

| Item | Evidence/source | Why it matters |
| --- | --- | --- |
| `PRE_IMPLEMENTATION_READY_SHA` / ARCH-FREEZE certification | [SHA ladder](./SFX_SHA_LADDER.md) currently has unset readiness/FRZ pins | No blanket right to start SFX-P5 |
| ADR-SFX-001..006 acceptance; OD-SFX-05/06/09/14 | [gap analysis](./SFX_GAP_ANALYSIS.md), [index](./SFX_INDEX.md) | Initial W3/ENTRY contract and gate; do not fabricate signatures |
| Inventory repin + `GATE-SFX-ENTRY` | [SHA ladder](./SFX_SHA_LADDER.md), [master plan](../../../../../governance/tickets/plans/SFX-MASTER-PLAN.md) | P5 and post-freeze implementation starts only after true gate evidence |
| C-SEF-01..04, `GATE-SEF-CONSUME` | [SEF master plan](../../../../../governance/tickets/plans/ASF-SEF-MASTER-PLAN.md) and [cross-program proposal](./r2/SFX_R2_SEF_CROSS_PROGRAM_AMENDMENT.md) | Assisted engineering handoff and joint test path |
| Delegated submits under HO-2 | R2 adoption OD + SEF CON-N-01 amendment / OD-SEF-R2 | Autonomous loop **cannot** start from SFX unilateral action |
| C-ECAP-01 / OD-ECAP-16 | [R2 WBS](./r2/SFX_R2_WBS_DAG_AMENDMENT.md) and operator decisions | Blocks R2-W8-03 enterprise rebind until resolved |
| REF usability rounds and restricted GEN content | [UX protocol](./SFX_USABILITY_PROTOCOL.md), [GEN seal](./SFX_GEN_SET_SEAL.md) | Required later qualification, **not** falsely labelled pre-freeze PASS |
| T-IGX-070-C historical G6 governance finding | [publication handoff](./SFX_R2_PUBLICATION_HANDOFF.md) | Separate formal lifecycle disposition; **not** an architecture defect |

**Go for docs:** design inputs, user outcome, architecture, requirement/contract sources, WBS, REF-1 acceptance, autonomous target and commercialization path are mapped here. **Stop for code:** do not equate this document, architecture adoption, or a documentation-only merge with `IMPLEMENTATION_AUTHORIZED`.

## 11. Implementer navigation / handoff protocol

1. Pin current `origin/development` SHA. Read this handoff + [SFX_INDEX](./SFX_INDEX.md), [master plan](../../../../../governance/tickets/plans/SFX-MASTER-PLAN.md) §§3/7/9/10/12/14/16/18 and [roadmap](../../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md). Preserve historical source/input versus current binding authority.
2. Read [R1 WBS](./SFX_WBS.md), [R2 WBS](./r2/SFX_R2_WBS_DAG_AMENDMENT.md), [readiness matrix](./r2/SFX_R2_IMPLEMENTATION_READINESS_MATRIX.md), 54+18 requirement sources and 17+6 schemas.
3. Obtain actual gate evidence for the proposed wave. If blocked, prepare the dependency or authorized documentation work **without** starting post-freeze code.
4. Start W3 against the existing owner boundaries. For the REF-1 slice, preserve HO-2 and await SEF consume evidence before R2-W4-06. Do not force autonomous driver into the first POC.
5. Produce one verifiable task/acceptance/evidence table per wave, run the required safety and regression suites, pin exact SHAs, and follow the existing independent certification mechanism. Do not create a new governance framework.
6. Report factory qualification separately from client production admission. Do not claim W8, R2 autonomous mode or a client production deployment until separately proven.

**Documentation completion criterion:** one consistent, readable route from the source business vision through adopted architecture, requirements, contracts, WBS, initial POC, autonomous target, safety and market delivery; explicit pending approvals identified without being mistaken for missing design. This handoff is **documentation-only** and may be editorially complete while post-freeze gates remain blocked.
