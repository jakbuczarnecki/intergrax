# Intergrax — Scenario Factory R2: Strategic Blueprint, Architecture & Commercial Roadmap

**Status:** INDEPENDENT AUDIT PROPOSAL / NOT CANONICAL / NOT OPERATOR-APPROVED  
**Date:** 2026-11-23  
**Repository:** `jakbuczarnecki/intergrax`  
**Verified branch / HEAD:** `development` / `3334b34eacfb99b4644d7b03b8fef9ffe02c6d88`  
**Governance:** This document does NOT authorize implementation, modify the existing `SFX-MASTER-PLAN.md`, accept ADRs, sign operator decisions, promote FRZ rows or waive ARCH-FREEZE, SEF, SFX, SCENARIO-GATE or PROD-Q gates. The canonical roadmap remains authoritative until an operator-approved amendment is published.

## Executive decision

**Recommend an R2 amendment, not a replacement of R1.** R1 has a sound Option B architecture and strong enterprise controls, but its product execution model is largely *single-build / human-directed improve*. R2 should introduce a bounded, evidence-driven **autonomous Goal → Build → Evaluate → Repair → Reevaluate** loop, while retaining SFX product semantics, SEF-X engineering authority and Intergrax's single scenario runtime authority. Commercialization should be staged: paid diagnosis now, internal assisted POC after authorized gates, private customer sandbox preview later, production delivery only through separate admissions.

## 1. Evidence and documents examined

| File | Role / audited finding |
|---|---|
| `governance/tickets/plans/SFX-MASTER-PLAN.md` | R1 binding plan; 14 phases P0–P13, 9 waves W0–W8, 17 SFX contract candidates, 54 requirements, 3 REFs; human submit HO-2; SFX-P8-03 single SEF handoff; no explicit autonomous goal-level iteration |
| `docs/project/maintainers/qualification/scenario_factory_x/SFX_INDEX.md` | `SFX_PRE_FREEZE_DOCS_CLOSED`; P5+ blocked; `PRE_IMPLEMENTATION_READY` not granted; six ADRs Proposed |
| `docs/project/maintainers/qualification/scenario_factory_x/SFX_WBS.md` | Detailed post-freeze execution tasks; update within existing waves instead of creating a second WBS |
| `docs/project/capabilities/architecture/SCENARIO_FACTORY.md` | Preferred Tier-2/Tier-3 Option B; consumes platform and SEF, no second runtime/authority |
| `docs/project/maintainers/qualification/scenario_factory_x/SFX_GAP_ANALYSIS.md` | 27 tracked gaps including SEF (05), SFX-facing UCA (21), CW qualification (22), JWT no-op (18), wildcard CORS (19), DLQ (20) |
| `docs/project/maintainers/qualification/scenario_factory_x/SFX_UX_DESIGN.md` | 11-step linear Studio flow; Improve triggered by user, not autonomously controlled by objectives |
| `docs/project/maintainers/qualification/scenario_factory_x/SFX_COMPETITIVE_EVIDENCE_MATRIX.md` | 15 workflow/agent products and 14 scoring dimensions as of 2026-11-12; insufficient autonomous coding harness comparison |
| `docs/project/maintainers/qualification/scenario_factory_x/design/SFX_METRICS.md` | G-1..G-5 / M-1..M-9; missing mission-level progress and autonomous candidate efficiency measures |
| `docs/project/maintainers/qualification/scenario_factory_x/SFX_GEN_SET_SEAL.md` | Six sealed unseen problems; integrity proved, confidentiality NOT technically assured (move to restricted location) |
| `governance/tickets/plans/ASF-SEF-MASTER-PLAN.md` | SEF-X EEA, 82 WBS tasks, bounded job repair loop, ticket/plan approval; engineering task FSM separated from scenario runtime |
| `docs/project/technical/devos/CURSOR_SOFTWARE_FACTORY.md` | Cursor-specific autonomous software development in isolated branches, bounded corrections and reviews; not the customer-facing SFX execution authority |
| `docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` | Closure and stage-order SSOT, ARCH-FREEZE predecessor to SFX implementation; SFX certification separate from scenario production admission |

**Documentation currency findings:** R1 plan frontmatter still marks W0 `in_progress` and W1/W2 `pending` whereas `SFX_INDEX.md` records their publication and `SFX_PRE_FREEZE_DOCS_CLOSED`; similarly `SFX_GAP_ANALYSIS.md` describes the `GEN_SEAL_SHA` rung as unset while `SFX_SHA_LADDER.md` now pins it. These are snapshot/drift observations, **not** authorization to rewrite programme closure. Reconcile the source roles and timestamps in R2 before treating checklist fields as current.

Evidence is at audited HEAD unless a document explicitly cites historical SHA. This is a documentation/code-surface audit and GitHub evidence review, **not** a freshly executed complete cross-platform test or certified security penetration test.

## 2. Target product and value proposition

**One-sentence product definition:** Scenario Factory helps a business discover a costly or risky manual process and autonomously generate, test and improve several isolated POC/MVP candidates from approved goals, evidence and existing enterprise systems—without making production changes.

**What it does not build:** A replacement ERP/CRM; a new generic execution engine; a new platform governance authority; a general-purpose autonomous SaaS deployment platform; production-ready software by merely completing code generation.

**Customer promise:** (1) understand the process, (2) validate business value, (3) obtain a reproducible demonstrable POC, (4) compare versions, (5) optionally commission a governed production implementation.

**Product personas:** process owner (explains problem), solution engineer (inspects code/blueprint), approver (delegates bounded sandbox work), independent evaluator (judges evidence), commercial owner (prepares proposal), future portal customer (restricted L0-L2).

**Two capabilities, not confused:**
1. *Autonomy of engineering:* drafting code, repairing tests and building versions; belongs to SEF-X EEA / configured coding backend.
2. *Autonomy of solution optimization:* evaluating business objective, choosing next experiment, requesting a better candidate, comparing versions; belongs to SFX product composition and existing evidence/governance mechanisms, NOT to a second coding engine.

## 3. Five product execution experiences

| Experience | Who drives choices | Authorized operations | Expected artifact |
|---|---|---|---|
| Manual | Human step-by-step | L0/L1; separately approved L2 | Spec, blueprint, scaffold, reports |
| Assisted | Agent recommends, human approves milestone | L0/L1; authenticated L2 as approved | Draft plan, one candidate and evaluation |
| Bounded autonomous (core R2) | Agent selects only within approved mission envelope | L2 isolated actions, immutable scope and cost/iteration ceilings; no L3 | Multiple POC iterations, champion selection, evidence dossier |
| Internal professional delivery | Internal engineer + supervised agents | As above, with privileged business reviews as required | Client POC/MVP handoff |
| Customer self-service preview (later) | Customer initiates from portal | Tenant-bound restricted preconfigured templates, synthetic/sanitized data, resource quotas, no production credentials | Shareable sandbox preview and proposal |

*Fully autonomous* here means no trivial prompts or task-by-task human steering **inside a previously approved sandbox work envelope**. It never means unrestricted external actions, ignoring human approvals for risky scope changes, or automatic production deployment. Customer-facing public preview must be opt-in separately after security, privacy, abuse and cost controls are certified.

## 4. Architecture: preserve canonical Option B

```
         Customer / Internal Studio         Developer IDE/CLI
                  |                                |
          Tier-3 API / Auth / MCP / Studio front doors
                  |  (schema, policy, tenant and operation parity)
                  v
  Tier-2 agents/scenario_factory — PRODUCT COMPOSITION ONLY
  Business intake / source inventory / adaptive discovery / spec
  Goal & acceptance model / blueprint / capability resolution
  Autonomy-policy view / iteration intent planner / candidate ranker
  Evidence-based evaluator / comparison / explainable ROI / reporter
  Versioned WorkArtifact and read-model adapters
                  |                        |
                  | engineering handoff    | sandbox execution
                  v                        v
     SEF-X Engineering Execution      Intergrax Canonical Scenario Runtime
     Authority / CodingExecution      HostTaskExecutionPort -> NexusLoop
     Backend (authorized)             -> PolicyEngine -> ToolRuntime
     plan / code / tests / PR         -> persistence / evidence / HITL
     repair bounded / workspaces         |
                  |                       |
                  +---- evidence/digests --+
                       |
                 Candidate selection, independent review
                       |
               Sandbox demo / dossier / external admission
```

**Semantic ownership matrix:**
- Business problem, discovery, goals, user-facing candidate comparison and ROI: SFX Tier-2.
- Coding jobs, workers, worktrees, plans, leases, repo edits, bounded code repair, coding backend substitution, engineering PR/evidence: **SEF-X**. Do not copy its FSM into SFX.
- Scenario execution and runtime policy: **Intergrax canonical runtime**; never call coding agent as business runtime.
- Artifacts, CAS revisions, approval bindings, sessions, idempotency, checkpoints, audit and evidence: existing Intergrax owners.
- Product interfaces: **Tier-3** Studio + API + MCP + CLI, sharing one Tier-2 schema and service implementation.
- Production admission: **external** scenario-program authority and per-integration PROD-Q. SFX exports dossier only.

**Additional risk:** The separate `scripts/cursor/igx_factory.py` is a DX implementation pattern, not production SFX/SEF. Reuse tests/patterns (plan steps, worktrees, independent reviewer, bounded correction) only after authorized interface and tenancy review. Do not turn a Cursor-specific script into a platform-level semantic engine.

## 5. Canonical model additions proposed for R2

Proposed names below are **draft descriptors**, not approved contract IDs or an authorization to introduce new persistence primitives. Prefer extending the existing 17 SFX contracts when compatible. Treat any new schema and its version as an ADR/COMPAT/EVOL decision.

| Concept | Purpose | Mapping / owner |
|---|---|---|
| `BusinessGoal` / `OutcomeAcceptance` | Business objective, measurable truth set, failure boundaries, exclusion scope, measurable value hypothesis | SFX-CON-01/03/07 additive contract review |
| `SourceIntakeManifest` | Source type, owner permission, classification, origin, tenant, snapshot, permitted processing and data lineage | SFX discovery/spec artifact; existing evidence/CW |
| `AutonomyEnvelope` | Authorized actor, tenant, allowed operations, datasets/fixtures, budget, max iterations, deadline, max parallel, egress deny, stop conditions | SFX policy view bound to existing human approval authority; no new authority |
| `DelegationGrantRef` | Signed scope/digest/expiry, issuer, revocation and verification reference | Existing DecisionBinding / SEF authority — **requires approval design** |
| `CandidateIteration` | Goal digest, blueprint revision, engineering job reference, test set, verdict, changed claims, cost and trace | CW append-only WorkArtifactVersion; SEF job by reference |
| `ScenarioEvaluationVerdict` | Pass/fail by acceptance criterion, confidence/calibration, regressions, security status, next-action class | Existing EvalRunner/evidence, SFX view |
| `CandidateComparison` | Pareto/frontier summary: accuracy, safety, runtime, cost, human effort, explanations; champion rationale | SFX derived, reproducible view |
| `EscalationRequest` | Missing business decision, scope increase, unclear external effects, policy change, stagnation or budget exhausted | Existing HITL / decisions / no silent default |
| `CommercialDeliveryPackage` | Business findings, demo link, source limitations, POC run evidence, cost and ROI assumptions, path to production | SFX dossier/report, not production admission |

**Minimum formal identities:** `tenant_id`, `workspace_id`, `scenario_id`, `goal_digest`, `blueprint_digest`, `candidate_id`, `candidate_revision`, `engineering_job_id`, `evaluation_suite_digest`, `evidence_refs`, `policy_or_grant_ref`, `artifact_digest`, `code_sha`, `run_id`. Ensure IdP human attribution for approvals, not a generic API-key service principal.

## 6. Bounded autonomous goal loop — normative proposed behavior

1. **INTAKE:** Receive goal, documents/records, user-declared constraints, systems and success metrics. Do not fetch private systems without explicit grants. Label untrusted content.
2. **DISCOVER:** Identify unknowns, ask only decision-critical questions, extract sources and evidence; do not invent schemas, API capabilities, KPI baselines or permissions.
3. **SPECIFY:** Derive explicit business assertions, forbidden outcomes, gold dataset or oracle, policy boundaries, independent evaluator, test fixtures, budget and candidate-selection rule. Fail closed if acceptance cannot be defined.
4. **BLUEPRINT:** Deterministically resolve Intergrax capabilities, design roles, tools, integration profile, exceptions/HITL, typed contracts. Mark `REUSE`, `CONFIGURE`, `EXTEND`, `MISSING` without invoking runtime-worker UCA outside its validated scope.
5. **APPROVE ENVELOPE:** Human accepts *one pinned scope* including `goal_digest`, test/eval set, authority L2, allowed assets, max spend, max iterations, deadline, expiry, data bounds, external-effects prohibition and revocation.
6. **GENERATE BASELINE:** Use deterministic scaffold, generate conformance tests and initial metrics; do not present business-logic-empty scaffold as a POC.
7. **BUILD CANDIDATE:** SEF-X executes one bounded engineering job under a lease. The coding backend may itself plan, delegate, implement, test and repair within its approved role. Build must produce exact digests and engineering evidence. Approval of autonomous submission by SFX is **not currently allowed by HO-2** and requires a governed SFX↔SEF-X amendment; until then humans submit each engineering intent.
8. **QUALIFY:** Run generated-code SAST/dependency/tenant/network/secrets/budget gates before any L2 execution; then scenario tests and business truth-set evaluations through canonical runtime. Capture negative tests, uncertainty and errors.
9. **EVALUATE:** Compare all business acceptance assertions; report `PASS`, `FAIL`, `INSUFFICIENT_EVIDENCE`, `BLOCKED_POLICY`, `BLOCKED_ENVIRONMENT` separately. Never label a successful code run a successful business result.
10. **SELECT / REPAIR:** If improving, write a versioned improvement hypothesis: what failed, supporting evidence, expected measurable impact, allowed changes, cost. Only then request a new SEF job. Preserve previous best candidate and re-test unaffected regression criteria. No hidden prompts to approve each safe fix within the approved scope.
11. **STOP / ESCALATE:** Stop on all acceptance criteria satisfied; safety violation; missing business information; permission revocation; new external system/side effect; unsupported backend; no progress after policy-defined rounds; repeated identical failures; budget/deadline/iterations exceeded. No silent extension of boundaries.
12. **FREEZE CHAMPION:** Select the best qualifying candidate only among non-regressing security-compliant versions, with independent check and exact-SHA evidence; mark `SANDBOX_TESTED` / `SANDBOX_CERTIFIED` only according to the existing authorized independent gates.
13. **PRESENT:** Produce a runnable restricted demo, source-context map, version diff, business KPI report, remaining gaps and realistic production roadmap; SFX exports `ADMISSION_DOSSIER_READY` when independently qualified, never `PRODUCTION_ADMITTED`.

**Suggested product-facing states (derived status, not new execution FSM):** `DISCOVERING`, `NEEDS_INPUT`, `GOAL_READY`, `AWAITING_GRANT`, `GENERATING`, `EVALUATING`, `IMPROVING`, `WAITING_REVIEW`, `ACCEPTANCE_MET`, `BOUND_REACHED`, `ESCALATED`, `UNRESOLVED`, `SANDBOX_READY`. These are projections of SFX artifact records and SEF job references, **not** a second runtime authority. Preserve canonical R1 lifecycle FSM; introduce a versioned sub-lifecycle for iterations only after ADR approval.

### Mode policy matrix

| Action | Manual | Assisted | Autonomous L2 | Customer sandbox later |
|---|---|---|---|---|
| Ask for missing business decision | Human | Agent prompts | Agent pauses & batches important questions | Restricted, bounded |
| Produce/change spec and blueprint | Human | Agent proposes | Agent within signed goal; new revision | Limited templates |
| Generate/edit code | Human engineer | SEF on command | SEF authorized bounded jobs | Isolated managed backend only |
| Auto-repair failed tests | Human | Suggestions | Within budget and unchanged scope | Stricter cap |
| Trigger new candidate | Human | On user approval | Allowed only under approved delegated envelope *after HO-2 amendment* | Pre-approved safe operations only |
| Access real credentials/live external systems | Per separate permission | Per separate permission | **DENY by default** | **DENY** |
| Mark scenario production-ready | **DENY** | **DENY** | **DENY** | **DENY** |
| Merge to production branch / publish externally | Human governed path | Human governed path | No auto merge/push by default | Never |

### SEF-X cross-track required amendment

R1 §3.2 `HO-2` currently says **human submits every EngineeringIntent; SFX never auto-submits**. SEF-X requires ticket + plan approval and has `repairBounded` within each job. These controls cannot be bypassed by a new SFX button. Propose a carefully designed **scoped delegation grant**: human approves an immutable mission envelope once; SEF validates each derived intent's digest, scope, ticket ownership, backend permissions and remaining budget, and may accept follow-up sandbox jobs without repeating trivial human approval *only if the SEF authority explicitly permits it*. Scope change requires a new grant. Engineering merge remains independent/operator-governed. New grant is a **capability/decision binding on existing authority**, not SFX self-authorization.

Required decisions: SFX↔SEF operator OD for HO-2 amendment; SEF FSM/ticket delegation semantics (OD-SEF-04); CodingExecutionBackend allowlist (OD-SEF-05); mandatory SEF egress and evidence provenance (C-SEF-01..03); SFX operator review of autonomy policy (new OD ID allocated by operator, not invented here).

## 7. Source ingestion and data boundaries

**First MVP inputs:** free-text problem, structured interview, synthetic/sanitized PDFs/CSV/JSON, sample invoices and purchase orders, API examples, policy documents. **Later:** authenticated read-only connectors with explicit tenant permission and source snapshots; write capabilities are independently scoped.

- Never treat document text or retrieved content as executable instructions. Bind citations and provenance to artifact hashes.
- Record source owner, data classification, region/location constraint, legal basis or contractual processing permission where applicable, retention category, expiry, and redaction policy.
- Enterprise customers may require dedicated environment and encryption/security review; this is not assumed solved by existing SFX auth.
- Upload size, content-type, malware scanning, decompression bombs, prompt injection, exfiltration, and cross-tenant probes are mandatory for portal preview.
- Never place real credentials or production financial effects into the L2 demo. Secret refs only; deny egress except explicitly approved fixture paths.
- For self-service: verify anti-abuse rate limits, tenant quota/cost per customer, deletion/export, identity verification and isolated worker tenancy before public availability.

## 8. Market relevance, not feature-count parity

R1 competitor matrix (15 products) is useful for workflow-building UX but does **not** directly benchmark 2026 autonomous coding harnesses such as Claude Code/Managed Agents, Codex, or UiPath Maestro Flow. Extend the benchmark to:
1. Bounded agentic software generation and iterations (goal → code → evaluate → revise).
2. Durable task execution, delegation, review, checkpoint, cancellation, budget and human escalation.
3. Business-process evidence, deterministic+agentic composition and explainable results.
4. Change-safe scenario lifecycle, reuse and vendor portability.
5. Time/cost/quality measured on identical controlled problems, not vendor marketing.

Official product references (retrieved 2026-11-23):
- Anthropic Claude Managed Agents: https://platform.claude.com/docs/en/managed-agents/overview
- Anthropic Claude Code cloud & subagents: https://code.claude.com/docs/en/claude-code-on-the-web
- OpenAI Agents API: https://developers.openai.com/api/docs/guides/agents-api/overview
- UiPath Maestro Flow: https://www.uipath.com/newsroom/uipath-launches-maestro-flow
- LangGraph persistent/interrupt design: https://docs.langchain.com/oss/javascript/langgraph/thinking-in-langgraph

## 9. Proposed roadmap amendment to existing W0–W8

**Do not declare a new competing canonical roadmap.** Amend `SFX-MASTER-PLAN.md` to R2, then sync `SFX_WBS.md`, `SFX_INDEX.md`, capability architecture and plan pairs, `SFX_GAP_ANALYSIS.md`, ADRs, requirements traceability, UX/metrics, SEF plan/ADR and canonical roadmap ledger. No post-freeze implementation without ARCH-FREEZE CLOSED + GATE-SFX-ENTRY and per-wave approvals. New unit identifiers below are **PROPOSALS**, pending conflict-free WBS ID allocation.

### PRE-FREEZE — R2 design delta only (authorized documentation scope via new governed ticket)

| Unit | Work | Acceptance |
|---|---|---|
| R2-D01 | Audit R1 vs goal-level autonomy and market competitor capabilities; record gaps + ownership | Exact-SHA comparative evidence; no fabricated competitor tests; one-owner-per-gap |
| R2-D02 | Define autonomy semantics and three modes, grant policy, customer-safety posture and STOP rules | Reviewed policy matrix; explicit L2/L3 separation and immutable budget |
| R2-D03 | Design SFX goal/iteration/evaluation records and mapping to existing contracts/CW | No second store/FSM; draft schemas, digests, recovery, backward compatibility |
| R2-D04 | Cross-track SEF-X HO-2/delegation change proposal and OD backlog | Owner acknowledgment; no implicit authorization; SEF ticket FSM/authority acceptance path |
| R2-D05 | Vertical product slicing, invoice-first demo, reference/GEN evaluation plan | Replayable synthetic truth set, economic rubric, feature-flagged entry gating |
| R2-D06 | Updated WBS/DAG/traceability and independent adversarial architecture review | Every new requirement → WBS → test → gate; 0 unmapped critical items |

**Pre-freeze deliverable:** `R2_DESIGN_READY` documentation-only, not `PRE_IMPLEMENTATION_READY`, not SFX implementation authority. Do not reveal the sealed GEN set; move it to genuinely restricted storage.

### W3 / P5 — product substrate and mission contracts (after GATE-SFX-ENTRY)

| Unit | Implementation | Objective acceptance |
|---|---|---|
| R2-W3-01 | Extend SFX_CON-01/03/07 for measurable business goals and acceptance truth | Typed schema round-trip, incomplete goals reject with clear errors |
| R2-W3-02 | Versioned `SourceIntakeManifest` and data classification mapping | Tenant/ACL/provenance tests, redacted storage, immutable digests |
| R2-W3-03 | Autonomy envelope and grant-reference adapters (without authorizing job submission) | Grant signature/ref, revocation, expiry, allowed assets/ops, cost budget denial tests |
| R2-W3-04 | Candidate/iteration/eval/compare artifact views via existing CW and evidence store | Append-only, CAS conflict, crash/replay, cross-tenant deny, no parallel authority |
| R2-W3-05 | Extend CLI/API operation catalog and versioned schema map for later Studio/IDE | L0/L1 parity; no L2 from offline CLI; reproducible digests |

### W4 / P6–P8 — Discovery, goal plans, build and bounded autonomy

| Unit | Implementation | Objective acceptance |
|---|---|---|
| R2-W4-01 | Goal-first discovery with document/record ingestion and missing-info questions | Deterministic blocking-field validation, provenance for every inferred fact |
| R2-W4-02 | Business target/oracle generation (tests, negative cases, explicit assumptions) | Independent assertions catch wrong outcome even when technical run passes |
| R2-W4-03 | Goal-aware blueprint and capability-resolution diff | No duplicate catalog or runtime; reuse coverage measured |
| R2-W4-04 | `DETERMINISTIC_SCAFFOLD` baseline, worktree isolation and test suite creation | Same digest → same scaffold tree, business logic incomplete clearly indicated |
| R2-W4-05 | SFX candidate planner as product-level *decision* projection (not engineering FSM) | For each candidate a typed hypothesis and evaluation request; no side effects |
| R2-W4-06 | SEF backend adapter to plan/code/test/repair one candidate; respect R1 HO-2 initially | Proper SEF job/evidence refs, no unauthorized autopilot |
| R2-W4-07 | After approved SEF cross-track amendment: grant-bound autonomous follow-up jobs | One prior human grant; n safe L2 cycles without extra prompts; violation pauses |
| R2-W4-08 | Crash/revoke/budget/stall/cancel/resume handling on job boundary | No duplicate external effects; fails closed; journal has complete reproducible chain |

### W5 / P9–P10 — evaluation, security, Studio experience

| Unit | Implementation | Objective acceptance |
|---|---|---|
| R2-W5-01 | Outcome evaluator per acceptance criterion, plus test oracle and evidence citation | Wrong business output never passes solely on code-level success |
| R2-W5-02 | Improvement hypothesis, regression selector, safe best-candidate preservation | Bad candidate never overwrites champion; independent comparison reproducible |
| R2-W5-03 | Autonomous driver: execute → evaluate → propose next within grant and policy | Positive, budget, stagnation, injection, revoke and timeout tests; no infinite loop |
| R2-W5-04 | Studio mission cockpit: modes, goal, budget, timeline, version diff, champion vs alternatives | One click start only with approved envelope; clear pause/stop/why controls |
| R2-W5-05 | Studio-facing source map, assumptions and evidence trace | Business user can see where every material claim comes from |
| R2-W5-06 | Auth/tenant/no-op JWT, CORS, secrets, upload and egress boundary hardening | Auth fail-closed, 0 cross-tenant data, 0 real effectful tools in L2 |
| R2-W5-07 | Generate Playwright/UI tests and API/CLI/MCP parity | PC-1..6 and all mode permissions green on same SHA |

**Private internal demo gate:** after W5 end-to-end reference slice passes, allow controlled demonstrations ONLY in sandbox, with explicit `NOT PRODUCTION` labeling and customer data restrictions. This is a product milestone, not SFX formal closure.

### W6 / P11 — value realization and business delivery

| Unit | Implementation | Objective acceptance |
|---|---|---|
| R2-W6-01 | Candidate comparison (quality, time, cost, manual touch, robustness) | Champion selected using predeclared metrics; no post hoc goal edits |
| R2-W6-02 | ROI and customer opportunity assessment from sourced baselines | Every numeric claim points to assumption or measured evidence |
| R2-W6-03 | Portable POC package: scenarios, fixture data, typed contracts, runbook, tests, evidence, limitations | Reproducible by independent reviewer in clean sandbox |
| R2-W6-04 | Customer-facing demo/safe preview packaging | Tenant-bound non-production URL/artifact, expiry, share permission and no L3 |
| R2-W6-05 | Commercial handoff report: current process, suggested process, gaps, next MVP steps, integration risks | Client-ready briefing + separate production scope/qualification list |

### W7 / P12 — evidence-based pilots and generalization

| Unit | Implementation | Objective acceptance |
|---|---|---|
| R2-W7-01 | REF-1 invoices with errors, duplicates, missing purchase orders; repeated candidate loop | Baseline + autonomous candidate improvements; independent truth set; exact cost/time |
| R2-W7-02 | REF-2 API-to-API drift, idempotency, unknown outcome, dead-letter limitations | No duplicates under retry; explicit unknown and external-dependency cases |
| R2-W7-03 | REF-3 RAG + HITL, malicious documents, citations | No unauthorized effects; all substantive answers grounded by sources |
| R2-W7-04 | 6 previously sealed GEN cases (no pre-reveal tuning) | Seal verified, independent evaluation, rerun full suite after material changes |
| R2-W7-05 | UX rehearsals with operator/internal reviewers; do not silently delete R1 P12-05 rounds | No critical safety ambiguity; any replacement of R1 external usability rounds requires a separate operator-approved amendment |
| R2-W7-06 | Multi-backend portability with at least two independently qualified coding backend configurations, if available | Same goal/contract -> conformant outputs; report non-equivalent results honestly |
| R2-W7-07 | Time/cost/productivity benchmarking against a documented human/assisted baseline | Exact SHA/fixture/model/cost; no fake blanket performance claims |

### W8 / P13 — certification and commercial readiness

| Unit | Implementation | Objective acceptance |
|---|---|---|
| R2-W8-01 | Full R1+R2 requirement traceability, generated-code and autonomy threat-model replay | All mandatory tests/evidence green, security stop conditions verified |
| R2-W8-02 | SEF GATE-SEF-CONSUME and HO-1..7 plus autonomy delegation extension | Independent evidence at certified SHA, no override of approval policy |
| R2-W8-03 | ENT-AUDIT CHANGED-SINCE, ACT-N11 rebind, ECAP, independent IIAC | No new platform authority or unapproved frozen-core changes |
| R2-W8-04 | SFX product certification + operator closure (existing OD-SFX-11) | Factory CLOSED only, not scenario production-admitted |
| R2-W8-05 | Private commercially usable pilot gate / pricing and delivery operations | Support boundaries, demo SLA targets as proposed, cost-per-run controls, disclosure, enterprise data policy |

### AFTER W8 — explicitly separate initiatives

**Private customer self-service portal:** harden multi-tenant accounts, verified customer identity, per-tenant cost quotas, invitation-based sharing, abuse/malware/DLP controls, data retention, isolated ephemeral sandboxes, revocation, source-access governance, billing/customer terms. New authorization/gate required. It should not block first internal sale of diagnosis and managed POCs.

**Production automation:** external SCENARIO-GATE programme activation + owner-signed per-scenario admission + PROD-Q live integrations + STRICT+PRODUCT environment + rollout/monitoring/incident response. Not an automatic consequence of W8/SFX product closure.

## 10. Two parallel commercial paths

1. **Now: expert-led paid discovery.** Intake problem and representative, legally permitted sanitized data; create business-process map, acceptance definition, risk register, preliminary blueprint, proposal and estimate. Do not claim factory software is operational.
2. **After GATE-SFX-ENTRY / authorized sandbox: managed proof.** Deliver independently verified synthetic-data POC, measured KPI and repeatable run. A POC is not a production solution.
3. **After SFX certification: managed POC/MVP factory.** Sell bounded packages based on scope, iterations, datasets and acceptance obligations; charge engineering/management and license/support separately as contracts require.
4. **Later: gated private self-service.** Customer can discover, build limited candidate, evaluate, request consult. No anonymous arbitrary code execution, production connectors, or cost-unbounded sessions.

**Commercial packaging proposal:** paid Process Diagnostic, Fixed-Scope POC Sprint, MVP Pilot, Enterprise Deployment & Continuous Improvement. Price points require cost and willingness-to-pay evidence; none are fabricated here. Reusable platform IP remains separate from client-specific scenario/configuration deliverables under explicit contracts.

## 11. Measurement and acceptance for commercial usefulness

R1 already proposes TTFS business ≤45 minutes and engineer blueprint ≤20 minutes; both are **proposed**, not measured. Do not promise them as an SLA. Extend metrics:

| ID draft | Definition | Why it matters |
|---|---|---|
| AUTO-01 | Time business statement → first spec passing deterministic acceptance validation | Discovery efficiency |
| AUTO-02 | Time first approved goal → first runnable L2 business-logic candidate (not scaffold) | Actual Time-to-POC |
| AUTO-03 | Time to first independently accepted sandbox result; measured separately from any human wait | Customer value |
| AUTO-04 | Total autonomous iteration count, repair attempts, number of escalations, avoided trivial approvals | Autonomy effectiveness |
| AUTO-05 | Acceptance criteria passed / total; hard negative and safety failures tracked separately | Business quality |
| AUTO-06 | Candidate-level improvement delta vs previous champion; cost and security non-regression | Safe optimization |
| AUTO-07 | Total tokens/currency per first accepted candidate; per iteration and backend/model | Unit economics |
| AUTO-08 | Human working minutes to accepted POC; distinguish approval time and operator implementation time | Automation ROI |
| AUTO-09 | Recovery success after forced crash, idempotency and no-effect proof | Reliability |
| AUTO-10 | Unknown-domain GEN score vs REF score, independently computed | Generalization |
| AUTO-11 | Ratio of reused/configured platform capability bindings to candidate bindings | Platform leverage |
| AUTO-12 | How many customer proposals convert to paid POC and then production projects | Market learning; only real customer observations |

**Candidate hard safety invariants:** 0 unauthorized tool calls, 0 cross-tenant accesses, 0 exposed secrets, 0 unapproved production effects, 0 self-certified generated scenarios, 0 bypassed source authorizations, 0 silent scope/budget extensions. Do not trade these for TTFS.

**Go/No-Go:** A demo is valid only if its signed goal, truth set, scorecard and run evidence can be reproduced by an independent evaluator. Incomplete acceptance is explicitly `UNRESOLVED`, not a marketed success.

## 12. Required cross-program decisions before coding

1. **Preserve R1 Option B** and single semantic authority; accept R2 only as an amendment with audit.
2. **Approve bounded autonomy model** and clarify whether approval of an immutable scope may cover multiple sandbox iterations; if not, full autonomy claim is not currently implementable.
3. **SEF-X HO-2 / CON-N-01:** add independently reviewed delegated sandbox-intent semantics or explicitly keep assisted-only. This is a **cross-track change**, not SFX unilateral reinterpretation.
4. **SEF-X production of evidence:** ensure CodingExecutionBackend can expose stable ref/iteration/cost, code/test output, workspace lease, authenticated actor and audit. Require egress controls C-SEF-02.
5. **State reuse:** define candidate/goal records as CW versioned artifacts; record no new state store or execution engine. Assign current-HEAD requalification of CAP-08/30/31.
6. **Choose commercial first milestone:** internal `Invoice-to-POC` sandbox E2E as the earliest release signal, then multi-domain proofs and independent SFX certification.
7. **OD backlog:** R1 OD-SFX-05/06/07/08/09/10/13/14 and proposed 17–20 need responsible operator disposition; allocate any new OD numbers centrally without collisions.
8. **Roadmap edit:** map each proposed unit to existing W3–W8 tasks, budget/role/lane and evidence; update the canonical programme ledger only via approved ticket and correct SHA path.

## 13. Explicit risks and countermeasures

| Risk | Countermeasure |
|---|---|
| 'Autonomy' interpreted as bypassing approvals | Signed immutable narrow delegation envelope, verify on each job, central human authorizing grant and revocation |
| A second runtime or job scheduler appears in SFX | SFX stores references/views; engineering jobs in SEF; scenario runs in Nexus; checkpoints in existing owner |
| Endless agent code-fix loops | Max iterations/timeout/budget; progress delta; fail/stop/stagnation categories; preserved champion |
| Repeated wrong but test-passing business outputs | Independent acceptance oracle, negative tests, outcome verification, sample provenance |
| Generated code escapes sandbox | Workspace lease, egress deny, SAST, secrets, supply-chain/allowlist, malicious-upload testing |
| POC presented as certified/production | Visible L2 badges and disclaimers, separate independent certification and admission |
| High engineering cost without customer value | Evidence-based ROI, time-to-first-POC and cost/success tracking, business pricing validation |
| Web portal becomes abuse target | Invite-only phased rollout; quota, identity, scanning, isolation and data lifecycle controls before public access |
| Frozen core inadvertently modified | Strict Option B, EVOL/ECAP/IIAC changed-since and authorized architecture reopening for exceptions |
| GEN test leaks to implementers | Restrict sealed content; only independent reveal; invalidate contaminated tests |

## 14. Formal next task for the SFX owner (read-only planning first)

1. Audit all R1 source files at a pinned HEAD and independently validate every finding in this proposal.
2. Prepare `SFX_R2_AUTONOMOUS_GOAL_TO_POC_AMENDMENT` as a **planning package**, not a new canonical owner.
3. Add structured R2 gap and requirements rows with unique IDs, source path, risk class, owner, WBS mapping, test requirement, gate, and operator decision.
4. Prepare proposed amendments to existing ADR-SFX-001/002/004/006, and an explicit SEF-X contract/OD package for HO-2; a new ADR is only warranted if accepted design cannot fit existing semantic authorities.
5. Create one updated DAG overlay that retains SFX W0–W8 and all formal gates, putting autonomous iteration and independent business validation on the earliest authorized implementation path.
6. Submit one independent architecture/security review dossier covering authority, revocation, limits, tenant, source provenance, no-new-runtime, evidence sufficiency, product market fit.
7. Seek operator approval of R2 design and associated cross-track decisions. Only then amend canonical WBS/roadmap and authorize post-freeze implementation via standard tickets.

**Bottom line:** R2's priority is not a more sophisticated editor or more agents. It is a reliable, bounded, repeatable *Business Goal → Validated POC* factory with autonomous iteration, measurable improvement and a clear route to commercialization.