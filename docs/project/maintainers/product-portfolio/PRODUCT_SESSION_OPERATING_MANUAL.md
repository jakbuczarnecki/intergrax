# Product Session Operating Manual

**Document type:** Normative operating manual  
**Owner:** Product Session operating model  
**Audience:** Future LKW and new-product Product Sessions / human operator / model executor  
**Purpose:** Define exactly how an independently running Product Session develops one product while preserving Intergrax platform boundaries and Portfolio Control authority.

**Related contracts:**

| Document | Role |
|----------|------|
| [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) | Constitution / governance |
| [PRODUCT_BOOTSTRAP_RULES.md](PRODUCT_BOOTSTRAP_RULES.md) | How new products start |
| [PORTFOLIO_CONTROL_OPERATING_MANUAL.md](PORTFOLIO_CONTROL_OPERATING_MANUAL.md) | How central control operates |
| [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md) | How canonical audit engine plugs into gates |
| This manual | How each Product Session operates |

**Core operating principle:**

```text
BUILD THE PRODUCT FIRST.
OBSERVE PLATFORM REUSE SECOND.
```

---

## 1. Product Session mission

A Product Session exists to build **one credible product**.

Its primary questions are:

1. Does this product solve a real user/business problem?
2. Can we build a convincing product outcome?
3. What product architecture is actually required?
4. Which existing Intergrax capabilities genuinely help?
5. Where does real product pressure reveal platform strengths or gaps?

The Product Session is **not** rewarded for:

- maximizing Intergrax reuse percentage;
- using every platform capability;
- inventing shared abstractions;
- proving the platform thesis;
- producing G4 requests;
- keeping architecture similar to LKW.

A valid result may show:

- strong reuse;
- weak reuse;
- missing platform capabilities;
- mostly product-owned behavior;
- commercial weakness;
- need to stop the product.

---

## 2. One session = one product

Each Product Session owns exactly **one** product:

| Product Session | Product |
|-----------------|---------|
| LKW Product Session | Local Knowledge Workspace (LKW) |
| Contract Recovery Product Session | Contract-to-Invoice Leakage / Recovery Operator |
| Supplier Disruption Product Session | Supplier Disruption Response Operator |
| Third-Party Risk Product Session | Third-Party Risk Decision Operator |
| Deployment Guardian Product Session | Deployment / Change Guardian |

A Product Session must **not**:

- redesign another product;
- coordinate the whole portfolio;
- make cross-product priority decisions;
- edit another product's architecture or roadmap;
- self-approve shared-platform generalizations.

It may inspect another product only when:

- understanding a canonical shared capability;
- Portfolio Control requests evidence;
- a G4 decision requires bounded comparison;
- existing platform ownership cannot otherwise be determined.

Avoid broad cross-product exploration by default.

---

## 3. Product Session ownership

### Product Session owns

- product problem definition after accepted G0;
- detailed domain architecture;
- product lifecycle / state model;
- workflows;
- UX / API semantics;
- product-specific policies and decision meaning;
- integration requirements;
- product roadmap;
- implementation;
- tests;
- product proof evidence;
- G0 / G1 / G2 preparation;
- T0 preparation;
- G3 / G5 evidence preparation;
- G4 escalation package;
- T1 input preparation;
- product-specific documentation.

### Portfolio Control owns

- gate acceptance;
- cross-product impact;
- G4 disposition;
- final portfolio status;
- recommendation / priority;
- independent verification;
- T1 portfolio acceptance;
- central control cards, status, ledger, and decision log.

### Canonical audit engine owns

- audit campaign / findings / remediation lifecycle under `docs/audit_results/`.

A Product Session does **not** own Portfolio Control decisions, cross-product G4 approval, final T1 acceptance, canonical audit methodology, global portfolio prioritization, or public visual README design.

---

## 4. Source-of-truth discipline

For work inside a Product Session:

**Implementation truth (priority order):**

1. exact code at SHA;
2. tests / executable proof;
3. accepted architecture;
4. implementation plan;
5. completion report.

**Product definition:**

- pre-G0: selection record — [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md);
- after G0: accepted G0.

**Architecture:**

- accepted G1 artifact.

**Reuse experiment:**

- frozen T0 + [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md).

**Audit:**

- canonical `docs/audit_results/` campaign.

**Portfolio state:**

- Portfolio Control artifacts — [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md), control cards, [PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md), [DECISION_LOG.md](DECISION_LOG.md).

**Rule:**

```text
Do not change a Product Session artifact in order to make Portfolio Control status look better.
```

---

## 5. Session start procedure

Every Product Session cycle starts with:

1. resolve current `development` HEAD;
2. read product control card — see [products/](products/);
3. read current product architecture / plan if they exist;
4. determine current accepted gate;
5. determine next allowed action;
6. inspect only relevant recent changes;
7. identify unresolved Portfolio Control decisions, G4 items, and audit findings;
8. identify task scope;
9. work only inside authorized product scope.

For new products before G0: do **not** browse platform implementation deeply. Product need must be defined first.

---

## 6. Product-first rule

Before architecture or platform mapping, establish:

- user;
- buyer;
- real pain;
- current alternative;
- wedge;
- primary workflow;
- outcome;
- pilot hypothesis;
- MVP success;
- non-goals;
- risks.

**Do not ask:**

```text
What Intergrax capability can this product demonstrate?
```

**Ask:**

```text
What must this product do to create user value?
```

Platform analysis follows product architecture need. Architecture-before-platform-mapping is mandatory.

---

## 7. New product bootstrap sequence

For the four new products — see [PRODUCT_BOOTSTRAP_RULES.md](PRODUCT_BOOTSTRAP_RULES.md):

```text
Selection / admission
    ↓
G0 preparation
    ↓
Portfolio Control G0 acceptance
    ↓
wedge freeze
    ↓
G1 product architecture
    ↓
Portfolio Control G1 acceptance
    ↓
Platform Capability Audit
    ↓
G2 / T0 preparation
    ↓
Portfolio Control G2 acceptance / T0 freeze
    ↓
application scaffold
    ↓
implementation
    ↓
G3 vertical slice
    ↓
normal product development
    ↓
G4 whenever material shared-platform pressure appears
    ↓
G5 major proof / MVP
    ↓
G6 canonical consumer audit + T1
    ↓
G7 market validation
    ↓
G8 portfolio decision
```

Product Session **must not** skip G2 before the first implementation commit.

---

## 8. LKW special operating mode

LKW predates the multi-product bootstrap methodology.

Therefore LKW Product Session:

- continues from current authoritative implementation plan;
- remains **ACTIVE** existing reference product;
- does **not** create retroactive G0 / G1 / T0 fiction;
- does **not** receive retroactive T1 reuse scoring;
- still escalates future material shared-platform pressure through G4;
- may receive canonical PLATFORM CONSUMER AUDIT;
- supplies product / proof evidence to Portfolio Control;
- follows the same platform-boundary discipline as other products.

At LKW Product Session launch: read exact current repo state; do not rely on historical summaries. Authoritative execution status: [LKW IMPLEMENTATION_PLAN.md](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md). Control card: [products/LKW.md](products/LKW.md).

---

## 9. G0 preparation

New Product Session prepares, but does **not** self-accept, G0.

**Required content:**

- product hypothesis;
- user;
- buyer;
- pain;
- current alternatives;
- why insufficient;
- wedge;
- primary workflow;
- MVP success;
- commercial hypothesis;
- pilot hypothesis;
- caveats;
- non-goals;
- risks;
- supporting selection / market evidence.

**Mandatory internal falsification:**

```text
Would we build this if Intergrax did not exist?
```

If answer becomes **NO**: report this honestly to Portfolio Control. Do not distort product to pass.

---

## 10. G1 architecture procedure

Design architecture from **product semantics first**.

Must define as applicable:

- user flow;
- domain entities / state;
- lifecycle;
- business invariants;
- external systems;
- side effects;
- evidence;
- approvals;
- recovery;
- failure behavior;
- security;
- tenant boundaries;
- sensitive data;
- API;
- frontend interaction;
- operations / deployment.

**Only after this:** map required responsibilities to Intergrax.

**Do not:**

- copy LKW architecture;
- force Hybrid Ask / RAG / chat / agent-loop concepts;
- reuse vocabulary merely because platform already has it;
- place functionality in platform because reuse sounds attractive.

---

## 11. Platform Capability Audit before T0

For every required non-product responsibility ask:

- Does Intergrax already own this responsibility?
- Is capability actually implemented or only planned?
- What is canonical public contract?
- Can product consume unchanged?
- Can product use configuration / adapter?
- Is capability actually LKW-owned rather than platform-owned?
- Is there a real reusable gap?
- Would local implementation duplicate platform responsibility?
- Would shared extension warp another product?
- Is G4 required?

Output feeds T0. This is product / platform capability analysis, **not** canonical `docs/audit_results/` campaign unless separately requested.

---

## 12. T0 preparation

For four new products only. Before first product implementation commit:

- freeze exact start SHA;
- define product hypothesis / user / workflow;
- document diversity from LKW;
- link accepted G1;
- define full platform responsibility matrix;
- define Critical Reuse Set;
- record expected reuse / gaps;
- record ambiguities;
- freeze M1–M6 methodology;
- freeze PASS / PARTIAL / FAIL rules per [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md).

T0 classifications are **expectations / hypotheses**. They are **not** evidence of reuse. Do **not** edit T0 after implementation to improve score.

---

## 13. Scaffold rule

Application scaffold is allowed only after accepted G2 / T0 for new products.

Scaffold should be minimal. Do **not** build:

- generic private agent framework;
- private runtime;
- private registry system;
- private policy engine;
- private observability system;
- product-local clone of platform infrastructure.

Scaffold exists to host product semantics, not establish new universal mechanics.

---

## 14. Implementation loop

Normal Product Session implementation cycle:

1. choose next product outcome;
2. define bounded implementation task;
3. identify product-owned vs reused platform responsibilities;
4. implement smallest coherent vertical progress;
5. test;
6. update architecture / plan if semantics changed;
7. inspect platform pressure;
8. if no G4 trigger → continue;
9. if G4 trigger → **STOP** shared-platform work and escalate.

Do not route every implementation decision through Portfolio Control. Only material events / gates require central review.

---

## 15. G3 vertical slice

Product Session must produce **real product semantics**.

| Product | Example vertical slice |
|---------|------------------------|
| Contract Recovery | contract / invoice → discrepancy → evidence-backed recovery finding |
| Supplier Disruption | disruption → affected order / item → mitigation outcome |
| Third-Party Risk | vendor request → evidence → defensible decision |
| Deployment Guardian | release / change → multi-system evidence → GO / NO-GO |
| LKW | existing authoritative product roadmap / proofs govern |

**Do not claim G3 for:**

- boot;
- health;
- database connectivity;
- generic model call;
- scaffold;
- mocked business outcome.

---

## 16. G4 trigger rule

Product Session **must stop** before material shared-platform change.

Trigger G4 when implementation would require any material change to:

- shared contracts;
- shared runtime behavior;
- shared identity semantics;
- governance;
- HITL;
- tool execution;
- integration framework;
- evidence / runtime history;
- retries / recovery / idempotency infrastructure;
- observability;
- common registries;
- shared persistence abstraction;
- provider / backend abstraction;
- any other reusable cross-product platform responsibility.

**Do not self-approve:**

- `EXTENDED_GENERALLY`;
- `GENUINE_PLATFORM_GAP`;
- product-specific modification in shared core.

---

## 17. G4 escalation package

When escalating, Product Session provides concise evidence:

- product need;
- concrete blocked workflow;
- exact current SHA;
- relevant architecture section;
- existing platform mechanism inspected;
- why existing capability is insufficient;
- proposed product-local alternative, if any;
- proposed general platform extension, if any;
- why it is general;
- expected impact;
- urgency;
- implementation currently blocked / not blocked.

Do **not** decide impact on all products. Portfolio Control does that independently.

---

## 18. Work after G4

Possible Portfolio Control disposition:

| Disposition | Product Session action |
|-------------|------------------------|
| **PRODUCT_OWNED** | implement in product |
| **REUSE_EXISTING_PLATFORM** | use existing canonical mechanism |
| **CONFIGURE_EXISTING_PLATFORM** | adapt / configure without semantic core change |
| **GENUINE_PLATFORM_GAP** | approved shared-platform work may proceed under separately authorized scope |
| **REJECT_GENERALIZATION** | redesign product-local solution / reuse existing platform |
| **REQUIRE_AUDIT** | wait for canonical audit outcome |
| **DEFER** | do not implement disputed shared change |

Product Session follows disposition.

---

## 19. Platform work ownership

If G4 accepts a shared platform extension:

Do **not** automatically let Product Session silently patch platform. The actual implementation ownership must be explicit.

Possible models:

- separately scoped platform task / session;
- Product Session implements under explicit G4-authorized shared scope;
- another designated maintainer stream implements.

Detailed cross-session handoffs are governed by [CROSS_SESSION_COORDINATION.md](CROSS_SESSION_COORDINATION.md). No platform mutation without explicit ownership and accepted scope.

---

## 20. Product leak / private platform anti-patterns

Product Session must actively avoid:

- product branches in shared core;
- private execution runtime;
- private identity system;
- private policy engine;
- private HITL mechanism;
- private tool gateway;
- private retry / recovery framework;
- private evidence journal;
- duplicate event / observability infrastructure;
- hidden universal registries;
- direct vendor / backend dependency where canonical port should own it;
- untyped metadata escape hatches across platform boundaries;
- replacing reusable platform mechanism merely because local code is easier.

But also avoid the opposite failure: forcing legitimate domain behavior into platform.

---

## 21. Product ownership examples

**Usually product-owned:**

- domain entities;
- domain workflow;
- product-specific state;
- UX;
- business rules / meaning;
- vendor-risk decision semantics;
- recovery claim semantics;
- disruption mitigation semantics;
- deployment readiness semantics;
- application-specific presentation.

**Usually platform-owned mechanism where applicable:**

- execution infrastructure;
- identity;
- policy enforcement mechanics;
- HITL mechanics;
- tool / integration execution;
- retry / recovery infrastructure;
- canonical evidence / execution history;
- observability;
- provider-neutral infrastructure contracts.

Ownership follows **responsibility**, not directory convenience.

---

## 22. T0 deviation procedure

If implementation reveals that a frozen T0 responsibility was incorrectly classified:

Do **not** silently edit T0.

Before implementing affected path:

- stop;
- perform bounded architecture review;
- record explicit versioned deviation;
- old classification;
- proposed new classification;
- rationale;
- date / SHA;
- why change is not result-gaming;
- request independent Portfolio Control review.

Then follow [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) rules.

---

## 23. G5 product proof / MVP

Product Session owns building product proof.

Proof must show user / business outcome, not infrastructure trivia.

Evidence should include:

- exact execution path;
- relevant inputs;
- product decision / action;
- evidence / provenance;
- failure mode;
- constraints / limitations;
- reproducibility where applicable.

**Do not claim:**

- production readiness from PoC;
- commercial validation from demo;
- platform reuse from one successful run.

Portfolio Control independently evaluates G5.

---

## 24. G6 / T1 preparation

For four preregistered products:

Product Session prepares:

- current exact SHA;
- T0 reference;
- final responsibility implementation inventory;
- evidence for expected classifications;
- known deviations;
- relevant tests / proofs;
- G4 history;
- platform changes introduced during experiment.

Then canonical **PLATFORM CONSUMER AUDIT** occurs per [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md).

Product Session does **not**:

- assign final audit verdict;
- rewrite findings;
- self-approve M1–M6;
- self-declare reuse PASS.

After audit: prepare T1 using canonical evidence. Portfolio Control independently accepts / rejects result.

---

## 25. Audit findings / remediation

If canonical audit creates findings:

Product Session may implement findings assigned to product scope.

But:

- finding lifecycle remains in `docs/audit_results/` campaign register;
- Product Session roadmap may reference IDs;
- Product Session must not create duplicate finding state;
- remediation follows AUDIT_REMEDIATION_PROTOCOL in audit campaign.

Product gate status and finding remediation status remain **separate**.

---

## 26. G7 market evidence

Product Session may collect / prepare:

- pilot feedback;
- customer interviews;
- usage evidence;
- willingness-to-pay evidence;
- buyer objections;
- sales friction;
- competitive evidence;
- integration feasibility feedback.

Do not convert anecdote into validation. Separate:

- product evidence;
- customer evidence;
- commercial evidence.

Portfolio Control uses G7 evidence for portfolio decisions.

---

## 27. G8 input

Product Session may recommend: **ACCELERATE**, **CONTINUE**, **REDUCE**, **PAUSE**, **STOP**.

But Portfolio Control owns portfolio decision.

Product Session should report:

- evidence;
- blockers;
- next milestone;
- expected cost;
- unresolved risks;
- market signal;
- platform pressure;
- opportunity to learn;
- reasons to stop.

Do not argue for continuation merely because work has already been invested.

---

## 28. Commercial vs platform outcome

Keep two dimensions **independent**.

A product may:

- commercially succeed + strongly reuse platform;
- commercially succeed + expose weak platform reuse;
- commercially fail + provide strong platform evidence;
- commercially fail + provide weak platform evidence.

Do not merge these into one verdict.

---

## 29. Documentation responsibility

Product Session owns detailed:

- architecture;
- roadmap;
- implementation documentation;
- product proofs;
- product evidence;
- product-specific operational docs.

Portfolio Control card is **not** product architecture. Public Product Presentation Document is **not** product architecture. VIS / public documentation may consume accepted facts later.

Product Session must keep authoritative product docs accurate enough that Portfolio Control and VIS can source factual claims safely.

---

## 30. Public claim input

When a product fact may be used publicly, Product Session should be able to point to:

- authoritative product document;
- implementation / proof evidence;
- exact limitation;
- current status.

Do not write marketing language as implementation truth. Detailed cross-session handoffs are governed by [CROSS_SESSION_COORDINATION.md](CROSS_SESSION_COORDINATION.md).

---

## 31. Concurrency rules

All sessions share `development`.

Product Session must:

- resolve HEAD at task start;
- never assume branch did not move;
- pin exact reviewed / implemented commits;
- preserve unrelated changes;
- avoid reset / rebase / stash / clean / amend / force push;
- stage only task-owned files;
- detect overlap before editing shared files;
- distinguish concurrent commits from own work;
- fast-forward push only.

If shared file changed concurrently: re-read it before editing. If conflict is material: stop / report rather than overwrite.

---

## 32. Task discipline

Implementation tasks should be bounded.

Before a task establish:

- outcome;
- exact scope;
- expected files / entrypoints;
- platform dependencies;
- tests;
- non-goals;
- G4 risk.

**Avoid:**

- repo-wide exploration without hypothesis;
- broad refactors during feature work;
- opportunistic platform cleanup;
- unrelated documentation rewrites;
- massive context loading.

Accuracy > token savings, but targeted evidence > bulk reading.

---

## 33. Product Session completion report

Every meaningful completed task should report:

- task ID / milestone;
- start SHA;
- final SHA;
- commit SHA(s);
- exact changed files;
- tests / evidence;
- product outcome achieved;
- architecture / plan changes;
- platform components reused;
- platform pressure encountered;
- G4 required? yes / no;
- unresolved issues;
- next product action.

The report is a claim submitted for independent Portfolio Control verification when a material gate / event is involved.

---

## 34. When to contact Portfolio Control

**Required:**

- G0 ready;
- G1 ready;
- G2 / T0 ready;
- before first implementation after T0 acceptance;
- G3 ready;
- material G4 pressure;
- G5 ready;
- G6 / T1 ready;
- significant G7 evidence;
- G8 recommendation;
- product pause / stop;
- material public claim requiring central acceptance;
- cross-product conflict.

**Not required:**

- ordinary local implementation commit;
- minor refactor within accepted architecture;
- routine tests;
- local product-specific UX change.

---

## 35. Questions every Product Session must answer

At any moment:

- What user problem are we solving?
- What is the current accepted gate?
- What is the next allowed action?
- What is product-owned?
- What platform responsibilities do we consume?
- Which are actually reused vs expected?
- Have we introduced any private platform mechanism?
- Have we touched shared platform semantics?
- Is G4 required?
- What exact evidence supports current milestone?
- What would falsify our product hypothesis?
- Are we changing the product to fit Intergrax?
- Are we overstating product / platform evidence?
- What is the shortest path to next meaningful user outcome?

---

## 36. Failure modes

Explicitly prohibit:

- platform-first product design;
- LKW cloning;
- architecture before product need;
- implementation before T0 for new products;
- retroactive T0;
- metric gaming;
- self-approved G4;
- product-specific branches in shared core;
- private platform duplication;
- unjustified new universal abstraction;
- treating planned platform capability as implemented;
- claiming reuse without evidence;
- claiming G3 from scaffold;
- claiming commercial validation from demo;
- treating tests as complete proof;
- self-declaring G6 PASS;
- copying canonical audit findings into private tracking systems;
- using public docs as product source of truth.

---

## 37. LKW vs new product summary

| Dimension | LKW | Four new products |
|-----------|-----|-------------------|
| Program role | Existing reference product | Newly selected applications |
| Program State | **ACTIVE** | **SELECTED** / Pre-bootstrap |
| Retroactive bootstrap | **No** — no retroactive G0 / G1 / T0 | G0 required |
| T0 / T1 | **No** retroactive T0 / T1 reuse scoring | T0 required before implementation |
| Operating path | Continue authoritative roadmap | Full bootstrap sequence |
| G4 | Applies prospectively | Applies when triggered |
| Canonical consumer audit | Valid | Valid at G6 |
| Control card | [products/LKW.md](products/LKW.md) | [products/contract-recovery.md](products/contract-recovery.md), [products/supplier-disruption.md](products/supplier-disruption.md), [products/third-party-risk.md](products/third-party-risk.md), [products/deployment-guardian.md](products/deployment-guardian.md) |

---

## 38. Operating loop

**New product:**

```text
Understand product need
    → prepare current gate
    → Portfolio Control acceptance
    → implement next product outcome
    → continuously detect platform pressure
    → escalate G4 when required
    → collect product evidence
    → submit next material gate
```

**LKW:**

```text
Read current authoritative state
    → implement current product roadmap
    → detect platform pressure
    → escalate G4 when required
    → collect proof / product evidence
    → submit material checkpoints to Portfolio Control
```

---

## 39. Handoff principle

Product Session should be independently productive. Portfolio Control should **not** become a synchronous dependency for ordinary work.

The boundary is:

```text
Product Session controls product execution.
Portfolio Control controls gates, cross-product truth and shared-platform generalization.
```

Detailed cross-session handoffs are governed by [CROSS_SESSION_COORDINATION.md](CROSS_SESSION_COORDINATION.md).
