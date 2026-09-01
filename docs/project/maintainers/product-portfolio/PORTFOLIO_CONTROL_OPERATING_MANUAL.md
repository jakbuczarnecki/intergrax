# Portfolio Control Operating Manual

**Document type:** Normative operating manual  
**Owner:** Portfolio Control  
**Audience:** Future Portfolio Control Session / human operator / model executor  
**Purpose:** Define exactly how central control supervises multiple independently evolving Intergrax products without becoming another Product Session.

**Related contracts:**

| Document | Role |
|----------|------|
| [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) | Constitution / governance |
| [PRODUCT_BOOTSTRAP_RULES.md](PRODUCT_BOOTSTRAP_RULES.md) | How new products start |
| [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md) | How canonical audit engine plugs into gates |
| This manual | How central control actually operates |

---

## 1. Mission

Portfolio Control exists to answer three continuous questions:

1. Are the products themselves progressing on credible evidence?
2. Is Intergrax remaining a reusable platform rather than accumulating product-specific hacks or private duplicate infrastructure?
3. Given current evidence, where should project attention and investment go?

The session is **skeptical by default**.

**No reward for:**

- more features;
- more shared abstractions;
- more products;
- PASS;
- keeping every product alive.

**Valid outcomes include:**

- CONTINUE;
- ACCELERATE;
- REDUCE;
- PAUSE;
- STOP;
- product-owned implementation;
- platform extension;
- rejection of a proposed platform extension;
- evidence that Intergrax reuse is weaker than expected.

**Core operating principle:**

```text
VERIFY REPO → UNDERSTAND PRODUCT STATE → CHECK GATE → CHECK PLATFORM IMPACT
→ DECIDE → UPDATE CENTRAL CONTROL ARTIFACTS
```

**Never:**

```text
READ PRODUCT SESSION REPORT → TRUST IT → COPY STATUS
```

---

## 2. Operating scope

Portfolio Control supervises:

| Product | Session |
|---------|---------|
| Local Knowledge Workspace (LKW) | LKW Product Session |
| Contract-to-Invoice Leakage / Recovery Operator | Contract Recovery Product Session |
| Supplier Disruption Response Operator | Supplier Disruption Product Session |
| Third-Party Risk Decision Operator | Third-Party Risk Product Session |
| Deployment / Change Guardian | Deployment Guardian Product Session |

**Parallel specialist streams** may exist, including:

- **COMM** - LKW proof work within its authorized roadmap;
- **VIS** - public visual/documentation presentation.

They are **evidence/content providers** where relevant, not Portfolio Control substitutes.

Detailed cross-session handoffs remain **MP-20**.

**Portfolio Control is NOT responsible for:**

- detailed product implementation;
- detailed product architecture ownership;
- acting as the developer for all products;
- replacing Product Sessions;
- replacing canonical audit methodology;
- designing public README visuals;
- owning COMM proof implementation;
- inventing product-market evidence.

---

## 3. Source-of-truth hierarchy

### Product implementation questions

| Priority | Source |
|----------|--------|
| 1 | Exact implementation at current/pinned SHA |
| 2 | Tests / executable proof evidence |
| 3 | Accepted product architecture |
| 4 | Product implementation roadmap/plan |
| 5 | Product Session completion report |
| 6 | Portfolio Control summaries |

### Audit findings

The canonical audit campaign register in `docs/audit_results/` wins. See [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md).

### Current product/portfolio status

[PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md) is the central index **only after independent verification**.

### Product definition

| Stage | Authority |
|-------|-----------|
| Pre-G0 | Frozen selection record - [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) |
| After G0 | Accepted G0 |
| After G1 | Accepted architecture for architecture semantics |

### Reuse evidence

| Question | Authority |
|----------|-----------|
| T0/T1 methodology | [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) |
| Implementation conformance | Canonical audit engine - `docs/audit_results/` |
| Portfolio decisions | Portfolio Control consumes both |

**Explicit rule:**

> A Product Session report is a **claim to verify**, not evidence by itself.

---

## 4. Session start procedure

Every Portfolio Control working cycle should begin with a bounded synchronization pass.

**Required sequence:**

1. Resolve current `development` HEAD.
2. Read [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md).
3. Identify products/events changed since last Portfolio Control checkpoint.
4. Inspect only relevant Product Session artifacts / commits.
5. Check whether any product crossed or is approaching a gate.
6. Check whether any material shared-platform change occurred.
7. Check whether any canonical audit campaign relevant to active products changed.
8. Check unresolved [PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md) items.
9. Check whether product priority/recommendation inputs materially changed.
10. Update central artifacts **only after verification**.

Do **not** scan the entire repo every cycle. Use **delta-driven inspection**.

---

## 5. Event-driven control model

Portfolio Control should **not** continuously micro-review every product commit.

**Material events requiring attention:**

| Event | Action |
|-------|--------|
| Product Session declares G0 ready | Gate review |
| G1 architecture ready | Gate review |
| G2/T0 ready | Gate review + freeze check |
| First implementation about to begin | Scaffold gate verification |
| G3 vertical slice ready | Product evidence review |
| Product identifies shared-platform pressure | Route to G4 |
| Shared platform modification requested | Route to G4 before implementation |
| G5 proof/MVP ready | Major proof review |
| G6/T1 requested | Canonical audit + T1 review |
| Canonical audit creates relevant findings | Consume; do not duplicate |
| Major market/customer evidence arrives | Portfolio re-evaluation |
| Material blocker or product failure | Status / recommendation review |
| Cross-product conflict appears | G4 + cross-product matrix |
| Product requests pause/stop/reprioritization | G8 / recommendation update |
| Product/COMM/VIS claim intended for public presentation materially changes | Public claim control |

Normal local implementation between accepted gates remains **Product Session ownership** unless a platform-impact trigger appears.

---

## 6. Product checkpoint review

When a Product Session reports completion/readiness, Portfolio Control **MUST** independently establish:

- exact product/session commit SHA;
- exact changed files relevant to claim;
- whether architecture/plan/proof actually supports claimed status;
- whether tests/evidence match the claim;
- whether gate prerequisites were satisfied;
- whether central product card/status are stale;
- whether platform boundaries were crossed;
- whether another product is affected.

**Possible verdicts:**

| Verdict | Meaning |
|---------|---------|
| **ACCEPTED** | Claim verified; gate may advance |
| **ACCEPTED WITH EXPLICIT GAP** | Accepted with documented limitation |
| **REJECTED / NOT READY** | Claim insufficient; return to Product Session |
| **REQUIRES CANONICAL AUDIT** | Adversarial evidence needed before decision |
| **REQUIRES G4** | Material platform pressure must be classified first |

Do **not** invent PASS semantics that conflict with canonical audit verdicts. These are Portfolio Control gate/checkpoint decisions, not audit finding status.

---

## 7. G0–G8 operating procedure

For each gate: entry condition → what Portfolio Control verifies → what it must NOT do → acceptance output → next allowed action.

Gate meanings remain defined in [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md). Audit integration per [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md).

---

### G0 - Product baseline

**Entry:** Product Session declares product baseline ready (four new products) or baseline-ingestion review due (LKW).

**Portfolio Control verifies:**

- real product problem;
- user/buyer;
- pain/current alternatives;
- wedge;
- primary workflow;
- MVP success criteria;
- commercial/pilot hypothesis;
- caveats/non-goals/risks;
- independence from Intergrax existence.

**Critical falsification question:**

> Would this still be worth exploring if Intergrax did not exist?

If **NO** → reject G0.

**Must NOT:**

- optimize wedge for platform capabilities;
- design architecture;
- score reuse.

**Acceptance output:** G0 accepted; product baseline frozen.

**Next allowed action:** G1 preparation.

**LKW:** baseline-ingestion review only; no retroactive bootstrap.

---

### G1 - Product architecture

**Entry:** Product Session declares architecture ready.

**Portfolio Control verifies** architecture starts from product need:

- user flow;
- domain lifecycle/state;
- external systems;
- consequential actions;
- evidence;
- approvals;
- recovery;
- security/tenancy;
- API/frontend/operations.

Then verify platform mapping. Check for:

- forced LKW vocabulary;
- platform-first design;
- premature shared abstractions;
- product semantics leaking into core;
- ignored canonical capabilities.

Invoke canonical audit **conditionally** per [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md) when architectural uncertainty warrants it.

**Must NOT:** accept architecture that exists primarily to justify platform reuse.

**Acceptance output:** G1 accepted; architecture becomes authoritative for design semantics.

**Next allowed action:** Platform capability audit / G2 (T0) preparation.

---

### G2 - T0 reuse baseline

**Entry:** Product Session declares T0 ready. **Four new products only.**

**Portfolio Control verifies:**

- G0 accepted;
- G1 accepted;
- no product implementation commit has started;
- exact start SHA frozen;
- diversity from LKW;
- platform responsibility matrix;
- Critical Reuse Set;
- expected reuse/gaps;
- M1–M6 methodology;
- PASS/PARTIAL/FAIL rules;
- ambiguities recorded.

Reject retroactive or score-optimizing T0 edits.

**Must NOT:** allow scaffold or implementation before T0 freeze.

**Acceptance output:** T0 frozen at pinned SHA.

**Next allowed action:** application scaffold / implementation.

**LKW:** not applicable retroactively.

---

### G3 - First real vertical slice

**Entry:** Product Session declares first meaningful end-to-end outcome ready.

**Portfolio Control verifies** actual product outcome.

**Reject as insufficient:**

- application starts;
- health endpoint;
- DB connection;
- one model call;
- generic agent;
- scaffold-only tests;
- mocked path lacking product semantics.

Require observable domain outcome appropriate to product.

Check platform boundary pressure exposed by slice. If material shared change appears → route to **G4** before shared change proceeds.

**Must NOT:** accept scaffold as vertical slice.

**Acceptance output:** G3 accepted; first real product workflow evidenced.

**Next allowed action:** continued product development; G4/G5 as triggered.

---

### G4 - Material platform pressure

**Entry:** Product Session escalates before implementing material shared-platform change.

This is one of Portfolio Control's **central responsibilities**.

For every proposed material shared-platform change answer:

1. What exact product need triggered it?
2. Is the need real and currently required?
3. Does existing platform capability already satisfy it?
4. Is product using existing capability incorrectly?
5. Would local implementation duplicate a platform responsibility?
6. Is proposed platform extension genuinely general?
7. Does it encode product-specific semantics?
8. What happens to LKW?
9. What happens to every other active product?
10. Would they reuse unchanged/configured, require adoption, reveal conflict, or be unaffected?
11. Is canonical audit required?
12. What reuse classification is currently justified?

**Possible disposition:**

| Disposition | Meaning |
|-------------|---------|
| `PRODUCT_OWNED` | Keep in product |
| `REUSE_EXISTING_PLATFORM` | Use existing mechanism unchanged |
| `CONFIGURE_EXISTING_PLATFORM` | Use via intended configuration/adapter |
| `GENUINE_PLATFORM_GAP` | Shared extension may proceed |
| `REJECT_GENERALIZATION` | Proposed extension is not general |
| `REQUIRE_AUDIT` | Canonical audit needed before decision |
| `DEFER` | Insufficient evidence |

Program canonical classifications (`REUSED_UNCHANGED`, `REUSED_CONFIGURED`, `EXTENDED_GENERALLY`, `PRODUCT_OWNED`, `PLATFORM_LEAK`) apply per [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md).

**Must NOT:** allow Product Session self-approval of `EXTENDED_GENERALLY`.

**Acceptance output:** G4 disposition recorded; implementation path authorized or blocked.

**Next allowed action:** product-owned work, platform work, audit, or redesign - per disposition.

A G4 decision must occur **BEFORE** material shared-platform modification.

---

### G5 - MVP / major proof

**Entry:** Product Session declares major proof or MVP milestone ready.

**Portfolio Control verifies:**

- milestone matches accepted product baseline and roadmap;
- evidence is product-meaningful, not scaffold-only;
- platform boundaries respected;
- prior G4 dispositions honored for any shared changes used.

Invoke canonical audit **conditionally** when material claims require adversarial evidence per [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md).

**Must NOT:** treat tests alone as complete commercial or product proof.

**Acceptance output:** Proof accepted, partial, or failed relative to product goals.

**Next allowed action:** G6 preparation, G7, continued development, or G8 as warranted.

---

### G6 - T1 reuse audit

**Entry:** Product Session requests T1 / reuse evaluation. **Preregistered new products.**

**Required sequence:**

```text
PLATFORM CONSUMER AUDIT @ exact SHA
        ↓
canonical findings / conformance matrix
        ↓
T1 (PRODUCT_REUSE_PROOF)
        ↓
M1–M6
        ↓
Portfolio Control gate decision
```

**Portfolio Control verifies:**

- T0 was frozen before implementation;
- audit campaign pinned to exact SHA;
- T1 compares against original T0, not rewritten expectations;
- audit classifications are not collapsed into T1 classifications without reasoning;
- no duplicate finding registers created in portfolio artifacts.

**Must NOT:** accept aggregate reuse claim without independent audit + T1 evidence.

**Acceptance output:** Reuse outcome recorded; PASS/PARTIAL/FAIL accepted or rejected.

**Next allowed action:** G7, G8, ledger update, or remediation per audit protocol.

**LKW:** no retroactive T0/T1.

---

### G7 - Market validation

**Entry:** Product Session submits external or pilot evidence.

**Portfolio Control verifies:**

- evidence supports or refutes commercial/pilot hypothesis from G0;
- limitations and scope of validation are explicit;
- public claims are not overstated.

**Must NOT:** invent market evidence; conflate platform learning with commercial validation.

**Acceptance output:** Validation recorded; portfolio action informed.

**Next allowed action:** G8 or recommendation/priority update.

---

### G8 - Continue / accelerate / reduce / pause / stop

**Entry:** Portfolio Control periodic review or material trigger.

**Portfolio Control evaluates** using evidence from all prior gates and current portfolio inputs.

**Must NOT:** keep weak products alive solely for platform exercise.

**Acceptance output:** Recommendation and/or program state change recorded in central artifacts.

**Next allowed action:** per disposition - including PAUSE, STOP, or ACCELERATE.

---

## 8. Cross-product impact check

For every **accepted** shared-platform change, Portfolio Control evaluates **all five products**, including the origin product.

**Required matrix:**

| Product | Current use | Effect of change | Adoption required? | Risk/conflict |
|---------|-------------|------------------|--------------------|---------------|
| LKW | | | | |
| Contract Recovery | | | | |
| Supplier Disruption | | | | |
| Third-Party Risk | | | | |
| Deployment Guardian | | | | |

**Possible outcomes:**

- unaffected;
- compatible by configuration;
- requires later adoption;
- reveals conflict;
- invalidates proposed generalization.

Do **not** claim reuse merely because another product could theoretically use it. Actual reuse evidence remains evidence-bound.

---

## 9. Platform impact ledger procedure

[PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md) is for **accepted** cross-product platform-impact evidence.

**Record an entry when** there is a material, accepted event such as:

- new product reuses shared capability unchanged/configured;
- product forces genuinely general extension;
- proposed generalization fails;
- conflict between product needs emerges;
- previous platform assumption is invalidated.

**Do NOT create ledger entries for:**

- speculative future reuse;
- LKW historical consumption merely reconstructed retrospectively;
- unresolved G4 hypotheses;
- raw canonical audit findings;
- ordinary product-local implementation.

Ledger classification must use program canonical reuse vocabulary where appropriate.

---

## 10. Canonical audit engine usage

Reference [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md).

**Portfolio Control:**

- requests audit;
- scopes audit question;
- consumes audit evidence;
- does **not** create another audit format.

**At G6** for preregistered new products:

```text
PLATFORM CONSUMER AUDIT @ exact SHA
→ canonical findings/conformance matrix
→ T1
→ M1–M6
→ Portfolio Control decision
```

**At G4:** audit conditional according to ambiguity/materiality.

Do **not** duplicate findings into Portfolio Control artifacts. Link to campaign registers only.

---

## 11. T0 / T1 anti-gaming control

Portfolio Control must actively protect the experiment from hindsight bias.

| Phase | Rule |
|-------|------|
| Before implementation | Freeze T0 |
| During implementation | If responsibility genuinely changes ownership/classification, require the versioned deviation mechanism in [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) **before** affected implementation |
| At T1 | Compare against original T0, not rewritten expectations |

**Reject:**

- denominator manipulation;
- removing failed responsibilities;
- retroactively labeling duplicated platform concern `PRODUCT_OWNED`;
- calling a product-specific core hack a general extension;
- treating missing evidence as reuse;
- forcing irrelevant Intergrax capabilities into denominator.

---

## 12. Product-to-platform boundary test

Portfolio Control should repeatedly ask:

**Product owns:**

- business semantics;
- domain state;
- workflow meaning;
- UX;
- product-specific decision meaning;
- domain acceptance.

**Platform owns** reusable enforcement/mechanisms such as, when applicable:

- execution;
- identity;
- governance/policy boundary;
- HITL mechanics;
- tool/integration execution;
- canonical evidence/runtime history;
- retry/recovery/idempotency infrastructure;
- observability;
- common infrastructure contracts.

Do **not** move something to platform merely because two products might someday reuse it.

**Rule:**

> Product need first. General reusable responsibility second. Speculative reuse never justifies platform expansion.

---

## 13. Product leak / private platform detection

**Escalate** when a Product Session creates or proposes:

- private execution runtime;
- private identity model;
- private policy engine;
- private HITL infrastructure;
- private tool gateway;
- private retry/recovery framework;
- private evidence journal;
- duplicate observability/event system;
- private platform-level registry;
- direct provider/vendor dependency bypassing canonical abstraction;
- product-specific branches in shared core.

**Possible outcome:**

- product design correction;
- G4;
- PLATFORM CONSUMER AUDIT;
- canonical finding if audited.

Do **not** automatically force product-specific domain logic into platform.

---

## 14. Portfolio prioritization

Portfolio Control periodically evaluates each product using evidence such as:

- market/customer signal;
- buyer clarity;
- pilot accessibility;
- product feasibility;
- progress toward meaningful MVP;
- next milestone cost;
- integration burden;
- competitive pressure;
- platform learning value;
- dependencies;
- blockers;
- opportunity cost.

Platform learning is **ONE factor**, not the objective function.

Do **not** preserve a commercially weak product merely because it exercises useful Intergrax capabilities.

**Recommendation** (separate from priority):

| Value | Meaning |
|-------|---------|
| ACCELERATE | Increase attention or pacing |
| CONTINUE | Maintain current course |
| REDUCE | Lower investment or scope emphasis |
| PAUSE | Recommend suspension pending evidence |
| STOP | Recommend program exit |

**Priority:**

| Value | Meaning |
|-------|---------|
| HIGH | Among highest portfolio attention items |
| MEDIUM | Standard attention |
| LOW | Lower relative attention |

Recommendation and Priority remain **separate dimensions**.

---

## 15. Pause / stop discipline

Portfolio Control must be willing to stop work.

**Reasons may include:**

- weak product hypothesis;
- failed G0;
- incumbent advantage destroys wedge;
- pilot/integration cost disproportionate;
- repeated architecture distortion to fit platform;
- no credible path to product evidence;
- platform experiment already learned enough;
- another product dominates opportunity;
- unresolved risk exceeds value.

Stopping a product is **NOT** platform failure by itself.

Commercial failure may still provide valid platform evidence.

Platform failure may occur even if the product itself is commercially promising.

Keep those conclusions **separate**.

---

## 16. Central artifact ownership

**Portfolio Control owns** and may update after evidence verification:

- [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md)
- [PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md)
- [DECISION_LOG.md](DECISION_LOG.md)
- `products/*.md` control cards
- central portfolio [README.md](README.md)
- Portfolio Control program integration/manual documents

**Product Sessions own:**

- detailed architecture;
- detailed roadmap/implementation plan;
- code;
- tests;
- product proof artifacts;
- G0/G1/T0 preparation artifacts

**Canonical audit engine owns:**

- audit campaign lifecycle / findings / remediation state - `docs/audit_results/`

**Public documentation / VIS owns:**

- visual/public presentation

Do **not** duplicate detailed product architecture into control cards.

---

## 17. Central artifact update rules

| Artifact | Update when |
|----------|-------------|
| **PORTFOLIO_STATUS** | Only after repo/evidence verification; records current central state, not aspirations |
| **Control card** | Gate changes; canonical product source changes; latest accepted proof/audit changes; recommendation/priority changes materially; presentation artifact becomes canonical/registered |
| **PLATFORM_IMPACT_LEDGER** | Append evidence-backed accepted platform-impact events |
| **DECISION_LOG** | Material portfolio/program decisions only; no implementation chatter |
| **README** | Navigation / central operating map; not a status dump |

---

## 18. Public claim control

Portfolio Control does **not** design public copy, but protects claim integrity.

If VIS/public documentation wants a material claim, Portfolio Control should be able to identify:

- authoritative product source;
- proof/audit evidence;
- current limitations;
- whether claim is accepted for public presentation.

Do **not** let public documentation become a source of truth.

COMM proof claims for LKW must be verified/accepted before being elevated as program evidence.

Detailed handoff belongs to **MP-20**.

---

## 19. Concurrency / shared development branch

The program operates while multiple sessions may advance shared `development`.

**Rules:**

- never assume one global frozen SHA for all products;
- pin exact SHA for each reviewed claim/gate/audit;
- distinguish product/session commit from current branch HEAD;
- preserve concurrent unrelated work;
- revalidate when later shared changes invalidate earlier evidence;
- do not attribute unrelated commits to a Product Session;
- compare exact commit ranges when needed.

Canonical audit campaign SHA rules remain governed by [AUDIT_PROTOCOL.md](../../../audit_results/AUDIT_PROTOCOL.md).

---

## 20. Stale status / conflict resolution

When sources disagree:

| Conflict | Resolution |
|----------|------------|
| Plan vs implementation | Implementation beats plan for what exists |
| Informal prose vs accepted architecture | Accepted architecture wins for intended design |
| Copied summary vs audit register | Canonical audit finding register wins |
| Product Session claim vs Portfolio Control | Product Session claim does not override verification |

If [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md) is stale → fix only after evidence verification.

If two Product Sessions request incompatible platform changes → **neither wins by order or urgency**; escalate to cross-product G4 analysis.

---

## 21. Minimum portfolio control review record

For a material gate/checkpoint, Portfolio Control should be able to state:

- product;
- gate/event;
- reviewed SHA(s);
- authoritative artifacts checked;
- claim under review;
- evidence supporting/refuting claim;
- platform impact;
- other products checked;
- audit required? yes/no + reason;
- verdict/disposition;
- central artifacts updated;
- next allowed action.

This is an **operating checklist**. Do **not** create a new persistent review database or file tree merely for this record. Use existing owned artifacts and the canonical audit system as appropriate.

---

## 22. Daily / periodic operating loop

```text
A. Synchronize          - HEAD, PORTFOLIO_STATUS, delta since last checkpoint
B. Detect material changes
C. Verify product evidence
D. Process pending gates
E. Process G4 / platform pressure
F. Process audit / T0 / T1 evidence
G. Re-evaluate portfolio priorities if inputs changed
H. Update central artifacts
I. Publish concise current-state handoff
```

Do **not** force all steps when nothing relevant changed.

---

## 23. Questions Portfolio Control must always be able to answer

At any point the central session should answer:

- What are the five products?
- What state/gate is each in?
- What is each product's next allowed action?
- What evidence supports that state?
- What changed since previous control checkpoint?
- Which shared capabilities are actually reused?
- Which reuse is only hypothetical?
- Which product created current platform pressure?
- Has any product duplicated/bypassed platform responsibilities?
- Which platform extensions were genuinely product-driven?
- What effect did each shared change have on all other products?
- Are any products being distorted to prove Intergrax?
- What canonical audits are active/relevant?
- Which unresolved findings affect product/platform decisions?
- Which product should receive attention next and why?
- Which public claims are actually supportable?

---

## 24. Failure modes / anti-patterns

**Explicitly prohibit:**

- trusting Product Session completion summaries;
- running repository-wide audits for every gate;
- copying canonical findings into portfolio docs;
- platform-first product design;
- retroactive T0;
- speculative generalization;
- private platform duplication;
- Product Session self-approval of G4;
- marking planned capability as implemented;
- marking READY_FOR_REVIEW as accepted;
- treating tests as complete proof;
- confusing product success with platform success;
- confusing commercial failure with platform failure;
- prioritizing products solely for platform-learning value;
- keeping all products active by default;
- using public docs as truth;
- creating a second Portfolio Control audit engine.

---

## 25. Initial state at session launch

Document current launch assumptions. At actual future session launch, Portfolio Control **MUST** re-read current repo and update assumptions if they have changed.

### LKW

- **ACTIVE** existing reference product;
- baseline-ingested;
- detailed current state obtained from authoritative LKW plan/implementation at session launch;
- **no retroactive T0/T1**.

### Four new products

| Product | State at MP-17 closeout |
|---------|-------------------------|
| Contract Recovery | SELECTED; Pre-bootstrap; G0 pending |
| Supplier Disruption | SELECTED; Pre-bootstrap; G0 pending |
| Third-Party Risk | SELECTED; Pre-bootstrap; G0 pending |
| Deployment Guardian | SELECTED; Pre-bootstrap; G0 pending |

All four: no accepted G1; no T0; no scaffold; no G3; no reuse evidence.

**Do not claim** the six sessions already exist. This manual defines procedure only; session launch is **MP-22**.

---

## 26. Exit / handoff format

At the end of a meaningful Portfolio Control cycle, produce a concise operator handoff containing:

- current HEAD;
- changed products/events reviewed;
- gate decisions;
- G4 decisions;
- audit activity;
- accepted platform-impact changes;
- recommendation/priority changes;
- blockers;
- next required Portfolio Control event.

Do **not** generate lengthy narrative if nothing changed.

---

## Related documents

| Question | Document |
|----------|----------|
| Program constitution | [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) |
| New product bootstrap | [PRODUCT_BOOTSTRAP_RULES.md](PRODUCT_BOOTSTRAP_RULES.md) |
| Audit integration | [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md) |
| Current portfolio state | [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md) |
| Reuse methodology | [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) |
| Canonical audit engine | [docs/audit_results/README.md](../../../audit_results/README.md) |
