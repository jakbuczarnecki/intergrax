# Cross-Session Coordination Rules

**Document type:** Normative coordination contract  
**Owner:** Portfolio Control operating model  
**Audience:** Portfolio Control, all Product Sessions, specialist streams, human/model operators  
**Purpose:** Define exactly what moves between sessions, who may decide what, and how shared development continues without authority confusion.

---

## Core principle

```text
Product Sessions own product truth.
Portfolio Control owns central acceptance and cross-product consequence.
Canonical audit engine owns audit findings.
VIS owns presentation.
COMM owns its authorized LKW proof work.
```

No session may silently absorb another session's authority.

---

## 1. Topology

### Core operating topology — six sessions

| # | Session |
|---|---------|
| 1 | Portfolio Control Session |
| 2 | LKW Product Session |
| 3 | Contract Recovery Product Session |
| 4 | Supplier Disruption Product Session |
| 5 | Third-Party Risk Product Session |
| 6 | Deployment Guardian Product Session |

### Specialist streams outside the six

| Stream | Role |
|--------|------|
| **VIS-3A** | Public visual/documentation presentation |
| **COMM** | Authorized LKW proof development and hardening |

VIS-3A and COMM are **not** Product Sessions and **not** Portfolio Control.

### Counts — do not confuse them

| Count | Value | Meaning |
|-------|-------|---------|
| **Public product count** | **Five** | LKW plus four newly selected applications |
| **Operating session count** | **Six** | Five Product Sessions plus Portfolio Control |

Portfolio Control is an operating session but **not** a public product.

---

## 2. Authority matrix

Ownership verbs: **OWNS** · **PREPARES** · **VERIFIES** · **CONSUMES** · **PRESENTS** · **NO AUTHORITY**

| Concern | Product Session | Portfolio Control | Audit Engine | COMM | VIS-3A |
|---------|-----------------|-------------------|--------------|------|--------|
| Product definition | **OWNS** | VERIFIES (gate acceptance) | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY |
| Product architecture | **OWNS** (accepted G1 artifact) | VERIFIES (G1 acceptance) | NO AUTHORITY | NO AUTHORITY | CONSUMES (accepted facts only) |
| Implementation | **OWNS** | VERIFIES (material gates) | NO AUTHORITY | OWNS (authorized LKW proof work only) | NO AUTHORITY |
| Tests | **OWNS** (product scope) | VERIFIES (gate evidence) | CONSUMES (audit scope) | OWNS (proof tests) | NO AUTHORITY |
| Product proof | **PREPARES** | VERIFIES / accepts | NO AUTHORITY | PREPARES (LKW proof artifacts) | NO AUTHORITY |
| Gate preparation | **PREPARES** | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY |
| Gate acceptance | NO AUTHORITY | **OWNS** | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY |
| G4 classification | PREPARES (escalation) | **OWNS** (disposition) | CONSUMES (when audit required) | NO AUTHORITY | NO AUTHORITY |
| Platform impact | PREPARES (claim) | **OWNS** (classification + ledger) | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY |
| Audit finding | CONSUMES (links IDs) | CONSUMES (requests campaign) | **OWNS** | NO AUTHORITY | NO AUTHORITY |
| Remediation lifecycle | PREPARES (product/platform work) | VERIFIES (closure evidence) | **OWNS** (canonical status) | NO AUTHORITY | NO AUTHORITY |
| T0 preparation | **PREPARES** (four new products) | VERIFIES | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY |
| T0 acceptance | NO AUTHORITY | **OWNS** | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY |
| T1 preparation | **PREPARES** | NO AUTHORITY | CONSUMES (audit evidence) | NO AUTHORITY | NO AUTHORITY |
| T1 acceptance | NO AUTHORITY | **OWNS** | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY |
| Recommendation / priority | PREPARES (input) | **OWNS** | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY |
| Public claim eligibility | NO AUTHORITY | **OWNS** (verification) | CONSUMES (audit evidence) | NO AUTHORITY | CONSUMES (after acceptance) |
| Public visual narrative | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY | **OWNS** (presentation) |
| LKW proof execution | NO AUTHORITY | VERIFIES (accepted proof evidence) | NO AUTHORITY | **OWNS** (authorized roadmap) | NO AUTHORITY |
| README visual layout | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY | NO AUTHORITY | **OWNS** |

---

## 3. Core truth flow

```text
Product Session
  → product-owned implementation / architecture / proof evidence
  → Portfolio Control verification
  → accepted central fact / status / decision
  → downstream public eligibility
  → VIS presentation
```

Public documentation must **never** become upstream source of product truth.

---

## 4. Product Session → Portfolio Control handoff

### Material handoff required for

| Event | Trigger |
|-------|---------|
| G0 ready | Product baseline complete |
| G1 ready | Product architecture complete |
| G2 / T0 ready | Reuse baseline ready (four new products) |
| First implementation authorization boundary | Bootstrap rules satisfied |
| G3 ready | Initial product slice evidence |
| G4 escalation | Material shared-platform pressure |
| G5 ready | Product milestone evidence |
| G6 / T1 ready | Reuse audit + T1 package |
| Major G7 evidence | Portfolio decision input |
| G8 recommendation | Program completion recommendation |
| Pause / stop | Product session recommendation |
| Material public claim | Status or proof claim for public use |
| Cross-product conflict | Incompatible platform semantics |

### Required handoff payload

Every material handoff must include:

1. **product / session** — which Product Session submits
2. **event / gate** — what is being requested
3. **exact relevant SHA** — commit evidence applies to
4. **authoritative artifact(s)** — docs, tests, proof outputs
5. **claim / request** — what Product Session asks Portfolio Control to accept
6. **evidence** — repo-verifiable support
7. **known limitations** — explicit gaps or caveats
8. **platform pressure yes/no** — whether G4 may be required
9. **shared files / components touched or proposed**
10. **requested Portfolio Control action** — accept, classify, defer, etc.

This defines **semantic handoff content** only. No transport protocol or tool dependency is required.

Ordinary local implementation within product scope requires **no** Portfolio Control handoff.

---

## 5. Portfolio Control → Product Session response

### Possible responses

| Response | Meaning |
|----------|---------|
| **ACCEPTED** | Claim / gate / evidence accepted as stated |
| **ACCEPTED WITH EXPLICIT GAP** | Accepted with documented limitation |
| **REJECTED / NOT READY** | Evidence insufficient or claim unsupported |
| **REQUIRES G4** | Material platform pressure must be classified first |
| **REQUIRES CANONICAL AUDIT** | Audit campaign required before proceeding |
| **DEFERRED** | Decision postponed pending evidence or dependency |
| **PRODUCT_OWNED** | Remains product-local; no central action needed |
| **REUSE_EXISTING_PLATFORM** | Existing capability sufficient; no shared change |
| **CONFIGURE_EXISTING_PLATFORM** | Configuration only; no generalization |
| **GENUINE_PLATFORM_GAP** | Shared platform extension authorized |
| **REJECT_GENERALIZATION** | Product-specific logic must not enter shared core |
| **PAUSE / STOP / priority change** | Central recommendation or state change |

### Response must include

1. **reviewed SHA(s)**
2. **evidence checked**
3. **decision**
4. **consequence** — what this means for portfolio / product
5. **next allowed product action**
6. **central artifacts updated or not** — control card, status, ledger, decision log
7. **audit / G4 requirement** — if applicable

Product Session must **not** reinterpret a Portfolio Control decision into stronger permission than stated.

---

## 6. G4 handoff

Product Session must **STOP** before material shared-platform change.

```text
Product Session → G4 escalation → Portfolio Control
```

Portfolio Control independently checks:

- existing platform capability
- misuse vs genuine gap
- impact on other products
- LKW impact
- generality of proposed change
- product leakage into shared core
- audit need

Then returns G4 disposition.

### No Product Session self-approval of

- **EXTENDED_GENERALLY**
- **GENUINE_PLATFORM_GAP**
- product-specific shared-core behavior

---

## 7. Shared platform work after G4

If G4 authorizes shared-platform work, **implementation ownership must be explicit**.

### Possible execution owner

- dedicated platform session / task
- originating Product Session under explicit shared-platform scope
- another designated maintainer stream

### Required before work

- accepted G4 disposition
- exact shared responsibility
- allowed files / components
- expected affected products
- tests / evidence expectations
- adoption expectations
- rollback / compatibility concern where relevant

### After work

```text
implementation owner reports exact commit / evidence
  → Portfolio Control verifies
  → cross-product impact re-evaluated
  → PLATFORM_IMPACT_LEDGER updated only if accepted impact exists
```

G4 approval is **not** proof of successful generalization. Verification follows implementation.

---

## 8. Cross-product impact broadcast

When shared platform changes **materially**, Portfolio Control determines which Product Sessions need notification.

### Notification should state

- shared capability changed
- exact commit
- whether adoption is required
- whether behavior is expected unchanged
- whether revalidation is required
- known compatibility risk
- whether product roadmap must react

Avoid broadcasting every internal refactor. Only **material semantic / platform-contract changes** require cross-session signal.

---

## 9. T0 coordination

New Product Session (four newly selected products) prepares T0.

Portfolio Control verifies and finally accepts.

After freeze:

- Product Session may **not** silently edit T0.
- If deviation needed: Product Session submits **versioned deviation** → Portfolio Control independently reviews → only then affected implementation proceeds.

Audit engine does **not** own T0. T0 methodology: [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md).

LKW has no retrospective T0 — historical reference baseline differs.

---

## 10. G6 / T1 coordination

Required sequence:

```text
Product Session declares G6 readiness
  ↓
Portfolio Control verifies preregistration and scope
  ↓
canonical PLATFORM CONSUMER AUDIT @ exact SHA
  ↓
audit campaign owns findings / conformance
  ↓
Product Session prepares T1 using audit evidence
  ↓
Portfolio Control independently evaluates T1 / M1–M6
  ↓
Portfolio Control accepts / qualifies / rejects reuse result
```

No participant may collapse audit verdict and T1 result into one thing.

---

## 11. Canonical audit coordination

Portfolio Control may request audit per [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md).

### Audit engine owns

- campaign
- finding IDs
- finding lifecycle
- severity
- remediation status
- verification

Product Session and Portfolio docs may **LINK** relevant finding IDs only. **No duplicate finding register.**

Remediation owner may be product or platform workstream, but canonical status stays in the audit campaign (`docs/audit_results/`).

---

## 12. LKW ↔ COMM

COMM currently owns authorized LKW proof development and hardening.

### LKW Product Session owns

- LKW product state
- LKW implementation roadmap
- product architecture
- product implementation truth
- product-specific product evidence

### COMM owns

- proof work explicitly inside COMM roadmap
- proof execution / hardening artifacts
- evidence produced by that proof work

### COMM does NOT own

- LKW product status
- Portfolio gate acceptance
- commercial validation
- global platform classification
- Portfolio recommendation / priority
- public visual narrative

### Flow

```text
COMM proof evidence
  → LKW Product truth context where relevant
  → Portfolio Control verification / acceptance
  → eligible public fact
  → VIS-3A presentation
```

### COMM proof success does NOT automatically mean

- LKW commercially validated
- platform universally proven
- another product reuses Intergrax

---

## 13. LKW Product Session ↔ Portfolio Control

LKW is the **ACTIVE** existing reference product.

### Handoff events

- material product checkpoint
- future G4
- major proof result
- relevant canonical audit result
- public claim material change
- commercial / market evidence
- major blocker
- recommendation / priority question

No retroactive G0 / G1 / T0 / T1 for LKW.

---

## 14. VIS-3A / public documentation boundary

VIS-3A owns **HOW** accepted facts are shown:

- root README visual structure
- hero
- product cards
- visual platform / product relationship
- documentation navigation
- Product Presentation Document visual / template system
- public narrative composition

VIS-3A does **NOT** own **WHAT** is true.

### VIS-3A may not independently upgrade

| From | To |
|------|-----|
| SELECTED | ACTIVE |
| planned | implemented |
| proof | commercial validation |
| partial evidence | validated platform claim |
| concept | shipped product |
| audit finding | resolved |
| hypothesis | demonstrated reuse |

---

## 15. Public fact eligibility

For a material public claim, VIS-3A should be able to trace:

- product
- claim
- authoritative source
- accepted implementation / proof / audit evidence
- current limitations
- Portfolio Control acceptance status

### Pre-G0 new product public facts

May come only from:

- frozen selection record — [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md)
- explicit Pre-bootstrap state
- approved product hypothesis wording

Do not expose internal uncertain findings as public claims without contextual review.

---

## 16. VIS-3A request → Portfolio Control

When VIS needs a stronger material claim, VIS requests **fact validation**, not implementation.

### Request includes

- proposed factual claim
- target product / platform area
- source currently used
- intended public context

### Portfolio Control returns

| Response | Meaning |
|----------|---------|
| **ACCEPTED AS WRITTEN** | Claim supported as stated |
| **ACCEPTED WITH QUALIFIER** | Claim allowed with explicit limitation |
| **NOT SUPPORTED** | Claim exceeds current evidence |
| **STALE — REVERIFY** | Underlying evidence may have changed |
| **REQUIRES PRODUCT SOURCE** | Product Session artifact needed |
| **REQUIRES PROOF / AUDIT EVIDENCE** | Additional evidence required |

VIS controls wording and layout **after** factual boundary is established.

---

## 17. Product Session → VIS

Normally Product Session does **NOT** directly change public product truth.

```text
Product Session → authoritative product artifacts
  → Portfolio Control verifies / accepts material claims
  → VIS consumes accepted facts
```

For purely descriptive non-material wording, direct product-doc sourcing may be possible, but **never** for:

- implementation status
- proof strength
- platform reuse
- customer / commercial validation
- gate status

These require central accepted truth.

---

## 18. Portfolio Control ↔ VIS

### Portfolio Control does NOT

- design hero
- choose visual layout
- create graphical narrative
- own public style

### Portfolio Control DOES

- validate material status claims
- identify canonical evidence
- qualify limitations
- reject unsupported narrative claims

VIS is free to make presentation compelling **inside** those truth boundaries.

---

## 19. Product Session ↔ Product Session

Direct cross-product coordination should be **rare**.

### Product Sessions must NOT

- negotiate shared-platform ownership privately
- agree on shared abstractions without Portfolio Control
- modify each other's architecture
- coordinate hidden common infrastructure

### Direct exchange allowed for

- bounded factual clarification
- shared capability usage reference
- Portfolio-Control-requested comparison
- already accepted shared contract adoption

Material shared decisions route through Portfolio Control.

---

## 20. Parallel development / concurrency

All sessions share the `development` branch.

Every handoff involving code or docs must use **exact SHA**.

### Rules

- HEAD may move between messages
- preserve unrelated concurrent work
- no reset / rebase / stash / clean / amend / force push
- stage task-owned files only
- re-read shared file after concurrent modification
- distinguish reviewed commit from current HEAD
- do not attribute concurrent commit to wrong session

If a relevant shared change lands after evidence was produced, Portfolio Control decides whether revalidation is required.

---

## 21. Stale evidence

Evidence becomes potentially stale when:

- shared contract used by product changes
- product implementation changes materially
- audit scope invalidated
- proof dependency changes
- policy / governance / runtime semantics change
- relevant external assumption changes

Staleness does **not** automatically invalidate everything. Portfolio Control determines bounded revalidation scope.

Never silently keep old PASS / acceptance as current.

---

## 22. Blocker / conflict escalation

Escalate to Portfolio Control when:

- two products require incompatible platform semantics
- product blocked by missing shared capability
- product wants to bypass platform to move faster
- shared change risks another product
- ownership ambiguous
- Product Session and audit evidence disagree
- COMM proof claim conflicts with LKW current state
- VIS public claim conflicts with authoritative status

Portfolio Control may: decide directly · require G4 · require canonical audit · request product redesign · defer · pause work.

---

## 23. Decision log rule

Material coordination outcomes may require [DECISION_LOG.md](DECISION_LOG.md) entry **only** when they are real portfolio / program decisions.

### Log when

- choose one shared ownership model over incompatible alternatives
- pause product
- change portfolio recommendation materially
- adopt major cross-product operating rule
- accept / reject major shared-platform direction

### Do NOT log

- every handoff
- implementation chatter
- routine gate completion
- raw audit findings

---

## 24. Platform impact ledger rule

[PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md) receives accepted platform-impact evidence **after** Portfolio Control verification.

### Examples

- Product A reused shared capability unchanged
- Product B forced general extension
- cross-product conflict invalidated platform assumption
- proposed platform generalization rejected

### Do NOT copy into ledger

- speculative G4 hypothesis
- raw audit findings
- Product Session self-classification

---

## 25. Control card / portfolio status synchronization

Portfolio Control updates central control card and [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md) after accepted material events.

Product Sessions must not treat stale central status as permission to ignore newer Portfolio Control decision.

If control card or status lags, Portfolio Control repairs it after verifying evidence.

Product Sessions must not directly rewrite central truth unless explicitly authorized by Portfolio Control workflow.

---

## 26. Pause / stop handoff

Product Session may **recommend** PAUSE / STOP.

Portfolio Control evaluates:

- product evidence
- market evidence
- opportunity cost
- next milestone cost
- platform learning
- risks
- dependencies

Portfolio Control owns final central recommendation / state change.

Stopped product evidence remains historical. Do not erase valid platform learning.

---

## 27. Handoff minimalism

Do not create synchronous bureaucracy.

| Scope | Handoff |
|-------|---------|
| Ordinary local implementation | None |
| Material event | Small structured handoff |

Goal: Product Sessions stay independently productive while shared-platform and central truth remain controlled.

---

## 28. No shadow authority

Explicitly prohibited:

| Shadow authority | Why prohibited |
|------------------|----------------|
| VIS as source of product truth | Presentation ≠ ownership |
| COMM as source of portfolio status | Proof ≠ gate acceptance |
| Product Session self-accepting G4 | Cross-product consequence |
| Product Session self-accepting T1 | Central reuse verdict |
| Portfolio Control rewriting product architecture | Product truth boundary |
| Product Session rewriting canonical audit finding status | Audit engine ownership |
| Audit campaign deciding portfolio priority | Scope boundary |
| README wording deciding implementation status | Public ≠ upstream truth |
| Direct Product Session pact creating shared platform ownership | G4 bypass |

---

## 29. Six-session launch model

**MP-22** will later create exactly **six** launch prompts:

1. Portfolio Control
2. LKW
3. Contract Recovery
4. Supplier Disruption
5. Third-Party Risk
6. Deployment Guardian

VIS-3A and COMM do **not** receive prompts from this launch pack unless a future separate task explicitly creates them. They remain external specialist streams.

No sessions are launched by MP-20.

---

## 30. Coordination examples

### Example A — Contract Recovery discovers missing shared approval mechanism

```text
Contract Recovery Product Session identifies shared-platform pressure
  → G4 escalation to Portfolio Control
  → Portfolio Control inspects canonical platform
  → possibly requires canonical audit
  → G4 disposition (REUSE / CONFIGURE / GENUINE_GAP / REJECT_GENERALIZATION)
  → explicit implementation owner assigned
  → verify commit / evidence
  → PLATFORM_IMPACT_LEDGER if accepted impact
  → cross-product notification if material
```

### Example B — COMM strengthens LKW proof

```text
COMM completes authorized proof hardening
  → proof evidence artifacts at exact SHA
  → Portfolio Control verification
  → accepted / qualified public fact recorded
  → VIS-3A presents within factual boundary
```

COMM proof success does not upgrade LKW commercial validation or platform-wide reuse claims.

### Example C — VIS wants "five production-ready products"

```text
VIS-3A requests fact validation
  → Portfolio Control reviews authoritative status
  → NOT SUPPORTED (four products remain SELECTED / Pre-bootstrap)
  → VIS must use truthful current statuses
```

### Example D — Deployment Guardian wants direct GitHub-specific logic in shared core

```text
Deployment Guardian Product Session → G4 escalation
  → Portfolio Control analyzes provider-boundary vs generalization
  → likely REJECT_GENERALIZATION or CONFIGURE_EXISTING_PLATFORM
  → product keeps provider logic in product tier
```

### Example E — G6 / T1 reuse demonstration

```text
Product Session declares G6 ready
  → Portfolio Control verifies scope
  → canonical PLATFORM CONSUMER AUDIT @ SHA
  → audit findings owned by campaign
  → Product Session prepares T1 from audit evidence
  → Portfolio Control independently accepts / qualifies / rejects T1
```

Audit verdict and T1 acceptance remain separate artifacts.

---

## 31. Source-of-truth table

| Concern | Canonical owner |
|---------|-----------------|
| Product implementation truth | Product Session / exact repo SHA |
| Product architecture | Product Session accepted G1 artifact |
| Product gate acceptance | Portfolio Control |
| Portfolio status | Portfolio Control — [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md) |
| Platform impact classification | Portfolio Control — [PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md) |
| Audit findings | `docs/audit_results/` campaign |
| Audit remediation state | `docs/audit_results/` campaign |
| T0 / T1 methodology | [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) |
| LKW proof implementation | COMM where explicitly authorized |
| Public visual narrative | VIS-3A |
| Public material claim eligibility | Portfolio Control verification + authoritative evidence |

---

## Related documents

| Document | Role |
|----------|------|
| [PORTFOLIO_CONTROL_OPERATING_MANUAL.md](PORTFOLIO_CONTROL_OPERATING_MANUAL.md) | Central control behavior |
| [PRODUCT_SESSION_OPERATING_MANUAL.md](PRODUCT_SESSION_OPERATING_MANUAL.md) | Common product behavior |
| [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md) | Audit engine integration |
| [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) | Program constitution |
| [session-briefs/](session-briefs/) | Product-specific mission |
| [README.md](README.md) | Maintainer workspace index |
