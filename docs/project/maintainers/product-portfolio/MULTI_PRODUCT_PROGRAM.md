# Multi-Product Program Governance Contract

**Document type:** Maintainer-level normative program contract  
**Last defined:** 2026-08-19 (MP-11)

---

## 1. Status and document role

This document is the **operating constitution** for coordinated development and control of all products admitted to the Intergrax multi-product program.

It applies to:

| Role | Product |
|------|---------|
| Existing reference product | **Local Knowledge Workspace (LKW)** |
| Newly selected applications | **Contract-to-Invoice Leakage / Recovery Operator** |
| | **Supplier Disruption Response Operator** |
| | **Third-Party Risk Decision Operator** |
| | **Deployment / Change Guardian** |

**Not active program products at MP-11 closeout:**

| Position | Product |
|----------|---------|
| Challenger | **Autonomous Agent Governance Operator** - not an active product unless formally promoted later |
| Future / wildcard | **Prior Authorization Operator** - not part of the first portfolio |

**LKW inclusion rule:** LKW is **not** retroactively treated as selected by MP-1→MP-8. It joins the program because Portfolio Control must evaluate the effects of **all active Intergrax applications** together. See [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) §9.

**Authoritative companions (do not duplicate here):**

| Topic | Document |
|-------|----------|
| Selection history | [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) |
| Per-product reuse evidence methodology | [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) |
| Product-first development rule | [PRODUCT_FIRST_MVP.md](../plans/PRODUCT_FIRST_MVP.md) |
| LKW architecture | [LKW ARCHITECTURE.md](../../../../applications/local_workspace_application/docs/ARCHITECTURE.md) |
| LKW implementation roadmap | [LKW IMPLEMENTATION_PLAN.md](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) |
| Workspace index | [README.md](README.md) |

---

## 2. Program objective

### Core principle

```text
REAL PRODUCT NEED
    ↓
PRODUCT ARCHITECTURE / IMPLEMENTATION PRESSURE
    ↓
PLATFORM OBSERVATION
    ↓
EVIDENCE-BASED PLATFORM EVOLUTION
```

**Never:**

```text
PLATFORM CAPABILITY
    ↓
INVENT OR DISTORT PRODUCT TO EXERCISE IT
```

### Explicit rules

- Products exist to solve **real problems first**.
- Platform proof is **observed second**, from genuine product pressure.
- **Commercial failure** may still yield useful platform evidence if the product hypothesis and experiment were genuine.
- **Platform learning alone** is never sufficient reason to keep a weak product alive indefinitely.
- No product may be distorted to maximize reuse score or platform-exercise value.

---

## 3. Session model

Two mandatory session classes operate the program.

### Product Session

**One independent session per active product**, including LKW.

| Owns | Does NOT own |
|------|--------------|
| Product semantics | Final classification of material shared-platform changes |
| User problem and product brief | Cross-product portfolio status |
| Product architecture | Portfolio-wide priority decisions |
| Product roadmap | Aggregate platform-proof claims |
| Implementation slices | |
| Application code | |
| Product tests | |
| Product docs | |
| Product-level validation | |
| Detection and reporting of platform gaps | |
| T0/T1 product reuse evidence per [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) | |

### Portfolio Control Session

**One central independent session.**

| Owns | Does NOT normally own |
|------|------------------------|
| Portfolio-level oversight | Detailed product architecture design |
| Verification of material product milestones | Product feature implementation |
| Independent audits of product/session claims | Forcing identical implementation shapes across products |
| Cross-product architectural review | |
| Platform-impact classification | |
| Central program state | |
| Portfolio prioritization recommendations | |
| Continue / accelerate / pause / stop recommendations | |
| Aggregate platform-proof interpretation | |
| Detection of product-specific leakage into shared core | |
| Detection of private reimplementation of platform responsibilities | |
| Detection of false or generalized platform capabilities | |

**Portfolio Control asks:** *Is this product architecture valid relative to shared boundaries and other products?*

**Portfolio Control must not become:** *Central architect that forces all products into identical implementation shapes.*

---

## 4. Ownership boundaries

| Dimension | Owner |
|-----------|-------|
| Product semantics | Product Session |
| Product architecture | Product Session |
| Product roadmap | Product Session |
| Product implementation | Product Session |
| Product tests | Product Session |
| Product docs | Product Session |
| Product-level validation | Product Session |
| Product T0 preparation | Product Session |
| T0 independent acceptance | Portfolio Control |
| Material shared-platform change **proposal** | Product Session may propose |
| Final shared-platform **classification** | Portfolio Control |
| Shared Intergrax implementation task | Appropriate platform session after accepted decision |
| Portfolio priority | Portfolio Control recommendation / program decision |
| Aggregate proof interpretation | Portfolio Control |
| Portfolio status docs | Portfolio Control |

**Hard rule:** No Product Session may silently absorb shared-platform responsibility merely to complete its milestone.

---

## 5. Product independence rules

Each product:

- has its **own session**;
- has its **own architecture**;
- has its **own roadmap**;
- may progress at a **different speed**;
- may discover **different requirements**;
- may use shared platform capabilities differently through **configuration**;
- may remain **paused** while another advances;
- may be **stopped**;
- must **not** be distorted to maximize reuse score.

Products do **not** need synchronized milestone numbers.

**Example pacing (illustrative, not a schedule):**

| Product | Possible stage |
|---------|----------------|
| LKW | Advanced proof stage |
| Supplier Disruption | Architecture |
| Third-Party Risk | Wedge refinement |

---

## 6. Common review gates

G0–G8 are **logical checkpoints**, not synchronized project phases. Products enter gates when their actual state warrants review.

### Gate summary

| Gate | What is verified | Prepared by | Independently reviewed by | Decision produced |
|------|------------------|-------------|---------------------------|-------------------|
| **G0 - Product / Reference Baseline** | Product identity, problem, scope, and baseline artifacts exist and are internally consistent | Product Session | Portfolio Control | Baseline accepted or correction required |
| **G1 - Architecture** | Product architecture is product-first, boundary-respecting, and feasible relative to shared platform | Product Session | Portfolio Control | Architecture accepted, revised, or blocked |
| **G2 - T0 Reuse Baseline** | Frozen T0 record per [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md): hypothesis, responsibility matrix, Critical Reuse Set, starting commit SHA | Product Session | Portfolio Control | T0 accepted or returned for correction |
| **G3 - First Real Vertical Slice** | First end-to-end slice delivers observable user value; not scaffold-only | Product Session | Portfolio Control | Slice accepted or correction required |
| **G4 - Material Platform Pressure** | Proposed shared-platform change is classified before implementation proceeds | Product Session (escalation) | Portfolio Control | Classification accepted; implementation may proceed, stay product-owned, or await platform work |
| **G5 - MVP / Major Proof** | Major product proof milestone against roadmap and product brief | Product Session | Portfolio Control | Proof accepted, partial, or failed relative to product goals |
| **G6 - T1 Reuse Audit** | Final responsibility classifications, M1–M6, PASS/PARTIAL/FAIL per reuse contract | Product Session (evidence) | Portfolio Control | Reuse outcome recorded; no aggregate claim without Portfolio Control acceptance |
| **G7 - Market Validation** | External or pilot evidence for commercial hypothesis | Product Session | Portfolio Control | Validation recorded; portfolio action informed |
| **G8 - Continue / Accelerate / Pause / Stop** | Whether the product should continue in the program | Portfolio Control (recommendation) | Program decision | Portfolio action recorded |

### LKW handling

LKW does **not** replay already completed stages.

Instead:

1. **Baseline-ingestion review** of current architecture, roadmap, proofs, platform usage, and open work (G0-equivalent for reference product).
2. From that point onward, LKW participates in relevant **future gates** and **platform-impact reviews** at its actual pace.

LKW current role (summary only - authoritative detail in LKW docs):

- Tier-3 application (`local_workspace_application`) with Tier-2 agents (`local_indexer`, `local_search`, `local_synthesizer`).
- Knowledge-centric, primarily Ask/read workflow; hybrid indexed + live knowledge access.
- First business product environment after harness platform maturity; exercises RAG ingest/retrieve, governed live access, multi-agent orchestration, memory, policy, trace, and Tier-3 composition.
- Dual role: real product **and** harness validation reference - not a template that other products must clone.

---

## 7. G4 - Material Platform Pressure Gate

G4 is the **mandatory escalation gate** before implementing any change that may affect shared Intergrax platform boundaries.

### G4 triggers

A Product Session must escalate **before implementation** if a needed change may:

- modify a shared Intergrax contract;
- add or alter a shared abstraction;
- change runtime behavior;
- change shared lifecycle semantics;
- change identity / governance / evidence / recovery semantics;
- introduce product-specific branching into shared core;
- duplicate a capability the platform appears to own;
- require another active product to change;
- move responsibility between product and platform layers.

**Does not require G4:** minor local product configuration within existing platform contracts.

### G4 evaluation questions

Portfolio Control evaluates:

1. Is the need genuinely required by the product?
2. Is the existing platform capability being used correctly?
3. Is the need product-owned?
4. Is a general platform extension justified?
5. Would the extension remain useful without naming the originating product?
6. What effect would it have on LKW and all other active products?
7. Does it introduce coupling or backward pressure?
8. How is it classified under [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md)?

### Required classifications (exact)

| Classification | Meaning |
|----------------|---------|
| `REUSED_UNCHANGED` | Existing shared mechanism consumed without platform modification |
| `REUSED_CONFIGURED` | Existing mechanism reused through intended configuration/policy/adapter contract without changing core platform semantics |
| `EXTENDED_GENERALLY` | Real product pressure exposed a missing reusable capability; platform extended through a general contract |
| `PRODUCT_OWNED` | Behavior correctly belongs in the product: domain workflow, UX, business semantics, product-specific policy meaning |
| `PLATFORM_LEAK` | Product-specific branching, private infrastructure duplication, bypass of shared contracts, or product-specific behavior leaking into platform core |

**Hard rule:** A product must **not** continue through a material shared-platform change until the responsibility/classification decision is **accepted**.

---

## 8. Platform-change decision rules

### Valid `EXTENDED_GENERALLY` decision must

- solve the originating need;
- be expressed in **product-neutral** terms;
- avoid branching on product identity;
- preserve existing abstractions where possible;
- identify impact on current consumers;
- be independently reviewable;
- **not** be justified solely by anticipated future reuse.

### `PRODUCT_OWNED` means

- keep capability and semantics inside the application;
- do **not** promote to platform merely because another product might someday need something similar.

### `PLATFORM_LEAK` means

- defect, not acceptable reuse;
- must **not** be normalized as successful platform adoption.

---

## 9. Cross-product impact review

Any **accepted** material platform change must review impact on:

- LKW
- Contract-to-Invoice Leakage / Recovery Operator
- Supplier Disruption Response Operator
- Third-Party Risk Decision Operator
- Deployment / Change Guardian
- any future active program product

**Do not** require code changes in inactive or not-yet-built applications.

Review may conclude:

| Outcome | Meaning |
|---------|---------|
| unaffected | No product impact |
| compatible by configuration | Existing products can adopt without platform rework |
| requires later adoption | Valid generalization; adoption deferred per product pace |
| reveals conflict | Generalization may be invalid or needs revision |
| invalidates the proposed generalization | Reject or redesign extension |

**Evidence rule:** The central program must preserve evidence of where a capability originated and where it was later reused. The ledger artifact will be created in MP-12; this contract defines the rule only.

---

## 10. Asynchronous product pacing

- Applications progress at **different speeds**.
- No artificial synchronization of milestones across products.
- Portfolio review uses **actual product state**, not assumed uniform progress.
- One product may be **paused** while another **accelerates**.
- A fast product must **not** force speculative shared-platform work for slower products.
- A slow product must **not** block unrelated valid progress elsewhere.

---

## 11. Portfolio prioritization

Portfolio Control may recommend:

| Recommendation | Meaning |
|----------------|---------|
| **ACCELERATE** | Increase focus, resources, or review cadence |
| **CONTINUE** | Maintain current pace and scope |
| **REDUCE / DEFER** | Lower priority; defer next milestone |
| **PAUSE** | Stop active work; preserve state and evidence |
| **STOP** | End the product experiment; preserve evidence |

### Decision criteria (non-exhaustive)

- strength of user / market evidence
- commercial opportunity
- product progress
- technical feasibility
- cost / risk of next milestone
- competitive changes
- pilot accessibility
- cross-product dependencies
- platform learning value

**Explicit rule:** Platform-learning value is **only one criterion**. A weak product must **not** be kept alive merely because it exercises useful platform mechanisms.

---

## 12. Stop / pause integrity

Stopping a product is a **legitimate successful outcome** of the experiment.

If stopped:

- preserve product evidence;
- preserve architecture findings;
- preserve platform-impact findings;
- preserve why it was stopped;
- do **not** rewrite original selection history;
- do **not** present it publicly as a successful product.

Likewise: a commercially weak product may still produce **valid platform evidence** if the experiment was genuine.

---

## 13. Product-session workflow

Standard workflow consistent with Intergrax maintainer practice:

```text
roadmap
  → explain current task
  → scope confirmation
  → precise implementation instruction
  → implementation
  → independent exact-commit audit
  → accept / correct
  → next task
```

For architecture:

- design **product-first**;
- reuse existing platform capabilities where valid;
- **report** missing capability instead of silently inventing shared-core code.

---

## 14. Portfolio Control workflow

For each material review:

1. Read central program state.
2. Read product control / baseline context.
3. Inspect authoritative product architecture / roadmap.
4. Inspect exact implementation evidence when applicable.
5. Check platform boundaries.
6. Check impact on every active product.
7. Classify finding.
8. Record accepted decision in central program artifacts.
9. Recommend next product / portfolio action.

**Audit rule:** Do **not** trust a Product Session's completion summary without repo verification when implementation evidence exists.

---

## 15. No hidden shared-core branching

**Hard prohibition:** Do not add logic such as:

```text
if product == LKW
if product == supply_disruption
product-specific runtime forks
```

inside shared Intergrax core merely to satisfy one application.

If product-specific behavior is genuinely required, it belongs in product-owned configuration, composition, or domain logic - unless a truly general platform abstraction is accepted through G4.

---

## 16. Relationship to PRODUCT_REUSE_PROOF.md

[PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) is the **canonical methodology** for:

- T0 pre-registration baseline
- Critical Reuse Set
- M1–M6 metrics
- responsibility classifications
- PASS / PARTIAL / FAIL outcomes
- anti-gaming rules

| Document | Governs |
|----------|---------|
| **MULTI_PRODUCT_PROGRAM.md** (this document) | Coordinated multi-product operation |
| **PRODUCT_REUSE_PROOF.md** | Per-product reuse evidence |

Do not duplicate the reuse contract here.

---

## 17. Future central artifacts

Later MP tasks will create the following. **Do not create them in MP-11.**

| Artifact | Intended ownership |
|----------|-------------------|
| `PORTFOLIO_STATUS.md` | Portfolio Control - live portfolio state |
| `PLATFORM_IMPACT_LEDGER.md` | Portfolio Control - origin and reuse of platform changes |
| `DECISION_LOG.md` | Portfolio Control - accepted program decisions |
| `products/*` | Per-product control cards |
| Portfolio Control checkpoint / gate evidence | `PORTFOLIO_STATUS.md`, control cards, operating manuals, and canonical audits in `docs/audit_results/` - **not** a separate `reviews/*` workspace or parallel review engine |

---

## 18. Initial program state (MP-11 closeout)

| Item | State |
|------|-------|
| MP-1→MP-8 selection | Complete |
| MP-10 workspace / selection record | Complete |
| Governance contract (this document) | **Defined** |
| LKW baseline ingestion | **Not yet performed** |
| New application scaffolding | **Not started** |
| New-product architecture | **Not started** |
| Cross-product platform-proof claim | **None** |

---

## 19. Next step

**MP-12** - create:

- `PORTFOLIO_STATUS.md`
- `PLATFORM_IMPACT_LEDGER.md`
- `DECISION_LOG.md`

and define their exact operational schemas and ownership.
