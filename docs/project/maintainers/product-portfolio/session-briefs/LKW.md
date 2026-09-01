# Local Knowledge Workspace - Product Session Brief

**Document type:** Durable product-specific session mission artifact  
**Owner:** LKW Product Session (future launch)  
**Audience:** Future session operator / MP-22 Session Launch Pack assembler  
**Status:** Reference product - **ACTIVE** existing development

> **This is NOT the final session launch prompt.**  
> It is a durable product-specific mission and context artifact consumed later by **MP-22 Session Launch Pack**.  
> Do not treat this file as a conversational bootstrap prompt. Common operating behavior lives in [PRODUCT_SESSION_OPERATING_MANUAL.md](../PRODUCT_SESSION_OPERATING_MANUAL.md).

---

## 1. Session identity

| Field | Value |
|-------|-------|
| Product Session | LKW Product Session |
| Product | Local Knowledge Workspace (LKW) |
| Program role | Existing reference product |
| Program State | **ACTIVE** |
| Product Stage | Advanced existing product |
| Baseline type | **REFERENCE BASELINE** (not retroactive T0) |
| Control card | [products/LKW.md](../products/LKW.md) |

LKW predates the multi-product selection pipeline. It is the strongest current implemented application and proof surface in the program. It is **not** being retrofitted into G0/G1/T0/T1 bootstrap sequence.

---

## 2. Mission

Continue developing LKW as a **real knowledge-workspace product** - the program's existing reference product and strongest implemented application/proof surface.

The session must prove that LKW solves a coherent knowledge-workspace problem for users, not merely that Intergrax platform mechanics work. Platform proof may emerge as a consequence; it is not the mission.

**Product-first rule:**

```text
The product is not being built to demonstrate Intergrax.
Intergrax reuse is observed as a consequence of building the product.
```

---

## 3. Why this product exists independently of Intergrax

Organizations accumulate knowledge across indexed repositories and live systems. Users need governed answers with traceable provenance - not generic chat over documents.

LKW exists because knowledge workers need a workspace that:

- manages indexed and live sources as durable product capabilities;
- governs Hybrid Ask with answer truth, provenance, and admissibility;
- binds conversation context to workspace semantics;
- delivers inspectable, recoverable knowledge operations.

This problem stands on its own. A buyer or user would adopt LKW for knowledge-work outcomes, not because Intergrax exists.

---

## 4. Current authoritative starting state

Derive execution status **only** from the authoritative [LKW IMPLEMENTATION_PLAN.md](../../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md). Do not hard-code stale historical tasks if the plan has moved.

**Authoritative snapshot (from implementation plan):**

| Item | Status |
|------|--------|
| Product level | Backend Product Alpha - LKW MVP / Hybrid Knowledge Workspace |
| Current direct task | `LKW-PLUGIN-CAPABILITY-CONFIGURATION-1` - **READY_FOR_REVIEW** |
| Next direct task | `LKW-INDEXED-SOURCE-LIFECYCLE-1` - **PLANNED** |
| Major accepted blocks | Hybrid Ask, Knowledge Access, Conversational Frontend, Conversation Context (see plan for full list) |
| Reference baseline commit | `821eb7f6b2096de142822a29abc4546ee387a158` |
| G0 / G1 / T0 | **Not applicable** - no retroactive bootstrap |
| Implementation | **In progress** - real existing codebase |
| Cross-product reuse evidence | **NONE** (LKW is reference, not reuse proof for other products) |

**READY_FOR_REVIEW is not ACCEPTED.** Review-pending work is not closed product evidence.

---

## 5. Product hypothesis / current product truth

LKW is a **knowledge-centric hybrid workspace**: indexed evidence plus governed live access, unified under Hybrid Ask with provenance and policy.

Current product truth (accepted or implemented areas - summary; plan is authoritative):

- managed / indexed knowledge functionality;
- Workspace Ask and provider-neutral Hybrid Ask V2;
- indexed + live evidence types with unified provenance path;
- accepted Knowledge Access foundations and Conversation Context;
- conversational frontend for supported scope.

Not every architecture target is implemented. LKW remains pre–1.0 with planned lifecycle, hardening, and proof gates ahead.

---

## 6. Buyer / user and economic or operational job

| Dimension | Value |
|-----------|-------|
| Primary users | Knowledge workers, operators, workspace administrators |
| Economic / operational job | Reduce time-to-trusted-answer across indexed and live organizational knowledge; govern what may be asked, from where, and with what provenance |
| Success horizon | Ongoing workspace operations - not a one-shot crisis or financial recovery event |
| Value unit | Trustworthy, inspectable answers and manageable knowledge sources |

Commercial and customer validation are **separate** from current implementation and proof trajectory.

---

## 7. Primary workflow

```text
Configure workspace knowledge sources (indexed + live bindings)
  → user asks in workspace or connected frontend
  → Hybrid Ask resolves policy, evidence plan, and sources
  → governed answer with provenance / admissibility
  → inspect, recover, or reconfigure sources as product operations
```

Parallel tracks (Slack, vendor plugins, future integrations) extend reach but must not displace core workspace product coherence.

---

## 8. What makes this product different from LKW / other products

N/A for self-comparison. **Differentiation from other program products:**

| Product | LKW contrast |
|---------|--------------|
| Contract Recovery | Financial discrepancy → recovery money; LKW is knowledge truth, not spend reconciliation |
| Supplier Disruption | Crisis tempo mitigation of supply exposure; LKW is durable knowledge operations |
| Third-Party Risk | Vendor-risk decision defensibility; LKW is workspace Q&A, not vendor approval |
| Deployment Guardian | Cross-system release GO/NO-GO; LKW is knowledge access, not deployment authorization |

LKW is the only product with substantial existing implementation and reference baseline.

---

## 9. Product-specific wedge / kill questions

- Does LKW solve a **coherent knowledge-workspace problem**, not only prove platform mechanics?
- Are indexed and live sources managed as **real product capabilities**, not demo wiring?
- Does governed answer quality create **user value beyond generic RAG**?
- Are new mechanisms **product-specific**, or genuinely platform-owned and reusable?
- Is proof evidence **reproducible and understandable** publicly (COMM may assist; Portfolio Control accepts)?
- Is LKW **accumulating platform responsibilities privately** that should surface through G4?

---

## 10. Major failure modes / category traps

- **Generic RAG/chat wrapper** - answers without governance, provenance, or workspace semantics.
- **Architecture showcase** - impressive platform wiring without product usability.
- **LKW semantics as platform canon** - treating workspace-specific meaning as shared Intergrax abstractions.
- **COMM proof strength vs commercial validation** - strong technical proof mistaken for market proof.
- **Platform demo drift** - reducing LKW to "the Intergrax demo app."
- **Review-pending work treated as done** - READY_FOR_REVIEW counted as accepted product evidence.

---

## 11. Platform posture

LKW **consumes** shared Intergrax capabilities where they genuinely help. It does **not** self-approve:

- `EXTENDED_GENERALLY`;
- `GENUINE_PLATFORM_GAP`;
- shared core behavior that is actually LKW-owned semantics.

**G4 boundary:** When material shared-platform change is needed, **STOP** → escalate to Portfolio Control via **G4**. Known candidate: generic durable/indexed eligibility descriptor (see control card and implementation plan § B).

Before shaping other products around LKW patterns: remember LKW consumption does not prove reuse for finance, supply-chain, risk, or deployment workflows.

COMM may produce strong proof artifacts; COMM does **not** own Portfolio Control authority or central product state.

---

## 12. Current gate / first allowed action

**Continue the authoritative LKW roadmap** - not G0 restart.

First allowed action: advance `LKW-PLUGIN-CAPABILITY-CONFIGURATION-1` through review/acceptance, then proceed to `LKW-INDEXED-SOURCE-LIFECYCLE-1` per [IMPLEMENTATION_PLAN.md](../../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md).

Future material shared-platform pressure → **G4**.

---

## 13. Evidence the session must eventually produce

Future targets (not claimed today):

- **Strong product proof** - real workspace knowledge operations end-to-end.
- **Consumer/platform conformance evidence** where useful for Intergrax (bounded platform proof, certification matrix scope).
- **Real usage / market validation** - separately from technical proof; not invented here.

Example proof trajectory already indexed: [LKW_PLATFORM_PROOF.md](../../../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md), [GOVERNED_HYBRID_KNOWLEDGE_PROOF.md](../../../../../applications/local_workspace_application/docs/proof/GOVERNED_HYBRID_KNOWLEDGE_PROOF.md).

---

## 14. What the session must NOT claim yet

- LKW 1.0 completion or universal production readiness.
- Customer or commercial validation.
- Cross-product reuse by another independent product.
- That every consumed platform capability originated from LKW.
- ACCEPTED status for READY_FOR_REVIEW tasks.
- Retroactive T0/T1 reuse scoring.
- Public product presentation layout (VIS-3A owns presentation of approved facts).

---

## 15. Sources of truth

| Topic | Document |
|-------|----------|
| Common session behavior | [PRODUCT_SESSION_OPERATING_MANUAL.md](../PRODUCT_SESSION_OPERATING_MANUAL.md) |
| Portfolio control | [PORTFOLIO_CONTROL_OPERATING_MANUAL.md](../PORTFOLIO_CONTROL_OPERATING_MANUAL.md) |
| Live portfolio state | [PORTFOLIO_STATUS.md](../PORTFOLIO_STATUS.md) |
| Control card | [products/LKW.md](../products/LKW.md) |
| Execution / current task | [IMPLEMENTATION_PLAN.md](../../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) |
| Architecture | [ARCHITECTURE.md](../../../../../applications/local_workspace_application/docs/ARCHITECTURE.md) |
| Hybrid Ask | [HYBRID_ASK_ARCHITECTURE.md](../../../../../applications/local_workspace_application/docs/HYBRID_ASK_ARCHITECTURE.md) |

---

## 16. Handoff expectations

Contact Portfolio Control when:

- a **material product checkpoint** completes (accepted block with portfolio significance);
- **G4 pressure** arises from material shared-platform need;
- **proof / audit acceptance** requires central recording;
- **major commercial or market evidence** emerges (separate from COMM proof).

Do not write detailed transport protocol here - detailed cross-session handoffs are governed by [CROSS_SESSION_COORDINATION.md](../CROSS_SESSION_COORDINATION.md).

VIS-3A presents approved public facts; Portfolio Control and Product Session remain sources for product state and accepted program truth.
