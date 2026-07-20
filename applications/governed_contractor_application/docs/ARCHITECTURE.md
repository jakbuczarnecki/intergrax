# Governed Contractor Application — architecture

**Status:** GEC-0…GEC-6 (2026-07-20) — product-profile scaffold + Tier-2 mapping + Governed Continuation + side-effect policy + descriptive proof profile; HITL UX/product policy packs/ProofReceipt persistence/providers deferred  
**Platform reference:** [`docs/platform/governed_external_execution.md`](../../../docs/platform/governed_external_execution.md) — ownership · lifecycle · invariants  
**Vertical:** Governed External Contractor (GEC)  
**Capability target:** governed external contractor agents (generic; not a one-off partner integration)  
**Implementation tracker:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)  
**Application ADRs:** [`adr/README.md`](adr/README.md)  
**Partner handoff (planned):** [`PARTNER_HANDOFF.md`](PARTNER_HANDOFF.md) · [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md)  
**Tier-2 adapter:** [`agents/external_contractor_adapter/docs/ARCHITECTURE.md`](../../../agents/external_contractor_adapter/docs/ARCHITECTURE.md)

**Collaboration boundary:** Intergrax is **source-available** for evaluation and technical partner discovery. It is **not** open source. Production, commercial, and redistribution use require explicit permission — see repository [`COLLABORATION.md`](../../../COLLABORATION.md) and [`LICENSE`](../../../LICENSE). This vertical is a **proof path**, not a production-readiness or certification claim.

---

## 1. Problem statement

Organizations increasingly buy specialized **external contractor agents** (for example code review, research, or domain analysis) that already own Agent Cards, quote-first commercial flows, task status timelines, tool evidence, and deliverables.

What those products typically lack — and what Intergrax must supply without reimplementing the contractor — is a **governed application shell**:

- public task API and tenant configuration,
- Nexus orchestration and runtime policy,
- quote presentation and **HITL acceptance** before side effects continue,
- workspace and deliverable boundaries,
- correlation, trace, evidence, and receipt exposure for auditors and clients.

**Problem:** demonstrate that Intergrax can wrap an existing external contractor agent with Tier-3 governance while the external agent remains responsible for domain execution.

**Non-problem:** building a competing local contractor (for example a local code-review agent) inside Intergrax.

---

## 2. Design-partner context

GEC may be validated with one or more design partners that already ship A2A-style contractor agents. Partner identities, base URLs, API keys, and product-specific field maps **must not** be hardcoded into Intergrax core (`intergrax/`).

| Layer | Partner-specific content allowed? |
|-------|-----------------------------------|
| `intergrax/` contracts / runtime / integrations | **No** — provider-neutral only |
| Tier-2 `external_contractor_adapter` | Neutral mapping to Intergrax contracts; partner config via host/env |
| Tier-3 `governed_contractor_application` | Tenant/env wiring, policy bundles, proof docs, handoff samples |
| `docs/` partner handoff / public adoption | Named partner materials as **external mapping**, not core identity |

The product capability remains:

```text
governed external contractor agents
```

A single design partner is a **validation instance**, not the architecture axis.

---

## 3. Four-tier responsibility mapping

```text
Client
  → Governed Contractor Application (Tier-3)
  → Nexus Task
  → External Contractor Adapter (Tier-2)
  → External A2A Contractor Agent
```

| Tier | Owner package | Responsibility |
|------|---------------|----------------|
| **0** | `intergrax/` | Reusable contractor/quote contracts, integration surfaces, HITL, policy, Nexus, ProofReceipt, trace |
| **1** | `intergrax/runtime/` | Orchestration, task lifecycle, policy enforcement, HITL gates, evidence spine |
| **2** | `agents/external_contractor_adapter/` | Domain adapter: Agent Card discovery, external task/quote/status/deliverable mapping |
| **3** | `applications/governed_contractor_application/` | Public API, tenant/env, policy bundles, quote UI/API presentation, HITL surfaces, workspace, receipts |

Hard boundaries (never violate):

```text
intergrax/ MUST NOT import from agents/ or applications/
agents/ MUST NOT import from applications/
applications/ MAY import from agents/ and intergrax/
```

---

## 4. Trust boundaries

| Boundary | Inside Intergrax trust | Outside / untrusted until verified |
|----------|------------------------|------------------------------------|
| Client → Tier-3 API | Authn/z, tenant isolation, request validation | Raw client payloads |
| Tier-3 → Nexus / policy / HITL | Runtime decisions, acceptance gates | — |
| Tier-2 adapter → external agent | Normalized contracts, correlation ids | External Agent Card, quotes, status, tool claims, deliverable bytes |
| Deliverable workspace | Allowlisted paths under app data home | External publication targets |
| Receipts | Platform ProofReceipt / DocumentStore facts | Partner-specific receipt products (if any) |

Intergrax does **not** attest the external contractor's domain correctness. It attests **governed process**: policy, HITL, correlation, and what the harness observed at its boundaries.

---

## 5. Quote-first lifecycle

Target lifecycle (implemented from GEC-1 onward; documented here as architecture intent):

```text
intake
  → discover external Agent Card
  → create external task
  → retrieve quote
  → present quote (Tier-3)
  → HITL accept / reject (runtime-controlled)
  → continue external work only after acceptance
  → sync status timeline
  → retrieve deliverables into workspace
  → normalize evidence + compose GovernedProofProfile
  → (host) persist / expose receipts  ← beyond platform GEC-6 boundary
```

| Stage | Owner |
|-------|-------|
| Quote retrieval / forwarding | Tier-2 adapter |
| Surface governed continuation (`reason=QUOTE`) | Tier-2 adapter (mapping only) |
| Quote presentation | Tier-3 application |
| Quote acceptance / rejection | Runtime HITL + Tier-3 surfaces — **not** the adapter |
| Continue after acceptance | Nexus + policy after HITL decision; Tier-2 forwards evidence |

### Governed Continuation (GEC-4)

Reusable platform capability — not a quote lifecycle engine ([ADR-GOVERNED-CONTINUATION-001](../../../docs/adr/entries/2026-07-20/ADR-GOVERNED-CONTINUATION-001.md)):

```text
External Work quote → GovernedContinuationRequest(QUOTE)
  → existing Nexus ExecutionInterrupt → human/policy decision
  → QuoteAcceptanceEvidence → Nexus resume → Tier-2 forward → Protocol
```

| Owns | Does not own |
|------|----------------|
| Platform composition helpers + generic `ContinuationReason` | New interruption runtime |
| Tier-2 surface + evidence forward | Approval evaluation |
| Nexus interrupt / HITL / resume | Quote-specific interrupt types |

HITL UX presentation remains deferred; GEC-4 proves composition only.

---

## 6. HITL ownership

| Concern | Owner |
|---------|-------|
| HITL protocol, pause/resume, decision records | Intergrax runtime (Tier-0/1) |
| Quote accept / reject UX or API | Tier-3 application (deferred product surface) |
| Wallet / payment approval (if ever in scope) | Tier-3 + runtime — **never** adapter |
| Adapter reaction to continuation evidence | Tier-2 forwards pre-authorized evidence only |

The adapter **must not** invent acceptance, auto-approve quotes, bypass HITL, or resume Nexus.

---

## 7. Side-effect policy boundary (GEC-5)

Meaningful external side effects are authorized by the **platform policy boundary** before provider-bound execution ([ADR-POLICY-SIDE-EFFECT-001](../../../docs/adr/entries/2026-07-20/ADR-POLICY-SIDE-EFFECT-001.md)).

```text
Tier-3 composition root
  → settings.meaningful_side_effect_policy  (MeaningfulSideEffectEvaluator)
  → ExternalContractorAdapterAgent
  → ExternalWorkAdapter (describe action → evaluate → ALLOW/DENY/REQUIRE_HUMAN)
```

| Outcome | Host / adapter behavior |
|---------|-------------------------|
| ALLOW | Provider-bound call may proceed |
| DENY | Structured adapter result; no provider mutation |
| REQUIRE_HUMAN / ESCALATE | Compose `GovernedContinuationRequest` (GEC-4); no provider call |

| Allowed in adapter | Prohibited in adapter |
|--------------------|-----------------------|
| Describe proposed side effect + call injected evaluator | Embed spend limits / quote thresholds |
| Forward evidence only after ALLOW | Treat evidence or resume as allow |
| Observational reads (`get_quote`, timeline, …) without mutation policy | Escape workspace allowlists |

Quote **receipt** is observational (continuation surface only). Quote **acceptance** is meaningful (`ACCEPT_QUOTE`). Product policy packs / business rules remain deferred — host injects a deterministic evaluator when proving the path.

---

## 7.1 Governed proof profile (GEC-6)

> A proof profile is a description of governed execution, not a receipt, not an audit log, and not an authorization mechanism.

After a meaningful side effect succeeds under policy ALLOW, Tier-2 composes `GovernedProofProfile` ([ADR-GOVERNED-PROOF-001](../../../docs/adr/entries/2026-07-20/ADR-GOVERNED-PROOF-001.md)). The host may surface it later (GEC-7+); it must not treat the profile as a signed receipt or authorization token.

| Host may | Host must not (in GEC-6) |
|----------|--------------------------|
| Observe proof metadata on adapter results | Persist / sign / hash proofs as a product receipt |
| Join evidence refs to HITL / policy artifacts | Embed provider transport payloads into proofs |

`ProofReceipt` / DocumentStore remain the later persistence path — distinct from the descriptive profile.

---

## 8. External integration boundary

External work access is a **platform integration concern** (GEC-2 — Done):

```text
Intergrax runtime / Tier-2 adapter
        │
        ▼
ExternalWorkIntegration  (intergrax/integrations/contracts/external_work.py)
        │
   ┌────┼─────────────┐
   ▼    ▼             ▼
 A2A   REST         Future provider   (deferred — not in GEC-2)
```

| Item | Canonical owner |
|------|-----------------|
| Interaction model (request/snapshot/timeline/evidence/capabilities) | `intergrax.contracts.external_work` |
| Integration Protocol | `ExternalWorkIntegration` |
| Structured errors | `ExternalWorkError` + `ExternalWorkErrorCode` |
| DI slot | `IntegrationCategory.EXTERNAL_WORK` / `IntegrationProfile.external_work` |
| ADR | [`ADR-EXTWORK-002`](../../../docs/adr/entries/2026-07-20/ADR-EXTWORK-002.md) |

**Semantic operations:** `discover` · `create_work` · `get_work` · `get_quote` · `submit_quote_acceptance` · `cancel_work` · `get_timeline` · `get_deliverables` · `get_evidence`

**Rules:**

- Boundary transmits already-authorized `QuoteAcceptanceEvidence` — it does not decide accept/reject.
- Mutating ops require idempotency keys; reads may retry; mutating ops are not retried blindly.
- `ExternalProviderEvidenceRef` ≠ `GovernedProofProfile` ≠ Intergrax `ProofReceipt`.
- Concrete partner mappers (A2A/REST) and catalog slugs land in later phases; Tier-3 supplies credentials/config only.
- Tier-2 consumes the Protocol; it does not own the boundary.

---

## 9. Task and correlation identity

| Identity | Role |
|----------|------|
| Intergrax `task_id` / `run_id` | Primary governed execution identity |
| External contractor task id | Correlated foreign key maintained by adapter |
| Quote id | Bound to external task + Intergrax run |
| Correlation / idempotency keys | Required for create/continue/status sync |

Architecture rule: every external call that can mutate contractor state must carry deterministic idempotency material derived from Intergrax identity + lifecycle stage.

**GEC-1 / GEC-2 platform schemas** (canonical owner — not this application package):

| Concept | Module |
|---------|--------|
| Money | `intergrax.contracts.money.MoneyAmount` |
| Status / correlation / quote / acceptance / deliverable | `intergrax.contracts.external_work` |
| Create request / snapshot / timeline / provider evidence / capabilities | `intergrax.contracts.external_work` |
| Integration boundary | `intergrax.integrations.contracts.external_work.ExternalWorkIntegration` |
| ADRs | [`ADR-EXTWORK-001`](../../../docs/adr/entries/2026-07-20/ADR-EXTWORK-001.md), [`ADR-EXTWORK-002`](../../../docs/adr/entries/2026-07-20/ADR-EXTWORK-002.md) |

Nexus `TaskState` is **not** extended with commercial/quote stages — use `ExternalWorkStatus` at the external-work boundary.

---

## 10. Deliverable workspace boundary

Deliverables retrieved from the external agent land only in **allowlisted workspace paths** owned by the Tier-3 host (data home / sample or tenant workspace roots).

| Action | Gate |
|--------|------|
| Store deliverable bytes | Workspace allowlist + policy |
| Expose download/list API | Tier-3 serving |
| Publish outside workspace | Explicit approval path (future) — not adapter-owned |

---

## 11. Receipt and evidence model

| Artifact | Owner | Notes |
|----------|-------|-------|
| HOS / Nexus trace | Platform runtime | Unchanged spine |
| Provider evidence references | `ExternalProviderEvidenceRef` via integration boundary | GEC-2 — refs only, not proof |
| External tool evidence (normalized) | Adapter → platform evidence shapes | GEC-3+ — no partner hardcoding in core |
| Governed proof profile (descriptive) | Platform `GovernedProofProfile` composed by Tier-2 | GEC-6 |
| Governed contractor receipt (persistence) | Platform ProofReceipt + Tier-3 exposure | Later (post GEC-6) |
| Partner-native receipts | External product (optional) | Mapped, not reimplemented |

Provider-supplied evidence, descriptive proof profiles, and Intergrax-generated ProofReceipts remain distinct. GEC-6 defines the descriptive profile only; receipt persistence stays a later platform capability.

---

## 12. Provider-neutral partner adapter model

```text
Tier-3 governed_contractor_application
  → ExternalContractorAdapterAgent (inject ExternalWorkIntegration)
  → ExternalWorkAdapter (Tier-2 mapping only)
  → ExternalWorkIntegration (platform Protocol)
  → Deterministic fake (GEC-3 tests) | future provider (GEC-9/10)
```

GEC-3 proves the abstraction **without transport**. GEC-4 proves **Governed Continuation** composition (External Work = first consumer). Host may inject via `settings.external_work_integration`. Tier-2 owns mapping/correlation/normalization/forwarding; Tier-3 + runtime own governance decisions (HITL UX, policy packs, receipts) in later phases. Adding a new partner should require **mapping + configuration**, not a fork of Nexus or a new Tier-3 product core.

---

## 13. Failure and recovery behavior

| Failure | Expected behavior |
|---------|-------------------|
| Agent Card discovery failure | Fail closed; no silent mock success in live mode |
| Quote retrieval failure | Surface error; no HITL accept path |
| HITL timeout / reject | Do not continue external paid/side-effecting work |
| External status sync failure | Retry with backoff where safe; preserve correlation |
| Deliverable fetch failure | Partial failure recorded in evidence; no workspace escape |
| Adapter crash mid-flight | Resume via idempotent correlation; no duplicate external creates when keys present |

---

## 14. Non-goals

- Local code-review (or other domain) agent that competes with the external contractor
- Orchestration graph inside the Tier-2 adapter
- Partner-specific URLs/identities in `intergrax/`
- Duplicating HITL, Nexus, policy, trace, or ProofReceipt stacks
- Placing reusable quote/contractor contracts inside this Tier-3 package
- Production SLA, marketplace, or wallet product claims in GEC-0
- Marking future GEC phases as done without evidence

---

## 15. Public proof path

Planned public proof (GEC-9 → GEC-11):

1. Deterministic stub external contractor (offline CI)
2. Live design-partner integration (operator-gated)
3. End-to-end PASS matrix: intake → quote HITL → status → deliverable → receipt/trace

Public wording must remain **source-available evaluation / proof path** — not open-source redistribution and not production certification.

---

## 16. Planned partner handoff

See [`PARTNER_HANDOFF.md`](PARTNER_HANDOFF.md). Handoff materials will include:

- base URL and auth for the Tier-3 host,
- quote-first API sequence,
- correlation field map,
- sample request/response fixtures (when GEC-7+ exist),
- explicit list of partner-owned vs Intergrax-owned concerns.

---

## Scaffold baseline (GEC-0)

| Item | Value |
|------|-------|
| Package | `governed_contractor_application` |
| Profile | `product` |
| Route prefix | `/v1/governed_contractor` |
| Default port | `8000` |
| Mounted agent | `ExternalContractorAdapterAgent` |
| Default capability | `external_contractor.adapt` |
| Factory | `host/factory.py` → harness host runtime |
| Deploy triad | `docker/`, `BUILD_AND_DEPLOY.md`, gate tests |

GEC-0 delivers scaffold + architecture/plan docs only. Domain contracts and runtime lifecycle begin at **GEC-1**.
