# Governed Contractor Application — architecture

**Status:** GEC-0 bootstrap (2026-07-20) — product-profile scaffold; domain runtime not yet implemented  
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
  → normalize evidence + emit governed receipt
```

| Stage | Owner |
|-------|-------|
| Quote retrieval / forwarding | Tier-2 adapter |
| Quote presentation | Tier-3 application |
| Quote acceptance / rejection | Runtime HITL + Tier-3 surfaces — **not** the adapter |
| Continue after acceptance | Nexus + policy after HITL decision |

---

## 6. HITL ownership

| Concern | Owner |
|---------|-------|
| HITL protocol, pause/resume, decision records | Intergrax runtime (Tier-0/1) |
| Quote accept / reject UX or API | Tier-3 application |
| Wallet / payment approval (if ever in scope) | Tier-3 + runtime — **never** adapter |
| Adapter reaction to acceptance decision | Tier-2 continues or stops per runtime signal |

The adapter **must not** invent acceptance, auto-approve quotes, or bypass HITL.

---

## 7. Side-effect policy boundary

Meaningful side effects (external task mutation after quote, deliverable write, external publication) are subject to **runtime policy bundles** configured by the Tier-3 host.

| Allowed in adapter | Prohibited in adapter |
|--------------------|-----------------------|
| Call external APIs through governed integration/tools | Decide policy allow/deny |
| Report proposed side effects / tool evidence | Escape workspace allowlists |
| Retry idempotent reads / safe status sync | Publish externally without Tier-3/runtime approval |

Reusable policy mechanisms stay in platform policy infrastructure — GEC only supplies product-specific rule packs under the host when needed.

---

## 8. External integration boundary

External contractor access is an **integration concern**:

- GEC-2 defines a provider-neutral external contractor integration contract in platform space.
- Concrete partner adapters map partner schemas → that contract.
- Tier-2 agent consumes the integration; it does not embed partner SDKs as business logic ownership.
- Partner URLs and credentials live in Tier-3 environment configuration.

---

## 9. Task and correlation identity

| Identity | Role |
|----------|------|
| Intergrax `task_id` / `run_id` | Primary governed execution identity |
| External contractor task id | Correlated foreign key maintained by adapter |
| Quote id | Bound to external task + Intergrax run |
| Correlation / idempotency keys | Required for create/continue/status sync |

Architecture rule: every external call that can mutate contractor state must carry deterministic idempotency material derived from Intergrax identity + lifecycle stage (exact schemas in GEC-1/GEC-2).

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
| External tool evidence (normalized) | Adapter → platform evidence shapes | No partner hardcoding in core |
| Governed contractor receipt | Platform ProofReceipt (or equivalent) + Tier-3 exposure | GEC-6 |
| Partner-native receipts | External product (optional) | Mapped, not reimplemented |

GEC reuses existing ProofReceipt / DocumentStore infrastructure; it does not invent a parallel receipt product inside the application.

---

## 12. Provider-neutral partner adapter model

```text
Partner product
  → partner mapping (handoff / config)
  → ExternalContractor integration contract (platform)
  → ExternalContractorAdapterAgent (Tier-2)
  → Governed Contractor Application (Tier-3)
```

Adding a new partner should require **mapping + configuration**, not a fork of Nexus or a new Tier-3 product core.

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
