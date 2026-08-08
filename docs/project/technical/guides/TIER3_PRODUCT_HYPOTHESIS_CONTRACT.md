# Tier-3 Product Hypothesis Contract

**Status:** Normative authoring guide (Tier-3 applications)  
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)  
**Canon:** [`TIER3_APPLICATION_ENVIRONMENT.md`](../../architecture/TIER3_APPLICATION_ENVIRONMENT.md) · [`SYSTEM_INVARIANTS.md`](SYSTEM_INVARIANTS.md) · [`MATURITY_TAXONOMY.md`](MATURITY_TAXONOMY.md)

---

## Purpose

This document defines the minimal product hypothesis contract required before creating or materially changing a Tier-3 Intergrax application environment.

A Tier-3 application is not just a technical host.  
It must represent a product hypothesis: a specific user, painful workflow, expected outcome, risk boundary and measurable success criteria.

This guide complements:

- [`docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../../architecture/TIER3_APPLICATION_ENVIRONMENT.md)
- [`docs/project/technical/guides/SYSTEM_INVARIANTS.md`](SYSTEM_INVARIANTS.md)
- [`docs/project/technical/guides/MATURITY_TAXONOMY.md`](MATURITY_TAXONOMY.md)

For scaffold → host factory → first `run_task()` workflow after the hypothesis is written, see [`APPLICATION_CREATION_GUIDE.md`](APPLICATION_CREATION_GUIDE.md). Use **this file first** when deciding whether an application should exist at all.

---

## When this contract is required

Complete the contract **before**:

1. Running `python -m intergrax.scaffold new-application` or `new-stack` for a new product host.
2. Materially changing user-facing scope: roster, mutating tools, HITL boundary, deployment posture, or success criteria.
3. Promoting a lab spike to a product or production candidate.

Skip only for throwaway local spikes that will not merge — and still answer question 1 in architecture §45 checklist.

---

## Required contract

Copy the template below into the application tree (see [Artifact location](#artifact-location)). Every field **MUST** be filled with concrete, testable content — not placeholders.

```markdown
## Product Hypothesis Contract

- Application id:
- Application name:
- Target user:
- User context:
- Painful workflow:
- Current alternative:
- Why Intergrax / harness / agentic execution is valuable here:
- Primary user goal:
- Expected output:
- Agent roster:
- Required tools / integrations:
- Required memory / RAG / context sources:
- Human review boundary:
- Risk tier:
- Success metric:
- Failure metric:
- Non-goals:
- Production readiness target:
- Evidence required before production:
```

---

## Field guidance

| Field | What to write |
|-------|----------------|
| **Application id** | Stable slug matching `applications/<app_id>/` and manifest `app_id`. |
| **Application name** | Human-readable product name. |
| **Target user** | Primary persona — role, seniority, domain. |
| **User context** | When/where the workflow happens; constraints (time, compliance, devices). |
| **Painful workflow** | Specific steps that are slow, error-prone, or expensive today. |
| **Current alternative** | Manual process, SaaS, scripts, or status quo — honestly stated. |
| **Why Intergrax / harness / agentic execution is valuable here** | Why agents + governed runtime beat a simple script or form — routing, policy, memory, multi-step reasoning, audit trail. |
| **Primary user goal** | One sentence outcome the user cares about. |
| **Expected output** | Deliverable shape: report, draft, ticket, file, API response, etc. |
| **Agent roster** | Tier-2 agents and roles; cite capability tokens where known. |
| **Required tools / integrations** | External systems the host must reach through ToolRuntime — not vendor SDKs in Tier-3. |
| **Required memory / RAG / context sources** | Approved memory views, indexes, corpora; retention expectations. |
| **Human review boundary** | What always requires human approval before external effect; HITL routes (see [`RELIABILITY_FAILURE_AND_HITL.md`](../../architecture/RELIABILITY_FAILURE_AND_HITL.md)). |
| **Risk tier** | Product-level risk: `low` · `medium` · `high` · `critical` — aligned with skill/agent governance tiers; drives policy and release gates. |
| **Success metric** | Observable metric that validates the hypothesis (latency, accuracy, adoption, cost, quality). |
| **Failure metric** | Observable signal that the hypothesis is wrong or unsafe — stop or rollback criteria. |
| **Non-goals** | Explicit out-of-scope items to prevent scope creep. |
| **Production readiness target** | Four-axis maturity target per [`MATURITY_TAXONOMY.md`](MATURITY_TAXONOMY.md) — e.g. A4 / I3 / P2 / E3 — not a vague "L5". |
| **Evidence required before production** | Tests, smoke scenarios, audit slices, operational windows, sign-offs. |

---

## Artifact location

Store the filled contract at:

```text
applications/<app_id>/PRODUCT_HYPOTHESIS.md
```

For stacks with multiple hosts, one contract per deployable application environment. Cross-link from `applications/<app_id>/ARCHITECTURE.md` when that file exists — do not duplicate the full contract in platform plan docs.

---

## Review checklist for Cursor

Before scaffold or material Tier-3 change, verify:

- [ ] Every template field is filled with concrete content — no TBD for production-bound work.
- [ ] **Painful workflow** and **current alternative** describe a real user problem, not only harness features.
- [ ] **Why Intergrax** explains agentic/harness value — not "because we use Intergrax".
- [ ] **Agent roster** maps to Tier-2 agents; business logic stays in agents, not `host/factory.py`.
- [ ] **Human review boundary** names actions that must not run without HITL when risk tier ≥ `medium`.
- [ ] **Success metric** and **failure metric** are measurable — not subjective-only.
- [ ] **Production readiness target** uses four-axis vocabulary from [`MATURITY_TAXONOMY.md`](MATURITY_TAXONOMY.md).
- [ ] **Non-goals** prevent silent expansion into adjacent products.

Technical host checklist (manifest, roster wiring, deploy triad): architecture [`TIER3_APPLICATION_ENVIRONMENT.md`](../../architecture/TIER3_APPLICATION_ENVIRONMENT.md) §45.

---

## Related documents

| Document | Why |
|----------|-----|
| [`TIER3_APPLICATION_ENVIRONMENT.md`](../../architecture/TIER3_APPLICATION_ENVIRONMENT.md) | Host profile, manifest, production gates |
| [`APPLICATION_CREATION_GUIDE.md`](APPLICATION_CREATION_GUIDE.md) | Author workflow after hypothesis is approved |
| [`AGENT_AUTHOR_MINIMAL_PATH.md`](AGENT_AUTHOR_MINIMAL_PATH.md) | Minimal Tier-2 agent path for roster members |
| [`SYSTEM_INVARIANTS.md`](SYSTEM_INVARIANTS.md) | Cross-layer MUST/MUST NOT |
| [`MATURITY_TAXONOMY.md`](MATURITY_TAXONOMY.md) | Four-axis maturity vocabulary |
| [`RELIABILITY_FAILURE_AND_HITL.md`](../../architecture/RELIABILITY_FAILURE_AND_HITL.md) | HITL and failure handling canon |
