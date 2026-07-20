# ADR-GOVERNED_CONTRACTOR-001: GEC vertical — product host + external adapter split

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-20 |
| **Deciders** | Platform / GEC bootstrap |
| **Related** | [`ARCHITECTURE.md`](../ARCHITECTURE.md) · [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) · [`ADR-EXTERNAL_CONTRACTOR_ADAPTER-001`](../../../../agents/external_contractor_adapter/docs/adr/ADR-EXTERNAL_CONTRACTOR_ADAPTER-001.md) |

## Context

Intergrax needs a vertical proof that a **governed Tier-3 application** can wrap an **existing external contractor agent** (quote-first, status, deliverables, evidence) without reimplementing the contractor domain inside the harness.

Scaffolding must use canonical CLI paths. Agent slug and application slug differ (`external_contractor_adapter` vs `governed_contractor`), so a single `new-stack` invocation is insufficient.

## Decision

1. Create Tier-2 adapter via `python -m intergrax.scaffold new-agent external_contractor_adapter --capability external_contractor.adapt --pattern reflex`.
2. Create Tier-3 host via `python -m intergrax.scaffold new-application governed_contractor --profile product --agents external_contractor_adapter --port 8000`.
3. Treat GEC as a **generic capability** (governed external contractor agents), not a one-off partner product.
4. Keep reusable quote/contractor contracts and integration surfaces in `intergrax/`; keep partner URLs/identities out of core.
5. Align host default capability to `external_contractor.adapt` (scaffold `new-application` otherwise defaults custom agents to `<slug>.basic`).

Rejected:

- Copying `attestation_demo` / `boundary_demo` trees by hand
- Placing orchestration or HITL acceptance inside the adapter
- Embedding design-partner identity in `intergrax/`

## Consequences

### Positive

- Clear four-tier split and canonical package layout
- Product profile deploy triad available from day one
- Documentation establishes non-goals before runtime work (GEC-1+)

### Negative

- Capability id must be manually aligned after split scaffold (known scaffold limitation)
- Reflex pattern may later need an ADR if lifecycle mapping outgrows single-step reflex

## Compliance

- Tier boundaries preserved
- Source-available / collaboration wording retained — no open-source or production-ready claims
- No GEC-1 domain runtime introduced in this ADR’s closeout

## Implementation notes

- App: `applications/governed_contractor_application/`
- Agent: `agents/external_contractor_adapter/`
- Verify: host + agent pytest smoke; deploy-triad check for `governed_contractor`
