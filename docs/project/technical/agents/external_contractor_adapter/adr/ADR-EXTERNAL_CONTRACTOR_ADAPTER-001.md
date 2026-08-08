# ADR-EXTERNAL_CONTRACTOR_ADAPTER-001: Tier-2 domain adapter (not orchestrator)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-20 |
| **Deciders** | Platform / GEC bootstrap |
| **Related** | [`ARCHITECTURE.md`](../ARCHITECTURE.md) · [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) · Host [`ADR-GOVERNED_CONTRACTOR-001`](../../../applications/governed_contractor_application/adr/ADR-GOVERNED_CONTRACTOR-001.md) |

## Context

GEC requires a Tier-2 agent that speaks to an external contractor product. Without an explicit boundary, the agent package risks becoming a second Nexus (orchestration, HITL, policy) or a competing domain contractor.

## Decision

1. Classify `ExternalContractorAdapterAgent` as a **domain adapter agent**.
2. Own only: Agent Card discovery, external task create/correlate, quote retrieval/forwarding, status sync, deliverable retrieval, evidence/receipt **normalization**.
3. Prohibit: quote acceptance, payment approval, policy decisions, workspace escape, external publication approval, governance bypasses.
4. Scaffold with ACP **reflex** + capability `external_contractor.adapt`; evolve pattern only via a later agent ADR if required.
5. Depend on provider-neutral platform integration/contracts (GEC-1/GEC-2); never hardcode partner URLs.

## Consequences

### Positive

- Clear ownership vs Tier-3 host and runtime
- Reuse of Nexus / HITL / policy / ProofReceipt stacks
- Partner substitution via mapping/config

### Negative

- Adapter alone cannot demonstrate end-to-end governance without host phases GEC-4…
- Reflex scaffold is a starting point, not a final lifecycle claim

## Compliance

- No `applications/` imports
- Tier boundaries preserved
- No GEC-3 runtime implementation claimed Done in GEC-0

## Implementation notes

- Package: `agents/external_contractor_adapter/`
- Host mount: `applications/governed_contractor_application/manifest.py`
- Verify: `uv run pytest agents/external_contractor_adapter/tests -q`
