# ADR-EXTERNAL_CONTRACTOR_ADAPTER-002: Mapping ownership and fake-provider proof (GEC-3)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-20 |
| **Deciders** | Platform / GEC |
| **Related** | [`ARCHITECTURE.md`](../ARCHITECTURE.md) · [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) · [ADR-EXTERNAL_CONTRACTOR_ADAPTER-001](ADR-EXTERNAL_CONTRACTOR_ADAPTER-001.md) · [ADR-EXTWORK-002](../../../../docs/project/technical/adr/entries/2026-07-20/ADR-EXTWORK-002.md) |

## Context

GEC-1/GEC-2 introduced platform contracts and `ExternalWorkIntegration`. Without an explicit Tier-2 principle, the adapter risks absorbing governance (HITL, policy, receipts) or transport (HTTP/A2A), or inventing a parallel execution framework.

## Decision

1. **Tier-2 owns mapping only** — request/snapshot/quote/timeline/deliverables/evidence normalization, correlation preservation, and idempotency forwarding.
2. **Tier-2 does not own governance** — no quote accept/reject, policy, wallet/payment, ProofReceipt, HITL decisions, retry/poll/resume engines, or Nexus `TaskState` commercial extensions.
3. **Consume only `ExternalWorkIntegration`** — inject via agent constructor / host settings; never construct providers or branch on provider type/protocol.
4. **Prove with a deterministic in-memory fake** in agent tests (`tests/fakes`) — not an A2A/REST stub and not the GEC-8/9 partner stub. No networking.

## Consequences

### Positive

- Clear reuse of GEC-1/GEC-2 abstractions without transport
- Governance remains in runtime + Tier-3 phases (GEC-4+)
- Partner substitution stays behind the Protocol

### Negative

- End-to-end governed proof still requires GEC-4…GEC-11
- Host without injection reports `external_work_integration_missing` (by design for GEC-3)

## Compliance

- No `applications` imports in the agent package
- No transport libraries in mapping modules
- Sync Protocol calls only

## Implementation notes

- `external_work_adapter.py`, `steps/domain_job.py`, agent DI
- Verify: `uv run pytest agents/external_contractor_adapter/tests -q`
