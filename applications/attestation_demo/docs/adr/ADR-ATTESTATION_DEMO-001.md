# ADR-ATTESTATION_DEMO-001: Partner PoC — unsigned EBE in API response

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-13 |
| **Deciders** | Platform + partner (BoundaryAttest) |
| **Related** | [`ARCHITECTURE.md`](../ARCHITECTURE.md) · [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) |

## Context

BoundaryAttest (external) provides portable signed receipts. Intergrax must demonstrate governed tool execution and export **neutral execution-boundary facts** without becoming a receipt product or implying platform attestation.

Partner agreed on:

- Intergrax emits **unsigned** boundary events at `RuntimeToolInvoker`
- Partner signs locally (`client_observed` recommended)
- PoC v1 delivers events in the **trigger API response** only (no webhook)

## Decision

1. Add Tier-1 **Execution Boundary Export (EBE)** — `execution_boundary_event.v1`, `signed: false`
2. Ship Tier-3 **`attestation_demo`** host with `POST /v1/attestation_demo/poc/run` returning `boundary_events[]`
3. Use Tier-2 **`boundary_demo_agent`** + `records.put` as the reference side-effecting tool
4. Defer webhook sink, HarnessKernel step-level export, and host-side signing to later phases

Rejected for PoC v1:

- Embedding BoundaryAttest in Intergrax
- Intergrax signing receipts or claiming `server_attested`
- Webhook-first delivery (partner chose synchronous API response)

## Consequences

### Positive

- Clear tier split: platform facts vs partner receipts
- Minimal partner integration path (HTTP + JSON)
- Reuses standard H-APP host scaffold and debug journal for comparison

### Negative

- Partner must run adapter externally
- Unsigned events alone do not prove independent third-party trust
- `records` lab wiring uses in-memory document store (not production persistence)

## Compliance

- Tier boundaries preserved — no `agents` or `applications` imports in `intergrax`
- BoundaryAttest remains external; no vendor SDK in platform
- Trust documentation states `client_observed`; no `server_attested` claim from Intergrax

## Implementation notes

- Platform: `intergrax/runtime/attestation`, hook in `RuntimeToolInvoker`
- Host: `applications/attestation_demo`
- Agent: `agents/boundary_demo`
- Verify: `uv run pytest applications/attestation_demo/tests -q`
