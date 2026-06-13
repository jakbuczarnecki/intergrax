# ADR-OBS-002: Unsigned Execution Boundary Export (EBE)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-13 |
| **Deciders** | Harness platform + partner (AgentReceipt PoC) |
| **Related** | [`architecture/OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md) §18 · [ADR-OBS-001](../2026-06-08/ADR-OBS-001.md) · `applications/attestation_demo/adr/ADR-ATTESTATION_DEMO-001.md` |

## Context

External partners (e.g. AgentReceipt) need portable evidence that a governed tool action occurred **without** access to Intergrax's internal journal. The Harness Observability Spine (HOS) answers operator questions inside the organization; it does not natively produce vendor-neutral, externally consumable tool-boundary facts.

Partner PoC requirements (agreed):

- Intergrax emits **unsigned** facts at `RuntimeToolInvoker`
- Partner signs receipts locally (`client_observed` recommended)
- PoC v1 delivers events in the **trigger API response** only (no webhook)

Alternatives considered:

1. **Extend HOS `RuntimeEvent` with receipt fields** — rejected: couples internal journal to external receipt semantics; violates vendor-neutral spine.
2. **Agent-level export in Tier-2** — rejected: duplicates boundary; agents must not own attestation logic.
3. **Intergrax signs boundary events in PoC v1** — rejected: overstates trust without key management; partner chose local signing.
4. **Optional EBE side channel at invoker** — **accepted**: unsigned `execution_boundary_event.v1`, memory buffer, Tier-3 profile.

## Decision

Add **Execution Boundary Export (EBE)** as an optional Tier-1 side channel:

1. **Schema:** `execution_boundary_event.v1` with explicit `signed: false`
2. **Hook:** `RuntimeToolInvoker` post-execution via `ExecutionBoundaryEmitter` (non-blocking)
3. **Policy:** `ExecutionBoundaryExportProfile` on `ApplicationEnvironmentProfile` (`side_effects_only` default for PoC)
4. **Delivery (PoC v1):** per-run `BoundaryEventBuffer` surfaced in Tier-3 API response
5. **Reference host:** `applications/attestation_demo` + `agents/boundary_demo` + `records.put`

Deferred: webhook sink, HarnessKernel step-level export, host-side event signing.

## Consequences

### Positive

- Clear separation: HOS internal journal vs external boundary facts
- Partner integrates via HTTP + JSON without Intergrax fork
- Honest trust model documented (`client_observed`, not `server_attested`)

### Negative

- Additional configuration surface on application hosts
- Unsigned events alone do not prove independent third-party trust
- PoC uses in-memory document store (lab profile)

## Compliance

- Tier boundaries preserved — EBE in `intergrax/runtime/attestation/`; no `agents/` imports in platform
- HOS unchanged — no receipt logic in unified journal
- AgentReceipt remains external; no vendor SDK in Intergrax
- Application ADR: `ADR-ATTESTATION_DEMO-001` records Tier-3 PoC decisions

## Verification

```bash
uv run pytest tests/unit/runtime/attestation/ -q
uv run pytest applications/attestation_demo/attestation_demo_tests -q
```
