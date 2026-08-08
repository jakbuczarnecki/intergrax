# ADR-EXTWORK-002: Provider-neutral ExternalWorkIntegration boundary

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-20 |
| **Deciders** | Platform / GEC bootstrap |
| **Related** | [`ADR-EXTWORK-001`](ADR-EXTWORK-001.md) · GEC host plan · agent adapter plan · Platform consolidation [`governed_external_execution.md`](../../../platform/governed_external_execution.md) |

## Context

GEC-1 introduced reusable external-work domain contracts (`MoneyAmount`, `ExternalWorkStatus`, correlation, quote, acceptance, deliverables). GEC needs a stable way to talk to many external work providers (A2A contractors, REST SaaS, future protocols) without:

- embedding partner SDKs in Tier-2/Tier-3,
- copying A2A Agent Cards into core,
- inventing a second orchestration or receipt system,
- deciding HITL/policy/payment outcomes inside the transport layer.

Existing integration Protocols (`IssueTracker`, `WorkflowOrchestratorBackend`, …) are sync, typed, and DI-friendly via `IntegrationProfile` — but none model quote-first external work, acceptance evidence submission, or provider-evidence vs Intergrax-proof separation.

## Decision

1. Name the canonical boundary **`ExternalWorkIntegration`** (not contractor-specific) under `intergrax/integrations/contracts/external_work.py`.
2. Keep the interaction model (create request, snapshot, timeline, provider evidence refs, capability tokens, error codes) in `intergrax/contracts/external_work.py` so domain types remain reusable beyond the integrations package.
3. Use a **sync** `@runtime_checkable` `Protocol` — matching existing integration contracts; do not add a parallel async API.
4. Extend `IntegrationCategory.EXTERNAL_WORK` + `IntegrationProfile.external_work` for DI / instance binding. Defer catalog slug + `PROVIDER_CATEGORY_CONTRACT_REGISTRY` entry until a real provider package exists (GEC-8+).
5. Errors are `ExternalWorkError(IntegrationError)` with `ExternalWorkErrorCode` — no transport exception leakage.
6. The boundary **transmits** `QuoteAcceptanceEvidence`; it never decides acceptance (same for cancel/pay/publish).
7. Provider evidence refs (`ExternalProviderEvidenceRef`) stay distinct from Intergrax ProofReceipt / GEC-6.

Rejected alternatives:

- Putting the Protocol in the Tier-2 adapter or Tier-3 app — violates platform-first ownership.
- Creating `ExternalWorkRegistry` — duplicates `IntegrationProfile` / catalog binding.
- Async-only Protocol — inconsistent with current integration style.
- A2A Agent Card as the discovery model — couples core to one protocol.

## Consequences

### Positive

- One reusable facade for multiple external-work kinds
- GEC adapter/host remain consumers
- Deterministic fakes prove the contract without network
- Later A2A/REST mappers implement the same Protocol

### Negative

- No catalog slug until first provider lands (instance binding only)
- Full retry middleware not provided in GEC-2 (codes document retryability)

## Compliance

- No `applications.*` / `agents.*` imports in boundary modules
- No HTTP/A2A/REST/partner execution in GEC-2
- Tier boundaries preserved
- Linked GEC architecture/plan docs updated

## Implementation notes

- Protocol: `intergrax/integrations/contracts/external_work.py`
- Domain: `intergrax/contracts/external_work.py` (GEC-2 models)
- Tests: `tests/unit/integrations/test_external_work_integration.py`, extended `tests/unit/contracts/test_external_work.py`
- Verify: `uv run pytest tests/unit/integrations/test_external_work_integration.py tests/unit/contracts/test_external_work.py -q`
