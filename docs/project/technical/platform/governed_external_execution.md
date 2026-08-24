# Governed External Execution

**Status:** Architecture closure (GEC-0…GEC-6.1) + host platform completion (PC-1…PC-10) — 2026-07-21  
**Role:** Primary architectural reference for the reusable platform capability  
**Not:** Implementation tracker, product UX spec, or ADR substitute  
**Host lifecycle:** [`governed_external_work_host_lifecycle.md`](governed_external_work_host_lifecycle.md)

This document consolidates ownership, lifecycle, invariants, and extensibility for **governed external execution**. Decision rationale lives in ADRs; vertical product detail lives in the GEC host and Tier-2 adapter docs.

---

## Terminology

| Term | Meaning |
|------|---------|
| **Governed External Execution** | Platform capability: intent → policy-gated external side effect → descriptive proof |
| **External Work** | Domain vocabulary and integration boundary (`external_work` contracts + `ExternalWorkIntegration`) |
| **GEC (Governed External Contractor)** | First vertical product proving the capability (`governed_contractor_application` + `external_contractor_adapter`) |
| **Governed Continuation** | Composition over Nexus interrupt / HITL / resume — not a second runtime |
| **Meaningful side effect** | External action that may create commitment, mutation, disclosure, access change, or irreversible consequence |
| **GovernedProofProfile** | Descriptive post-execution proof metadata — not a receipt, audit log, or authorization token |
| **Execution Evidence & Host Attestation** | Downstream host capability after proof — see [`execution_evidence_and_host_attestation.md`](execution_evidence_and_host_attestation.md) |
| **ProofReceipt (execution evidence)** | Portable host-attested export of a governed boundary event — **not** owned by GEC |
| **ProofReceipt (DocumentStore)** | Separate LKW persistence product (`intergrax.proofs.receipts`) |

---

## Authoritative ADRs

| ADR | Decision |
|-----|----------|
| [ADR-EXTWORK-001](../adr/entries/2026-07-20/ADR-EXTWORK-001.md) | Provider-neutral External Work + money contracts; do not extend Nexus `TaskState` |
| [ADR-EXTWORK-002](../adr/entries/2026-07-20/ADR-EXTWORK-002.md) | Sync `ExternalWorkIntegration` Protocol; transmits evidence, never decides |
| [ADR-GOVERNED-CONTINUATION-001](../adr/entries/2026-07-20/ADR-GOVERNED-CONTINUATION-001.md) | Continuation as Nexus composition; Tier-2 surfaces / forwards only |
| [ADR-POLICY-SIDE-EFFECT-001](../adr/entries/2026-07-20/ADR-POLICY-SIDE-EFFECT-001.md) | Policy before meaningful external side effects; fail closed |
| [ADR-GOVERNED-PROOF-001](../adr/entries/2026-07-20/ADR-GOVERNED-PROOF-001.md) | Proof profiles describe; do not authorize, sign, or persist |

Domain composition notes (not ADR substitutes):

- [`docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md`](../../architecture/RELIABILITY_FAILURE_AND_HITL.md) — GEC-4…GEC-6 composition sections
- [`agents/external_contractor_adapter/docs/ARCHITECTURE.md`](../../../../agents/external_contractor_adapter/docs/ARCHITECTURE.md) — Tier-2 mapping consumer
- [`applications/governed_contractor_application/docs/ARCHITECTURE.md`](../../../../applications/governed_contractor_application/docs/ARCHITECTURE.md) — Tier-3 host vertical

---

## End-to-end lifecycle

Platform capability ends at `GovernedProofProfile`. Host product surfaces begin after that boundary.

```text
User intent
  │
  ▼
Tier-3 host (tenant / API / wiring)
  │  injects ExternalWorkIntegration + MeaningfulSideEffectEvaluator
  ▼
Provider-neutral request
  │  ExternalWorkCreateRequest / correlation / idempotency
  │  real Nexus task_id + run_id (never synthesized)
  ▼
Tier-2 adapter (mapping only)
  │  discover / normalize / correlate
  ▼
Policy evaluation  ← MeaningfulSideEffectRequest
  │
  ├─ DENY ──────────────────────────────► structured stop (no provider mutation)
  │
  ├─ REQUIRE_HUMAN / ESCALATE
  │     ▼
  │   Governed Continuation (optional)
  │     surface GovernedContinuationRequest
  │     → Nexus ExecutionInterrupt
  │     → HITL / policy decision
  │     → continuation evidence (e.g. QuoteAcceptanceEvidence)
  │     → Nexus resume
  │     → Tier-2 forwards evidence
  │     → policy again before mutation
  │
  └─ ALLOW
        ▼
      Meaningful side effect
        │  provider-bound Protocol method (create / accept / cancel)
        ▼
      GovernedProofProfile  ← descriptive composition (mandatory)
        │
========│========  PLATFORM BOUNDARY  ========
        ▼
Host capabilities begin
  (GovernedExecutionResult → governed EBE → HostAttestation → ProofReceipt;
   persistence / attestation recovery / offline CLI demo;
   UX, publication, partner transport, …)
```

Downstream (not GEC):

- [`execution_evidence_and_host_attestation.md`](execution_evidence_and_host_attestation.md)
- [`governed_external_work_host_lifecycle.md`](governed_external_work_host_lifecycle.md)

External Work specialization of the optional continuation branch uses `ContinuationReason.QUOTE` only.

Synchronous mapping sketch (observational reads omitted):

```text
discover → policy(CREATE) → create_work → GovernedProofProfile
  → [no acceptance] get_quote / timeline → optional QUOTE continuation surface
  → [acceptance] policy(ACCEPT_QUOTE) → submit_quote_acceptance → GovernedProofProfile
```

---

## Ownership matrix

Exactly one primary owner per capability. “Does not” columns eliminate ambiguity.

| Capability | Owner | Does | Does not |
|------------|-------|------|----------|
| **Host (Tier-3)** | `applications/governed_contractor_application` | Inject integrations + policy evaluator; tenant/env; public API; workspace allowlists; present quotes / HITL surfaces (product); observe proof metadata | Own External Work contracts; evaluate policy rules in the adapter; implement providers; treat proof as receipt/authz |
| **Tier-2 adapter** | `agents/external_contractor_adapter` | Map/normalize External Work; preserve correlation + idempotency; surface/forward continuation; describe side effects; compose `GovernedProofProfile` after ALLOW + success | Decide accept/reject; resume Nexus; embed policy rules; persist/sign proofs; poll/retry engines; import `applications.*` |
| **Provider** | Implements `ExternalWorkIntegration` | Execute discover/create/quote/accept/cancel/timeline/deliverables/evidence behind the Protocol | Evaluate policy; authorize acceptance; compose Intergrax proofs; own Nexus identity |
| **Policy** | Runtime `PolicyEngine` / injected `MeaningfulSideEffectEvaluator` (prefer `RuntimePolicyBundleEvaluator` for attested packs) | Authorize or deny meaningful side effects before provider calls against a concrete `ImmutableRuntimePolicyBundle`; map REQUIRE_HUMAN to continuation | Resume Nexus; execute provider transport; invent business rules inside Tier-2 |
| **Continuation** | Platform composition + Nexus/HITL | Pause for governance; carry decision evidence; resume orchestration | Execute external work; decide commercial outcomes inside Tier-2; introduce `ContinuationRuntime` |
| **Proof (`GovernedProofProfile`)** | Platform contract; Tier-2 composes | Describe who/what/run/policy refs/evidence refs/correlation after success | Authorize, resume, sign, hash, store, publish |
| **Execution Evidence & Host Attestation** | Host + platform contracts (downstream of GEC) | Compose governed EBE, sign, emit portable ProofReceipt, verify offline | Own provider execution; live inside Tier-2; authorize side effects |
| **DocumentStore ProofReceipt** | Platform persistence (LKW path) | Persist structured proof results | Replace descriptive proof or governed attested export |
| **Future Audit** | Platform audit storage (later) | Durable audit of governed executions | Live inside Tier-2 mapping modules |

### Stage owners (intent → proof)

| Stage | Sole owner |
|-------|------------|
| User intent / product API | Host |
| Nexus task / run orchestration | Runtime Nexus |
| External Work contracts | Platform (`intergrax/contracts`) |
| Integration Protocol | Platform (`ExternalWorkIntegration`) |
| Mapping / normalization / correlation forward | Tier-2 |
| Policy allow/deny/require-human | Policy engine (+ host rule packs) |
| HITL accept/reject decision | Runtime HITL + host surfaces |
| Continuation surface + evidence forward | Tier-2 (mapping only) |
| Provider-bound mutation | Provider (after ALLOW) |
| Descriptive proof composition | Tier-2 (mandatory after success) |
| Receipt / audit / attestation products | Downstream Execution Evidence (host) — see dedicated platform doc; beyond GEC closure |

---

## Platform invariants

Collected from accepted ADRs and GEC-3…GEC-6 architecture (not invented here).

1. **Policy precedes meaningful side effects.** No provider-bound mutation without prior `PolicyDecision` ALLOW (fail closed on missing evaluator, principal, run identity, or indeterminate).
2. **Evidence never authorizes.** `QuoteAcceptanceEvidence` / continuation evidence is not an allow decision; policy still gates the subsequent side effect.
3. **Proof never authorizes.** `GovernedProofProfile` is descriptive only — never allow, resume, or evaluate policy.
4. **Continuation never executes work.** It surfaces interruption and forwards decision evidence; it does not call mutating provider methods.
5. **Tier-2 never persists proofs.** No signing, hashing, DocumentStore writes, or receipt products in the adapter.
6. **Providers never evaluate policy.** The Protocol transmits already-authorized evidence; it does not decide accept/reject/pay/publish.
7. **Real execution identity is preserved.** `task_id` and `run_id` are distinct; run identity is forwarded from Nexus context — never synthesized from `task_id` or placeholders.
8. **No meaningful side effect succeeds without proof.** After ALLOW + successful mutation, composition of `GovernedProofProfile` is mandatory (not best-effort).
9. **Correlation and idempotency are preserved.** Keys derived from Intergrax identity + stage are forwarded; mutating calls are not blindly retried.
10. **Host owns receipts (future).** Persistence and product receipt exposure remain host/platform receipt paths — distinct from descriptive proof.
11. **Proof profiles are descriptive only.** Not receipts, audit logs, or authorization mechanisms.
12. **External Work status ≠ Nexus `TaskState`.** Commercial/quote stages stay in `ExternalWorkStatus`.
13. **Observational reads are not mutation gates.** e.g. `get_quote` may surface continuation; it is not itself a meaningful side-effect gate.
14. **Contracts stay transport-free.** No HTTP/A2A/REST/partner SDK types in External Work, continuation, side-effect, or proof contracts.
15. **Tier boundaries hold.** `intergrax` ↛ agents/applications; agents ↛ applications.

---

## Extensibility — adding a new governed external provider

Target path:

```text
1. Implement ExternalWorkIntegration (sync Protocol)
2. Bind via IntegrationProfile.external_work / host settings
3. Inject into ExternalContractorAdapterAgent (or equivalent consumer)
4. Supply MeaningfulSideEffectEvaluator (bundle-backed for attestation) + principal + real run_id
5. Host orchestrator: ProviderInvocation → GovernedExecutionResult → attested receipt
```

### Must not require changes

| Layer | Change required to add a provider? |
|-------|-------------------------------------|
| Policy engine / `MeaningfulSideEffectRequest` | **No** — inject rules/evaluator; do not fork the engine |
| `GovernedProofProfile` / proof contracts | **No** — compose with existing fields |
| Governed Continuation contracts | **No** — reuse reasons/evidence refs; add domain evidence mappers only if a new reason needs them |
| Governance abstractions (Nexus interrupt, HITL, `PolicyDecision`) | **No** |
| Tier-2 mapping core | **No** provider branches — only if a new *domain* mapping concern appears (not transport) |

### May require (local to the new provider)

- Protocol implementation + credentials/config in the host
- Optional catalog slug / `PROVIDER_CATEGORY_CONTRACT_REGISTRY` entry when a real provider package lands
- Host policy *packs* / thresholds for the product (not adapter-embedded rules)
- Transport mapper (A2A/REST/…) behind the Protocol

### Extensibility verdict

Adding a provider **does not** require changing the policy engine, proof contracts, continuation contracts, or governance abstractions — provided the provider stays behind `ExternalWorkIntegration` and the host injects policy. Gap if violated: any `if provider == …` in Tier-2/core or provider-side authorization would break neutrality and must be rejected in review.

---

## Future platform capabilities

Outside GEC-0…GEC-6.1 closure. Documented as boundaries only — no contracts or implementations claimed here.

| Capability | Boundary vs governed external execution |
|------------|----------------------------------------|
| **Execution Evidence & Host Attestation** | Implemented downstream capability — [`execution_evidence_and_host_attestation.md`](execution_evidence_and_host_attestation.md); does not reopen GEC |
| **DocumentStore Proof Receipt** | Persistence / queryable LKW receipt product; does not replace `GovernedProofProfile` or governed attested export |
| **Persistent Audit** | Durable audit store for governed executions; complementary to descriptive proof and HOS trace |
| **Harness Execution Boundary Events** | Tool/step BoundaryAttest export (`execution_boundary_event.v1`) — sibling of governed EBE |
| **Verification (critic)** | Critic / verification engines that judge outcomes; distinct from descriptive proof and receipt signature verify |
| **Replay** | Reconstruct or re-drive runs from trace; does not authorize side effects |
| **Provider transport & catalog** | A2A/REST mappers, partner stubs, catalog slugs — behind Protocol; not governance |
| **Product HITL UX / policy packs** | Host product surfaces and business rule packs — consumers of continuation + policy, not new runtimes |

---

## Architecture review notes (closure)

Review of ownership from user intent through `GovernedProofProfile` found **no undocumented architectural decision** requiring a new ADR. Residual product work (HITL UX, receipts, live providers) remains intentionally outside this platform closure.

**Consistency fixes applied at closure (docs only):**

- Soft wording that treated post-ALLOW proof composition as optional was aligned to the mandatory invariant in consumer/domain docs where incorrect.
- This file is the single consolidation point so domain pairs and vertical docs can reference rather than restate the full matrix.

---

## Verification (unchanged behavior)

```bash
uv run pytest tests/unit/contracts/test_governed_continuation.py tests/unit/contracts/test_governed_proof.py tests/unit/runtime/policy/test_meaningful_side_effect_policy.py agents/external_contractor_adapter/tests applications/governed_contractor_application/tests -q
```

Partner-facing readiness / five-point matrix (claims boundaries, demo path): [`docs/project/integrations/impeachmentright_validation_readiness.md`](../../integrations/impeachmentright_validation_readiness.md).
