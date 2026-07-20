# ADR-GOVERNED-PROOF-001: Governed proof profiles describe, but do not own, execution evidence

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-20 |
| **Deciders** | Platform / GEC |
| **Related** | [ADR-GOVERNED-CONTINUATION-001](ADR-GOVERNED-CONTINUATION-001.md) · [ADR-POLICY-SIDE-EFFECT-001](ADR-POLICY-SIDE-EFFECT-001.md) · [ADR-EXTWORK-002](ADR-EXTWORK-002.md) · GEC-6 · Platform consolidation [`governed_external_execution.md`](../../../platform/governed_external_execution.md) |

## Context

After a governed external side effect, auditors and product surfaces need a stable answer to: who initiated, which Nexus run, what action/resource/provider, which policy outcome, which governance evidence, and which correlation/idempotency values.

Reuse audit found:

| Existing capability | Role vs GEC-6 |
|---------------------|---------------|
| `ProofReceipt` / DocumentStore | Persistence + receipt product — **not** the descriptive profile |
| `MeaningfulSideEffectRequest` | Pre-execution request — not post-execution proof |
| `PolicyDecision` / `PolicyAction` | Outcome vocabulary to **reference** |
| `ContinuationEvidenceRefs` / `QuoteAcceptanceEvidence` | Governance artifacts to **reference by id** |
| Trace / runtime events / provenance | Observability spine — complementary, not a proof profile |

Without an explicit descriptive contract, consumers risk embedding transport payloads, duplicating policy/continuation models, or treating Tier-2 composition as signing/persistence ownership.

## Decision

1. Introduce a minimal reusable `GovernedProofProfile` (+ `GovernanceEvidenceRef`) in `intergrax.contracts.governed_proof`.
2. The profile is **descriptive only** — it never authorizes, resumes, evaluates policy, signs, hashes, stores, or publishes.
3. Record `PolicyAction` and policy rule/reason strings; do not recompute decisions or embed full nested policy/HITL objects.
4. Reference governance evidence by kind + id (External Work: `quote_acceptance_evidence` › `acceptance_id`).
5. Preserve existing `task_id`, `run_id`, `correlation_id`, and `idempotency_key` — never mint new identity.
6. External Work is the first consumer; Tier-2 may compose the profile after a meaningful side effect succeeds under policy ALLOW.

Rejected: reusing `ProofReceipt` as the GEC-6 deliverable; embedding provider SDK/HTTP payloads; Tier-2 receipt stores or cryptographic attestation.

## Consequences

### Positive

- Clear separation: proof profile (description) vs ProofReceipt (later persistence)
- Reuses policy / continuation vocabulary without duplication
- Provider-neutral and reusable beyond External Work

### Negative

- Persistence, signatures, and audit storage remain later platform work
- Product surfaces must join refs to stored artifacts themselves

## Compliance

- No persistence, signing, hashing, or receipt generation in Tier-2
- No transport / partner SDK fields on the profile
- Tier boundaries preserved

## Implementation notes

- Contract: `intergrax/contracts/governed_proof.py`
- Consumer: `agents/external_contractor_adapter/external_work_adapter.py`
- Verify: `uv run pytest tests/unit/contracts/test_governed_proof.py agents/external_contractor_adapter/tests -q`
