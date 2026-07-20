# ImpeachmentRight — full validation (Execution Evidence)

**Date:** 2026-07-20  
**Verdict:** `FULLY_DEMONSTRABLE`  
**Upstream GEC canon:** [`docs/platform/governed_external_execution.md`](../platform/governed_external_execution.md)  
**Capability canon:** [`docs/platform/execution_evidence_and_host_attestation.md`](../platform/execution_evidence_and_host_attestation.md)  
**Readiness (lifecycle through proof):** [`impeachmentright_validation_readiness.md`](impeachmentright_validation_readiness.md)

---

## Closure questions

| # | Question | Answer |
|---|----------|--------|
| 1 | Is `governed_execution_boundary_event.v1` produced by the host? | **Yes** — `governed_contractor_application.host.execution_evidence` |
| 2 | Is it host-signed? | **Yes** — Ed25519 `HostAttestor` (DI; test attestor offline) |
| 3 | Is there one portable attested export? | **Yes** — `execution_evidence.proof_receipt.v1` |
| 4 | Does it bind policy decision, runtime policy bundle identity/digest, provider invocation reference, task/run identity and proof? | **Yes** — event sections + digests (not full pack body embed) |
| 5 | Can it be verified offline? | **Yes** — `verify_proof_receipt` (no provider network) |
| 6 | What remains explicitly outside scope? | DocumentStore/public registry persistence, replay, live/paid providers, wallets, remote KMS/HSM, default HTTP product UX packaging |

---

## Three original partner gaps

| Gap | Status | Evidence |
|-----|--------|----------|
| Host-signed execution boundary event | **Closed** | Host composes + signs governed EBE |
| One attested export joining policy / invocation / identity / proof | **Closed** | Portable `ProofReceipt` |
| Explicit runtime policy bundle identity and digest | **Closed** | `ImmutableRuntimePolicyBundle` + `PolicyDecision` refs |

---

## Demonstrable lifecycle

```text
CREATE_EXTERNAL_WORK
  → policy evaluation (decision bound to immutable pack identity/digest) → ALLOW → provider create
  → GovernedProofProfile → governed ExecutionBoundaryEvent → HostAttestation → verified ProofReceipt

QUOTE continuation → QuoteAcceptanceEvidence → ACCEPT_QUOTE
  → policy evaluation (decision bound to immutable pack identity/digest) → ALLOW → provider accept
  → GovernedProofProfile → governed ExecutionBoundaryEvent → HostAttestation → verified ProofReceipt
```

**Wording note:** demo evaluator stamps `ImmutableRuntimePolicyBundle` identity onto
`PolicyDecision`; it is not a claim that production `RuntimePolicyBundle` / live
policy engines already consume that immutable pack as their sole rule source.

**Command:**

```bash
uv run pytest applications/governed_contractor_application/tests/host/test_partner_attested_execution_demo.py -q
```

Supporting:

```bash
uv run pytest \
  tests/unit/contracts/test_runtime_policy_bundle.py \
  tests/unit/execution_evidence \
  tests/unit/contracts/test_governed_proof.py \
  tests/unit/runtime/policy/test_meaningful_side_effect_policy.py \
  agents/external_contractor_adapter/tests \
  applications/governed_contractor_application/tests \
  -q
```

---

## Explicit non-claims

- Harness tool/step `execution_boundary_event.v1` (BoundaryAttest) ≠ governed `governed_execution_boundary_event.v1`
- DocumentStore `intergrax.proof_receipt.v1` ≠ portable `execution_evidence.proof_receipt.v1`
- Receipt / verification never authorize future execution
- No production key custody; test attestor is local Ed25519 only
