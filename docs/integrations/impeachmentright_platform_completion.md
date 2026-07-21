# ImpeachmentRight — Platform Completion Audit

**Date:** 2026-07-21  
**Branch:** `development`  
**Issue:** [#190](https://github.com/jakbuczarnecki/intergrax/issues/190)  
**Verdict:** `FULLY_PLATFORM_READY`

Conscious non-defect qualification only:

```text
governed_execution_boundary_event.v1  ≠  harness tool/step execution_boundary_event.v1
```

---

## Scope of this audit

Platform-side readiness for a provider-neutral contractor integration such as
`@impeachmentright`. Does **not** implement partner A2A transport, wallet,
payment processing, or paid smoke tasks.

---

## Public claims matrix

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | Complete provider-neutral governed external-work lifecycle | **SAFE TO CLAIM** | `GovernedExternalWorkOrchestrator`, filesystem/in-memory stores, `retry_attestation`, CLI `intergrax demo governed-contractor --offline` |
| 2 | Every meaningful mutation preceded by decision from concrete immutable pack | **SAFE TO CLAIM** | `RuntimePolicyBundleEvaluator` **interprets** `ImmutableRuntimePolicyBundle` rules directly → `EvaluatedPolicyDecision` (not post-hoc stamp) |
| 3 | Receipt verifies exact immutable pack used for the decision | **SAFE TO CLAIM** | `ProofReceipt.policy_bundle_artifact`; verifier recomputes digest, checks rule/action |
| 4 | Receipt ties decision to first-class invocation + task/run | **SAFE TO CLAIM** | `ProviderInvocation` / `ProviderInvocationOutcome` inside `GovernedExecutionResult`; strict attestation rejects `invocation:unknown` |
| 5 | Host recovers from attestation failure without repeating provider | **SAFE TO CLAIM** | Persisted GER + `retry_attestation`; restart test in `test_platform_completion.py` |
| 6 | Lifecycle reproducible offline via supported host command | **SAFE TO CLAIM** | CLI demo + separate `intergrax receipt verify` |

---

## Architecture (final)

```text
Tier-2 ExternalWorkAdapter
  ← RuntimePolicyBundleEvaluator (injected)
  → GovernedProofProfile

Host GovernedExternalWorkOrchestrator
  → ProviderInvocation (pre-call)
  → GovernedExecutionResult
  → governed EBE + policy_bundle_artifact
  → HostAttestor → ProofReceipt
  → stores / retry_attestation

Verifier (offline)
  → signature + event digest + bundle body recompute
```

Evaluation model (documented): **direct interpretation** of
`ImmutableRuntimePolicyBundle` rules (`match_action` + `effect`), not binding of
an independent live PolicyEngine snapshot after the fact.

---

## Commands executed

```bash
uv run pytest applications/governed_contractor_application/tests/host/test_partner_attested_execution_demo.py -q
uv run pytest applications/governed_contractor_application/tests/host/test_platform_completion.py \
  tests/unit/runtime/policy/test_runtime_policy_bundle_evaluator.py \
  tests/unit/contracts/test_governed_execution_result.py -q
uv run pytest applications/governed_contractor_application/tests/host/test_cli_verification_hardening.py -q

uv run intergrax demo governed-contractor \
  --offline \
  --store build/external_work_demo

uv run intergrax receipt verify \
  build/external_work_demo/export/accept_receipt.json \
  --store build/external_work_demo

uv run intergrax demo governed-contractor \
  --offline \
  --simulate-signing-failure \
  --store build/external_work_recovery_demo

uv run intergrax external-work retry-attestation \
  exec-offline-accept \
  --store build/external_work_recovery_demo

uv run intergrax receipt verify \
  build/external_work_recovery_demo/export/accept_receipt.json \
  --store build/external_work_recovery_demo
```

See also [`impeachmentright_cli_verification_hardening.md`](impeachmentright_cli_verification_hardening.md)
for the CLI/packaging hardening audit (explicit key sources, subprocess portability,
signer-failure recovery).

---

## Remaining partner-specific work

- Concrete adapter for partner A2A/REST endpoint (translation only)
- Partner wallet / payment unlock wiring behind evidence + policy
- Paid smoke task against live provider
- Partner-facing handoff packaging (not public reply until independent diff audit)

## Remaining production-hardening

- Remote KMS/HSM signer DI
- Public receipt registry / DocumentStore product path
- Distributed event store (explicitly out of scope)
- Default HTTP product surface beyond CLI demo (optional)

---

## Closing

```text
READY TO DRAFT PARTNER RESPONSE
```

Do not draft the public `@impeachmentright` reply in this change set — await
independent diff audit.
