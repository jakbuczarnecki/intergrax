# ImpeachmentRight — Governed External Execution validation readiness

**Date:** 2026-07-20  
**Scope:** Partner-facing readiness after GEC-0…GEC-6.1 architecture closure  
**Platform canon:** [`docs/platform/governed_external_execution.md`](../platform/governed_external_execution.md)  
**Related:** [`execution_evidence_and_host_attestation.md`](../platform/execution_evidence_and_host_attestation.md) · [`impeachmentright_full_validation.md`](impeachmentright_full_validation.md)

This document answers whether Intergrax can honestly demonstrate the maintainer’s five observable compatibility areas using repository-backed evidence. It does **not** restate platform ownership matrices; those live in the platform reference above.

---

## Executive verdict

**READY_NOW**

Platform + Tier-2 adapter implement the full governed lifecycle through mandatory `GovernedProofProfile`. A single offline demonstration path exists and passes. The host application mounts the agent and DI slots but does **not** yet expose a product HTTP/CLI orchestration of the quote→accept flow — that is a host product / demonstration packaging gap, not a platform capability gap.

Host Attestation / governed Execution Boundary Events / attested export are implemented as the downstream capability [`execution_evidence_and_host_attestation.md`](../platform/execution_evidence_and_host_attestation.md). Full closure verdict: [`impeachmentright_full_validation.md`](impeachmentright_full_validation.md).

---

## Platform readiness

Lifecycle verified in code and tests:

```text
intent → provider-neutral request → real task_id / run_id
  → meaningful side-effect policy → optional governed continuation
  → explicit ALLOW → provider-bound execution → mandatory GovernedProofProfile
```

| Invariant | Implementation evidence | Tests |
|-----------|-------------------------|-------|
| Policy precedes every meaningful provider side effect | `ExternalWorkAdapter._authorize_side_effect` before `create_work` / `submit_quote_acceptance` / `cancel_work` | `agents/external_contractor_adapter/tests/test_meaningful_side_effect_policy.py` (ordering + DENY) |
| Missing execution identity fails closed | Empty/`None` `run_id` → `side_effect_identity_missing`; no provider call | same + `test_governed_proof_profile.py` |
| Evidence does not authorize execution | `QuoteAcceptanceEvidence` forwarded only after separate ACCEPT_QUOTE policy ALLOW | `test_meaningful_side_effect_policy.py` (evidence + DENY) |
| Continuation does not execute work | `surface_continuation_blocker` / `with_continuation_surface` compose `GovernedContinuationRequest` only | `test_governed_continuation_composition.py` |
| Providers do not evaluate policy | `ExternalWorkIntegration` Protocol transmits evidence; fake has no policy API | `intergrax/integrations/contracts/external_work.py` + fake |
| Successful governed side effects always produce proof | `_compose_proof_after_authorized` mandatory after ALLOW + success | `test_governed_proof_profile.py` |
| Correlation / idempotency preserved | Forwarded on create/accept/cancel; not reminted by policy | `test_meaningful_side_effect_policy.py` + proof tests |
| Tier-2 no persistence / signing / publication | AST + source guards; compose-only | `test_tier2_performs_no_persistence_signing_or_receipt_generation` |

Platform unit contracts:

- `tests/unit/contracts/test_governed_proof.py`
- `tests/unit/runtime/policy/test_meaningful_side_effect_policy.py`
- `tests/unit/contracts/test_governed_continuation.py` (related)

---

## Agent readiness (`external_contractor_adapter`)

**Classification: production architecture for mapping + governance composition; demonstration architecture for partner-facing host exposure.**

Not scaffold-only: CREATE / ACCEPT_QUOTE / CANCEL are real Protocol-gated paths with policy + mandatory proof.

| Action | Side-effect class | Policy composition | Run identity | Provider order | Proof | Neutrality |
|--------|-------------------|--------------------|--------------|----------------|-------|------------|
| `CREATE_EXTERNAL_WORK` | meaningful (`create_work`) | `MeaningfulSideEffectRequest` before create | required Nexus `run_id` | policy → `create_work` → proof | yes (no governance evidence) | Protocol-only DI |
| `ACCEPT_QUOTE` | meaningful (`submit_quote_acceptance`) | policy before accept; evidence ≠ allow | from correlation / context | policy → accept → proof + evidence refs | yes (`QUOTE` + `GovernanceEvidenceRef`) | no transport / host imports |
| `CANCEL_EXTERNAL_WORK` | meaningful (`cancel_work`) | policy before cancel | required | policy → cancel → proof | yes (no acceptance evidence) | same |

Observational reads (`get_quote`, timeline, …) may surface QUOTE continuation; they are not mutation gates.

Ownership confirmed: no HITL decisions, no policy rule packs, no signing/persistence, no `applications.*` imports.

---

## Host readiness (`governed_contractor_application`)

| Question | Finding |
|----------|---------|
| Can host inject `ExternalWorkIntegration` + evaluator? | **Yes** — `host/agent_builders.py` via `settings.external_work_integration` / `settings.meaningful_side_effect_policy` |
| Default settings / factory wire a demo fake? | **No** — optional attrs; default run reports missing integration |
| HTTP `/v1/governed_contractor/run` complete quote→accept→proof flow? | **No** — message/capability only; no external-work metadata orchestration |
| CLI / deterministic app entry for full flow? | **No** |
| Existing host tests cover governed proof? | **No** — smoke: health / agents / hello run only |

**Host classification:** DI-ready scaffold; **not** a complete product demonstration surface for the five-point flow. Full scenario is executable today via Tier-2 construction (demo test below), not via the default host API.

---

## Reproducible demonstration

**Canonical command:**

```bash
uv run pytest agents/external_contractor_adapter/tests/test_partner_validation_demo.py -q
```

**What it shows (deterministic fake, offline):**

1. `CREATE_EXTERNAL_WORK` under policy ALLOW → create proof  
2. QUOTE `GovernedContinuationRequest` with distinct `task_id` / `run_id`  
3. `QuoteAcceptanceEvidence` (HITL/interrupt/policy refs)  
4. `ACCEPT_QUOTE` policy ALLOW → provider accept  
5. Final `GovernedProofProfile` with: `task_id`, `run_id`, `provider_id`, `action`, `policy_action` / `policy_rule_id`, `governance_evidence`, `correlation_id`, `idempotency_key`

Supporting gate tests: `test_governed_proof_profile.py`, `test_meaningful_side_effect_policy.py`, `test_governed_continuation_composition.py`.

---

## Five-point compatibility matrix

| External assessment area | Intergrax implementation | Evidence in repository | Status |
|--------------------------|--------------------------|------------------------|--------|
| 1. Agent / application / runtime boundary | Tier-2 adapter maps only; host injects Protocol + policy; Nexus owns run identity / interrupt | Platform doc · `external_work_adapter.py` · `agent_builders.py` · tier import rules | **Demonstrable now** (via Tier-2 demo; host boundary wiring demonstrable as DI, not full product UX) |
| 2. Policy before meaningful action | `MeaningfulSideEffectRequest` + evaluator before CREATE / ACCEPT_QUOTE / CANCEL | `test_meaningful_side_effect_policy.py` · `test_partner_validation_demo.py` | **Demonstrable now** |
| 3. Human checkpoint / escalation | Governed Continuation (`ContinuationReason.QUOTE`) over Nexus interrupt composition; Tier-2 surfaces/forwards only | `test_governed_continuation_composition.py` · demo continuation assert | **Demonstrable now** (composition + evidence forward; product HITL UX is host gap) |
| 4. Evidence after execution | Mandatory descriptive `GovernedProofProfile` + host-attested governed EBE / portable ProofReceipt | `test_governed_proof_profile.py` · `test_partner_attested_execution_demo.py` | **Demonstrable now** (descriptive proof + host-signed attested export) |
| 5. Controlled tools | Provider-bound mutations only behind Protocol after ALLOW; observational reads ungated as mutations | `PROVIDER_METHOD_SIDE_EFFECT_CLASS` · Protocol + fake | **Demonstrable now** (Protocol/tool boundary; not a partner SDK tool catalog) |

---

## Gap classification

### A. Documentation gap

- Partner-facing readiness lived only in vertical plans until this file.
- Platform canon already exists; this file links rather than duplicates.

### B. Demonstration gap

- Full flow was covered by multiple unit tests but lacked one named partner demo until `test_partner_validation_demo.py`.
- ~~Host HTTP/CLI still does not package the flow as a single operator command.~~ → **closed** via
  `uv run intergrax demo governed-contractor --offline` (see platform completion audit).

### C. Host product gap

- No default injected fake/evaluator in production settings.
- `/run` does not accept/orchestrate external-work metadata, quote acceptance, or proof presentation.
- Product HITL UX and Intergrax-authored policy *bundles* for the vertical are deferred (GEC-7+).

### D. Platform capability gap (not a GEC defect)

- ~~Host-signed governed Execution Boundary Event~~ → **closed** via Execution Evidence
- ~~Single cryptographically attested export~~ → **closed** (`execution_evidence.proof_receipt.v1`)
- ~~Runtime policy bundle identity/digest~~ → **closed** (`ImmutableRuntimePolicyBundle`)
- ~~Bundle-backed evaluation + pack body offline verify~~ → **closed** (`RuntimePolicyBundleEvaluator`, `policy_bundle_artifact`)
- ~~First-class provider invocation + atomic GER + host orchestrator/recovery~~ → **closed** (PC-3…PC-7)
- Still out of scope: DocumentStore/public registry persistence, replay, paid providers, wallets, remote KMS/HSM
- Contractor-local policy remaining on the partner side until they consume Intergrax runtime policy packs (partner adoption work + host packs)

---

## What can be claimed now

- Intergrax has a **provider-neutral** External Work contract + `ExternalWorkIntegration` boundary.
- A Tier-2 adapter maps CREATE / quote continuation / ACCEPT_QUOTE / CANCEL with **policy-before-mutation** and **mandatory `GovernedProofProfile`**.
- Real Nexus `task_id` / `run_id` are required and fail closed when missing.
- Quote acceptance evidence is **not** an allow token; policy re-evaluates before accept.
- Continuation surfaces interruption; it does not execute the side effect.
- Offline deterministic proof of the lifecycle is reproducible via the demo pytest **and**
  `uv run intergrax demo governed-contractor --offline`.
- Architecture is closed through GEC-6.1 per [`governed_external_execution.md`](../platform/governed_external_execution.md).

## What must not be claimed yet

- That harness tool/step BoundaryAttest EBE is the same artefact as governed `governed_execution_boundary_event.v1` (siblings).
- Production partner transport (A2A/HTTP), paid tasks, wallets, or live provider integration.
- That the default governed_contractor HTTP API already demos the full quote→accept→attested-receipt product flow (CLI + pytest do; HTTP product UX still optional).
- That `GovernedProofProfile` itself is a ProofReceipt, audit log, or authorization mechanism.
- Production key custody / remote KMS/HSM (local Ed25519 test attestor only).
- That Intergrax has replaced contractor-local policy packs on the partner’s side.

---

## Recommended response strategy

1. **Lead with attested demo** — `test_partner_attested_execution_demo.py` + platform canon for Execution Evidence.
2. **Map the five points honestly** using the matrix above; separate descriptive `GovernedProofProfile` from host-signed portable receipt.
3. **Close their three gaps with precise naming** — governed sibling `governed_execution_boundary_event.v1` (not harness tool/step `execution_boundary_event.v1`); portable `execution_evidence.proof_receipt.v1`; decision bound to `ImmutableRuntimePolicyBundle` identity/digest (not a claim that partner contractor-local packs are already replaced).
4. **Invite next step:** live provider behind `ExternalWorkIntegration` (partner translation adapter) + wallet/payment evidence.
5. **Do not oversell** production KMS/HSM, receipt registry, replay, or wallets.

**Platform completion audit:** [`impeachmentright_platform_completion.md`](impeachmentright_platform_completion.md) — verdict `FULLY_PLATFORM_READY`.

---

## Verification

```bash
uv run pytest \
  tests/unit/contracts/test_governed_proof.py \
  tests/unit/runtime/policy/test_meaningful_side_effect_policy.py \
  agents/external_contractor_adapter/tests \
  applications/governed_contractor_application/tests \
  agents/external_contractor_adapter/tests/test_partner_validation_demo.py \
  -q
```
