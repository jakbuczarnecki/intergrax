# ERL-QUAL-004 — Platform Capability Usage Map

**Qualification ID:** ERL-QUAL-004  
**Scenario slug:** `enterprise_payment_uncertainty_recovery`  
**Document role:** Enterprise architecture view of how this scenario exercises existing Integrax platform capabilities—business problem first, platform mapping second.

**Related artifacts:** [Scenario Specification](../SCENARIO_SPEC.md) · [Proof Architecture Design](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md) · [Enterprise Reliability Layer](../../../../docs/project/architecture/ENTERPRISE_RELIABILITY_LAYER.md)

---

## 1. Business Problem

Enterprise transaction processing depends on **external payment systems** whose outcomes are not always visible at the moment of request completion.

After a high-value customer order, the commerce application initiates payment capture with an external provider. In production, several conditions are routine rather than exceptional:

- **External payment uncertainty** — the provider may have accepted or completed a charge while the integration receives no definitive success or failure signal (timeouts, partial responses, network partition mid-request).
- **Incomplete information** — internal order, inventory, and payment-correlation state cannot be updated honestly until authoritative external truth is known; both “paid” and “not paid” remain plausible.
- **Need for reliable continuation** — downstream steps (duplicate capture, fulfillment, inventory release) are materially risky if executed on guesswork; the business requires pause, verification, and governed resume or compensation.
- **Audit requirements** — regulators, finance, and operations must reconstruct what the organization knew, when reconciliation ran, what decisions were taken, and why risky automation did or did not proceed—without conflating silence with failure or success.

This scenario qualifies Integrax behavior around that problem. It does **not** define a new payment product; it demonstrates that a representative enterprise workflow can remain safe under genuine external ambiguity.

---

## 2. Scenario End-to-End Flow

The proof narrative follows the same spine for Variants A, B, and C; terminal branches differ after reconciliation establishes (or fails to establish) external truth.

```text
Payment request
        ↓
External UNKNOWN
        ↓
Reliability case
        ↓
Reconciliation
        ↓
Evidence
        ↓
Evaluation
        ↓
Resolution
        ↓
Governance
        ↓
Recovery
```

| Stage | Business meaning | What must hold |
| --- | --- | --- |
| **Payment request** | Order workflow invokes payment capture through a declared external-effect contract. | Capture is a real side effect; outcome may be ambiguous on the wire. |
| **External UNKNOWN** | Integration cannot classify the capture as confirmed success or failure. | Ambiguity is modeled as uncertainty, not collapsed into a generic error. |
| **Reliability case** | Platform admits managed uncertainty and correlates the effect to a reliability case. | Risky continuation is gated; case lifecycle is explicit. |
| **Reconciliation** | System-of-record probe when reachable (lab: external reality / PostgreSQL SoR). | Truth is discovered before material retry—not assumed from timeout. |
| **Evidence** | Classification and reconciliation facts are materialized for audit (`ExternalEffectEvidence`, scenario payment bundles). | Facts are durable on the production observability path, not proof-only logs. |
| **Evaluation** | Payment-specific quality of evidence is judged before resolution (PSP confirmation, settlement posture, probe alignment). | Enterprise rules stay in scenario plugins; platform returns bounded evaluation outcomes. |
| **Resolution** | Truth maps to continue, compensate, or escalate paths (`ResolutionDecision`). | Decision is separated from execution retry. |
| **Governance** | Enterprise policy gates risky continuation (thresholds, Variant C escalation). | Governance does not execute payments or self-approve HITL. |
| **Recovery** | Reliability case transitions and lifecycle handoff to execution (pause, resume, containment). | No parallel orchestration engine; UER applies lifecycle mutations. |

Variant terminals (fixture-controlled):

- **A** — Reconciliation confirms paid → safe continuation without duplicate capture.
- **B** — Reconciliation confirms failed → controlled recovery / compensation alignment.
- **C** — Reconciliation unavailable or inconclusive → governance escalation; **UNRESOLVED** or governed containment.

---

## 3. Platform Capability Mapping

Capabilities listed here are **existing** ERL and execution-runtime surfaces documented in [`ENTERPRISE_RELIABILITY_LAYER.md`](../../../../docs/project/architecture/ENTERPRISE_RELIABILITY_LAYER.md) and exercised or prepared in this scenario package. The scenario supplies payment semantics via plugins and adapters; the platform owns orchestration and contracts.

| Capability | Business need | Platform responsibility | Scenario responsibility |
| --- | --- | --- | --- |
| **Reliability** (uncertainty + case lifecycle) | Stop treating “no answer” as terminal failure; own the journey from UNKNOWN to closed case. | UNKNOWN admission, reliability case lifecycle coordination, gating risky steps, journal-friendly case transitions. | Declare payment external-effect contracts; correlate order/payment identifiers; application workflow respects pause and handoff. |
| **Reconciliation** | Discover authoritative payment truth without blind capture retry. | Reconciliation gateway pattern, probe execution envelope, plugin registry for `ReconciliationStrategy` / probe executor. | **Payment Reconciliation Plugin** (`ScenarioExternalRealityReconciliationPlugin`): SoR lookup via `ExternalRealityLookupPort` (in-memory or PostgreSQL lab adapter). |
| **Evidence** | Persist what was observed for audit and downstream decisions. | `ExternalEffectEvidence` shape, evidence persistence on observability spine, correlation identifiers. | Payment reconciliation evidence model, dataset-backed materialization, mapping from probe results to platform evidence. |
| **Evidence evaluation** | Judge whether reconciliation output is sufficient or conflicting before resolution. | `evaluate_external_effect_evidence`, `EvidenceEvaluatorStrategy`, bounded outcomes (`READY_FOR_DECISION`, `INSUFFICIENT_EVIDENCE`, etc.). | **Payment Evidence Evaluator Plugin** (`PaymentEvidenceEvaluatorPlugin`): PSP/settlement interpretation via `PaymentReconciliationEvidenceLookupPort` (explicit composition, not platform business rules). |
| **Resolution** | Map established truth to generic continue / compensate / escalate decisions. | `ResolutionStrategy` SPI, `ResolutionDecision` actions, orchestration without executing domain side effects. | **Payment Resolution Strategy Plugin** (`PaymentResolutionStrategyPlugin`): maps payment truth to platform resolution actions. |
| **Governance** | Enforce enterprise thresholds before risky continuation (especially when truth is missing). | `GovernanceStrategy` SPI, `GovernanceDecision` (`allow`, `deny`, `approval_required`). | **Payment Governance Policy Plugin** (`PaymentGovernancePolicyPlugin`): amount thresholds, currency policy, business context lookup. |
| **Recovery** | Resume or contain execution after decisions—without duplicating the runtime. | Recovery lifecycle coordination, `RecoveryStrategy` SPI (platform), handoff to `ExecutionLifecyclePort` / UER. | Variant B/C recovery **semantics** (compensation alignment, escalation narrative); **Payment Recovery Plugin** on `RecoveryStrategy` is **future**—not yet registered in scenario wiring. |
| **Execution runtime** | Execute workflow steps, pause/resume, emit traces and lifecycle facts. | Unified Execution Runtime (UER): run/attempt identity, pause/resume/cancel, checkpoint hooks, tool invocation path. | Scenario application (`application/`) implements order → payment workflow; external payment lab (`external_payment/`) behind ports; proof harness projects—not replaces—runtime events. |

---

## 4. Cross-Cutting Capabilities

These neighbors are not payment-specific; they explain **why** the scenario requires production-path instrumentation.

| Capability | Why needed | Problem solved |
| --- | --- | --- |
| **Tracing** | Payment capture and reconciliation must link to execution identity (`TraceEvent`, `ToolCallTrace`). | Operators and auditors can follow which run attempted capture and which probe answered—without proof-only reconstructions. |
| **Diagnostics** | Reconciliation exhaustion, governance blocks, and compensation failures need structured, redaction-safe signals. | Reduces war-room guesswork; supports Variant C “why escalation” without leaking PAN/cardholder data (`DiagnosticPayload.redact`). |
| **Observability** | UNKNOWN admission, reconciliation lifecycle, resolution, and governance must appear on the runtime journal / event spine. | Satisfies Application Observability Test in [Scenario Specification § A](../SCENARIO_SPEC.md#a-scenario)—proof is projection, not the sole recorder. |
| **Audit evidence** | Qualification and enterprise review require durable records of uncertainty and resolution. | Demonstrates evidence-based decisions and post-incident replay; aligns with ERL “Audit Evidence” and proof themes in [Proof Architecture § 8](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#8-evidence-model). |

---

## 5. Plugin Extension Model

Integrax extends enterprise behavior through **contracts and SPIs** in `intergrax/contracts/enterprise_reliability/`. The scenario implements enterprise payment logic in `erl_integration/` without importing payment semantics into `intergrax/`.

### Platform

- Lifecycle orchestration for reconciliation, evidence evaluation, resolution, governance, and recovery handoff.
- Neutral contracts: `ExternalEffectEvidence`, probe requests/results, `ResolutionDecision`, `GovernanceDecision`, plugin descriptors and registry (`EnterpriseReliabilityPluginRegistry`).
- Generic decision outcomes and execution isolation (ERL does not execute captures or shipments).

### Scenario

- Payment semantics, business rules, external adapters, and lab fixtures.
- Vendor-neutral dataset (`dataset/`) and provisioning (`contracts/provisioning/`, PostgreSQL materialization).
- Enterprise plugin implementations registered at lab composition time.

### Current plugins (implemented in scenario)

| Plugin (document name) | SPI / entry | Implementation |
| --- | --- | --- |
| Payment Reconciliation Plugin | `ReconciliationStrategy` + probe executor | `erl_integration/plugins/external_reality_reconciliation.py` |
| Payment Evidence Evaluator Plugin | `EvidenceEvaluatorStrategy` | `erl_integration/plugins/payment_evidence_evaluator.py` (wired via explicit `evaluator_strategy` in composition—see [plugin doc](ERL_QUAL_004_PAYMENT_EVIDENCE_EVALUATOR_PLUGIN.md)) |
| Payment Resolution Strategy Plugin | `ResolutionStrategy` | `erl_integration/plugins/payment_resolution_strategy.py` |
| Payment Governance Policy Plugin | `GovernanceStrategy` | `erl_integration/plugins/payment_governance_policy.py` |

Registration reference: `erl_integration/wiring.py` (`register_scenario_reconciliation_plugins`).

### Future

| Plugin | SPI | Intent |
| --- | --- | --- |
| **Payment Recovery Plugin** | `RecoveryStrategy` | Encode scenario-specific recovery/compensation choices after resolution and governance, returning platform `RecoveryDecision` without forking UER. |

---

## 6. Ownership Boundary

### Platform owns

- Reliability **lifecycle** and case state machine (coordination, not business workflow).
- **Contracts** and plugin SPIs for reconciliation, evidence, evaluation, resolution, governance, recovery.
- **Orchestration** of ERL capabilities and handoff to execution lifecycle ports.
- **Generic decisions** (bounded enums and payloads)—not `PAYMENT_APPROVED` domain labels in core.
- **Observability** persistence boundaries and normalized runtime events.

### Scenario owns

- **Payment semantics** (capture, settlement, order correlation, variant fixtures A/B/C).
- **Business rules** in plugins and policies (thresholds, evidence cross-checks, resolution mapping).
- **External adapters** (simulated provider, PostgreSQL SoR, in-memory lab lookups).
- **Policies** and enterprise governance context materialized from dataset/provisioning.

**Proof harness** (future `proof/`) projects canonical events and asserts invariants; it does not own payment truth or replace application workflow—see [Scenario Specification APPLICATION vs PROOF](../SCENARIO_SPEC.md#application-vs-proof-harness).

---

## 7. Enterprise Validation

Use this checklist when reviewing whether ERL-QUAL-004 reads as a **real enterprise architecture** rather than a platform demo script.

The scenario demonstrates:

- [ ] **Realistic business problem** — external payment uncertainty after capture, with revenue and audit stakes.
- [ ] **Vendor-neutral architecture** — contracts and dataset-first SoR; no mandated PSP brand in platform core.
- [ ] **Real infrastructure usage** — lab PostgreSQL and Docker foundation for materialized truth ([PostgreSQL Infrastructure](ERL_QUAL_004_POSTGRESQL_INFRASTRUCTURE.md)); deferred vector/streaming per vendor infra decision.
- [ ] **Plugin extensibility** — reconciliation, evidence evaluation, resolution, governance implemented as scenario plugins on existing SPIs.
- [ ] **Evidence-based decisions** — reconciliation and evaluation precede resolution; governance gates continuation.
- [ ] **Auditability** — observability spine + proof projection themes (UNKNOWN, reconciliation, decisions, lifecycle).
- [ ] **Separation of platform and business logic** — `intergrax/` free of payment product rules; scenario `erl_integration/` owns enterprise interpretation.

---

## Document consistency

This map aligns with:

- ERL hub capabilities and boundaries in [`ENTERPRISE_RELIABILITY_LAYER.md`](../../../../docs/project/architecture/ENTERPRISE_RELIABILITY_LAYER.md).
- End-to-end flow and evidence themes in [ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md).
- Observability contract in [SCENARIO_SPEC.md § A](../SCENARIO_SPEC.md#observability--explainability--diagnostics-contract).
- Per-plugin ownership docs under `docs/ERL_QUAL_004_PAYMENT_*_PLUGIN.md`.

No additional platform capabilities are asserted beyond what is documented or implemented in this repository path.
