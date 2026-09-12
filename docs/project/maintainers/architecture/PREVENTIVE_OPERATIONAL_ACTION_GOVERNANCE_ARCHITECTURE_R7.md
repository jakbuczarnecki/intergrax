# Preventive Operational Action Governance (R7)

**Task:** `PREVENTIVE-OPERATIONAL-ACTION-GOVERNANCE-R7`

**Status:** Architecture frozen

**Invariant:** `PROPOSAL_IS_NOT_EXECUTION` — Preventive Intelligence proposes; Governance authorizes; External Operation path executes.

---

## 1. Authority model

| Layer | Role |
| ----- | ---- |
| Preventive Intelligence | Risk, recommendation, evidence — **no execution** |
| `PreventiveActionProposal` | WHAT SHOULD HAPPEN |
| Decision / Governance | Permission |
| `PreventiveActionAdmissionGate` | ALLOW / DENY / REQUIRES_APPROVAL |
| `ExternalOperationAdmission` | Same verdicts on translated intent |
| Execution Runtime | Single spine via `ExternalOperationExecutionGate` |

Predictive or proposal **confidence is never authorization**.

---

## 2. Lifecycle

`PreventiveActionLifecycleState`:

```text
PROPOSED → WAITING_APPROVAL | APPROVED | REJECTED
APPROVED → EXECUTED | FAILED
```

`REQUIRES_APPROVAL` is a first-class path — no silent auto-execution.

---

## 3. Contracts (`intergrax/contracts/preventive/actions/`)

- `PreventiveActionProposal` — no `execute()` surface
- `PreventiveActionType` — namespaced, versioned catalog
- `PreventiveActionProvider` — translate proposal → `ExternalOperationIntent` only
- `PreventiveActionAdmissionGate` — admission port
- `PreventiveActionAuditRecord` — operator reconstructability
- `PreventiveActionOutcomeEvaluation` — learning loop input

---

## 4. Runtime (`intergrax/runtime/prevention/actions/`)

- `PolicyPreventiveActionAdmissionGate` — fail-closed policy
- `resolve_preventive_governance_chain` — Decision → Governance → preventive + external admission
- `GovernedPreventiveActionOrchestrator` — proposal, translate, audit; reuses external gate
- `PreventiveActionOutcomeEngine` — feeds predictive analyzer quality store
- `project_preventive_action_history` — diagnostic read attachment

---

## 5. Execution reuse

Forbidden: `PreventiveExecutionEngine`, synthetic `RuntimeEvent`, local retry authority.

Required: `ExternalOperationIntent`, `ExternalOperationAdmission`, `ExternalOperationExecutionGate`, platform execution identity, diagnostic failure recorder.

---

## 6. Diagnostic integration

Failed preventive actions emit real `EXTERNAL_OPERATION_FAILED` runtime events. Central Diagnostic Engine assesses via existing reconstruction — no `PreventiveProblemEngine`.

`DiagnosticInvestigationView.preventive_action_history` exposes Risk → Recommendation → Approval → Execution → Outcome trail.

---

## 7. Security

- Tenant isolation on admission context
- `assert_no_secrets_in_preventive_action_audit` (delegates to external-operation secret patterns)
- Provider `ProviderPayloadBounds` (timeout, size, retry ceiling)

---

## 8. Architectural prohibitions

- Preventive layer executing actions autonomously
- Second execution engine or Problem Store or Diagnostic Engine
- Provider bypassing governance
- Confidence-based auto-execution
