# LLM External Operation Admission — Architecture R1

| Field | Value |
|-------|-------|
| **Status** | Accepted (R1) |
| **Date** | 2026-09-11 |
| **Contract module** | `intergrax/contracts/external_operations/` |
| **Runtime module** | `intergrax/runtime/external_operations/admission/` |

---

## Principle

LLM **suggests**; the platform **admits**; providers **execute** only admitted attempts.

Forbidden: `execute(command_string)` without intent, admission, and audit.

---

## Boundary model

```text
LLM proposal
    → Decision System (reference only)
    → Governance
    → ExternalOperationAdmission (ALLOW | DENY | REQUIRES_APPROVAL)
    → ExternalOperationAttempt lifecycle
    → ExternalOperationProvider SPI
    → Real provider I/O
```

---

## Contracts

| Symbol | Role |
|--------|------|
| `ExternalOperationIntent` | Who / why / what / tenant scope |
| `ExternalOperationAdmission` | Admission port (no I/O) |
| `OperationAdmissionDecision` | Verdict + operator reason |
| `ExternalOperationAttempt` | CREATED → ADMITTED → EXECUTING → terminal |
| `ExternalOperationAuditRecord` | Reconstructable audit chain |
| `ExternalOperationProvider` | Provider SPI (payload bounds, risk profile) |
| `ExternalOperationEvidence` | Typed diagnostic evidence |

---

## Security

- Tenant isolation enforced at admission (`intent.tenant_id` vs context).
- Secrets redacted / rejected in audit and evidence (`sanitize_external_operation_text`, `assert_no_secrets_in_audit_payload`).
- Providers cannot bypass admission, own audit stores, or hidden retry loops (`max_retries` on `ProviderPayloadBounds`).

---

## LLM adapter integration

`LLMAdapter.bind_external_operation_admission_gate` wires `ExternalOperationExecutionGate`.
When bound, `_admit_llm_provider_intent` runs before W4-C physical lifecycle (`LlmExternalOperationAttempt`).

---

## Diagnostics

Failures surface as `ExternalOperationFailed` + `ExternalOperationEvidence` (not exception-only), consumable by the Central Diagnostic Engine via existing runtime event / failure evidence paths (DIAG R2).
