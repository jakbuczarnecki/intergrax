# LLM External Operation Admission — Qualification R1

| Field | Value |
|-------|-------|
| **Status** | Qualified (unit matrix) |
| **Date** | 2026-09-11 |
| **Architecture** | [`LLM_EXTERNAL_OPERATION_ADMISSION_ARCHITECTURE_R1.md`](../architecture/LLM_EXTERNAL_OPERATION_ADMISSION_ARCHITECTURE_R1.md) |

---

## Test matrix

| Area | Test | Module |
|------|------|--------|
| Contracts | `test_external_operation_contract_validation` | `tests/unit/contracts/test_external_operation_contract_validation.py` |
| Admission | `test_operation_requires_admission_before_execution` | same |
| Governance | `test_denied_operation_never_reaches_provider` | `tests/unit/runtime/external_operations/test_llm_external_operation_admission_hardening_r1.py` |
| Approval | `test_human_approval_required_blocks_execution` | same |
| Provider isolation | `test_provider_failure_is_contained` | same |
| Audit | `test_operation_attempt_is_reconstructable` | same |
| Diagnostics | `test_failed_external_operation_creates_evidence` | same |
| Security | `test_secret_not_present_in_audit` | same |
| Tenant isolation | `test_operation_attempt_cannot_cross_tenant` | same |

---

## Definition of Done

| Area | R1 |
|------|-----|
| Intent contract | PASS |
| Admission boundary | PASS |
| Provider SPI | PASS |
| Attempt lifecycle | PASS |
| Audit chain | PASS |
| Governance integration | PASS |
| Diagnostic evidence | PASS |
| Security isolation | PASS |
| Tenant isolation | PASS |
| Regression tests | PASS (scoped pytest) |
| Documentation | PASS |

---

## Run

```powershell
uv run pytest tests/unit/contracts/test_external_operation_contract_validation.py tests/unit/runtime/external_operations/test_llm_external_operation_admission_hardening_r1.py -q
```
