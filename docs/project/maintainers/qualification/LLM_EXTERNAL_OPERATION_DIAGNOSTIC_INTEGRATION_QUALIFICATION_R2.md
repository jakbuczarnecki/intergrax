# LLM External Operation — Diagnostic Integration Qualification R2

| Matrix | Module |
|--------|--------|
| Identity, admission, failure, diagnostic, boundary, isolation, retry, tenant, secrets, predictive | `tests/unit/runtime/external_operations/test_llm_external_operation_diagnostic_integration_r2.py` |

## Enterprise gates

| Area | Gate |
|------|------|
| Architecture | Single execution authority; single diagnostic engine; no local Problem store |
| Runtime | Durable `EXTERNAL_OPERATION_FAILED`; identity propagation; provider isolation |
| Security | Tenant isolation; secret-safe evidence |
| Diagnostics | FailureBoundary at external execution; findings reach assessment |

## Run

```powershell
uv run pytest tests/unit/runtime/external_operations/test_llm_external_operation_diagnostic_integration_r2.py -q
```
