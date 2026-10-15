# PROD-Q-R1-R7-Q-P3-A2-R1 — HTTP Caller Authentication Qualification Evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (implementation + focused qualification; globals unchanged) |
| **Parent ADR** | [`PROD_Q_R1_R7_Q_P3_A2_HTTP_CALLER_AUTHENTICATION_AND_TENANT_AUTHORIZATION_ADR.md`](PROD_Q_R1_R7_Q_P3_A2_HTTP_CALLER_AUTHENTICATION_AND_TENANT_AUTHORIZATION_ADR.md) — **PROPOSED** (not ratified to ACCEPTED) |
| **Architecture basis** | Alternative A — `fastapi_core` `AuthProvider` + `require_auth` + `context_only` identity |
| **Does not close** | P3-A · P3 · R7-Q · global FRZ rows |

## Authentication and authorization contract

| Layer | Owner | Mechanism |
|-------|-------|-----------|
| Authenticate HTTP caller | `AuthProvider` via `RequestContextMiddleware` | `require_auth` on strict GCA `/run` and `/agents` |
| Caller → serving tenant | `authorize_strict_production_serving_http_caller` | `AuthContext.tenant_id` == `StrictProductionServingExecutionTenantBinding.authoritative_execution_tenant_id` |
| Task tenant (body) | `GovernedContractorRunService._resolve_identity` | `context_only` — no body upgrade |
| Session tenant gate (P3-A1) | `StrictProductionServingTenantAuthorityTaskExecutionPort` | unchanged ordering |
| Admitted root principal | `bind_strict_serving_http_caller_principal` + `get_strict_serving_http_caller_principal` | per-request `RequestIdentity` from API key / JWT context |

## Strict ingress profile

- API keys required when `serving_runtime_handoff` is bound (`validate_strict_production_serving_http_ingress_composition`).
- `identity_source=context_only` required for strict serving HTTP.
- Alternate execution HTTP surfaces (task control, interaction intake, MCP) rejected at factory composition when strict handoff is present.

## Route inventory (strict GCA)

| Route | Strict behavior |
|-------|-----------------|
| `POST …/run` | `require_auth` + caller authorization + principal bind |
| `GET …/agents` | `require_auth` |
| Task control / interaction / MCP | Factory fail-closed if enabled with strict handoff |
| Health | Open (orchestrator readiness) |

## AUTH-01 — AUTH-10 verdicts

| ID | Verdict | Evidence |
|----|---------|----------|
| AUTH-01 | **PASS** | `test_p3a2_t01_anonymous_run_denied_before_execution` |
| AUTH-02 | **PASS** | `test_p3a2_t07_body_tenant_tamper_denied` + P3-A1 regression |
| AUTH-03 | **PASS** | `test_p3a2_t03_authenticated_matching_tenant_succeeds` |
| AUTH-04 | **PASS** | `test_p3a2_t02_invalid_api_key_denied` |
| AUTH-05 | **PASS** | `test_p3a2_t05_forged_tenant_header_without_auth_denied` |
| AUTH-06 | **PASS** | `test_p3a2_t06_admitted_principal_reflects_api_key_user_not_host_service` |
| AUTH-07 | **PASS** | `test_p3a2_t04_wrong_tenant_api_key_denied_before_admission` |
| AUTH-08 | **PARTIAL** | Session/handoff gates (D3/D4); no new HTTP-only stale-URL test in this package |
| AUTH-09 | **N/A** | Closed-world — no delegation model (ADR §13) |
| AUTH-10 | **PASS** | `test_p3a2_t10_caller_deny_skips_colocated_admission` + inner delegate spies |

## Regression (bounded run @ implementation)

```text
uv run pytest -p no:xdist -q \
  tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_a2_http_caller_auth.py \
  tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_a_cross_tenant_strict.py \
  tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py \
  tests/unit/applications/test_strict_production_serving_http_caller_authorization.py \
  tests/unit/api/fastapi_core/auth/test_auth_integration.py
```

**Result:** 25 passed (agent session).

## FRZ local contributions (globals stay OPEN)

| FRZ | Contribution |
|-----|----------------|
| FRZ-SEC-01 | Caller tenant bound before execution |
| FRZ-SEC-04 | HTTP auth on strict `/run` |
| FRZ-PRD-08 | Removes anonymous strict execution ingress |
| FRZ-REG-02/03/09 | P3-A2 adversarial qualification module |

## Audit note

Independent auditors must verify behavior on the **implementation commit** on GitHub; this document is not sufficient for closure.
