# PROD-Q-R1-R7-Q-P3-A2 — HTTP Caller Authentication Qualification Evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (P3-A2-R1 implementation + P3-A2-R2 audit remediation; globals unchanged) |
| **Parent ADR** | [`PROD_Q_R1_R7_Q_P3_A2_HTTP_CALLER_AUTHENTICATION_AND_TENANT_AUTHORIZATION_ADR.md`](PROD_Q_R1_R7_Q_P3_A2_HTTP_CALLER_AUTHENTICATION_AND_TENANT_AUTHORIZATION_ADR.md) — **PROPOSED** (not ratified to ACCEPTED) |
| **Architecture basis** | Alternative A — `fastapi_core` `AuthProvider` + `require_auth` + `context_only` identity |
| **Qualified ingress profile** | **API-key map required** at strict composition (`validate_strict_production_serving_http_ingress_composition`); JWT/OIDC profile **OPEN** |
| **Does not close** | P3-A · P3 · R7-Q · global FRZ rows |

## Authentication and authorization contract

| Layer | Owner | Mechanism |
|-------|-------|-----------|
| Authenticate HTTP caller | `AuthProvider` via `RequestContextMiddleware` | `require_auth` on strict GCA `/run` and `/agents` |
| Caller → serving tenant | `authorize_strict_production_serving_http_caller` | `AuthContext.tenant_id` == `StrictProductionServingExecutionTenantBinding.authoritative_execution_tenant_id` |
| Task tenant (body) | `GovernedContractorRunService._resolve_identity` | `context_only` — no body upgrade |
| Session tenant gate (P3-A1) | `StrictProductionServingTenantAuthorityTaskExecutionPort` | unchanged ordering |
| Admitted root principal | `bind_strict_serving_http_caller_principal` → `build_strict_production_serving_host_task_execution_governance` → `admit_strict_production_serving_root_governance_identity` | per-request `RequestIdentity` from authenticated context; **not** `ProductionControlPlaneGovernance.principal` |

## Strict ingress profile

- API keys required when `serving_runtime_handoff` is bound (`validate_strict_production_serving_http_ingress_composition`).
- `identity_source=context_only` required for strict serving HTTP.
- Alternate execution HTTP surfaces (task control, interaction intake, MCP) rejected at factory composition when strict handoff is present.

## Static route inventory — `create_governed_contractor_backend_app` (strict + handoff)

Verified by composition code (not feature flags alone).

| Route / surface | Reachable in strict handoff profile | Execution / disclosure | Disposition |
|-----------------|-------------------------------------|-------------------------|-------------|
| `GET /health` | Yes (`create_app` → `health_router`) | Readiness only | Open — orchestrator probe (ADR §7) |
| `POST /runs`, `GET /runs/{id}`, `POST /runs/{id}/cancel` | Yes (`create_app` → `runs_router`) | `DefaultExecutionAdapter` no-op — **not** `HostTaskExecutionPort` / agent harness | `require_scope` + auth; **not** governed-contractor agent dispatch |
| `POST {prefix}/run` | Yes (`mount_governed_contractor_routes`, `strict_ingress`) | Agent execution | `require_auth` + caller authorization + principal bind |
| `GET {prefix}/agents` | Yes | Registry metadata | `require_auth` |
| Product observability dashboard | If env enables diagnostics pane | Operator diagnostics | Mounted when enabled — **not** in default strict qual fixture; tenant-scoped read wiring |
| Task control / interaction / MCP | No | — | `validate_strict_production_serving_http_ingress_composition` **fail-closed** when flags enabled |
| Scheduler (`include_scheduler`) | Background | Host execution | Not HTTP ingress |

**Bypass audit:** No reachable unauthenticated HTTP path to `HostTaskExecutionPort` for strict handoff composition.

## AUTH-01 — AUTH-10 verdicts

| ID | Verdict | Evidence |
|----|---------|----------|
| AUTH-01 | **PASS** | `test_p3a2_t01_anonymous_run_denied_before_execution` (P3-A2-R1 historical) |
| AUTH-02 | **PASS** | `test_p3a2_t07_body_tenant_tamper_denied` + P3-A1 regression (historical) |
| AUTH-03 | **PASS** | `test_p3a2_t03_authenticated_matching_tenant_succeeds` (historical) |
| AUTH-04 | **PASS** | `test_p3a2_t02_invalid_api_key_denied` (historical) |
| AUTH-05 | **PASS** | `test_p3a2_t05_forged_tenant_header_without_auth_denied` (historical) |
| AUTH-06 | **PASS** | **R2:** `test_strict_governance_admits_bound_http_caller_not_task_body_or_host_principal` + ContextVar tests in `test_strict_production_serving_http_caller_principal_admission.py` (final admitted `principal_id` seam). P3-A2-T-06 HTTP test captures identity conversion only — supplementary. |
| AUTH-07 | **PASS** | `test_p3a2_t04_wrong_tenant_api_key_denied_before_admission` (historical) |
| AUTH-08 | **PARTIAL** | Session/handoff gates (D3/D4); no new HTTP-only stale-URL test in this package |
| AUTH-09 | **N/A** | **R2:** No on-behalf-of / delegation in strict HTTP identity model — `test_auth09_strict_http_profile_has_no_delegation_semantics`; ADR §13. Unsupported delegation inputs have no code path on GCA strict `/run`. |
| AUTH-10 | **PASS** | `test_p3a2_t10_caller_deny_skips_colocated_admission` (historical) |

## P3-A2-R2 bounded unit run (@ remediation commit)

Permitted node IDs only:

```text
tests/unit/applications/test_strict_production_serving_http_caller_principal_admission.py
tests/unit/applications/test_strict_production_serving_http_caller_authorization.py::test_auth09_strict_http_profile_has_no_delegation_semantics
```

Command:

```text
uv run pytest -p no:xdist -q \
  tests/unit/applications/test_strict_production_serving_http_caller_principal_admission.py \
  tests/unit/applications/test_strict_production_serving_http_caller_authorization.py::test_auth09_strict_http_profile_has_no_delegation_semantics
```

## P3-A2-R1 regression (historical — do not re-run for closure)

```text
uv run pytest -p no:xdist -q \
  tests/qualification/prod_q/test_prod_q_r1_r7_q_p3_a2_http_caller_auth.py \
  ...
```

**Historical result @ R1 implementation SHA:** 25 passed (agent session).

## FRZ local contributions (globals stay OPEN)

| FRZ | Contribution |
|-----|----------------|
| FRZ-SEC-01 | Caller tenant bound before execution |
| FRZ-SEC-04 | HTTP auth on strict `/run` |
| FRZ-PRD-08 | Removes anonymous strict execution ingress |
| FRZ-REG-02/03/09 | P3-A2 adversarial qualification module (R1) + R2 admitted-principal unit seam |

## NOT PROVEN (explicit)

| Item | Status |
|------|--------|
| Global FRZ-SEC-01 / FRZ-SEC-04 promotion | **OPEN** — independent GitHub SHA audit |
| ADR ratification to ACCEPTED | **OPEN** — operator decision |
| P3-A / P3 / R7-Q closure | **OPEN** — parent not closed |
| JWT/OIDC strict profile | **OPEN** — separate qualification |
| AUTH-08 full HTTP stale-session | **PARTIAL** |
| Product observability dashboard auth in strict prod | **NOT PROVEN** in this package (pane gating / deployment) |

## Audit note

Independent auditors must verify behavior on the **implementation commit** on GitHub; this document is not sufficient for closure.
