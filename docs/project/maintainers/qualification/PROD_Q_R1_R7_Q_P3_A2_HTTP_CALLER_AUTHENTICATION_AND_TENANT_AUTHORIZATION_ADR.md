# PROD-Q-R1-R7-Q-P3-A2 — HTTP Caller Authentication & Caller-to-Tenant Authorization

| Field | Value |
|-------|-------|
| **Status** | **PROPOSED — READY FOR AUDIT** (architecture proposal; **P3-A2-R1 implementation** tracked in [`PROD_Q_R1_R7_Q_P3_A2_HTTP_CALLER_AUTHENTICATION_QUALIFICATION_EVIDENCE.md`](PROD_Q_R1_R7_Q_P3_A2_HTTP_CALLER_AUTHENTICATION_QUALIFICATION_EVIDENCE.md) — **not** ratified to ACCEPTED) |
| **Revision** | **R0** |
| **Decision date** | 2026-10-15 |
| **Branch** | `development` |
| **Architecture inventory SHA** | `0e84b37649eadb2d188cd5b9bde984e81868e064` |
| **Parent** | **PROD-Q-R1-R7-Q-P3** — **NOT CLOSED** |
| **Sibling evidence** | **PROD-Q-R1-R7-Q-P3-A** — P3-A1 session-bound tenant gate **PASS** (scoped); finding **F-03** open |
| **Baseline ADR** | [`PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md`](PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md) (deployment / activation authority — does **not** close HTTP caller trust) |
| **Related** | [`PROD_Q_R1_R7_Q_P3_A_CROSS_TENANT_STRICT_QUALIFICATION_EVIDENCE.md`](PROD_Q_R1_R7_Q_P3_A_CROSS_TENANT_STRICT_QUALIFICATION_EVIDENCE.md) · [`PROD_Q_R1_R7_Q_P3_P0_ADVERSARIAL_QUALIFICATION_INVENTORY_AND_GATE_PLAN.md`](PROD_Q_R1_R7_Q_P3_P0_ADVERSARIAL_QUALIFICATION_INVENTORY_AND_GATE_PLAN.md) |
| **Does not close** | **PROD-Q-R1-R7-Q-P3-A** (full security) · **PROD-Q-R1-R7-Q** · **PROD-Q** · any global **FRZ-*** PASS |

---

## 0. Scope separation (mandatory)

| # | Concern | Owner / mechanism today | P3-A2 scope |
|---|---------|-------------------------|-------------|
| 1 | Control-plane principal authorization | `ProductionControlPlaneGovernance` + `ControlPlaneMutationAuthorizationBoundary` at `deploy_and_activate` | **Out of scope** (proven P3-T-01) |
| 2 | Serving-session tenant binding | `StrictProductionServingExecutionTenantBinding` from `ServingRuntimeHandoff` + `ApplicationEnvironmentTenantResolver` | **Partial** — binding exists; HTTP must not override it |
| 3 | HTTP caller authentication | `intergrax.fastapi_core` `AuthProvider` + optional `ApiKeyAuthProvider` | **In scope** — not enforced on GCA `/run` |
| 4 | Caller-to-execution-tenant authorization | **Not implemented** at HTTP ingress | **In scope** — primary gap (F-03) |
| 5 | Physical execution admission | `ColocatedServingUnitAdmissionTaskExecutionPort` → `admit_execution` | **Out of scope** for design (ordering invariant defined here) |

**P3-A1** proves: untrusted body `tenant_id` cannot diverge from authoritative serving binding before colocated admission (**P3-T-02**). It does **not** prove that only authorized human/service callers may invoke `/run`.

---

## 1. Exact current trust graph (strict GCA production @ inventory SHA)

```text
[HTTP client]
    │  POST /v1/governed_contractor/run  (no Depends(require_auth))
    │  body: GovernedContractorRunRequestV1.tenant_id (client-supplied)
    ▼
RequestContextMiddleware
    │  auth_provider.authenticate(request)  → AuthContext | None
    │  (no fail-closed if unauthenticated)
    ▼
GovernedContractorRunService.run_task
    │  Task(tenant_id=body.tenant_id, user_id=body.user_id, …)
    ▼
HostTaskExecutionPort stack (factory order, outer → inner):
    1. StrictProductionServingTenantAuthorityTaskExecutionPort
         assert_task_tenant_matches_strict_production_serving_authority
    2. ColocatedServingUnitAdmissionTaskExecutionPort
         handoff.colocated_execution_admission.admit_execution(unit_ref)
    3. build_environment_host_task_execution → ExecutionRuntime path
         admit_root_governance_identity:
           admit_strict_production_serving_root_governance_identity(
             principal = ProductionControlPlaneGovernance.principal  # process/session, NOT HTTP)
```

**Trusted inputs at execution:**

| Input | Source | HTTP caller influence |
|-------|--------|------------------------|
| `authoritative_execution_tenant_id` | `ServingRuntimeHandoff.identity.application_environment_id` | **None** (session) |
| `task.tenant_id` | HTTP body | **Full** (validated against binding only) |
| `AdmittedRootGovernanceIdentity.principal_id` | `ProductionControlPlaneGovernance.principal` | **None** |
| `AdmittedRootGovernanceIdentity.tenant_id` | Binding | **None** |
| Colocated admission eligibility | Store-backed handoff + unit state | **None** at HTTP |

**Policy vs wiring gap:** `GovernedContractorBackendSettings` defaults `identity_source=context_only` in `prod` and requires API key material unless `GOVERNED_CONTRACTOR_BACKEND_ALLOW_UNAUTHENTICATED=true`, but `governed_contractor_application/serving/fastapi_router.py` **does not read** `identity_source` or `RequestContext` (contrast: `legal_application/serving/fastapi_router.py` `_resolve_identity`).

---

## 2. Supported ingress modes (closed-world)

| Mode | How invoked | HTTP authentication | Caller identity reaches execution | `/run` without gateway auth |
|------|-------------|---------------------|-----------------------------------|-----------------------------|
| **M1 — Strict production ASGI (D4)** | `create_governed_contractor_strict_production_asgi_app(session)` | `ApiKeyAuthProvider` if `api_keys_map` non-empty; else `NoAuthProvider` unless `BACKEND_ALLOW_UNAUTHENTICATED` prevents startup in `prod` | **Not from HTTP** — governance principal fixed at host build | **Yes today** — route has no `require_auth`; network isolation is not identity |
| **M2 — Strict production + disaster flag** | Same with `GOVERNED_CONTRACTOR_BACKEND_ALLOW_UNAUTHENTICATED=true` | `NoAuthProvider` | Same as M1 | **Yes** — explicit operator break-glass only |
| **M3 — Non-strict / dev factory** | `create_governed_contractor_backend_app` without handoff | Optional API keys; default dev allows unauthenticated settings | Harness / lab governance mappers | **Yes** — not a production qualification target |
| **M4 — Colocated sidecar / future gateway** | Not implemented as first-class contract in repo | **Undocumented** — no sanctioned “trusted header” `AuthProvider` | N/A | **Must not** be assumed safe without ADR-approved provider |

**Sanctioned ingress integration contracts present in codebase:**

- `intergrax.fastapi_core.auth.provider.AuthProvider` (+ `ApiKeyAuthProvider`, `JwtAuthProvider`, `CompositeAuthProvider`)
- `intergrax.fastapi_core.auth.dependency.require_auth` / `require_scope`
- `intergrax.applications._shared.harness_auth` — **separate** harness API-key middleware for task-control/MCP surfaces (`require_harness_api_key`), not wired on GCA `/run`

**No** production-qualified “edge gateway strips client headers and injects signed identity” module exists in-tree; any such deployment is **operator-owned** until bound to an approved `AuthProvider` or mutual-TLS front door.

---

## 3. Identity and authorization ownership (target architecture)

| Decision | Canonical owner | Forbidden |
|----------|-----------------|-----------|
| Authenticate HTTP caller | `AuthProvider` resolved in `RequestContextMiddleware` + route-level `require_auth` on execution-bearing routes | Route-local API-key checks; harness-only middleware on strict `/run` |
| Map credential → tenant/workspace | `ApiKeyIdentity.tenant_id` / JWT claims via existing fastapi_core providers | Second tenant registry; shadow Governance |
| Authorize caller for **this** strict serving session | **New thin port** (recommended name): `StrictProductionServingHttpCallerAuthorizationPort` implemented using existing `ApplicationEnvironmentTenantResolver` + `AuthContext` — compares authenticated tenant (and optional subject) to `StrictProductionServingExecutionTenantBinding` | Reusing body `tenant_id` as proof of caller authority |
| Session execution tenant | `StrictProductionServingExecutionTenantBinding` (unchanged — P3-A1) | Re-deriving tenant from HTTP body |
| Root governance identity at dispatch | `admit_strict_production_serving_root_governance_identity` — principal must transition from **authenticated caller** (`AuthContext` → `RequestIdentity`), not only static `ProductionControlPlaneGovernance.principal` | Harness identity mapper; silent fallback to host service principal for all clients |
| Control-plane mutations | Unchanged — `ProductionControlPlaneGovernance` | — |

**Ordering invariant (no-bypass):**

```text
HTTP authenticate → HTTP caller-to-tenant authorize → build Task from trusted context
  → StrictProductionServingTenantAuthorityTaskExecutionPort (body tenant vs binding)
  → ColocatedServingUnitAdmissionTaskExecutionPort
  → admit_root_governance_identity (caller-derived principal)
```

---

## 4. Threat-model matrix AUTH-01 — AUTH-10

| ID | Scenario | Authoritative component (target) | Expected decision | Execution evidence required |
|----|----------|-----------------------------------|-------------------|----------------------------|
| **AUTH-01** | Anonymous caller, valid body `tenant_id` matching session | `require_auth` on `/run` + prod settings (keys required) | **401** before `HostTaskExecutionPort` | Qualification HTTP test: no `Authorization` / API key → 401; inner delegate not called |
| **AUTH-02** | Authenticated tenant A, body claims tenant B (session tenant B) | Caller authorization port + P3-A1 gate | **403** at caller authorization (preferred) or **ValueError** at tenant gate if body not rewritten from auth | Test: API key tenant A, body `tenant_id=env-prod` (B) → deny before admission |
| **AUTH-03** | Authenticated tenant B, body/context tenant B, session B | Full stack | **200** / task executed | Positive control mirroring P3-T-02 with real API key |
| **AUTH-04** | Missing / invalid / expired credential | `AuthProvider` + `require_auth` | **401** fail-closed | Tests: wrong key, empty key, expired JWT (when JWT profile enabled) |
| **AUTH-05** | Forged `X-*` identity headers without valid credential | Only `AuthProvider` may set `AuthContext`; no trust of raw tenant headers | **401** — context remains unauthenticated | Test: forged tenant header, no key → 401; no execution |
| **AUTH-06** | Host `ProductionControlPlaneGovernance.principal` treated as HTTP caller | `admit_strict_production_serving_root_governance_identity` must use **caller** `RequestIdentity` after HTTP auth | Deny anonymous; admitted principal ≠ static host principal unless caller is that service account | Test: distinct API key `user_id` reflected in admitted identity audit hook |
| **AUTH-07** | Auth succeeds but no tenant mapping | Caller authorization port | **403** `caller_tenant_not_authorized_for_serving_session` | Test: valid key for tenant A on session bound to B |
| **AUTH-08** | Valid caller, wrong environment/revision (stale URL / wrong port) | Session opener + handoff validation (existing) + optional `allowed_hosts` / deployment routing | **No traffic-ready app** or handoff mismatch at open — not HTTP auth alone | Reuse D3/D4 reconciliation tests + new HTTP test against wrong session factory |
| **AUTH-09** | Delegated / multi-tenant on-behalf-of | **Not supported** in closed-world @ SHA | **403** unless future ADR adds scoped delegation evidence | Document N/A; no guesswork |
| **AUTH-10** | Unauthorized provider effects | Caller deny **before** `admit_execution` | Inner delegate + provider dispatch **not** invoked | Spy on `TaskBoundAgenticDelegate.execute` + admission counters (P2-C / P3-A pattern) |

**Current state @ SHA (pre-implementation):** AUTH-01, AUTH-02 (caller leg), AUTH-04–07, AUTH-10 (caller leg) are **not proven**; P3-A1 covers only **body tenant vs binding** (subset of AUTH-02).

---

## 5. Alternatives and recommendation

### 5.1 Alternative A — Application ingress (fastapi_core)

Enforce authentication and `context_only` identity on GCA `/run` using existing `AuthProvider` contracts and the **legal_application** `_resolve_identity` pattern; add caller-to-session tenant check against `StrictProductionServingExecutionTenantBinding` injected at factory composition (app state or closure — **not** a new global registry).

**Pros:** Matches `GovernedContractorBackendSettings` prod rules; reuses `ApiKeyAuthProvider` / future `JwtAuthProvider`; single ASGI trust boundary auditable in qualification.  
**Cons:** Application must hold binding reference for authorization dependency.

### 5.2 Alternative B — Trusted gateway identity propagation

Terminate TLS and OAuth/mTLS at edge; forward only via headers/API keys that the app validates through a **configured** `AuthProvider` (e.g. JWT signed by gateway, or API key minted per tenant).

**Pros:** Centralized IdP, rate limits, WAF.  
**Cons:** Requires explicit provenance contract (issuer, audience, header stripping); **not** “trust `X-Tenant-Id` from client”; must still call same in-app caller authorization port.

### 5.3 Alternative C — Network isolation only

Private VPC / localhost bind without app auth.

**Rejected** for production qualification: violates FRZ-SEC-01/04 intent for HTTP execution entry; equivalent to AUTH-01 accept.

### **Recommendation (operator decision target)**

**Primary: Alternative A** as the canonical production contract, with **Alternative B** allowed only as a **deployment profile** that terminates in the same `AuthProvider` + caller authorization port (no duplicate semantics).

**Rationale @ inventory SHA:** Settings already mandate API keys + `context_only` in prod, but `/run` ignores them; implementing A closes the documented policy–code gap with minimal new architecture. No sanctioned gateway identity module exists to adopt as Alternative B alone.

---

## 6. Existing contracts to reuse

| Contract | Path | Reuse |
|----------|------|-------|
| `AuthProvider` / `AuthContext` | `intergrax/fastapi_core/auth/**` | Authenticate all strict execution HTTP routes |
| `require_auth`, `require_scope` | `intergrax/fastapi_core/auth/dependency.py` | Route dependencies |
| `ApiKeyConfig` / `ApiKeyIdentity` | `intergrax/fastapi_core/auth/api_key.py` | Tenant + scopes on keys |
| `RequestContext` | `intergrax/fastapi_core/context.py` | `context_only` identity |
| `GovernedContractorBackendSettings.identity_source` | `applications/governed_contractor_application/host/settings.py` | Prod policy already set |
| `StrictProductionServingExecutionTenantBinding` | `strict_production_serving_execution_tenant_authority.py` | Session tenant authority (P3-A1) |
| `ApplicationEnvironmentTenantResolver` | `intergrax/agent_distribution/control_plane_governance.py` | Align caller tenant with environment authority (same family as control-plane) |
| `admit_strict_production_serving_root_governance_identity` | same module | Admitted root — extend principal input to caller |
| Legal serving reference | `legal_application/serving/fastapi_router.py` | `_resolve_identity` pattern |
| Harness task control auth | `harness_auth.require_harness_api_key` | **Do not** reuse for strict `/run` (different principal model) |

---

## 7. Required changes — direct-caller inventory (strict GCA app)

Routes mounted by `create_governed_contractor_backend_app` / strict ASGI factory:

| Route / surface | Auth today | Must change for P3-A2 |
|-----------------|------------|---------------------|
| `POST {prefix}/run` | **None** | **Yes** — `require_auth` + identity resolution + caller authorization |
| `GET {prefix}/agents` | **None** | **Yes** — read scope or same auth (information disclosure) |
| `POST /v1/runs` (fastapi_core) | `require_scope` | Verify strict prod disables or scopes runs router consistently |
| Harness task control (`wire_harness_task_control`) | `require_harness_api_key` when enabled | Align policy: strict prod should disable lab task control **or** map harness auth to same tenant authority (prefer **disable** in strict session profile) |
| Interaction intake routes | No fastapi_core auth by default | Same as task control — disable or enforce auth in strict prod |
| MCP mount | Harness coupling auth | Strict prod profile: document deny-by-default unless auth wired |
| Health | Typically open | Acceptable with readiness gating at orchestrator |

**Composition changes (minimal package):**

1. `governed_contractor_application/serving/fastapi_router.py` — inject binding + settings; `Depends(require_auth)`; resolve tenant/user from context; reject body conflicts (`context_only`).
2. `governed_contractor_application/host/factory.py` — pass `strict_serving_tenant_binding` into router/service; attach caller authorization callable.
3. `strict_production_host_task_execution_wiring.py` / factory — supply `RequestIdentity` from **per-request** context (middleware state), not only static `production_governance.principal`, for strict serving execution path.
4. Settings — fail closed if `prod` + strict handoff + `BACKEND_ALLOW_UNAUTHENTICATED`.

**No changes** to P3-A1 tenant gate semantics (retain outer ordering).

---

## 8. Explicit no-bypass invariants

1. No HTTP execution route may call `HostTaskExecutionPort` without successful `AuthProvider` authentication in strict production.
2. No route may trust `GovernedContractorRunRequestV1.tenant_id` without `context_only` checks against `AuthContext`.
3. Caller tenant must be authorized for `StrictProductionServingExecutionTenantBinding.authoritative_execution_tenant_id` before task construction.
4. `ProductionControlPlaneGovernance.principal` must not implicitly authenticate all HTTP clients (AUTH-06).
5. No second tenant registry or shadow Governance object at HTTP layer.
6. Caller denial must occur **before** `ColocatedServingUnitAdmissionTaskExecutionPort.admit_execution`.
7. `BACKEND_ALLOW_UNAUTHENTICATED` incompatible with traffic-ready strict session qualification.

---

## 9. Minimal implementation package (bounded next task)

| Item | Estimate |
|------|----------|
| GCA router + factory wiring | 1 module pair |
| Caller authorization port (≤1 small module in `intergrax/applications/_shared/`) | 1 file |
| Principal threading into `build_strict_production_serving_host_task_execution_governance` | Factory + wiring |
| Strict prod settings guard | Settings validation |
| Qualification module `test_prod_q_r1_r7_q_p3_a2_http_caller_auth*.py` | AUTH-01–07, AUTH-10 |

**Out of package:** JWT IdP integration, gateway vendor adapters, multi-tenant delegation (AUTH-09).

---

## 10. Focused security-test matrix (qualification)

| Test ID | Maps to | Assertion |
|---------|---------|-----------|
| P3-A2-T-01 | AUTH-01 | Anonymous POST `/run` → 401, no inner execute |
| P3-A2-T-02 | AUTH-04 | Invalid API key → 401 |
| P3-A2-T-03 | AUTH-03 | Valid key, matching tenant, P3-T-02 body → success path |
| P3-A2-T-04 | AUTH-02 / AUTH-07 | Valid key tenant A, session tenant B → 403 before admission |
| P3-A2-T-05 | AUTH-05 | Forged tenant header without key → 401 |
| P3-A2-T-06 | AUTH-06 | Admitted / audit principal reflects API key user, not static host principal only |
| P3-A2-T-10 | AUTH-10 | On deny, colocated `admit_execution` not called (mock/spy) |

Replay: existing `test_prod_q_r1_r7_q_p3_a_cross_tenant_strict.py` + `test_prod_q_r1_r7_q_p2_c_http_execution_admission.py` must remain green.

---

## 11. P2 / P3 regression impact

| Area | Impact |
|------|--------|
| P2-C HTTP admission | Tests must send valid API keys after implementation; update fixtures only |
| P3-A1 tenant gate | Unchanged semantics; remains second line of defense |
| P3-T-01 / T-02 evidence | **Remains valid** — control-plane and body-vs-binding proofs |
| D4 strict launcher | No activation change; ASGI app auth behavior changes |
| FRZ checklist rows | No global status change from this ADR alone |

---

## 12. FRZ contribution map (local only — globals stay OPEN)

| FRZ ID | P3-A2 intended contribution after implementation + qualification |
|--------|-------------------------------------------------------------------|
| **FRZ-SEC-01** | Tenant isolation includes **caller** tenant, not only body/handoff match |
| **FRZ-SEC-04** | Closes HTTP auth bypass on strict `/run` (complements D4 activation deny) |
| **FRZ-PRD-08** | Removes production HTTP bypass for execution |
| **FRZ-REG-02/03/09** | New adversarial qualification tests |

**Global FRZ-SEC-01/04 remain OPEN** until independent audit of implementation commit on GitHub.

---

## 13. Remaining architecture dependencies

| Dependency | Status |
|------------|--------|
| Operator-approved production mutation policy profile (P3 gate dim 6) | OPEN — separate from HTTP caller |
| JWT/OIDC `AuthProvider` configuration contract for enterprise IdP | Optional profile; API key sufficient for first qualification wave |
| Gateway strip/propagate specification | Document in deployment runbook when Alternative B used |
| AUTH-09 delegation model | Not defined — explicit non-goal |
| Task control / MCP / interaction routes in strict prod | Product decision: disable vs unify auth |

---

## 14. P3-A acceptance boundaries and next task

| Claim | Status |
|-------|--------|
| P3-A1 session-bound tenant validation | **PASS** (scoped, audited) |
| HTTP caller authentication | **NOT PROVEN** |
| Caller-to-execution-tenant authorization | **NOT PROVEN** |
| P3-A security closure | **BLOCKED** (F-03) |
| Global FRZ-SEC-01 / FRZ-SEC-04 | **OPEN** |

**Recommended next bounded implementation child:** **PROD-Q-R1-R7-Q-P3-A2-R1** — implement §7–§9, run §10 qualification, operator audit @ implementation SHA.

---

## Audit note

Changes delivered under this child are **documentation only**. Independent auditors must verify behavior against the **implementation** commit on GitHub; this ADR is not sufficient for final closure.
