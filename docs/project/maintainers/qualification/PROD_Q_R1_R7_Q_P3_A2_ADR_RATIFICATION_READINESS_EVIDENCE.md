# PROD-Q-R1-R7-Q-P3-A2-ADR-RATIFICATION-READINESS — Evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (operator ratification package; **not** ADR ACCEPTED) |
| **AUDITED_HEAD** | `9e46aecc8a03ae50ab5a8046950952e48418ca23` |
| **START_HEAD (task)** | `9e46aecc8a03ae50ab5a8046950952e48418ca23` |
| **ADR** | [`PROD_Q_R1_R7_Q_P3_A2_HTTP_CALLER_AUTHENTICATION_AND_TENANT_AUTHORIZATION_ADR.md`](PROD_Q_R1_R7_Q_P3_A2_HTTP_CALLER_AUTHENTICATION_AND_TENANT_AUTHORIZATION_ADR.md) — **PROPOSED — READY FOR AUDIT** |
| **Implementation** | **P3-A2-R1** `495b001d2` · **P3-A2-R2** `88da0c903` |
| **Qualification body** | [`PROD_Q_R1_R7_Q_P3_A2_HTTP_CALLER_AUTHENTICATION_QUALIFICATION_EVIDENCE.md`](PROD_Q_R1_R7_Q_P3_A2_HTTP_CALLER_AUTHENTICATION_QUALIFICATION_EVIDENCE.md) |
| **Parent** | **PROD-Q-R1-R7-Q-P3** — **BLOCKED** ([`PROD_Q_R1_R7_Q_P3_FINAL_ADVERSARIAL_ACCEPTANCE_MATRIX.md`](PROD_Q_R1_R7_Q_P3_FINAL_ADVERSARIAL_ACCEPTANCE_MATRIX.md)) |
| **Ratification disposition (this package)** | **A — READY FOR OPERATOR RATIFICATION** (API-key strict profile only) |

## 1. Root cause — ADR inconsistency

§0–§7 and §14 @ ADR inventory SHA (`0e84b376…`) correctly described **pre-R1** gaps (no `require_auth` on GCA `/run`, static host principal). **P3-A2-R1/R2** implemented Alternative A; qualification evidence records **PASS** for AUTH-01–07, 10. **§14 was stale** (“NOT PROVEN”, next child R1). Reconciled in ADR **§15** @ `9e46aecc` — **no** change to fundamental Alternative A recommendation.

## 2. Closed-world inventory

### Semantic owners

| Responsibility | Module / symbol |
|----------------|-----------------|
| `AuthProvider` contract | `intergrax/fastapi_core/auth/provider.py` |
| API key provider | `intergrax/fastapi_core/auth/providers/api_key/provider.py` (`ApiKeyAuthProvider`) |
| `AuthContext` | `intergrax/fastapi_core/auth/context.py` |
| Request context + auth attachment | `intergrax/fastapi_core/middleware/request_context.py` (`RequestContextMiddleware`) |
| `require_auth` | `intergrax/fastapi_core/auth/dependency.py` |
| Backend settings / keys | `GovernedContractorBackendSettings` (`applications/governed_contractor_application/host/settings.py`) |
| Caller → session tenant | `authorize_strict_production_serving_http_caller` |
| Auth → `RequestIdentity` | `request_identity_from_strict_production_http_auth` |
| Per-request caller principal | `bind_strict_serving_http_caller_principal` / `get_strict_serving_http_caller_principal` |
| P3-A1 tenant gate | `StrictProductionServingTenantAuthorityTaskExecutionPort` |
| Admitted root principal | `build_strict_production_serving_host_task_execution_governance` → `admit_strict_production_serving_root_governance_identity` |
| Strict ingress composition guard | `validate_strict_production_serving_http_ingress_composition` |

### Composition owners

| Surface | Owner |
|---------|-------|
| GCA backend app + strict validation | `applications/governed_contractor_application/host/factory.py` |
| GCA HTTP routes | `applications/governed_contractor_application/serving/fastapi_router.py` |
| Strict ASGI session factory | `create_governed_contractor_strict_production_asgi_app` (D4 chain) |

### Direct production callers (strict path)

- `fastapi_router.run_task` → `authorize_strict_production_serving_http_caller` → `bind_strict_serving_http_caller_principal` → `host_execution.execute`
- `strict_production_host_task_execution_wiring.build_strict_production_serving_host_task_execution_governance` → admission closure reads ContextVar principal

## 3. Trust graph — historical vs current

| Stage | @ inventory `0e84b376` | @ HEAD `9e46aecc` |
|-------|-------------------------|-------------------|
| `/run` auth | None | `require_auth` |
| Body tenant | Trusted for task | `context_only` — auth context only |
| Caller vs session tenant | Not checked | `authorize_strict_production_serving_http_caller` |
| Admitted `principal_id` | Host governance principal | HTTP caller `RequestIdentity` (R2) |

## 4. AUTH-01 — AUTH-10 matrix

| ID | ADR expectation | Implementation path | Test symbol | Evidence SHA | Continuity @ `9e46aecc` | Verdict | Qual scope | Blocks ratification? |
|----|-----------------|---------------------|-------------|--------------|-------------------------|---------|------------|------------------------|
| AUTH-01 | Anonymous → 401, no execute | `require_auth` + strict route | `test_p3a2_t01_anonymous_run_denied_before_execution` | R1 `495b001d2` | Symbols present | **PASS** | P3-A2-R1 qual | No |
| AUTH-02 | Cross-tenant caller/body | caller auth + P3-A1 gate | `test_p3a2_t07_body_tenant_tamper_denied` | R1 | Present | **PASS** | P3-A2 + P3-A1 | No |
| AUTH-03 | Matching tenant success | full stack | `test_p3a2_t03_authenticated_matching_tenant_succeeds` | R1 | Present | **PASS** | P3-A2-R1 | No |
| AUTH-04 | Invalid key → 401 | `ApiKeyAuthProvider` + `require_auth` | `test_p3a2_t02_invalid_api_key_denied` | R1 | Present | **PASS** | P3-A2-R1 | No |
| AUTH-05 | Forged headers, no cred | middleware only sets auth | `test_p3a2_t05_forged_tenant_header_without_auth_denied` | R1 | Present | **PASS** | P3-A2-R1 | No |
| AUTH-06 | Admitted principal = caller | ContextVar + `admit_strict_production_serving_root_governance_identity(principal=…)` | `test_strict_governance_admits_bound_http_caller_not_task_body_or_host_principal` | R2 `88da0c903` | Present | **PASS** | P3-A2-R2 unit | No |
| AUTH-07 | Wrong tenant key → 403 | `authorize_strict_production_serving_http_caller` | `test_p3a2_t04_wrong_tenant_api_key_denied_before_admission` | R1 | Present | **PASS** | P3-A2-R1 | No |
| AUTH-08 | Stale URL / wrong session | D3/D4 handoff + launcher | D3/D4 qual docs (not HTTP E2E) | D4 `0dd51fe…` etc. | Unchanged | **PARTIAL** | P3 parent | No (per P3 matrix **G-08** optional) |
| AUTH-09 | Delegation unsupported | no OBO path | `test_auth09_strict_http_profile_has_no_delegation_semantics` | R2 | Present | **N/A** | P3-A2-R2 | No |
| AUTH-10 | Deny before admission | caller auth before port stack | `test_p3a2_t10_caller_deny_skips_colocated_admission` | R1 | Present | **PASS** | P3-A2-R1 | No |

## 5. Architecture audit (11 invariants)

| # | Claim | Classification |
|---|-------|----------------|
| 1 | Anonymous cannot `/run` | **PASS** |
| 2 | Invalid API key cannot execute | **PASS** |
| 3 | Forged tenant headers cannot grant identity | **PASS** |
| 4 | Tenant A cannot execute under session B | **PASS** |
| 5 | Body cannot upgrade identity (`context_only`) | **PASS** |
| 6 | Per-request principal at admission boundary | **PASS** (R2) |
| 7 | Context isolation cross-request | **PARTIAL** — thread ContextVar test only; async ASGI **NOT PROVEN** |
| 8 | Strict execution routes bypass auth | **PASS** — composition rejects alt surfaces |
| 9 | Task-control / interaction / MCP | **PASS** — fail-closed at factory when strict handoff |
| 10 | Health / read-only disposition | **PASS** (explicit open health); `/runs` non-harness path documented |
| 11 | Governance / physical admission ownership | **PASS** — unchanged ordering |

## 6. Enterprise audit matrix (selected rows)

| Row | Source | Verdict |
|-----|--------|---------|
| Contracts over implementations | `AuthProvider`, `RequestIdentity`, binding dataclasses | **PASS** |
| Layer boundaries | Tier-3 app → `intergrax` fastapi_core + applications `_shared` | **PASS** |
| Exactly-one semantic owner (HTTP auth) | `AuthProvider` | **PASS** |
| Composition owner (GCA strict) | `factory.py` + `fastapi_router.py` | **PASS** |
| Strong typing on boundaries | caller auth / principal modules | **PASS** |
| No shadow auth on `/run` | ingress validation | **PASS** |
| Pluginable `AuthProvider` | fastapi_core | **PASS** |
| Bypass resistance (strict handoff) | qual + ingress validator | **PASS** |
| Fail-closed factory | `validate_strict_production_serving_http_ingress_composition` | **PASS** |
| Per-request context | ContextVar bind in `run_task` | **PARTIAL** (sync/async concurrency) |
| Evidence continuity | `test_p3a2_g00_qualification_baseline_ancestor` | **PASS** |
| Regression protection | qual module + R2 unit gate | **PASS** |

No global enterprise **PASS**.

## 7. FRZ mapping (globals **OPEN**)

| FRZ | Checklist wording (canonical) | Current evidence | Historical SHA | New evidence this task | Unproven remainder | Stage |
|-----|------------------------------|------------------|----------------|------------------------|-------------------|-------|
| FRZ-SEC-01 | tenant isolation proven | P3-A cross-tenant + P3-A2 caller bind + P3-A1 gate | P3-A `e24aa42a…` | Reconciliation @ `9e46aecc` | Global tenant isolation | CTRL-X, PROD-Q |
| FRZ-SEC-04 | production auth bypass = 0 | Strict `/run` auth + factory composition | D4, P2-C | P3-A2 qual (local) | Platform-wide bypass = 0 | CTRL-X, PROD-Q |
| FRZ-PRD-08 | production-only bypass count = 0 | Same local chain | D1/D2/P2 | P3-A2 | Global count | PROD-Q |
| FRZ-REG-02 | every corrected blocker has regression gate | P3-A2 qual + R2 unit | R1/R2 | Doc only | QUAL-X global | QUAL-X |
| FRZ-REG-03 | negative tests detect violations | P3-A2 adversarial tests | R1 | — | — | QUAL-X |
| FRZ-REG-09 | critical invariants protected | AUTH tests + ingress gate | R1/R2 | — | — | QUAL-X |
| FRZ-SEC-07 | security claim bounded by threat model | P3-A2 bounds API-key strict profile | AW-7C-P0-3B pattern | Explicit limits §15.9 | Dashboard optional surface | CTRL-X |

## 8. Findings classification

| Finding | Class |
|---------|-------|
| ADR §14 stale vs implementation | **TRACKED FREEZE DEBT** — resolved in ADR §15 (this commit) |
| **G-01** ADR not ACCEPTED | **IN-SCOPE BLOCKER** for **P3** closure — **not** implementation blocker |
| AUTH-08 HTTP-only stale URL | **TRACKED FREEZE DEBT** — PARTIAL within P3 scope |
| Async ASGI ContextVar interleaving | **TRACKED FREEZE DEBT** — optional **G-08** |
| Observability dashboard strict auth | **NOT PROVEN** — owner: product observability wiring / future qual |
| JWT/OIDC profile | **TRACKED FREEZE DEBT** — separate profile |

**No IN-SCOPE BLOCKER** for Alternative A API-key strict implementation @ HEAD.

## 9. Tests

**Policy:** no pytest replay in this ratification task.

**Continuity:** static review @ `9e46aecc` — qual and unit symbols above match imported modules.

**Historical commands:** see qualification evidence doc (R1 25 passed @ R1 SHA; R2 bounded unit run documented).

## 10. P3 / integration context (pointers only)

- P3-A cross-tenant: [`PROD_Q_R1_R7_Q_P3_A_CROSS_TENANT_STRICT_QUALIFICATION_EVIDENCE.md`](PROD_Q_R1_R7_Q_P3_A_CROSS_TENANT_STRICT_QUALIFICATION_EVIDENCE.md)
- P2-C HTTP admission: referenced from P3-A2 qual fixtures
- D3/D4: strict launcher / handoff — AUTH-08 partial contribution
- P3-INTEGRATION-GATE: parent matrix — **E.1** full replay not required for this child

## 11. Operator decision (not pre-applied)

After independent GitHub audit @ `9e46aecc`:

- **ACCEPT** ADR Alternative A (API-key strict ingress) → update ADR status to **ACCEPTED** with decision date, then proceed **G-01** closure steps in P3 matrix.
- **REJECT** → record blocker; keep **PROPOSED**.

**Recommended P3 status:** **BLOCKED** (unchanged) until ADR ratification + remaining parent gates.

## Audit note

Independent auditors must verify behavior on the **implementation commit** on GitHub; this document is not sufficient for closure.
