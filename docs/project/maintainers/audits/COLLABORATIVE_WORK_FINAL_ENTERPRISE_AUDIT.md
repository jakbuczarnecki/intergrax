# Collaborative Work — Final Enterprise Audit

**Audit ID:** COLLAB-WORK-FINAL-REVIEW-1  
**Date:** 2026-09-02  
**Baseline ancestor:** `139f77d36bdff3323581f58d38069f9a043b6197` (contained in HEAD)  
**HEAD audited:** `2e22ac11af20aeee2a3cfdcd7db46587a1076a7c`  
**Branch:** `development`  
**Classification:** **ENTERPRISE_READY_FOR_E2E**

---

## 1. Executive verdict

Collaborative Work (MP-1) is a **coherent enterprise platform capability** for shared human + AI authority semantics. Domain concepts (membership, delegation, base authority, policy rules, operation profiles, enforcement) are owned by `intergrax/collaborative_work/` and `intergrax/contracts/collaborative_work.py`. Platform mechanics (relational provider selection, PostgreSQL/SQLite connection lifecycle, meaningful-side-effect runtime policy, governed continuation / HITL bridge, provider qualification runner) are **reused**, not duplicated.

No parallel CW provider registry, diagnostics engine, observability bus, or approval store was found. Vendor code is confined to adapter modules (`postgresql_repository.py`, `sqlite_repository.py`) and Integrations binders.

**Residual risk is bounded:** MP-2+ collaborative business objects (WorkItem, Artifact, Decision, Activity) are not implemented; LKW has not yet adopted durable CW persistence; enforcement gate does not yet correlate `side_effect_scope_id` with resource scope (platform contract evolution). None of these block MP-1 E2E qualification.

---

## 2. Current architecture

```text
Principal (contract)
  → WorkspaceMembership (repository)
  → PrincipalAuthorityGrant (repository)
  → AuthorityDelegation (repository, optional)
  → CollaborativeWorkAuthorityResolver
  → CollaborativePolicyEvaluator (workspace + resource rules)
  → CollaborativeWorkEnforcementGate
        → RuntimePolicyEngine / MeaningfulSideEffectPolicyEvaluator (runtime layer)
        → compose_policy_decisions
  → MeaningfulSideEffectAuthorizationBoundary (runtime adoption)
  → execution / side effect (caller-owned)
```

**Persistence materialization:**

```text
IntegrationProfile
  → resolve_collaborative_work_repositories (persistence_provider.py)
  → CollaborativeWorkMaterializationBinder.bind_* (unbound)
  → CollaborativeWorkPersistenceFactory.materialize_* (configured)
  → CollaborativeWorkRepositories bundle
```

Prebuilt provider instances implementing `CollaborativeWorkPersistenceProvider` are handled without a second registry.

---

## 3. Platform reuse matrix

| AREA | CURRENT IMPLEMENTATION | PLATFORM COMPONENT REUSED | DUPLICATION RISK | ENTERPRISE RISK | STATUS | RECOMMENDATION |
|------|------------------------|---------------------------|------------------|-----------------|--------|----------------|
| Persistence connection/DSN | PG/SQLite adapters delegate to `PostgreSQLConnectionProvider`, `PostgreSQLIntegrationConfig`, SQLite bundle | Integrations relational_store | Low | Low | OK | None |
| Provider resolution | `resolve_collaborative_work_repositories` → `resolve_slug` / `get_entry` | Integrations registry | Low | Low | OK | None |
| Provider materialization | `CollaborativeWorkMaterializationBinder` + `CollaborativeWorkPersistenceFactory` | Integrations catalog factories | Low | Low | OK | None |
| Provider qualification | `repository_qualification_suite.py` + shared runner | `intergrax.core.qualification.*` | Low | Low | OK | None |
| Policy evaluation (runtime) | Gate injects `MeaningfulSideEffectPolicyEvaluator` | `RuntimePolicyEngine` | Low | Low | OK | None |
| Policy composition | `compose_policy_decisions` | Reuses `PolicyDecision` / `PolicyAction` | Low | Low | OK | None |
| HITL / approval | `REQUIRE_HUMAN` → `MeaningfulSideEffectAuthorizationBoundary` → governed continuation bridge | Platform human runtime | Low | Low | OK | No `CollaborativeWorkApprovalStore` |
| Execution lifecycle | Not owned by CW | Nexus / UAEP | None | Low | OK | Preserve WorkItem ≠ Task |
| Diagnostics | No CW engine | N/A at CW library layer | Low | Low | Boundary OK | Wire RuntimeEvent at host adoption |
| Observability | `audit_payload` on `PolicyDecision` only | Platform contracts available | Low | Medium | Partial | Emit platform signals at adoption boundary |
| Secret handling | Config via Integrations; no DSN in domain models | `PostgreSQLIntegrationConfig` | Low | Low | OK | None |
| Idempotency | DB-enforced `collaborative_idempotency` table | Domain-owned semantics | Low | Low | OK | None |

---

## 4. Domain / platform / provider ownership

| Owns (CW domain) | Reuses (platform) | Must not duplicate |
|------------------|-------------------|--------------------|
| Membership, delegation, authority grants | Connection pooling, SSL, driver load | Generic DB manager |
| Policy rules, operation profiles | Integrations slug/options merge | Second provider registry |
| Repository ports, CAS, idempotency fingerprints | Transaction/session providers | Generic retry framework |
| Authority resolver, policy composer, enforcement gate | Runtime policy engine | Parallel policy engine |
| Qualification suite semantics | Qualification runner infrastructure | Local qualification store |

**Violation found:** None material.

`persistence.py` imports concrete adapters at composition root — acceptable; domain modules (`authority.py`, `enforcement_gate.py`, `policy_composition.py`, `repository.py`, `policy_source.py`) contain no vendor imports (verified by `test_vendor_neutrality.py`).

---

## 5. Tenant / workspace isolation

All repository contracts require `tenant_id` + `workspace_id` on every get/create/update. PostgreSQL schema enforces composite primary keys and uniqueness scoped to `(tenant_id, workspace_id, …)`.

Qualification suite and contract tests prove cross-tenant reads return `None`. No global lookup by business ID alone without scope keys was found in production repository methods.

**Status:** OK.

---

## 6. Authority / delegation

`CollaborativeWorkAuthorityResolver`:

- Reloads membership/delegation/grants from repositories; ignores caller-supplied authority fields.
- Fails closed on missing/inactive membership, missing/inactive delegation, temporal invalidity, insufficient scopes.
- Delegated acting uses **delegator** base authority (intersection), not amplification.
- Role name does not grant authority; `PrincipalAuthorityGrant.authority_scopes` required.

**Status:** OK — matches CW-INV-03, CW-INV-10, CW-INV-12, CW-INV-17.

---

## 7. Policy / enforcement

`compose_policy_decisions` uses explicit precedence: DENY < REQUIRE_HUMAN/ESCALATE < MODIFY (normalized to DENY) < ALLOW. Missing mandatory layers → DENY. `UNKNOWN` applicability → DENY.

`CollaborativeWorkEnforcementGate` loads profile from repository; caller cannot override applicability. Missing/inactive profile → DENY. Runtime layer requires `MeaningfulSideEffectRequest` with identity cross-checks.

`MeaningfulSideEffectAuthorizationBoundary` evaluates gate before side effects; only `ALLOW` permits continuation.

**Bypass search:** No production path found that executes meaningful side effects while skipping the gate when boundary is wired. Tier-3 demo (`governed_contractor_application`) uses the boundary explicitly.

**Gap (P2):** Gate validates `resource`, `principal_id`, `action` alignment but not `side_effect_scope_id` correlation with `resource_scope` after GEC scope-id contract hardening.

---

## 8. Persistence

| Backend | Durable | Multi-instance | CAS | Uniqueness | Idempotency | Qualified |
|---------|---------|----------------|-----|------------|-------------|-----------|
| InMemory | No | No | Yes (lock) | Yes | Yes | Reference only |
| SQLite | Yes (file) | Single-writer honest | Yes | DB constraints | DB table | `cw.sqlite.repository.v1` QUALIFIED |
| PostgreSQL | Yes | Proven (A/B bundles, reconnect) | Yes | DB constraints | DB table | `cw.postgresql.repository.v1` PRODUCTION_QUALIFIED |

**Fail-closed:** No silent fallback from PostgreSQL to SQLite or in-memory when configured backend fails.

**Boundary:** CW adapters own domain SQL/schema; Integrations owns DSN, connection provider, schema search_path, driver import boundary (`import_psycopg`).

---

## 9. Provider abstraction

**Correct flow confirmed:**

```text
CW domain ports
  → persistence_provider.resolve_collaborative_work_repositories
  → Integrations registry (sqlite/postgresql binders)
  → open_*_collaborative_work_repositories
  → vendor repository implementations
```

- No `TypeError` probing or `hasattr` capability detection in authoritative wiring.
- `isinstance(factory, CollaborativeWorkMaterializationBinder)` is explicit typed protocol check.
- `postgresql_repository.py` does not import `psycopg` directly (session provider boundary).

---

## 10. Concurrency / CAS / idempotency

- Updates require `expected_revision`; mismatch raises typed `*RevisionConflict`.
- PostgreSQL integration tests prove concurrent update one-winner across independent connections.
- Idempotency: same key + same semantic fingerprint → stable replay; different fingerprint → `*IdempotencyConflict`.
- Cross-instance idempotency enforced via `collaborative_idempotency` table (PG/SQLite).

**Honest limits:** SQLite is not claimed for multi-instance production (architecture doc CW-INV-19).

---

## 11. Observability / diagnostics

CW library layer does **not** implement a private telemetry or diagnostics subsystem. Enforcement and composition attach structured `audit_payload` to `PolicyDecision`.

Runtime adoption (`MeaningfulSideEffectAuthorizationBoundary`) bridges to governed continuation for `REQUIRE_HUMAN` / `ESCALATE` using platform human runtime — no parallel approval store.

**Gap (P2):** CW repository/gate paths do not directly emit `RuntimeEvent` / `PlatformProblemSignal`; observability depends on host/runtime wiring above the boundary. Documented as intentional library/host split.

---

## 12. Evidence / auditability

Enforcement results carry:

- `operation_id`, `profile_revision`, `authority_scope`
- Per-layer decisions in `PolicyCompositionResult`
- `audit_payload` with `contributing_layers`, `determining_layer`

Sufficient for authority/policy investigation at the enforcement boundary. Full execution evidence (trace spine) remains platform-owned downstream.

---

## 13. Provider qualification relationship

Qualification proves provider persistence capability via `cw.postgresql.repository.v1` / `cw.sqlite.repository.v1` suites. It does **not** grant principal authority or bypass policies. Status is historical compatibility fact, separate from runtime authorization.

Multi-provider proof and real PostgreSQL execution qualification **PASS** at audit HEAD.

---

## 14. Security

| Check | Result |
|-------|--------|
| Authority fail-closed | PASS |
| Delegation non-amplifying | PASS |
| No silent provider fallback | PASS |
| No reflection wiring | PASS |
| Secret persistence in CW paths | Not observed |
| Typed public APIs | PASS |

**P0/P1 security findings:** None.

---

## 15. Test proof quality

| Suite | Strength | Notes |
|-------|----------|-------|
| `tests/unit/collaborative_work/` (218 tests) | Strong | Contracts, gate, composition, vendor neutrality, binding |
| `tests/integration/collaborative_work/test_postgresql_repository.py` | Strong | Real Docker PG, multi-connection concurrency |
| `repository_qualification_suite.py` | Strong | Cross-tenant, reopen, semantic checks |
| `test_provider_qualification_multi_provider_proof.py` | Strong | Neutral runner, sqlite + stub |
| In-memory adapters | Reference | Correctly marked `reference_only` |
| Enforcement gate tests | Fixed during audit | Drifted on `side_effect_scope_id` required field |

**Weak proofs:** None that invalidate enterprise claims for MP-1 scope.

---

## 16. Documentation truth

| Doc claim | Code truth | Match? |
|-----------|------------|--------|
| MP-1 core implemented | Yes | Yes |
| WorkItem / Artifact / Decision future (MP-2+) | Not in codebase | Yes |
| PostgreSQL production-qualified | Qualification + integration tests pass | Yes |
| SQLite local/lightweight | Yes | Yes |
| No central CW diagnostics engine | Yes | Yes |
| LKW first consumer | Plan states; **no CW imports in LKW app yet** | Partial — adoption pending |
| Final review pending | This audit | Closing |

No material doc overclaim for implemented MP-1 capabilities.

---

## 17. Capability map

| CAPABILITY | IMPLEMENTED? | ENTERPRISE-READY? | QUALIFIED? | LIMITATIONS | NEXT ACTION |
|------------|--------------|-------------------|------------|-------------|-------------|
| Shared membership | Yes | Yes | Via PG/SQLite suites | API/HTTP layer out of scope | LKW adoption |
| Delegation | Yes | Yes | Via PG/SQLite suites | — | — |
| Authority grants | Yes | Yes | Via PG/SQLite suites | — | — |
| Policy composition | Yes | Yes | Unit proofs | MODIFY → DENY at boundary | — |
| Enforcement gate | Yes | Yes | Unit + runtime bridge | `side_effect_scope_id` correlation P2 | Harden validation |
| Collaborative persistence | Yes | Yes (PG) | PG PRODUCTION_QUALIFIED, SQLite QUALIFIED | — | — |
| SQLite | Yes | Local/dev | QUALIFIED | Not multi-instance | — |
| PostgreSQL | Yes | Yes | PRODUCTION_QUALIFIED | Requires `integrations-postgresql` extra | — |
| Provider abstraction | Yes | Yes | Multi-provider proof | MySQL/Oracle not bound | Future adapters |
| Provider qualification | Yes | Yes | Runner + suites | — | — |
| Requalification | Yes | Platform-owned | Unit tests | — | — |
| Observability | Partial | Adoption-dependent | N/A | No CW-local telemetry | Host wiring |
| Diagnostics | Partial | Library outside execution host | N/A | By design | Document host boundary |
| WorkItem | No | No | No | MP-2 | Future gate |
| Artifacts | No | No | No | MP-3 | Future gate |
| Decisions | No | No | No | MP-4 | Future gate |
| Activity | No | No | No | MP-6 | Future gate |

---

## 18. Findings

### P0

None.

### P1

None.

### P2

1. **TEST-DRIFT-1:** `test_enforcement_gate.py` helper omitted required `side_effect_scope_id` after platform contract change — **fixed in this audit** (narrow test-only correction).
2. **ENFORCE-SCOPE-1:** Enforcement gate does not validate `MeaningfulSideEffectRequest.side_effect_scope_id` against trusted `resource_scope` / operation profile.
3. **ADOPTION-1:** LKW (`local_workspace_application`) does not yet wire durable CW repositories or enforcement boundary.
4. **OBS-1:** CW library does not emit canonical `RuntimeEvent` / `PlatformSignal`; observability relies on runtime host above `MeaningfulSideEffectAuthorizationBoundary`.

### P3

1. Update plan status from "final review pending" to "MP-1 audit closed" after acceptance.
2. Document explicit library-vs-host diagnostics boundary in architecture hub §Integration boundaries.

---

## 19. Required follow-ups (post-audit)

1. LKW adoption of `resolve_collaborative_work_repositories` + enforcement boundary (planned consumer integration).
2. Add `side_effect_scope_id` correlation check in enforcement gate when resource scope is required.
3. MP-2 gate: WorkItem / Assignment persistence on same repository pattern.
4. Optional: platform observability hooks at `MeaningfulSideEffectAuthorizationBoundary.authorize`.

---

## 20. Test execution (audit session)

```text
uv run pytest tests/unit/collaborative_work/ -q
  → 218 passed

uv run pytest tests/integration/collaborative_work/test_postgresql_repository.py -m "integration and network" -q
  → 15 passed (Docker PG, INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN)

uv run pytest tests/unit/core/qualification/ -q
  → 208 passed

uv run pytest tests/integration/core/qualification/test_provider_qualification_multi_provider_proof.py -q
  → 7 passed

uv run pytest tests/integration/core/qualification/test_provider_qualification_execution_postgresql.py -m "integration and network" -q
  → 1 passed

uv run pytest tests/unit/runtime/policy/test_meaningful_side_effect_authorization.py -q
  → 5 passed
```

Environment note: `uv sync --extra integrations-postgresql` required for psycopg-backed unit tests (`test_postgresql_platform_reuse.py`).

---

## 21. Acceptance gate checklist

| # | Criterion | Pass |
|---|-----------|------|
| 1 | No parallel provider mechanism | Yes |
| 2 | No vendor leakage in domain/core | Yes |
| 3 | Tenant/workspace isolation correct | Yes |
| 4 | Authority fails closed | Yes |
| 5 | Delegation cannot amplify | Yes |
| 6 | Policy composition deterministic | Yes |
| 7 | Meaningful side effects governed | Yes (at boundary) |
| 8 | Persistence durable per backend claims | Yes |
| 9 | CAS/concurrency honest | Yes |
| 10 | Idempotency honest | Yes |
| 11 | No silent provider fallback | Yes |
| 12 | Diagnostics/observability reuse platform | Yes (boundary) |
| 13 | No private telemetry system | Yes |
| 14 | Evidence boundaries coherent | Yes |
| 15 | Qualification ≠ authority shortcut | Yes |
| 16 | Hard typed-wiring preserved | Yes |
| 17 | Real PostgreSQL proof passes | Yes |
| 18 | SQLite focused regression passes | Yes |
| 19 | Qualification regression passes | Yes |
| 20 | Docs do not materially overclaim | Yes |

---

## 22. Product truth — what Multiplayer can do today

**A user / operator CAN (via library APIs and qualified persistence):**

- Register explicit workspace membership for a principal in a tenant/workspace.
- Grant scoped base authority to principals (not inferred from role).
- Create delegations with time bounds and resource scope; revoked/expired delegations do not authorize.
- Store collaborative policy rules and operation policy profiles with revisioned updates.
- Evaluate effective collaborative authority and composed policy decisions fail-closed.
- Block or allow meaningful operations through the enforcement gate before side effects run.
- Persist all MP-1 authoritative state durably in SQLite (local) or PostgreSQL (multi-instance).
- Prove PostgreSQL/SQLite provider compatibility through the shared qualification runner.

**A user CANNOT yet:**

- Create or track shared **WorkItems**, artifacts, or collaborative **Decisions** as product objects.
- Use Multiplayer features end-to-end inside LKW without additional adoption wiring.
- Rely on SQLite for multi-instance production clustering.
- Get automatic activity feeds or full audit traces from CW alone (requires runtime/host integration).

---

*End of audit document.*
