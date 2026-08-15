# Collaborative Work — Implementation Plan

**Architecture (1:1):** [`architecture/COLLABORATIVE_WORK.md`](../../architecture/COLLABORATIVE_WORK.md)
**Feature coordination:** [`capabilities/plan/MULTIPLAYER_AI.md`](../../capabilities/plan/MULTIPLAYER_AI.md)
**Architecture governance:** [`architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md)
**ADR:** [ADR-MP-001](../../technical/adr/entries/2026-08-11/ADR-MP-001.md) · [ADR-MP-002](../../technical/adr/entries/2026-08-11/ADR-MP-002.md)

**Status:** Domain registered — **MP-1 core — production persistence gate OPEN** (final review pending)
**First consumer:** `applications/local_workspace_application` (LKW)

---

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Default:** this read-scope block + active `COLLAB-WORK-*` row only.
- **Architecture hub:** [`architecture/COLLABORATIVE_WORK.md`](../../architecture/COLLABORATIVE_WORK.md) read-scope block only.
- **Feature context:** [`capabilities/plan/MULTIPLAYER_AI.md`](../../capabilities/plan/MULTIPLAYER_AI.md) active `MP-*` section only.

---

## 1. Objective

Deliver platform-owned collaborative identity and authority primitives reusable across Tier-3 applications, with LKW as first consumer.

Delivery order: **platform architecture/contract slice → platform implementation → LKW adoption → LKW live proof**.

---

## 2. Delivery principles

1. **Single domain ownership** — no shared ownership with Nexus, Application Hosting, or LKW.
2. **Semantics before persistence** — contracts and effective-authority model precede storage choices.
3. **Policy enforces; Collaborative Work defines** — membership/delegation source of truth stays here.
4. **Execution stays in Nexus** — `RequestIdentity` and `DelegationSpec` remain execution contracts.
5. **Fail closed** — privileged collaborative mutations require provable effective authority.
6. **No LKW-owned platform types** — consumer integration only.

---

## 3. Implementation waves

### COLLAB-WORK-0 — Architecture and governance

| ID | Task | Status |
|----|------|--------|
| COLLAB-WORK-0A | ADR-MP-001 Collaborative Work Plane ownership | **Done** (MP-1A) |
| COLLAB-WORK-0B | ADR-MP-002 Principal / Membership / Delegation semantics | **Done** (MP-1A) |
| COLLAB-WORK-0C | Domain architecture + plan pair registration | **Done** (MP-1A) |
| COLLAB-WORK-0D | Multiplayer hub ownership synchronization | **Done** (MP-1A) |

COLLAB-WORK-0 closes with **0D Done**. Runtime implementation begins at **COLLAB-WORK-1A** after MP-1 review acceptance.

### COLLAB-WORK-1 — MP-1 architecture / contract slice

| Field | Value |
|-------|-------|
| **ID** | COLLAB-WORK-1A |
| **Priority** | P0 |
| **Status** | **APPROVED / CLOSED** |
| **Purpose** | Principal, WorkspaceMembership, Delegation, and effective-authority contract slice |
| **Dependencies** | MP-0 accepted; ADR-MP-001 and ADR-MP-002 accepted |
| **Exact scope** | Semantic contracts in `intergrax/contracts/collaborative_work.py`; effective-authority resolution boundary; fail-closed enforcement hook design |
| **REUSED** | `MeaningfulSideEffectRequest` policy enforcement; `RequestIdentity` as execution-intake bridge only; `PolicyDecision` / `PolicyAction` for authority outcomes |
| **NEW** | `CollaborativePrincipal`, `WorkspaceMembership`, `AuthorityDelegation`, `EffectiveAuthorityRequest`, `EffectiveAuthorityDecision` |
| **Explicit out of scope** | DB models, repositories, APIs, HTTP routes, membership/delegation services, LKW changes, tests for runtime code, MP-2+ rows |
| **Acceptance** | Contracts frozen; effective authority intersection documented; membership explicit; delegation non-amplifying; execution contracts not duplicated |
| **Proof requirements** | `tests/unit/contracts/test_collaborative_work.py` — contract, isolation, fail-closed, delegation validation |
| **Next step** | COLLAB-WORK-1B — membership/delegation persistence and concurrency |

| Field | Value |
|-------|-------|
| **ID** | COLLAB-WORK-1B |
| **Priority** | P0 |
| **Status** | **APPROVED / CLOSED** |
| **Purpose** | Membership / Delegation persistence and concurrency |
| **Dependencies** | COLLAB-WORK-1A approved; `intergrax/contracts/collaborative_work.py` revision semantics |
| **Exact scope** | Provider-neutral `WorkspaceMembershipRepository` and `AuthorityDelegationRepository` contracts; optimistic `expected_revision` updates; tenant/workspace scoped keys; create idempotency; in-memory reference adapters |
| **REUSED** | `WorkspaceMembership`, `AuthorityDelegation` contracts; repository/exception patterns from platform binding repositories; `threading.RLock` for in-memory adapter |
| **NEW** | `intergrax/collaborative_work/repository.py`, `intergrax/collaborative_work/in_memory_repository.py`, focused repository unit tests |
| **Explicit out of scope** | Effective-authority resolution; `CollaborativePrincipal` repository; SQL/Redis/Postgres backends; HTTP/API; LKW adoption; PolicyEngine integration; MP-2+ rows |
| **Acceptance** | Authoritative create/get/update semantics; no silent last-write-wins; scoped isolation; typed conflict/not-found outcomes; idempotent create replay |
| **Proof requirements** | `tests/unit/collaborative_work/test_in_memory_repository.py`; contract regression `tests/unit/contracts/test_collaborative_work.py` |
| **Next step** | COLLAB-WORK-1C — authoritative effective-authority state resolution |

| Field | Value |
|-------|-------|
| **ID** | COLLAB-WORK-1C |
| **Priority** | P0 |
| **Status** | **APPROVED / CLOSED** |
| **Purpose** | Authoritative effective-authority state resolution using reloaded `WorkspaceMembership` and `AuthorityDelegation` repository records — collaborative slice only (`acting principal ∩ active membership ∩ active/valid delegation ∩ requested scope structural checks`) |
| **Dependencies** | COLLAB-WORK-1B approved; `WorkspaceMembershipRepository` and `AuthorityDelegationRepository` ports |
| **Exact scope** | `CollaborativeWorkAuthorityResolver` in `intergrax/collaborative_work/authority.py`; authoritative rehydration; fail-closed deny paths; deterministic clock injection for delegation validity |
| **REUSED** | `EffectiveAuthorityRequest`, `EffectiveAuthorityDecision`, `EffectiveAuthorityDenialReason`, `PolicyDecision` / `PolicyAction`; repository ports from COLLAB-WORK-1B |
| **NEW** | Authoritative resolver implementation; focused unit tests in `tests/unit/collaborative_work/test_effective_authority.py` |
| **Explicit out of scope** | Workspace policy, resource policy, runtime/tool policy evaluation; principal base-authority source; membership role→capability RBAC mapping; delegator authority non-amplification proof beyond fail-closed `SCOPE_ONLY_INSUFFICIENT`; PolicyEngine changes; execution enforcement; MP-2+ rows |
| **Acceptance** | Caller-supplied membership/delegation never trusted for authority semantics; repository reload wins; direct and delegated acting paths fail closed when final authority cannot be proven; stale/tampered embedded evidence cannot authorize |
| **Proof requirements** | `tests/unit/collaborative_work/test_effective_authority.py`; repository and contract regressions |
| **Next step** | COLLAB-WORK-1D — principal base authority and collaborative authority composition |

| Field | Value |
|-------|-------|
| **ID** | COLLAB-WORK-1D |
| **Priority** | P0 |
| **Status** | **APPROVED / CLOSED** |
| **Purpose** | Authoritative principal base-authority source and first security-correct Collaborative Work ALLOW (`base authority ∩ membership ∩ delegation ∩ requested scope`) |
| **Dependencies** | COLLAB-WORK-1C approved; `CollaborativeWorkAuthorityResolver`; COLLAB-WORK-1B repository semantics |
| **Exact scope** | `PrincipalAuthorityGrant` contract; `PrincipalAuthorityRepository` port and in-memory adapter; resolver composition with delegator non-amplification; new denial reasons; collaborative-slice ALLOW only |
| **REUSED** | `EffectiveAuthorityDecision`, `PolicyDecision` / `PolicyAction`; COLLAB-WORK-1B revision/idempotency semantics; membership and delegation repositories |
| **NEW** | `PrincipalAuthorityGrant`, `PrincipalAuthorityRepository`, `InMemoryPrincipalAuthorityRepository`; base-authority resolution in resolver; repository and resolver tests |
| **Explicit out of scope** | Workspace/resource/runtime policy composition; role→permission RBAC; wildcard semantics; PolicyEngine changes; execution enforcement; MP-2+ rows |
| **Acceptance** | Base authority authoritative not caller-evidence; delegator base authority required for delegated acting; delegation cannot amplify; ALLOW scoped to collaborative slice only; membership role does not substitute base authority |
| **Proof requirements** | `tests/unit/collaborative_work/test_effective_authority.py`; `tests/unit/collaborative_work/test_in_memory_repository.py`; contract regressions |
| **Next step** | COLLAB-WORK-1E — policy composition and final enforcement decision boundary |

| Field | Value |
|-------|-------|
| **ID** | COLLAB-WORK-1E |
| **Priority** | P0 |
| **Status** | **APPROVED / CLOSED** |
| **Purpose** | Fail-closed composition of collaborative authority, workspace, resource, and runtime/tool policy decisions into one final enforcement decision |
| **Dependencies** | COLLAB-WORK-1D approved; `CollaborativeWorkAuthorityResolver`; `RuntimePolicyEngine.evaluate_meaningful_side_effect`; `PolicyDecision` / `PolicyAction` |
| **Exact scope** | `compose_policy_decisions` composition boundary; `PolicyCompositionInput` / `PolicyCompositionResult` contracts; mandatory/applicable layer semantics; audit provenance |
| **REUSED** | `PolicyDecision`, `PolicyAction`, `EffectiveAuthorityDecision`; `MeaningfulSideEffectRequest`; `RuntimePolicyEngine` / `PolicyEngine` meaningful-side-effect path |
| **NEW** | `intergrax/collaborative_work/policy_composition.py`; composition contracts; `tests/unit/collaborative_work/test_policy_composition.py` |
| **Explicit out of scope** | Workspace/resource policy evaluators; second PolicyEngine; side-effect execution; widespread runtime rewiring; MP-2+ rows |
| **Acceptance** | Collaborative ALLOW alone does not authorize execution; missing mandatory layer fails closed; DENY/REQUIRE_HUMAN/ESCALATE/MODIFY never weakened; contributing decisions auditable |
| **Proof requirements** | `tests/unit/collaborative_work/test_policy_composition.py`; resolver regression if integration requires |
| **Next step** | COLLAB-WORK-1F — authoritative workspace and resource policy source |

| Field | Value |
|-------|-------|
| **ID** | COLLAB-WORK-1F |
| **Priority** | P0 |
| **Status** | **APPROVED / CLOSED** |
| **Purpose** | Authoritative workspace and resource policy source returning real ``PolicyDecision`` values for composition |
| **Dependencies** | COLLAB-WORK-1E approved; `compose_policy_decisions`; `PolicyDecision` / `PolicyAction` |
| **Exact scope** | `CollaborativePolicyRule` contract; `CollaborativePolicyRepository` port; in-memory adapter; `CollaborativePolicyEvaluator`; exact-key lookup; fail-closed evaluation |
| **REUSED** | `PolicyDecision`, `PolicyAction`, `PolicyCompositionLayer`; COLLAB-WORK-1B repository revision/idempotency semantics |
| **NEW** | `intergrax/collaborative_work/policy_source.py`; policy repository extensions; `tests/unit/collaborative_work/test_collaborative_policy_source.py` |
| **Explicit out of scope** | Policy DSL; second PolicyEngine; operation applicability classifier; enforcement wiring; policy-management authorization/API; wildcard/inheritance; MP-2+ rows |
| **Acceptance** | Workspace/resource evaluators return authoritative ``PolicyDecision``; missing/inactive rules DENY; exact scope matching only; outputs compose with 1E; policy administration authorization unsolved |
| **Proof requirements** | `tests/unit/collaborative_work/test_collaborative_policy_source.py`; `tests/unit/collaborative_work/test_policy_composition.py` regressions |
| **Next step** | COLLAB-WORK-1G — trusted operation classification and final enforcement gate |

| Field | Value |
|-------|-------|
| **ID** | COLLAB-WORK-1G |
| **Priority** | P0 |
| **Status** | **APPROVED / CLOSED** |
| **Purpose** | Authoritative operation policy classification and reusable final enforcement gate orchestrating authority, workspace/resource/runtime policy, and composition |
| **Dependencies** | COLLAB-WORK-1E approved; COLLAB-WORK-1F approved; `compose_policy_decisions`; `CollaborativeWorkAuthorityResolver`; `CollaborativePolicyEvaluator`; `RuntimePolicyEngine.evaluate_meaningful_side_effect` |
| **Exact scope** | `CollaborativeOperationPolicyProfile` contract; profile repository port; in-memory adapter; `CollaborativeWorkEnforcementGate`; runtime identity validation; fail-closed classification |
| **REUSED** | `PolicyCompositionApplicability`; `compose_policy_decisions`; authority resolver; workspace/resource evaluators; `RuntimePolicyEngine` / `PolicyEngine` meaningful-side-effect path; COLLAB-WORK-1B revision semantics |
| **NEW** | `intergrax/collaborative_work/enforcement_gate.py`; operation profile contracts; profile repository; `tests/unit/collaborative_work/test_enforcement_gate.py`; `tests/unit/collaborative_work/test_operation_policy_profile_repository.py` |
| **Explicit out of scope** | Broad application adoption; LKW operation catalog; policy management APIs; durable SQL adapter; MP-2+ rows |
| **Acceptance** | Caller cannot control applicability or authority scope; active profile classifies layers; missing/disabled profile DENY; required resource/runtime validated; final ALLOW only when all required layers ALLOW; no operation execution in gate |
| **Proof requirements** | `tests/unit/collaborative_work/test_enforcement_gate.py`; `tests/unit/collaborative_work/test_operation_policy_profile_repository.py`; policy composition and policy source regressions |
| **Next step** | COLLAB-WORK-1H — durable collaborative state and first production enforcement adoption |

| Field | Value |
|-------|-------|
| **ID** | COLLAB-WORK-1H |
| **Priority** | P0 |
| **Status** | **APPROVED / CLOSED** |
| **Purpose** | Durable authoritative Collaborative Work security state and first canonical production enforcement adoption boundary |
| **Dependencies** | COLLAB-WORK-1G approved; `CollaborativeWorkEnforcementGate`; COLLAB-WORK-1B repository semantics |
| **Exact scope** | SQLite durable repository adapters for MP-1 authoritative entities; `open_sqlite_collaborative_work_repositories` composition factory; `MeaningfulSideEffectAuthorizationBoundary` runtime adoption boundary |
| **REUSED** | In-memory reference adapters; `SQLiteOptimizationArtifactRepository` persistence pattern; `RuntimePolicyEngine` meaningful-side-effect path; existing repository ports and revision/idempotency semantics |
| **NEW** | `intergrax/collaborative_work/sqlite_repository.py`, `serialization.py`, `persistence.py`; `intergrax/runtime/policy/meaningful_side_effect_authorization.py`; contract and adoption tests |
| **Explicit out of scope** | MP-2+ rows; broad connector/application wiring; observability routing layer; Alembic/Postgres vendor lock-in; second runtime or HITL engine |
| **Acceptance** | Durable adapters report `durable=True` / `reference_only=False`; CAS/idempotency/uniqueness parity with in-memory; shared boundary invokes gate before side effects; ALLOW-only continuation; fail closed on missing state |
| **Proof requirements** | `tests/unit/collaborative_work/test_repository_contracts.py`; `tests/unit/runtime/policy/test_meaningful_side_effect_authorization.py`; `tests/unit/collaborative_work/test_vendor_neutrality.py`; enforcement gate regressions |
| **Next step** | MP-1 CORE FINAL REVIEW |

| Field | Value |
|-------|-------|
| **ID** | COLLAB-WORK-1H-R2 |
| **Priority** | P0 |
| **Status** | **READY_FOR_REVIEW** |
| **Purpose** | MP-1 core security and typed-wiring closure — canonical principal membership, delegator membership validity, zero dynamic attribute mutation |
| **Dependencies** | COLLAB-WORK-1H approved lineage; `CollaborativeWorkAuthorityResolver`; repository ports |
| **Exact scope** | `get_for_principal` membership lookup; principal uniqueness in memory/SQLite; delegator active membership gate; Pydantic repository commands; AST typed-wiring proof |
| **REUSED** | COLLAB-WORK-1B revision/idempotency semantics; durable SQLite adapter; effective-authority resolver |
| **NEW** | CW-INV-16…18; `MISSING_DELEGATOR_MEMBERSHIP` / `DELEGATOR_MEMBERSHIP_NOT_ACTIVE`; closure tests |
| **Explicit out of scope** | LKW integration; MP-2+; observability; Postgres |
| **Acceptance** | One membership per principal/workspace; delegator must remain active member; no `getattr`/`setattr` family in scoped production paths; docs synchronized |
| **Proof requirements** | `tests/unit/collaborative_work/test_canonical_membership_closure.py`; `tests/unit/collaborative_work/test_typed_wiring_architecture.py`; authority and repository regressions |
| **Next step** | MP-1 CORE FINAL REVIEW |

| Field | Value |
|-------|-------|
| **ID** | COLLAB-WORK-1H-R3 |
| **Priority** | P0 |
| **Status** | **READY_FOR_REVIEW** |
| **Purpose** | SQLite canonical membership migration closure — schema parity with fresh databases |
| **Dependencies** | COLLAB-WORK-1H-R2 canonical membership semantics; durable SQLite adapter |
| **Exact scope** | Transactional rebuild of legacy `workspace_memberships` to `principal_id TEXT NOT NULL` plus unique principal membership |
| **REUSED** | Canonical `WorkspaceMembership` `record_json`; SQLite adapter `BEGIN IMMEDIATE` conventions |
| **NEW** | Legacy table rebuild; explicit duplicate/identity/deserialisation failure with rollback |
| **Explicit out of scope** | Authority semantics; LKW integration; MP-2+; Alembic/SQLAlchemy |
| **Acceptance** | Migrated schema matches fresh constraints; duplicates fail closed; original legacy table intact on failure; reopen is idempotent |
| **Proof requirements** | `tests/unit/collaborative_work/test_sqlite_membership_migration.py`; repository, canonical membership, authority, typed-wiring, vendor-neutrality regressions |
| **Next step** | MP-1 CORE FINAL REVIEW |

| Field | Value |
|-------|-------|
| **ID** | COLLAB-WORK-1J |
| **Priority** | P0 |
| **Status** | **READY_FOR_REVIEW** |
| **Purpose** | PostgreSQL durable backend and production parity — cross-process transactional concurrency proof |
| **Dependencies** | COLLAB-WORK-1H durable SQLite adapter; platform PostgreSQL integration (`psycopg`, `PostgreSQLIntegrationConfig`) |
| **Exact scope** | `PostgreSQLCollaborativeWorkStore` + typed repositories for all MP-1 authoritative ports; `open_postgresql_collaborative_work_repositories`; real PostgreSQL parity/concurrency/integration tests |
| **REUSED** | Repository ports; serialization; SQLite semantic reference; `infra/docker/postgresql/docker-compose.yml` |
| **NEW** | Production PostgreSQL adapter; schema constraints/indexes; multi-connection CAS/uniqueness/idempotency proofs; CW-INV-19 |
| **Explicit out of scope** | LKW integration; MP-2+; Alembic unless platform canon changes; CI platform redesign |
| **Acceptance** | Semantic parity with InMemory/SQLite; real PostgreSQL tests pass; no production fallback to SQLite; vendor neutrality and typed-wiring gates hold |
| **Proof requirements** | `tests/integration/collaborative_work/test_postgresql_repository.py`; repository/authority/enforcement/typed-wiring/vendor-neutrality regressions |
| **Next step** | MP-1 CORE FINAL REVIEW |

---

## 4. Out of scope (current phase)

- MP-2 WorkItem / Assignment implementation rows
- MP-3…MP-6 architecture or implementation rows
- LKW product adoption (MP-7)
- Runtime Python models beyond contract stubs when implementation gate opens
