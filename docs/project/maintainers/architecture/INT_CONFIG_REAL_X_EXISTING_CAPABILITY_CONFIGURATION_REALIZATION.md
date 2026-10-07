# INT-CONFIG-REAL-X-P0 — Existing Capability Configuration Realization Architecture Lock

## 1. Status and scope

| Field | Value |
| ----- | ----- |
| **Task** | `INT-CONFIG-REAL-X-P0` + `INT-CONFIG-REAL-X-P0-R1` + `INT-CONFIG-REAL-X-P0-R2` (architecture lock; children of `INT-CONFIG-REAL-X`) |
| **Parent** | `INT-CONFIG-REAL-X` — Existing Capability Configuration Realization |
| **Program baseline** | `234d3c03dce5708766f8e068afbd2ea496132002` (`development`) |
| **Source** | Scenario #24 GAP-01 — `CONFIGURE_EXISTING` disposition |
| **Canonical owner** | **Integrations** |
| **Artifact role** | Closed-world design record before `INT-CONFIG-REAL-X-P1` |
| **Production / tests** | **0** in P0 |
| **P0-R1** | Configuration authorization evidence boundary closure (scope-binding; production = 0) |
| **P0-R2** | Canonical Governance authorization invocation boundary closure (docs-only; production = 0) |
| **Status** | **P0-R2: READY FOR INDEPENDENT ARCHITECTURE AUDIT** — **P0-R1:** accepted in substance; **BLOCKED BY R2** (provenance seam) — **P0:** **BLOCKED** pending independent P0-R2 audit + **P0-R2-I1** |

**Scope:** lock reusable platform semantics for realizing **existing** integration/capability configuration under explicit tenant scope. P0 does **not** implement contracts, services, or qualification tests.

**Explicit non-closure:** this lock does **not** close `COMPAT-X`, `TENANT-X`, `TRACE-X`, or any global `FRZ-*` PASS.

**UCA:** consumer only after public contract certification; UCA must **not** implement GAP-01 realization.

---

## 2. Problem statement

Scenario recovery and upstream discovery can emit **`CONFIGURE_EXISTING`**: the required provider/capability **already exists** in the platform catalog, but a **tenant-scoped, validated configuration** must be realized before sanctioned composition can use it.

Today, generic integration configuration seams (`IntegrationProfile.options`, `resolve(..., config=...)`, pre-built `IntegrationBinding.instance`) support **host-level provider selection** and **legacy construction kwargs**. They do **not** establish:

- canonical tenant ownership for enterprise configuration realization;
- typed configuration **request** semantics;
- typed provider configuration **payload** contract;
- approved-configuration **evidence** semantics at the realization boundary;
- a **realization strategy** SPI;
- a tenant-scoped **configured-capability** result distinct from effective/authorized/executing state;
- prohibitions against provider/tenant substitution during realization;
- a configuration-specific **authority** boundary separate from Governance and Execution.

**STOP — ARCHITECTURE DECISION REQUIRED** applies to production implementation of `INT-CONFIG-REAL-X` until this P0 lock is independently accepted.

---

## 3. Non-goals

- Capability Acquisition, Marketplace acquisition, or minting new global capabilities
- New capability creation or global provider registration/mutation
- Second Integration Catalog or second provider resolver
- Governance policy decision minting inside Integrations realization
- Execution admission, ToolRuntime invocation, or business-operation execution
- Whole-platform `CONFIG-X` closure (parent precedes `CONFIG-X` but does not replace it)
- Silent promotion of `dict[str, Any]` profile options as the enterprise realization API
- `ConfigurationStrategyRegistry`, reflection discovery, service-locator strategy lookup
- Redesign of `IntegrationPlugin` authority (separate future decision if required)

---

## 4. Canonical ownership matrix

| Concern | Canonical owner |
| ------- | ---------------- |
| `CONFIGURE_EXISTING` disposition | Existing upstream discovery/recovery decision owner (Autonomous Work / recovery — **not** Integrations) |
| Configuration realization semantics | **Integrations** |
| Realization request/result contracts | **Integrations contracts** (`intergrax/integrations/contracts/**` or sanctioned Integrations-owned contract package) |
| Provider existence/discovery | Existing **Integration Catalog** (`intergrax/integrations/registry/catalog.py`) |
| Provider selection/materialization | Existing sanctioned Integrations resolver/composition (`resolve`, `resolve_from_profile`, catalog `factory`) |
| Provider-specific configuration validation | Injected **realization strategy** / provider extension through platform SPI |
| Tenant identity | Request subject/context; **immutable** through realization |
| Configuration authorization | **Governance** — not realization |
| Execution admission/lifecycle | **Execution** — not realization |
| Capability Acquisition | Explicitly **not** involved |
| Marketplace | Explicitly **not** involved |
| Observability | Records realization evidence only; **no** truth/authority creation |

Exactly one owner per row.

---

## 5. Existing-state inventory

| Seam | Current shape | Role today |
| ---- | ------------- | ---------- |
| `IntegrationProfile` | Typed per-category `IntegrationBinding` slots | Tier-3 host provider **selection** |
| `IntegrationProfile.options` | `dict[str, dict[str, Any]]` | Per-slug construction kwargs bag |
| `IntegrationProfile.options_for_slug(...)` | `dict[str, Any]` | Slug-keyed options lookup |
| `IntegrationBinding.instance` | `Any` | Pre-built integration object bypassing factory |
| `resolve(..., config: Mapping[str, Any])` | Merges profile options + override config → `entry.factory(**merged)` | Catalog materialization |
| `resolve_from_profile` | Profile slot → catalog resolve or pre-built instance | Sanctioned composition path |
| Integration Catalog | `get_entry(slug)` + `IntegrationEntry.factory` | **Single** provider registration/resolution authority — see [`INTEGRATION_REGISTRY_CANONICAL_AUTHORITY.md`](INTEGRATION_REGISTRY_CANONICAL_AUTHORITY.md) |
| `IntegrationPlugin` | Catalog registration + factory | External provider extension (unchanged authority in P0) |

**UCA boundary:** UCA owns gap/acquisition/qualification flows; GAP-01 realization is **transferred** to platform Integrations (`INT-CONFIG-REAL-X`). UCA must not implement realization.

**Governance boundary:** permission is admitted only through **`ControlPlaneMutationAuthorizationPort.authorize(...)`** in the sanctioned realization path; lower layers may verify typed **`ControlPlaneMutationAuthorizationEvidence`** scope continuity only after port admission — they must **not** decide permission, construct authority from DTOs alone, or import concrete Governance runtime.

**Execution boundary:** configured capability is handed to downstream consumers; Execution owns runnable work after Governance where required.

---

## 6. Architectural gap

```text
CONFIGURE_EXISTING (upstream)
        ↓
   [missing certified reusable realization contract]
        ↓
weak profile/options/config seams (not tenant-realization semantics)
```

Generic seams do **not** themselves establish tenant-scoped realization, typed payloads, strategy SPI, or configured ≠ effective ≠ authorized ≠ executing distinctions at an enterprise boundary.

---

## 7. Canonical flow (locked)

**Governance authorizes · Integrations realizes · Execution executes** — no layer absorbs another role.

```text
CONFIGURE_EXISTING disposition (upstream)
        ↓
ExistingCapabilityConfigurationRealizationRequest (immutable typed)
        ↓
public production entry: governed realization façade / coordinator (P1; e.g. ExistingCapabilityConfigurationRealizationFacade)
        ↓
project → ControlPlaneMutationRequest (deterministic; P1)
        ↓
ControlPlaneMutationAuthorizationPort.authorize(request)   # injected; Governance semantics
        ↓
ControlPlaneMutationAuthorizationResult (permitted + PolicyAction.ALLOW only)
        ↓
require permitted == True and decision.action == PolicyAction.ALLOW
        ↓
verify exact result.evidence scope (R1 checks; fail closed)
        ↓
pure realization core (internal; no permission decisions)
        ↓
ExistingCapabilityConfigurationRealizationStrategy (platform SPI; composed)
        ↓
validate target + tenant + typed payload (fail closed)
        ↓
typed existing-integration resolver port → sanctioned Catalog/resolver
        ↓
ExistingCapabilityConfigurationRealizationResult (tenant-scoped configured binding reference)
        ↓
later: effective composition (Integrations) → Governance (separate use paths) → Execution
```

**P0-R2 invariant:** `strategy` cannot become reachable without a call to the injected Governance **`ControlPlaneMutationAuthorizationPort`** in the same sanctioned operation path.

**Forbidden public production path:** `service.realize(request, authorization_evidence=caller_constructed_evidence)` when that path can reach a strategy without canonical port invocation.

P0-R1 locks evidence scope-binding; P0-R2 locks authorization invocation provenance; **P0-R2-I1** promotes the port in contracts/runtime; P1 implements façade, projection, composition wiring, and verification.

**Forbidden shortcuts:** acquisition fallback, Marketplace fallback, global provider mutation, second catalog/resolver, ToolRuntime, direct business side effects.

---

## 8. Request contract semantics

Immutable typed request equivalent: **`ExistingCapabilityConfigurationRealizationRequest`**.

| Semantic field | Requirement |
| -------------- | ----------- |
| `request_id` | Correlation identity for the realization attempt |
| `tenant_id` | Explicit tenant; missing → fail closed |
| Integration / capability category | Which catalog category slot is being configured |
| Target integration/provider identity | Exact provider/slug/manifest identity — **not** “discover any” |
| Typed configuration payload | See §9 — **not** `dict[str, Any]` at semantic boundary |
| Configuration schema/version identity | `configuration_type` + `configuration_version` (or equivalent stable ids) |
| Host/binding scope | Where applicable (host profile revision, binding scope) |
| Governance principal / context | Canonical `RequestIdentity` from upstream authenticated/governed context — Integrations must **not** synthesize principal |
| Configuration fingerprint | Stable identity of **desired** configuration (distinct from Governance `request_digest`) |
| Correlation to control-plane mutation | `request_id` (or equivalent) binds to projected `mutation_id`; not a substitute for typed authorization evidence |
| Correlation / trace reference | Observability continuity only |

**Rules:**

1. Tenant explicit; missing tenant fails closed.
2. Target provider explicit; request cannot mean discover-any or acquire-missing.
3. Target capability must already exist in catalog (resolver confirms).
4. Provider-specific settings are **not** arbitrary metadata bags at the generic boundary.
5. Request does **not** grant permission (Governance ≠ realization).

Exact Python field names deferred to P1 repository conventions.

---

## 9. Typed configuration payload (central decision)

The generic realization boundary must **not** use `dict[str, Any]`, `Mapping[str, Any]`, `object`, or untyped metadata for **provider-specific realization semantics**.

**Locked model:** platform-defined extensibility via **`IntegrationConfigurationPayload`** (Protocol / typed contract) carrying stable platform-owned identity:

```text
configuration_type
configuration_version
```

Provider-specific types (e.g. conceptually `RedisConfiguration`, `PostgresConfiguration`, `ExternalApiConfiguration`) **structurally implement** the platform contract.

- Generic realization code must **not** inspect arbitrary vendor fields.
- Provider strategy owns interpretation and validation of its typed payload.
- **No** closed central union of every vendor configuration in generic core.
- **No** vendor branches in generic realization service.

---

## 10. Realization strategy SPI

Platform structural contract equivalent: **`ExistingCapabilityConfigurationRealizationStrategy`**.

| Responsibility | Semantics |
| -------------- | --------- |
| `strategy_id` | Stable strategy identity |
| `can_realize(request) -> bool` | Structural match on category + configuration_type/version + target |
| `realize(request, existing target/context) -> typed realization result` | Validate payload; produce configured binding material |

**Hard restrictions (strategy must not):**

- change tenant or choose unrelated provider;
- acquire capabilities or register global providers;
- mutate Integration Catalog;
- grant authority or invoke Execution / ToolRuntime;
- return arbitrary `object` as the semantic result;
- create global configuration side effects.

Structural external implementation must be possible (P2 qualification).

---

## 11. Sanctioned composition (exactly one model)

**Two-layer realization (P0-R2 lock):**

| Layer | Owner | Responsibility |
| ----- | ----- | -------------- |
| Governed realization façade / coordinator | Integrations orchestration (e.g. **`ExistingCapabilityConfigurationRealizationFacade`**) | project request → **`ControlPlaneMutationAuthorizationPort.authorize`** → validate result → call pure core |
| Pure realization core | Integrations | validate target/config/state → select one strategy → realize; **no** permission decisions |

**Exactly one authorization invocation owner** within configuration realization: the governed façade (or equivalent Integrations-owned orchestration seam). The strategy, provider, resolver, and pure core **must not** call Governance or hold an authorization port.

**Pure core (locked; internal entry):**

```text
ExistingCapabilityConfigurationRealizationService(
    strategies: tuple[ExistingCapabilityConfigurationRealizationStrategy, ...],
    existing_integration_resolver: <typed port>,
)
```

The pure core **does not** call `ControlPlaneMutationAuthorizationPort`, `ControlPlaneMutationAuthorizationBoundary`, or any `intergrax/runtime/governance/**` implementation. It may accept already-validated **`ControlPlaneMutationAuthorizationEvidence`** only on an **internal / non-authority** API for traceability and scope continuity — not as a second public production entry.

**Sanctioned public production composition (locked):**

```text
host / sanctioned platform composition wires:
    authorization_port: ControlPlaneMutationAuthorizationPort   # concrete: ControlPlaneMutationAuthorizationBoundary
    realization_facade: ExistingCapabilityConfigurationRealizationFacade(
        authorization_port=authorization_port,
        realization_core=...,
    )
    ↓
facade.realize(realization_request)   # public production entry (P1 naming may vary; semantics locked)
    ↓
build canonical ControlPlaneMutationRequest from realization request
    ↓
authorization_port.authorize(governance_request)
    ↓
ControlPlaneMutationAuthorizationResult
    ↓
if result.permitted and result.decision.action == PolicyAction.ALLOW:
    verify exact result.evidence scope (R1)
    ↓
    realization_core.realize_internal(..., validated_evidence=result.evidence)   # internal only
```

Exact method signatures deferred to P1 code conventions.

Rationale: explicit DI on the **public Protocol**; no second registry; Governance semantic owner stays behind **`ControlPlaneMutationAuthorizationPort`**; Integrations depends on **`intergrax/contracts/control_plane_mutation.py`** only — not concrete runtime Governance.

**Forbidden:** `ConfigurationStrategyRegistry`, `RealizationProviderRegistry`, tenant→strategy global maps, vendor switches, service locators, reflection discovery; Integrations production imports of `intergrax/runtime/governance/**`; public **`realize(..., authorization_evidence=...)`** that bypasses port invocation.

If `IntegrationPlugin` redesign is later required to host strategies: record **`STOP — ARCHITECTURE DECISION REQUIRED`** as a separate decision; P1 preferred path uses explicit composition **without** modifying plugin registration authority.

---

## 12. Existing integration resolution reuse

**Integration Catalog / sanctioned `resolve` / `resolve_from_profile`** remain the **only** owners of provider existence and materialization.

**Typed port:** `existing-integration resolver` (conceptual) delegating to sanctioned paths.

| Allowed | Forbidden |
| ------- | --------- |
| Does target exist for category? | Discover arbitrary alternative providers |
| Resolve exact requested target | Choose vendor by business semantics |
| | Acquire missing provider |

Realization service is **not** a second provider resolver.

---

## 13. Legacy `IntegrationProfile.options` classification

**Decision: Option A (recommended, locked for P0)**

- `IntegrationProfile.options: dict[str, dict[str, Any]]` and `resolve(..., config: Mapping[str, Any])` remain **compatibility / internal construction** seams for Tier-3 host wiring.
- `INT-CONFIG-REAL-X` introduces a **typed realization boundary** that may adapt into existing provider construction **only** inside a bounded Integrations-owned adapter at the strategy/provider edge.
- Legacy options are **not** silently declared the canonical enterprise configuration realization contract.

**Option B (not selected):** harden all legacy seams before realization → would require mandatory child `INT-CONFIG-REAL-X-P1A` and keep parent blocked. **Not required** at P0 based on closed-world inventory.

---

## 14. Result contract semantics

Immutable typed result: **`ExistingCapabilityConfigurationRealizationResult`**.

| Semantic field | Requirement |
| -------------- | ----------- |
| `request_id` | Ties to request |
| `tenant_id` | Must equal request tenant |
| Target provider/integration identity | Must equal request target (no cross-provider unless separate governed contract — **not** introduced here) |
| Configuration type/version | From realized payload |
| Configured binding / capability reference | Tenant-scoped handle for downstream composition |
| Configuration fingerprint or stable identity | Deterministic identity for evidence |
| Realization evidence refs | Observability/audit pointers; no raw secrets |

**Must not imply:** `authorized`, `executed`, `globally activated`.

**Lifecycle state at result:** `realized/configured` only — not `effective`, not `executing`.

---

## 15. Tenant isolation (architecture lock)

**Tenant scope: YES.**

```text
realization_request.tenant_id
== governance_request.tenant_id (via principal.tenant_id)
== authorization_evidence.tenant_id
== strategy input tenant
== resolved target tenant scope
== result.tenant_id
```

**Forbidden:** tenant A request → tenant B binding; missing tenant → global config; tenant-scoped config → global provider mutation; strategy rewrites tenant.

**P0-R2:** tenant-scoped permission must be obtained through the **same** Governance port invocation that admits that exact realization operation. Preconstructed tenant-A evidence must not be passed into a tenant-B realization path. Missing authorization port → no global/default permission fallback.

Configuration is scoped by tenant/context. No process-global provider configuration mutation.

**P0 runtime verdict:** `N/A — WITH EVIDENCE` (no runtime change). Architecture must still answer roadmap §2.0.1 questions (see program tracker closure report).

---

## 16. Governance boundary and authorization evidence (P0-R1 lock)

```text
configuration validation ≠ configuration authorization
```

**Hard invariant:**

```text
Governance authorizes
Integrations realizes
Execution executes
```

Existing capability configuration realization is a **domain mutation** that **consumes** canonical control-plane mutation authorization; it does **not** own or redefine authorization semantics.

### 16.1 Canonical contract reuse (no second authorization contract)

Reuse **`intergrax/contracts/control_plane_mutation.py`** only — **no** config-specific duplicate:

| Contract | Role |
| -------- | ---- |
| `ControlPlaneMutationRequest` | Governance evaluation input for the projected mutation |
| `ControlPlaneMutationAuthorizationScope` | Continuation / exact scope identity where required |
| `ControlPlaneMutationAuthorizationEvidence` | Typed proof Integrations verifies before strategy |
| `ControlPlaneMutationAuthorizationResult` | Composition owner outcome (`permitted`, `decision`, `evidence`) |
| `ControlPlaneMutationPolicyEvaluator` | Governance policy evaluation (composition-injected) |
| `control_plane_mutation_request_digest` | Canonical authorization-request identity binding |
| `ControlPlaneMutationAuthorizationPort` | **Architecture target (P0-R2-I1):** public typed Protocol — `authorize(request) → ControlPlaneMutationAuthorizationResult`; **absent from contracts at R2 baseline** |

**Semantic owner:** **`ControlPlaneMutationAuthorizationPort`** = **Governance** (permission, policy evaluation, ALLOW/DENY, HITL continuation, evidence production). **Contract home:** `intergrax/contracts` owns the public abstraction location, not policy semantics.

**Canonical runtime implementation (locked):** **`ControlPlaneMutationAuthorizationBoundary`** in `intergrax/runtime/governance/control_plane_mutation_authorization.py` — the sole sanctioned concrete implementation mechanism introduced by this architecture wave; **P0-R2-I1** makes structural conformance mechanical.

**Reference architecture pattern (not semantic equivalence):** **`MeaningfulSideEffectAuthorizationPort`** — public `@runtime_checkable` Protocol in `intergrax/contracts`, concrete Governance runtime behind it, consumers depend only on the Protocol. Configuration realization reuses this **pattern** only; **`ControlPlaneMutationAuthorizationPort`** and **`MeaningfulSideEffectAuthorizationPort`** remain **distinct** domain boundaries (no generic `AuthorizationPort` super-abstraction).

**Forbidden new Integrations-owned authorization types** (examples): `ConfigurationRealizationAuthorizationEvidence`, `ConfigurationApprovalToken`, `ConfigurationGovernanceDecision`, `ConfigurationPermission`.

**Forbidden:** `approval_evidence_ref: str` alone as sufficient authorization proof. It may remain optional provenance inside the canonical request/evidence; realization requires typed **`ControlPlaneMutationAuthorizationEvidence`**.

**Forbidden:** Integrations production dependency on `intergrax/runtime/governance/**` (e.g. `control_plane_mutation_authorization.py`). Concrete **`ControlPlaneMutationAuthorizationBoundary`** stays in sanctioned platform composition only.

### 16.2 Mutation classification

```text
evaluation_point = GovernanceEvaluationPoint.CONTROL_PLANE_MUTATION
```

Configuration realization changes configured platform state and consumes the existing control-plane mutation Governance contract. **Do not** add a new `GovernanceEvaluationPoint` for this capability in P0-R1.

Stable **`mutation_type`** for P1 (documented constant):

```text
integration_configuration.realize.v1
```

### 16.3 Realization → control-plane mutation projection (P1 semantics)

Deterministic semantic projection (implementation deferred to P1):

```text
ExistingCapabilityConfigurationRealizationRequest
        ↓ project
ControlPlaneMutationRequest
        ↓ Governance
ControlPlaneMutationAuthorizationResult
        ↓ when permitted + ALLOW
ControlPlaneMutationAuthorizationEvidence
        ↓ Integrations verifies exact scope
realization strategy may run
```

| `ControlPlaneMutationRequest` field | Configuration realization semantic source |
| ----------------------------------- | ------------------------------------------- |
| `mutation_id` | realization `request_id` or deterministic mutation identity |
| `mutation_type` | `integration_configuration.realize.v1` |
| `principal` | upstream authenticated/governed principal — **not** synthesized by Integrations |
| `tenant_id` (via `principal.tenant_id`) | must equal realization `tenant_id` |
| `resource_scope` | exact host/binding/configuration scope |
| `resource_type` | stable integration-configuration resource type (P1 constant; vendor-neutral) |
| `resource_id` | exact target provider/integration identity |
| `current_revision` | actual current configured state/revision for target scope |
| `target_revision` | requested configuration identity/fingerprint |
| `risk_classification` | supplied by **Integrations / INT-CONFIG** mutation-domain classification (`ControlPlaneMutationRisk`); Governance **consumes** it for authorization — Governance does **not** own configuration-domain risk semantics |
| `approval_evidence_ref` | optional provenance only |
| `task_id` / `run_id` | preserved when execution/work context exists |

Integrations must **not** invent alternate fields when canonical fields carry the semantics.

### 16.4 Resource identity and revision binding

Authorization must cover the **exact** mutation — no tenant-wide or provider-wide approval:

```text
evidence.tenant_id == request.tenant_id
evidence.resource_id == requested exact provider/integration target
evidence.resource_scope == requested host/binding/configuration scope
evidence.target_revision == requested configuration identity/fingerprint
evidence.current_revision == actual current configuration revision used by realization
```

Authorization for `current=A, target=B` must **not** authorize `current=A, target=C`, `current=D, target=B`, or any scope drift.

Stale evidence bound to prior `current_revision`, `target_revision`, or `request_digest` cannot authorize a changed realization request. No closest-match approval reuse.

### 16.5 Request digest continuity

Distinct concepts (both required where applicable):

```text
configuration fingerprint = identity of desired configuration payload
request_digest            = identity of full governed ControlPlaneMutationRequest
```

Before strategy execution:

```text
authorization_evidence.request_digest
==
control_plane_mutation_request_digest(project(realization_request))
```

P1 must use canonical **`control_plane_mutation_request_digest`** — **no** configuration-specific ad-hoc hash for authority verification.

### 16.6 Policy action and result vs evidence

Realization proceeds only when Governance evidence represents:

```text
policy_action == PolicyAction.ALLOW
```

and composition has:

```text
result.permitted == True
and result.decision.action == PolicyAction.ALLOW
```

Hard fail-closed for realization (no strategy/provider side effects):

```text
DENY
REQUIRE_HUMAN
ESCALATE
MODIFY
```

Integrations must **not** start HITL workflows or convert continuation-required into permission.

**Boundary (P0-R2):** the governed façade obtains **`ControlPlaneMutationAuthorizationResult`** via **`ControlPlaneMutationAuthorizationPort.authorize`**; the pure core receives **already-admitted** evidence on an internal API and verifies scope continuity. **`ControlPlaneMutationAuthorizationEvidence(...)`** and **`ControlPlaneMutationAuthorizationResult(...)`** construction prove **shape** only — **not** sanctioned producer provenance. **Typed evidence ≠ authority provenance.** Neither provenance nor scope checks alone are sufficient.

### 16.7 Integrations-side evidence verification (P1 minimum)

Before `strategy.can_realize()` / `strategy.realize()` / resolver/provider mutation:

| Check | Required |
| ----- | -------- |
| authorization evidence present | yes |
| `policy_action == PolicyAction.ALLOW` | yes |
| `tenant_id` matches realization request | yes |
| `mutation_type` matches projected mutation | yes |
| `resource_type` matches | yes |
| `resource_id` matches target | yes |
| `resource_scope` matches | yes |
| `current_revision` matches actual current state | yes |
| `target_revision` matches requested fingerprint | yes |
| `task_id` / `run_id` match when present on request | yes |
| `request_digest` matches canonical projection | yes |

Any mismatch → fail closed; preferred: **strategy calls = 0**, **resolver/provider mutation calls = 0**.

### 16.8 Principal ownership

Integrations must **not** synthesize `RequestIdentity`, `user_id`, `auth_subject`, or `PrincipalType`. If P1 cannot obtain a canonical principal without new identity semantics → **`STOP — ARCHITECTURE DECISION REQUIRED`** (no `system-user`, `default-admin`, or `configuration-service` identities).

### 16.9 Governance evidence is not execution authority

`ControlPlaneMutationAuthorizationEvidence` authorizes **only** this configuration realization mutation. It does **not** authorize provider business operations, Tool invocation, later side effects, execution admission, or future provider use.

```text
configured != authorized for use
```

Future use follows separate Governance/Execution boundaries.

### 16.10 P1 dependency graph

```text
intergrax/integrations/**  →  intergrax/contracts/control_plane_mutation.py   ALLOWED

intergrax/integrations/**  →  intergrax/runtime/governance/**                 FORBIDDEN
```

Sanctioned host: Governance boundary → typed result/evidence → Integrations realization service.

**Invariant:** `configured != authorized`; realization does not widen authority. **No second authorization contract.**

### 16.11 P0-R2 — Authorization invocation boundary (provenance lock)

**Audit finding (closed-world):** at architecture baseline, typed **`ControlPlaneMutationAuthorizationEvidence`** can validate tenant, resource, revision, digest, and policy action, but its **presence alone** does not mechanically prove canonical Governance performed authorization. **Scope integrity ≠ authority provenance.**

**Pre-audit fact:** production contracts include `ControlPlaneMutationRequest`, `ControlPlaneMutationAuthorizationEvidence`, `ControlPlaneMutationAuthorizationResult`, `ControlPlaneMutationPolicyEvaluator`, and runtime **`ControlPlaneMutationAuthorizationBoundary`** — but **not** **`ControlPlaneMutationAuthorizationPort`**. R2 locks additive introduction in exactly one canonical module: **`intergrax/contracts/control_plane_mutation.py`** (preferred; request/result/evidence already live there). Do **not** split across a second module unless repository organization mandatorily requires it — **one** canonical definition.

**Minimum port semantics (architecture target for P0-R2-I1):**

```python
@runtime_checkable
class ControlPlaneMutationAuthorizationPort(Protocol):
    def authorize(
        self,
        request: ControlPlaneMutationRequest,
    ) -> ControlPlaneMutationAuthorizationResult:
        ...
```

One responsibility: evaluate one typed control-plane mutation request through canonical Governance semantics and return one typed authorization result. No configuration-specific methods, vendor hooks, callbacks, `dict` metadata, `object`, or `Any`.

**Authority provenance model:**

```text
permission provenance
=
successful call to injected ControlPlaneMutationAuthorizationPort
for the exact projected mutation
within the sanctioned realization operation
```

plus `result.permitted == True` and `result.decision.action == PolicyAction.ALLOW`, plus all R1 exact evidence checks (tenant, mutation type, resource type/id/scope, current/target revision, request digest, task/run identity where applicable).

**No cryptographic token requirement in this architecture:** no signed evidence, JWT, MAC, signature authority, opaque capability token, or config authorization token for in-process admission — the enterprise invariant is **exactly one sanctioned call path** that invokes the Governance port before work.

**Public API rule (P1):** future public capability entry = governed façade/coordinator (conceptually under **`ExistingCapabilityConfigurationRealizationPort`**). Sanctioned shape:

```text
public production entry
→ authorization port
→ pure realization core
```

Never: public production entry → raw evidence parameter → pure core. **Handcrafted evidence cannot admit realization** on a sanctioned public path.

**Strategy boundary:** strategy receives only an already-admitted typed realization context. **Strategy does not receive Governance authority** (no authorization port, policy evaluator, Governance runtime boundary, or approval coordinator).

**Dependency direction:**

```text
Integrations orchestration  →  ControlPlaneMutationAuthorizationPort (contracts)     ALLOWED
runtime Governance          →  implements ControlPlaneMutationAuthorizationPort       ALLOWED

Integrations                →  intergrax/runtime/governance/**                        FORBIDDEN
contracts                   →  runtime/governance                                     FORBIDDEN
Governance concrete impl    →  Integrations provider implementation                   FORBIDDEN
```

**Interaction with MSE:** **`ControlPlaneMutationAuthorizationPort`** governs **CONTROL_PLANE_MUTATION** configuration-state admission only. **`MeaningfulSideEffectAuthorizationPort`** governs consequential tool effects. Do **not** merge. Future provider business effects may separately require MSE or other boundaries.

### 16.12 P0-R2 — Fail-closed authorization invocation matrix

| Condition | Result |
| --------- | ------ |
| authorization port missing in production composition | fail closed / composition error |
| authorization port raises or fails | no realization |
| `result.permitted == False` | no realization |
| result decision action != `PolicyAction.ALLOW` | no realization |
| continuation required (`REQUIRE_HUMAN`, `ESCALATE`, `MODIFY`, `DENY`) | no realization |
| evidence scope mismatch | no realization |
| evidence digest mismatch | no realization |
| caller supplies handcrafted ALLOW evidence without port invocation | no sanctioned realization path |
| strategy attempts direct Governance call | architecture violation |
| alternate public raw-evidence realization entry appears | architecture violation |

### 16.13 P0-R2 — Implementation wave and sequencing

**Mandatory child:** **`INT-CONFIG-REAL-X-P0-R2-I1` — Canonical Control-Plane Authorization Port Promotion**

Strict future scope (not implemented in R2):

1. add **`ControlPlaneMutationAuthorizationPort`** to public contracts;
2. make existing **`ControlPlaneMutationAuthorizationBoundary`** structurally satisfy the Protocol;
3. targeted contract/runtime tests;
4. no Integrations realization implementation yet.

**Required sequence (no scope mixing):**

```text
P0-R2 architecture decision (this document)
→ P0-R2-I1 Governance public-port promotion
→ P0 architecture closure / reconciliation
→ P1 Integrations typed realization implementation
```

**P1 MUST NOT START BEFORE P0-R2-I1 INDEPENDENT ACCEPTANCE.**

P1 must not simultaneously modify Governance public authorization contracts, build Integrations realization contracts, and build the realization service — different semantic owners.

**R1 status:** **`INT-CONFIG-REAL-X-P0-R1`** = scope-binding design **accepted in substance**; cannot close parent until R2 provenance seam is resolved; preferred roadmap status **BLOCKED BY R2 / remediation dependency** — not independent final **CLOSED** until workflow permits post-R2 reconciliation.

---

## 17. Execution boundary

```text
configured capability
→ later consumer
→ Governance (where required)
→ canonical Execution
```

Never:

```text
configuration realization → ToolRuntime
configuration realization → direct provider business side effect
```

except strictly bounded configuration validation/materialization explicitly owned by the provider configuration boundary.

---

## 18. Fail-closed matrix

| Condition | Outcome |
| --------- | ------- |
| Target capability missing | Fail closed — **not** acquisition |
| Tenant missing | Fail closed |
| Provider mismatch | Fail closed |
| Configuration payload type unsupported | Fail closed |
| Configuration version unsupported | Fail closed |
| Strategy missing | Fail closed |
| Multiple strategies claim same realization | Ambiguity → fail closed |
| Authorization evidence absent | Fail closed — strategy calls = 0 |
| `policy_action != PolicyAction.ALLOW` | Fail closed — no realization |
| Governance continuation required (`REQUIRE_HUMAN`, `ESCALATE`, `MODIFY`, `DENY`) | Fail closed — no realization |
| Tenant mismatch (evidence vs request) | Fail closed |
| Provider/resource/`resource_id` mismatch | Fail closed |
| `resource_scope` mismatch | Fail closed |
| `target_revision` / configuration fingerprint mismatch | Fail closed |
| `current_revision` mismatch (stale authorization) | Fail closed |
| `mutation_type` mismatch | Fail closed |
| `request_digest` mismatch | Fail closed |
| `task_id` / `run_id` mismatch when present | Fail closed |
| Malformed authorization evidence | Fail closed |
| Authorization port missing at composition | Fail closed |
| Port invocation failure | Fail closed — no realization |
| Handcrafted evidence without port invocation on public path | No sanctioned path — fail closed |
| Strategy or provider calls Governance | Architecture violation — fail closed |
| Strategy attempts tenant/provider widening | Reject |
| Provider realization failure | Explicit typed failure |

No silent fallback to default provider, global tenant, env-selected alternate, acquisition, or another strategy.

---

## 19. Error model (architecture families)

Minimum conceptual families (enums optional in P0):

```text
TARGET_NOT_FOUND
UNSUPPORTED_CONFIGURATION
INVALID_CONFIGURATION
UNSUPPORTED_CONFIGURATION_VERSION
MISSING_AUTHORITY_EVIDENCE          # missing typed ControlPlaneMutationAuthorizationEvidence
IDENTITY_MISMATCH                 # resource/provider/scope/digest/revision mismatch
TENANT_MISMATCH                   # tenant continuity violation
# non-ALLOW policy_action → authorization not satisfied (typed failure family; enum names P1)
UNSUPPORTED_STRATEGY
STRATEGY_AMBIGUITY
REALIZATION_FAILED
```

Free-form exception text is **not** the canonical semantic contract.

---

## 20. Pluginability / replaceability (P1/P2 evidence plan)

Later qualification must prove:

1. External realization strategy implements platform Protocol without core modification.
2. Two implementations are structurally substitutable.
3. Provider-specific configuration types live outside generic core.
4. Generic core contains zero vendor branches.
5. No second registry required.

---

## 21. Configuration state lifecycle

```text
requested → validated → realized/configured
```

Do **not** conflate with `effective`, `authorized`, `executing`.

**Transition to `effective`:** existing sanctioned Integrations composition/resolution owner — `INT-CONFIG-REAL-X` does **not** invent a second activation lifecycle.

Distinctions locked:

```text
existing != configured != effective != authorized != executing
```

- **Existing:** provider/capability in catalog.
- **Configured:** validated tenant-scoped configuration realized.
- **Effective:** sanctioned composition may resolve/use binding.
- **Authorized:** Governance permits use.
- **Executing:** Execution owns runnable work.

---

## 22. Secrets and data handling

- Raw secrets must not become generic configuration metadata.
- Payloads should carry **secret references** where existing secret-store semantics permit.
- Tenant-scoped credential references remain tenant-scoped.
- Realization evidence must not expose raw secrets.
- Secret resolution ownership stays with existing security/secrets boundary (not implemented in P0).

---

## 23. P1 implementation boundary

**`INT-CONFIG-REAL-X-P1` — Typed Realization Contracts & Pure Service**

After independent **P0-R2** acceptance and **P0-R2-I1** port promotion, P1 may implement:

- `IntegrationConfigurationPayload`
- `ExistingCapabilityConfigurationRealizationRequest` / `Result`
- `ExistingCapabilityConfigurationRealizationStrategy`
- governed realization façade depending on **`ControlPlaneMutationAuthorizationPort`** (public production entry)
- `ExistingCapabilityConfigurationRealizationService` (pure core; internal evidence handoff typed as `ControlPlaneMutationAuthorizationEvidence` after façade admission)
- typed existing-integration resolver port
- deterministic projection to `ControlPlaneMutationRequest` + evidence verification using `control_plane_mutation_request_digest`

Reuse (do **not** duplicate): `ControlPlaneMutationRequest`, `ControlPlaneMutationAuthorizationEvidence`, `ControlPlaneMutationAuthorizationPort` (post-I1), `control_plane_mutation_request_digest`.

- Fail-closed composition rules; no catalog mutation; **no** concrete Governance runtime imports in Integrations; **exactly one** port invocation owner in the façade.
- Unit tests for generic service behavior only (no full provider rollout).

If canonical Governance contracts cannot express realization without architecture change → **`STOP — ARCHITECTURE DECISION REQUIRED`**.

Parent `INT-CONFIG-REAL-X` remains **BLOCKED** until P0 + P0-R1 (substance) + P0-R2 + P0-R2-I1 architecture audits accept this document.

---

## 24. P2 / CERT boundaries

| Wave | Scope |
| ---- | ----- |
| **INT-CONFIG-REAL-X-P2** | Reference provider configuration strategy + tenant isolation proof |
| **INT-CONFIG-REAL-X-CERT** | Adversarial configuration realization certification |

Insert **`INT-CONFIG-REAL-X-P1A`** (Typed Integration Configuration Boundary Hardening) only if implementation proves Option B is required.

---

## 25. Certification / exit criteria (P0 document)

P0 architecture evidence complete when independent audits accept:

1. Ownership matrix and gap analysis;
2. Typed request/payload/result/strategy/composition locks;
3. Catalog/resolver authority preserved;
4. Legacy options classified (Option A);
5. Tenant, Governance, Execution boundaries explicit;
6. Fail-closed matrix and error families;
7. Wave decomposition bounded;
8. **P0-R1:** canonical `ControlPlaneMutation*` reuse; projection matrix; digest/revision binding; ALLOW-only realization; no `runtime.governance` Integrations dependency; authorization evidence fail-closed matrix;
9. **P0-R2:** `ControlPlaneMutationAuthorizationPort` architecture lock; provenance vs scope; façade vs pure core; exactly-one port invocation; handcrafted evidence cannot admit public realization; **`ControlPlaneMutationAuthorizationBoundary`** as canonical implementation target; **P0-R2-I1** defined; I1 precedes P1.

P0 / P0-R1 / P0-R2 do **not** self-declare **CLOSED**.

---

## 26. Applicable FRZ evidence plan (remain OPEN)

P0 defines evidence plan only — **no PASS**.

| Family | IDs | Later wave |
| ------ | --- | ---------- |
| Configuration | FRZ-CFG-02, FRZ-CFG-03, FRZ-CFG-04, FRZ-CFG-07 (+ FRZ-CFG-01 via **CONFIG-X**) | P1/P2/CERT + **CONFIG-X** |
| Contracts | FRZ-CTR-01, FRZ-CTR-05, FRZ-CTR-06 | P1, P2, CERT |
| Strong typing | FRZ-TYP-01, FRZ-TYP-02, FRZ-TYP-03, FRZ-TYP-04, FRZ-TYP-06 | P1, P2, CERT |
| Pluginability | FRZ-PLG-01, FRZ-PLG-02 | P2, CERT |
| Replaceability | FRZ-RPL-01, FRZ-RPL-02 | P2, CERT |
| Governance | FRZ-GOV-01, FRZ-GOV-02, FRZ-GOV-04, FRZ-GOV-07, FRZ-GOV-09, FRZ-GOV-10 (authorization port invocation + evidence before work; control-plane mutation governed) | P0-R2-I1, P1, CERT, GOV-X2 |
| Traceability | FRZ-TRC-06, FRZ-TRC-07, FRZ-TRC-11 | P1, CERT, **TRACE-X** |
| Tenant | FRZ-TEN-01, FRZ-TEN-02, FRZ-TEN-05, FRZ-TEN-09, FRZ-TEN-10, FRZ-TEN-11, FRZ-TEN-12 | P2, CERT, **TENANT-X** |

**new global FRZ PASS = 0** · **new FRZ-TEN PASS = 0** (architecture evidence plan only).

---

## 27. Forbidden shortcuts (summary)

- Second provider catalog or resolver
- Global mutable configuration owner
- Reflection-based strategy lookup
- Arbitrary dict metadata as semantic configuration (`no dict[str, Any] semantic boundary`)
- Tenantless configuration contract
- Strategy changing tenant/provider
- Integrations granting Governance permission
- Public realization admitting caller-constructed authorization evidence without port invocation
- Direct Execution/ToolRuntime invocation
- Capability Acquisition / Marketplace fallback

---

> **Independent audit reminder:** Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
