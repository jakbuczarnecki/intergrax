# INT-CONFIG-REAL-X-P0 — Existing Capability Configuration Realization Architecture Lock

## 1. Status and scope

| Field | Value |
| ----- | ----- |
| **Task** | `INT-CONFIG-REAL-X-P0` (architecture lock; child of `INT-CONFIG-REAL-X`) |
| **Parent** | `INT-CONFIG-REAL-X` — Existing Capability Configuration Realization |
| **Program baseline** | `234d3c03dce5708766f8e068afbd2ea496132002` (`development`) |
| **Source** | Scenario #24 GAP-01 — `CONFIGURE_EXISTING` disposition |
| **Canonical owner** | **Integrations** |
| **Artifact role** | Closed-world design record before `INT-CONFIG-REAL-X-P1` |
| **Production / tests** | **0** in P0 |
| **Status** | **PROPOSED / READY FOR INDEPENDENT ARCHITECTURE AUDIT** |

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

**Governance boundary:** realization may **verify** that supplied approval/evidence references exist and match scope; it must **not** decide permission.

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

```text
CONFIGURE_EXISTING disposition (upstream)
        ↓
ExistingCapabilityConfigurationRealizationRequest (immutable typed)
        ↓
ExistingCapabilityConfigurationRealizationService (Integrations-owned)
        ↓
ExistingCapabilityConfigurationRealizationStrategy (platform SPI; composed)
        ↓
validate target + tenant + typed payload + evidence refs (fail closed)
        ↓
typed existing-integration resolver port → sanctioned Catalog/resolver
        ↓
ExistingCapabilityConfigurationRealizationResult (tenant-scoped configured binding reference)
        ↓
later: effective composition (Integrations) → Governance (if required) → Execution
```

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
| Authority/evidence reference | Proves configuration was approved where policy requires; missing when required → fail closed |
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

**Locked:**

```text
ExistingCapabilityConfigurationRealizationService(
    strategies: tuple[ExistingCapabilityConfigurationRealizationStrategy, ...],
    existing_integration_resolver: <typed port>,
)
```

Rationale: explicit DI, no second registry, no global singleton, no plugin-discovery authority duplication, structural replaceability.

**Forbidden:** `ConfigurationStrategyRegistry`, `RealizationProviderRegistry`, tenant→strategy global maps, vendor switches, service locators, reflection discovery.

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
request.tenant
== strategy input tenant
== resolved target tenant scope
== result.tenant
== realization evidence tenant
```

**Forbidden:** tenant A request → tenant B binding; missing tenant → global config; tenant-scoped config → global provider mutation; strategy rewrites tenant.

Configuration is scoped by tenant/context. No process-global provider configuration mutation.

**P0 runtime verdict:** `N/A — WITH EVIDENCE` (no runtime change). Architecture must still answer roadmap §2.0.1 questions (see program tracker closure report).

---

## 16. Governance boundary

```text
configuration validation ≠ configuration authorization
```

Realization may verify approval/evidence **exists** and **applies** to tenant/provider/configuration scope.

Realization must **not** decide policy permission. Governance remains semantic authority.

If no reusable typed governance evidence reference exists at P1: document required reference shape; defer concrete Governance wiring to `INT-CONFIG-REAL-X-P1` or `GOV-X2` — **do not** invent a new Governance authority inside Integrations.

**Invariant:** `configured != authorized`; realization does not widen authority.

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
| Approval evidence required but missing | Fail closed |
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
MISSING_AUTHORITY_EVIDENCE
IDENTITY_MISMATCH
TENANT_MISMATCH
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

- Introduce frozen public contracts for request/result/payload protocol/strategy/service.
- Pure `ExistingCapabilityConfigurationRealizationService` with explicit strategy tuple + resolver port.
- Fail-closed composition rules; no catalog mutation; no Execution/Governance calls.
- Unit tests for generic service behavior only (no full provider rollout).

Parent `INT-CONFIG-REAL-X` remains **BLOCKED** until P0 architecture audit accepts this document.

---

## 24. P2 / CERT boundaries

| Wave | Scope |
| ---- | ----- |
| **INT-CONFIG-REAL-X-P2** | Reference provider configuration strategy + tenant isolation proof |
| **INT-CONFIG-REAL-X-CERT** | Adversarial configuration realization certification |

Insert **`INT-CONFIG-REAL-X-P1A`** (Typed Integration Configuration Boundary Hardening) only if implementation proves Option B is required.

---

## 25. Certification / exit criteria (P0 document)

P0 complete when independent architecture audit accepts:

1. Ownership matrix and gap analysis;
2. Typed request/payload/result/strategy/composition locks;
3. Catalog/resolver authority preserved;
4. Legacy options classified (Option A);
5. Tenant, Governance, Execution boundaries explicit;
6. Fail-closed matrix and error families;
7. Wave decomposition bounded.

P0 does **not** self-declare **CLOSED**.

---

## 26. Applicable FRZ evidence plan (remain OPEN)

P0 defines evidence plan only — **no PASS**.

| Family | IDs | Later wave |
| ------ | --- | ---------- |
| Configuration | FRZ-CFG-01, FRZ-CFG-02, FRZ-CFG-03+ | P1/P2/CERT + **CONFIG-X** |
| Contracts | FRZ-CTR-01, FRZ-CTR-03, FRZ-CTR-05, FRZ-CTR-06 | P1, P2, CERT |
| Strong typing | FRZ-TYP-01..04, FRZ-TYP-06 | P1, P2, CERT |
| Pluginability | FRZ-PLG-01, FRZ-PLG-02 | P2, CERT |
| Replaceability | FRZ-RPL-01, FRZ-RPL-02 | P2, CERT |
| Governance | FRZ-GOV-09 (no authority widening); configured ≠ authorized | CERT, GOV-X2 |
| Tenant | FRZ-TEN-01, FRZ-TEN-02, FRZ-TEN-05, FRZ-TEN-06, FRZ-TEN-09..12 | P2, CERT, **TENANT-X** |

---

## 27. Forbidden shortcuts (summary)

- Second provider catalog or resolver
- Global mutable configuration owner
- Reflection-based strategy lookup
- Arbitrary dict metadata as semantic configuration (`no dict[str, Any] semantic boundary`)
- Tenantless configuration contract
- Strategy changing tenant/provider
- Integrations granting Governance permission
- Direct Execution/ToolRuntime invocation
- Capability Acquisition / Marketplace fallback

---

> **Independent audit reminder:** Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
