# PLATFORM-PLUGIN-AUDIT-1 - Production Architecture & Implementation Audit

> **Historical audit snapshot (2026-08-12).** For current Platform Plugins behavior see [`PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) and [`EXTENSION_AUTHOR_GUIDE.md`](../../technical/guides/EXTENSION_AUTHOR_GUIDE.md).

**Task:** `PLATFORM-PLUGIN-AUDIT-1`  
**Status:** `READY_FOR_REVIEW`  
**Branch:** `development`  
**Audited commit:** `f7b6eedf354d43b1459b8077a56f8acd3fdaaa3d` (PLATFORM-PLUGIN-9 closeout)  
**Audit date:** 2026-08-12  
**Auditor role:** Independent post-program review (not implementation)

**Index used (not treated as proof):** PLATFORM-PLUGIN-1 inventory · PLATFORM-PLUGIN-9 closeout · frozen architecture §25

---

## 1. Executive verdict

| Field | Value |
|-------|-------|
| **Overall verdict** | **APPROVED_WITH_GAPS** |
| **Production-ready (trusted-host model)** | **Yes** - for deployments that treat installed Python as trusted, enable third-party entry points explicitly, and run host qualification before registration |
| **Enterprise-ready** | **No** - missing unified inventory, verifiable qualification attestation, canonical platform version authority, cross-worker governance, and isolation beyond in-process trust |
| **Architecture remediation required** | **No** - gaps are documented limitations, hardening backlog, and enterprise-future work; core platform/domain split remains sound |
| **Blocking production (single-tenant / trusted host)** | **None** at CRITICAL severity |
| **Blocking enterprise multi-tenant control plane** | **Multiple** - see §12 and enterprise matrix §11 |

**Summary:** The Platform Plugin subsystem delivers a coherent **coordination layer** (`intergrax/core/plugins/`) over **domain-owned contracts**. Discovery, manifest validation, compatibility checks, and qualification vocabulary are implemented and tested. PLUGIN-9 closeout invariants hold at the audited SHA. Independent review confirms the implementation matches frozen architecture for the intended **trusted in-process** trust model. Residual risks concentrate on **operational visibility**, **failure isolation adoption**, **semantic-only qualification**, **explicit opt-in activation**, and **enterprise governance surfaces** - not on accidental second plugin frameworks or tier violations.

---

## 2. Audit scope and methodology

- **Repository:** `D:\Projekty\intergrax`, branch `development` only  
- **Baseline:** `origin/development = f7b6eedf354d43b1459b8077a56f8acd3fdaaa3d` - **verified ancestor of HEAD**  
- **No production code changes** in this task  
- **Per-domain read cap:** public contract · loader/bootstrap · config/DI · one example · focused tests  
- **Tests executed:** contract suite · PLUGIN-8 E2E · unit `tests/unit/core/plugins/` · scaffold qualification test  
- **NOT_FULLY_AUDITED:** every third-party provider implementation; full Tier-3 application wiring matrix; Windows-specific wheel packaging in CI

---

## 3. Dimension scores

| Dimension | Score | Notes |
|-----------|-------|-------|
| A. Architecture alignment | PASS_WITH_RISK | Domain ownership preserved; no universal runtime wrapper; version authority gap |
| B. Modularity & component design | PASS | Clean Tier-0 coordination modules; no tier violations in `core/plugins/` |
| C. Discovery architecture | PASS_WITH_RISK | Deterministic EP scan; `isolate` mode exists but domain loaders default `fail_fast` |
| D. Registration & global state | PASS_WITH_RISK | Process-scoped catalogs intentional; test reset helpers pervasive |
| E. Configuration / secrets / DI | PASS | Manifest secret rejection; domain wiring contexts authoritative |
| F. Security | PASS_WITH_RISK | Trusted in-process model explicit; defense-in-depth at domain gates |
| G. Multi-tenancy | PASS | Catalog metadata global; runtime materialization tenant-scoped in inspected paths |
| H. Lifecycle & resource ownership | PASS_WITH_RISK | Vocabulary only; domain-owned cleanup; some loaders instantiate at EP load |
| I. Compatibility & versioning | PASS_WITH_RISK | PEP 440 + specifier checks solid; no canonical runtime version API |
| J. Qualification | PASS_WITH_RISK | Immutable records; trivial fabrication by host code - intentional |
| K. Failure handling | PASS_WITH_RISK | Matrix incomplete in loaders; one bad EP can block group in `fail_fast` paths |
| L. Scalability | PASS_WITH_RISK | Global catalogs OK for single-app process; multi-app / 1000-plugin ops need discipline |
| M. Performance | PASS_WITH_RISK | Coordination mostly startup; tool-invocation EP scan on lookup path |
| N. Observability & auditability | PASS_WITH_RISK | Attribution fields exist; no unified operator inventory |
| O. Testing quality | PASS_WITH_RISK | Strong contract/E2E; `test_plugin_catalog_counts` failing at audit time |
| P. CI/CD | PASS_WITH_RISK | PLUGIN-9 gate on Linux ci_smoke; subset of full plugin tests |
| Q. Public API quality | PASS_WITH_RISK | Stable exports; loose `object` on some error attributes |
| R. Developer experience | PASS_WITH_RISK | Tools dual-mode exemplary; 9 surfaces external-EP-first |
| S. Enterprise readiness | PARTIAL | See §11 |

---

## 4. Architecture assessment (A)

**Verified:**

1. Platform coordination vs domain ownership - **preserved**. `intergrax/core/plugins/` coordinates discovery/manifest/compatibility/qualification vocabulary; domains own registration semantics (`register_tool_plugin`, RAG bootstrap registries, VK contribution catalog).
2. No universal runtime wrapper - **confirmed**. No monolithic `PlatformPlugin` runtime type in production paths.
3. No accidental second plugin architecture - **confirmed**. Single shared `discovery.py`; domain-specific loaders call it; VK uses separate contribution catalog by design.
4. Public/internal boundaries - **structurally enforceable** via tier boundaries and domain registries; not all surfaces have scaffold/local paths.
5. Tier layering - **`core/plugins` imports only Tier-0** (packaging, pydantic, importlib.metadata). No imports from `runtime/`, `agents/`, or `applications/`.
6. Domain contracts authoritative - **yes** per architecture §25 and code paths (e.g. `ToolPlugin`, `SecurityDefensePlugin`).
7. Host composition location - **Tier-3** `applications/_shared/*_wiring.py` and host factories.
8. Package coordination does not leak runtime semantics - manifest is metadata-only; EP values are code pointers.
9. DO-NOT-UNIFY decisions - **still justified** (conflict policy, VK catalog, lifecycle engines remain domain-owned).
10. Future composability - **yes**; additive EP groups and capability descriptors supported.

**Score rationale:** PASS_WITH_RISK due to documented absence of canonical Intergrax platform version authority (architecture §20.4, §29).

---

## 5. Component / module assessment (B)

| Module | Responsibility | Assessment |
|--------|----------------|------------|
| `discovery.py` | EP enumeration, load, factory resolution, `register_plugins` | Single-purpose; deterministic sort; factory invocation only in `_resolve_tier0_plugin_type` |
| `package_contract.py` | Pydantic manifest models, secret key rejection | Frozen models; canonical name/version normalization |
| `manifest_io.py` | pyproject TOML parsing, project identity cross-check | Fail-closed on conflicts and unknown keys |
| `platform_semantics.py` | Compatibility API, lifecycle/conflict vocabulary | Pure functions; no global state |
| `platform_qualification.py` | Trust model, qualification records, admission helper | Pure; no persistence/registry |
| `errors.py` | Typed exception hierarchy | Loose `object` on some attributes (see F011) |
| `catalog_bootstrap.py` | Unified Tier-0 bootstrap for integrations/tools/skills + security EP hook | Idempotent shipped flag; global `_tier0_shipped_done` |
| `memory_bootstrap.py` | Memory store EP discovery | Count-only bootstrap; on-demand rediscovery (see F008, F010) |

**No god module** in platform coordination. **No service locator** in `core/plugins/`. **Duplication:** security/policy/tool-invocation loaders repeat iter/load/instantiate pattern instead of `register_plugins` - acceptable domain variance (instance vs class plugins).

---

## 6. Discovery architecture (C)

| Question | Answer | Evidence |
|----------|--------|----------|
| One bad plugin blocks unrelated discovery? | **Within same EP group, often yes** (`fail_fast` default in `load_entry_point_plugins`) | `discovery.py` `on_load_failure="fail_fast"`; domain loaders do not pass `isolate` |
| Failures bounded per capability? | **Partial** - `load_entry_point_targets(..., isolate)` exists; RAG/security/policy loaders use fail-fast paths |
| Deterministic? | **Yes** - `sorted(specs, key=(name, value))` | `iter_entry_point_specs` |
| Duplicate names handled? | **Yes** - `PluginConflictError` with `ENTRY_POINT_NAME` | `load_entry_point_targets` |
| Malicious metadata uncontrolled? | **Bounded** - manifest validation rejects secrets/unknown keys; EP load executes installed code (trusted model) |
| Imports only when expected? | **Yes** for scan vs load separation | `iter_entry_point_specs` scan-only; `load_entry_point_value` on demand |
| Repeated discovery safe? | **Yes** but may repeat import/load work | No global EP cache |

**Security defense loader** always `override=True` on registration - duplicates silently override (F005).

---

## 7. Registration & global state (D)

**Global catalogs (process-scoped):**

- `integrations/registry/catalog.py` - `_CATALOG` dict
- Tool/skill/context catalogs - similar snapshots
- `defense_registry.py` - `_SHIPPED` + `_DYNAMIC`
- Bootstrap idempotency flags - `_tier0_shipped_done`, `_context_shipped_done`

**Classification:** **Acceptable architecture** for single-application-per-process deployments. **Enterprise scaling limitation** for:

- multiple Intergrax application configurations in one process without catalog isolation
- test order dependence mitigated by `clear_catalog` / `reset_*_for_tests` helpers (F007)

**Thread-safety:** Catalog mutations at bootstrap are not generally locked; assumes single-threaded startup - typical for CPython app servers (worker-per-process).

---

## 8. Configuration / secrets / DI (E)

**Verified against PLUGIN-5:**

| Check | Result |
|-------|--------|
| Secrets not in package metadata | `reject_secret_like_keys` + pydantic `extra=forbid` |
| EP values code-location only | `EntryPointSpec.value` is import target string |
| Domain config resolution | `IntegrationManifest.env_prefix`; profiles per domain |
| `ToolWiringContext` | Explicit composed dependencies; handlers must not self-resolve integrations |
| Integration env-prefix exception | Documented in architecture §12.3 |
| Memory factory kwargs | Host passes tenant/RAG kwargs in `build_session_turn_index_store` |
| Security `HookContext` | Defense plugins receive hook context, not raw secrets API |
| VK credential references | Binding-scoped in VK composition paths |

**No global DI container** in platform layer. `discover_plugins_enabled()` reads `INTERGRAX_DISCOVER_PLUGINS` - narrow, explicit env gate (F003).

---

## 9. Security findings (F)

**Threat model (explicit):** Installed third-party plugins are **trusted in-process Python**. No sandbox, no signing verification, no process isolation.

**Guarantees that exist:**

| Mechanism | Role |
|-----------|------|
| Explicit EP discovery opt-in | `INTERGRAX_DISCOVER_PLUGINS` / `discover_entry_points=False` defaults |
| Manifest secret rejection | Blocks credential fields in `[tool.intergrax.plugin]` |
| Platform compatibility check | Fail-closed for external packages without compatibility evidence |
| Production qualification gate | `require_production_qualification` before scaffold registration |
| Domain policy / security profiles | Select which defenses/policies activate |
| `ToolWiringContext` scoping | Limits injected services per registration |
| Tenant kwargs on memory/RAG paths | Runtime instances bound per tenant in inspected wiring |

**Not found:** path-based dynamic loading outside importlib EP; arbitrary plugin activation without host bootstrap.

---

## 10. Multi-tenancy (G)

- Plugin **classes** and catalog **metadata** are global - **safe** (immutable type objects).
- **Runtime instances** (integrations, stores, VK bindings) resolved with `tenant_id` in memory/RAG/VK wiring inspected.
- **Defense plugins** registered globally; inspection uses per-request `HookContext` - tenant data in context, not in plugin singleton state for shipped plugins.
- **Risk:** Third-party defense plugin implementing mutable global tenant cache would be a **plugin author defect** - platform does not enforce instance isolation.

**Score:** PASS for infrastructure; tenant safety depends on host wiring and plugin author discipline.

---

## 11. Lifecycle & resource ownership (H)

- **Discovery lifecycle vocabulary** - `PlatformPluginLifecycleState` enum only; **no runtime state engine**.
- **Materialization** - domain-owned (`build_registry_from_profile`, RAG managers, VK registries).
- **EP load** - `_resolve_tier0_plugin_type` may call factory and instantiate classes (`instantiate_entry_point_target` for security/policy).
- **Cleanup** - no universal plugin shutdown; integrations/RAG domains own client lifecycle.
- **Leak surface:** EP-loaded plugin instances in `_DYNAMIC` defense registry persist for process lifetime without unload API.

---

## 12. Scalability & performance (L, M)

| Axis | Assessment |
|------|------------|
| 10 plugins | No concern |
| 100 plugins | EP enumeration at startup acceptable; catalog dict lookups O(1) |
| 1000 plugins | Startup EP scan + import cost grows; no lazy catalog for all domains |
| Multi-app one process | Global catalogs - **contamination risk** without process split |
| Startup | `importlib.metadata.entry_points()` per group; RAG may call `discover_plugins_enabled()` per engine creation |
| Request path | `load_tool_invocation_pattern` scans all EPs per lookup - **O(N)** (F009) |
| `discover_session_turn_index_plugin_types` | Re-scans EPs when called - **O(N)** per wiring (F008) |
| Distributed workers | Each worker independent - **no distributed inventory** |

---

## 13. Observability (N)

**Present:** package identity in manifest; EP `distribution` on `EntryPointSpec`; qualification evidence codes; load errors logged in discovery duplicate warnings.

**Missing for operators:**

- unified view: installed vs discovered vs enabled vs qualified vs active vs failed
- lifecycle state transitions not recorded in runtime
- qualification results not persisted

**Classification:** **Missing enterprise capability** - acceptable for current program scope.

---

## 14. Failure matrix (K)

| Scenario | Owner | Exception/result | Isolation | Logging | Fail mode | Unrelated plugins |
|----------|-------|------------------|-----------|---------|-----------|-------------------|
| Bad manifest | `manifest_io` / pydantic | `PlatformPluginManifestValidationError` | Package-level | Parse error message | Closed | N/A (pre-load) |
| Invalid version | `package_contract` | `ValueError` → validation error | Package-level | In exception | Closed | N/A |
| Incompatible platform | `platform_semantics` | `PlatformIncompatibilityError` / admission denied | Package-level | In `result.reason` | Closed | N/A |
| Duplicate EP name | `discovery` | `PluginConflictError` | EP group | Warning if override | Configurable | Same group only |
| Import failure | `discovery` | `PluginLoadError` | **Group** in fail_fast | In exception | Closed | **Blocked in same group** |
| Callable factory failure | `discovery` | `PluginLoadError` | Same EP | In exception | Closed | Same group |
| Registration failure | Domain register fn | `ValueError` / domain errors | Domain | Domain-dependent | Closed | Domain-dependent |
| DI failure | Host wiring | Domain exceptions at materialization | Request/bootstrap | App logs | Closed | N/A |
| Contract validation failure | Domain manifest | Domain errors | Capability | Domain | Closed | N/A |
| Qualification failure | `require_production_qualification` | `ProductionQualificationRequiredError` | Host path | In exception | Closed | N/A |
| Runtime invocation failure | Tool/runtime invoker | Tool execution errors | Request | Runtime traces | Domain | N/A |

---

## 15. Public extension surface matrix (12 surfaces)

| # | Surface | Contract | Loader / registration | Config / DI | Example | Tests | EP adoption | Local path | Verdict |
|---|---------|----------|----------------------|-------------|---------|-------|-------------|------------|---------|
| 1 | Integrations | `IntegrationPlugin` | `bootstrap_catalogs` → `register_integration_plugin` | `IntegrationProfile`, `env_prefix` | `integrations/examples/custom_memory_kv` | `test_external_integration_entry_point.py` | Shared `register_plugins` | Documented `register_integration_plugin` | PASS |
| 2 | Tools | `ToolPlugin` | `bootstrap_catalogs` → `register_tool_plugin` | `ToolWiringContext` | Reference wheel + local embedded | PLUGIN-8 E2E + contract | Shared `register_plugins` | Scaffold + qualification | PASS |
| 3 | Skills | `SkillPlugin` | `bootstrap_catalogs` → `register_skill_plugin` | `SkillProfile` | `skills/examples/custom_pack` | Catalog bootstrap tests | Shared `register_plugins` | Documented register helper | PASS |
| 4 | Context | `ContextPlugin` | `bootstrap_context_catalog` | `ContextProfile` | Builtin + EP | `test_context_catalog_bootstrap.py` | Shared `register_plugins` | Host composition only | PASS_WITH_RISK (DX) |
| 5 | Memory stores | Duck-typed factories | `memory_bootstrap` / on-demand EP scan | Host factory kwargs | `tests/fixtures/plugin_packages/` | `test_memory_store_bootstrap.py` | `load_entry_point_plugins` | No scaffold | PASS_WITH_RISK |
| 6 | RAG chunkers | `BaseChunkingStrategy` | `register_plugins` in chunking engine | `RagProfile`, `discover_plugins_enabled()` | Domain strategies | `test_rag_plugin_discovery.py` | Shared utility | Not documented | PASS_WITH_RISK |
| 7 | RAG retrievers | Retriever plugin protocol | `retriever_bootstrap.register_plugins` | RAG bootstrap kwargs | Domain providers | RAG plugin discovery tests | Shared utility | Not documented | PASS_WITH_RISK |
| 8 | RAG rerankers | Reranker plugin protocol | `reranker_bootstrap.register_plugins` | RAG bootstrap kwargs | Domain providers | RAG plugin discovery tests | Shared utility | Not documented | PASS_WITH_RISK |
| 9 | Vendor Knowledge | `VendorKnowledgeProviderContribution` | `contribution_catalog` (not Tier-0) | `KnowledgeSourceBinding`, tenant scope | `acme_reference` | VK contribution tests | Domain catalog | Host composition | PASS (by design) |
| 10 | Security defenses | `SecurityDefensePlugin` | `load_security_defense_plugins` | `ApplicationSecurityProfile` | Security fixture package | `test_plugin_discovery.py` | Bespoke loader | Not documented | PASS_WITH_RISK |
| 11 | Policy rules | `PolicyRuleHandler` | `load_policy_rule_plugins` | Policy bundle / profile | Domain loaders | `test_plugin_discovery.py` | Bespoke loader | Not documented | PASS_WITH_RISK |
| 12 | Tool invocation patterns | `ToolInvocationPattern` | `load_tool_invocation_pattern` | `ToolInvocationMode` | Shipped modes + EP | `test_tool_invocation_registry.py` | Bespoke loader | Not documented | PASS_WITH_RISK |

---

## 16. Testing & CI assessment (O, P)

**Strengths:**

- `tests/contract/core/plugins/test_platform_plugin_contract.py` - cross-stage invariant suite (manifest, discovery, qualification, tier boundaries, no sandbox claims)
- `tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py` - real wheel build, isolated install, runtime invocation
- `tests/unit/core/plugins/test_plugin_discovery.py` - negative cases, isolate mode, security/policy loaders
- Deterministic EP ordering tests; factory-not-invoked-on-scan tests

**Weaknesses:**

| Issue | Evidence |
|-------|----------|
| Catalog count test failing at audit SHA | `test_plugin_catalog_counts.py::test_core_integration_preset_count` - expected ≥12 core integrations, got 11 (not in PLUGIN-9 gate) |
| Gate subset | CI runs contract + E2E + scaffold only on `ci_smoke` |
| Linux-only wheel proof | E2E builds wheel on runner OS; Windows packaging not gated |
| Global reset dependence | Widespread `clear_catalog`, `reset_*_for_tests` - tests prove behavior but mirror global-state architecture |
| No concurrency tests | Catalog bootstrap assumes single-threaded startup |

**Audit test run (2026-08-12):** 153 passed, 1 failed (`test_plugin_catalog_counts`) in `tests/unit/core/plugins/` + contract + E2E + scaffold.

---

## 17. Public API quality (Q)

**Exports from `intergrax.core.plugins`:** comprehensive, documented in `__init__.py` `__all__`; stable names aligned with architecture §20.

**Loose typing (technical debt, not blocking):**

- `PluginConflictError.conflict_kind: object | None` - should be `PlatformPluginConflictKind | None`
- `PlatformIncompatibilityError.result: object | None` - should be `PlatformCompatibilityResult | None`
- `ProductionQualificationRequiredError.result: object | None` - should be `PluginQualificationResult | None`

**Assessment:** PASS_WITH_RISK - API usable and typed at contract level; error attribute typing is loose coupling.

---

## 18. Developer experience (R)

**Strengths (PLUGIN-8 baseline):**

- Reference wheel package + local embedded example
- Scaffold generates qualification-before-registration wiring
- `EXTENSION_AUTHOR_GUIDE.md` + architecture §20 matrix

**Friction:**

- 9 of 12 surfaces lack scaffold/local registration parity with Tools
- Third-party EP discovery requires `INTERGRAX_DISCOVER_PLUGINS` - easy to misconfigure
- Debugging discovery requires understanding per-domain loader (not one CLI inventory)
- Context surface historically had doc/runtime mismatch (fixed in PLUGIN-9 per closeout)

---

## 19. Enterprise readiness snapshot (S)

| Capability | Status |
|------------|--------|
| Signed artifacts | ABSENT |
| Provenance / attestation | ABSENT |
| SBOM | ABSENT |
| Allowlists (package) | PARTIAL - env opt-in + host profiles |
| Plugin repository / catalog service | ABSENT |
| Policy-controlled activation | PARTIAL - profiles + explicit discovery |
| Tenant-aware governance | PARTIAL - runtime wiring; not catalog governance |
| Isolation (non-Python / subprocess) | ABSENT |
| Central qualification authority | ABSENT |
| Rollout / canary | ABSENT |
| Hot reload | ABSENT |
| Plugin health monitoring | ABSENT |
| Distributed inventory | ABSENT |
| Revocation | ABSENT |
| Observability / control plane | PARTIAL - domain logs; no unified inventory |
| Trusted in-process execution | CURRENT |
| Domain capability contracts | CURRENT |
| EP discovery + manifest coordination | CURRENT |
| Dual-mode Tools delivery | CURRENT |
| Semantic qualification records | CURRENT |

---

## 20. Findings (by severity)

### CRITICAL

None identified with evidence at audited SHA for the **intended trusted-host production model**.

### HIGH

#### PLUGIN-AUDIT-F001

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Domain** | Compatibility / platform coordination |
| **Evidence** | Architecture §20.4: «Explicit host input (`0.1.0` in E2E) - no global runtime version authority»; `check_platform_compatibility` requires caller-supplied version |
| **Impact** | Inconsistent compatibility decisions across hosts; upgrade risk undetected without host discipline |
| **Current behavior** | Host passes arbitrary `platform_version` string into compatibility/admission helpers |
| **Expected behavior** | Single authoritative Intergrax platform version for compatibility checks (enterprise) |
| **Remediation class** | ENTERPRISE_FUTURE |
| **Blocking production readiness** | No (documented limitation) |

#### PLUGIN-AUDIT-F002

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Domain** | Qualification |
| **Evidence** | `build_qualification_result(..., status=PRODUCTION_QUALIFIED)` callable by any host code; contract test `test_host_embedded_package_compatibility_not_fabricated` shows host can self-qualify |
| **Impact** | Qualification is **semantic metadata**, not verifiable attestation; unsuitable for external audit without host trust |
| **Current behavior** | Host constructs qualification records; platform validates shape and gates on status enum |
| **Expected behavior** | For enterprise: provenance-linked attestation (out of current program scope) |
| **Remediation class** | ENTERPRISE_FUTURE |
| **Blocking production readiness** | No (intentional host authority) |

#### PLUGIN-AUDIT-F003

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Domain** | Activation / security |
| **Evidence** | `discover_plugins_enabled()` defaults false; `bootstrap_catalogs(..., discover_entry_points=False)` default; `INTERGRAX_DISCOVER_PLUGINS` required for Tier-0 EP wiring in `tool_wiring.py` / `integration_wiring.py` |
| **Impact** | Fail-closed security posture; operators may believe pip install alone activates plugins |
| **Current behavior** | Installation ≠ activation; explicit env or parameter required |
| **Expected behavior** | Documented operator workflow (current) - enterprise may want policy-driven activation UI |
| **Remediation class** | DOCUMENTATION / ENTERPRISE_FUTURE |
| **Blocking production readiness** | No |

#### PLUGIN-AUDIT-F004

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Domain** | Discovery / failure isolation |
| **Evidence** | `load_entry_point_plugins` defaults `on_load_failure="fail_fast"`; security/policy/RAG Tier-0 paths do not use `isolate`; only unit test demonstrates isolate |
| **Impact** | One broken third-party EP in a group can prevent entire group bootstrap |
| **Current behavior** | First load error aborts group registration |
| **Expected behavior** | Architecture §22 TARGET: bounded per-capability failure where loaders support it |
| **Remediation class** | HARDENING |
| **Blocking production readiness** | No (mitigated by trusted packages + opt-in discovery) |

#### PLUGIN-AUDIT-F005

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Domain** | Security defenses |
| **Evidence** | `defense_plugin_loader.py` line 29: `register_security_defense_plugin(plugin, override=True)` always |
| **Impact** | Duplicate or shipped defense IDs silently overridden by EP plugins; weak conflict observability |
| **Current behavior** | Last EP wins for same `plugin_id` |
| **Expected behavior** | Configurable conflict policy aligned with other catalogs |
| **Remediation class** | HARDENING |
| **Blocking production readiness** | No |

#### PLUGIN-AUDIT-F006

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Domain** | Observability |
| **Evidence** | No unified inventory API across catalogs; lifecycle enum not tracked at runtime; architecture §19 TARGET list not implemented as operator surface |
| **Impact** | Operators cannot answer «what plugins are active/failed/qualified» without custom introspection |
| **Current behavior** | Per-domain snapshots and logs |
| **Expected behavior** | Enterprise control-plane inventory (future) |
| **Remediation class** | ENTERPRISE_FUTURE |
| **Blocking production readiness** | No |

### MEDIUM

#### PLUGIN-AUDIT-F007

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Domain** | Global state |
| **Evidence** | Module-level `_CATALOG`, `_DYNAMIC`, `_tier0_shipped_done`; extensive `clear_catalog` / `reset_*_for_tests` in tests |
| **Impact** | Multi-application-in-one-process and test pollution if resets omitted |
| **Remediation class** | ARCHITECTURE (enterprise) / accepted for current model |
| **Blocking production readiness** | No |

#### PLUGIN-AUDIT-F008

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Domain** | Memory stores / performance |
| **Evidence** | `discover_session_turn_index_plugin_types()` calls `load_entry_point_plugins` on each invocation |
| **Impact** | Repeated metadata scan/import work when wiring memory vector indexes |
| **Remediation class** | HARDENING |
| **Blocking production readiness** | No |

#### PLUGIN-AUDIT-F009

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Domain** | Tool invocation patterns / performance |
| **Evidence** | `load_tool_invocation_pattern` iterates all EPs until name match |
| **Impact** | O(N) EP scan per pattern load on request/config paths |
| **Remediation class** | HARDENING |
| **Blocking production readiness** | No |

#### PLUGIN-AUDIT-F010

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Domain** | Memory bootstrap |
| **Evidence** | `bootstrap_memory_stores` counts discovered plugins but does not register to a catalog; registration happens only via explicit wiring helpers |
| **Impact** | Misleading bootstrap semantics; operators may expect catalog registration |
| **Remediation class** | DOCUMENTATION |
| **Blocking production readiness** | No |

#### PLUGIN-AUDIT-F011

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Domain** | Public API |
| **Evidence** | `errors.py` - `conflict_kind` and `result` typed as `object | None` |
| **Remediation class** | DX |
| **Blocking production readiness** | No |

#### PLUGIN-AUDIT-F012

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Domain** | Developer experience |
| **Evidence** | Architecture §20.3 - local registration documented for Tools/Integrations/Skills only |
| **Remediation class** | DX |
| **Blocking production readiness** | No |

#### PLUGIN-AUDIT-F013

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Domain** | Testing / CI |
| **Evidence** | `test_plugin_catalog_counts.py` failure at audit: core integrations 11 < MIN 12 |
| **Remediation class** | BUG_FIX (test drift) |
| **Blocking production readiness** | No (outside PLUGIN-9 gate) |

#### PLUGIN-AUDIT-F014

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Domain** | CI/CD |
| **Evidence** | PLUGIN-9 gate in `.github/workflows/unit-tests.yml` runs subset on `ubuntu-latest`; wheel E2E not Windows-gated |
| **Remediation class** | HARDENING |
| **Blocking production readiness** | No |

### LOW

#### PLUGIN-AUDIT-F015

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Domain** | Performance |
| **Evidence** | No caching of `iter_entry_point_specs` results across domains at platform level |
| **Remediation class** | HARDENING |
| **Blocking production readiness** | No |

#### PLUGIN-AUDIT-F016

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Domain** | Lifecycle |
| **Evidence** | `PlatformPluginLifecycleState` exists but no runtime transition tracking |
| **Remediation class** | ENTERPRISE_FUTURE |
| **Blocking production readiness** | No |

### INFO

#### PLUGIN-AUDIT-F017

| Field | Value |
|-------|-------|
| **Severity** | INFO |
| **Domain** | Security model |
| **Evidence** | `PlatformPluginTrustModel` single value `TRUSTED_IN_PROCESS`; contract forbids sandbox/signing enum values |
| **Remediation class** | DOCUMENTATION |
| **Blocking production readiness** | No |

#### PLUGIN-AUDIT-F018

| Field | Value |
|-------|-------|
| **Severity** | INFO |
| **Domain** | Architecture |
| **Evidence** | VK EP group `intergrax.vendor_knowledge.providers` intentionally excluded from Tier-0 `discovery.py` EP constants - separate contribution catalog |
| **Remediation class** | DOCUMENTATION |
| **Blocking production readiness** | No |

---

## 21. Remediation recommendations

### Immediate blockers

None for trusted-host production at audited SHA.

### Hardening backlog

1. Adopt `on_load_failure="isolate"` in production EP loaders with structured error aggregation (F004).
2. Align security defense registration with catalog conflict policies (F005).
3. Cache EP spec scans for memory stores and tool invocation patterns (F008, F009, F015).
4. Fix or recalibrate `test_plugin_catalog_counts` (F013).
5. Tighten error attribute types on public exceptions (F011).

### Architecture follow-ups

1. Document multi-app-per-process catalog isolation requirements (F007).
2. Clarify memory bootstrap count-vs-register semantics in author guide (F010).

### Enterprise-future (input for PLATFORM-PLUGIN-ENTERPRISE-1)

1. Canonical platform version authority API (F001).
2. Qualification provenance / attestation model (F002).
3. Unified plugin inventory and lifecycle telemetry (F006, F016).
4. Policy-controlled activation beyond env flags (F003).
5. Signed artifacts, revocation, distributed inventory (§19 matrix).

---

## 22. Cross-assessment summaries

| Area | Assessment |
|------|------------|
| **Modularity** | Strong Tier-0 coordination; domain-owned semantics preserved; acceptable loader duplication |
| **Security** | Appropriate for trusted in-process model; explicit opt-in and qualification gates; not sandboxed |
| **Scalability** | Fine for tens–hundreds of plugins per worker; global catalogs and EP rescans limit multi-app and hot-path efficiency |
| **Multi-tenancy** | Infrastructure does not store tenant mutable state in platform layer; depends on host wiring |
| **Lifecycle** | Vocabulary-only at platform layer; domain cleanup contracts vary |
| **Failure isolation** | Utility supports isolation; production loaders mostly fail-fast |
| **Test / CI** | Contract + E2E strong; full unit plugin suite has drift; gate is subset |
| **Public API** | Stable and discoverable; minor typing debt |
| **DX** | Tools path is reference quality; other surfaces lag |

---

## 23. Validation evidence

| Command / suite | Result |
|-----------------|--------|
| `pytest tests/contract/core/plugins/test_platform_plugin_contract.py` | PASS |
| `pytest tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py` | PASS |
| `pytest tests/unit/scaffold/test_scaffold_local_extension_qualification.py` | PASS |
| `pytest tests/unit/core/plugins/` | 153 pass, 1 fail (`test_plugin_catalog_counts`) |

---

## 24. References

- [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md)
- [`PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md`](PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md)
- [`PLATFORM_PLUGIN_9_CLOSEOUT.md`](PLATFORM_PLUGIN_9_CLOSEOUT.md)
- [`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md) (roadmap)
- `intergrax/core/plugins/` - implementation
- `tests/contract/core/plugins/test_platform_plugin_contract.py` - conformance suite
