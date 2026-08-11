# PLATFORM-PLUGIN-1 — Global Extension Surface Inventory & Architecture Audit

**Task:** `PLATFORM-PLUGIN-1`  
**Status:** `READY_FOR_REVIEW`  
**Branch:** `development`  
**Canonical roadmap:** [`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md)  
**Architecture hub:** **none** (intentionally deferred to PLATFORM-PLUGIN-2)

**Legend:** **FACT** = repository evidence · **INFERENCE** = interpretation · **PROPOSAL** = future design (not current capability)

---

## A. Executive conclusion

**FACT:** Intergrax exposes **at least 22 materially distinct extension surfaces** across Tier-0 catalogs, domain-specific entry-point groups, host-composed wiring, and internal registries. There is **no** global plugin manifest, **no** shared lifecycle engine, and **no** unified trust/sandbox layer.

**INFERENCE:** Many surfaces are **intentionally domain-specific** (Vendor Knowledge contributions, security defenses, RAG component registries). Others share patterns (setuptools EP + catalog slug) but differ in conflict policy, opt-in flags, and materialization.

**INFERENCE:** Accidental duplication exists at the **discovery/loader** layer (`core/plugins/discovery.py` vs bespoke loaders) and **documentation** layer (Context EP implemented but guide marks "Planned").

**PROPOSAL:** A **canonical Platform Plugin Contract** is **conditionally justified** — as a **coordination and packaging layer** for third-party authors, not as a replacement for domain contracts. Final decision belongs to PLATFORM-PLUGIN-2.

**FACT:** Third-party code is **trusted installed Python**; no executable evidence of sandboxing or isolation for any extension surface.

---

## B. Extension surface inventory

Surfaces are numbered for reference. Taxonomy codes: `PEP` = PUBLIC_EXTERNAL_PLUGIN · `IP` = INTEGRATION_PROVIDER · `HCE` = HOST_COMPOSED_EXTENSION · `IEP` = INTERNAL_EXTENSION_POINT · `NE` = NOT_EXTENSIBLE.

| # | Subsystem | Surface | Purpose | Contract | Public? | Taxonomy |
|---|-----------|---------|---------|----------|---------|----------|
| 1 | Integrations | Tier-0 integration catalog | Register vendor backends by slug/category | `IntegrationPlugin` + `IntegrationManifest` | Yes (EP) | PEP / IP |
| 2 | Integrations | Shipped manifest registration | First-party providers without EP | `manifest.py` + factory | Internal | IEP |
| 3 | Tools | Tier-0 tool catalog | LLM-invokable tool bundles | `ToolPlugin` + `ToolBundleManifest` | Yes (EP) | PEP |
| 4 | Skills | Tier-0 skill catalog | Capability bundles (not LLM tools) | `SkillPlugin` + `SkillBundleManifest` | Yes (EP) | PEP |
| 5 | Context | Context plugin catalog | Context fragment providers | `ContextPlugin` | Yes (EP) | PEP |
| 6 | Memory | Memory store plugins | Swap profile/session/turn-index stores | duck-typed factory methods | Yes (EP) | PEP |
| 7 | RAG | Chunker plugins | Document splitting strategies | `BaseChunkingStrategy` / plugin class | Yes (EP) | PEP |
| 8 | RAG | Retriever plugins | Retrieval algorithms | `BaseRetriever` / `BaseRetrieverPlugin` | Yes (EP) | PEP |
| 9 | RAG | Reranker plugins | Reranking strategies | `BaseReranker` / `BaseRerankerPlugin` | Yes (EP) | PEP |
| 10 | Vendor Knowledge | Provider contributions | External VK source families | `VendorKnowledgeProviderContribution` | Yes (EP, opt-in) | PEP |
| 11 | Security | Defense plugins | UAEP hook security inspection | `SecurityDefensePlugin` | Yes (EP) | PEP |
| 12 | Policy | Policy rule handlers | Custom policy evaluation | `PolicyRuleHandler` | Yes (EP) | PEP |
| 13 | Nexus tools | Tool invocation patterns | Custom tool invocation modes | `ToolInvocationPattern` | Yes (EP) | PEP |
| 14 | Runtime | `RuntimePlugin` | Event bus / hooks / policy at app startup | `RuntimePlugin` dataclass | Host-only | HCE |
| 15 | Agents | `AgentRegistry` | Register Tier-2 agents | `Agent` + `AgentContract` | Host-only | HCE |
| 16 | Applications | Environment profiles | Integration/tool/skill/policy wiring | `ApplicationEnvironmentProfile` + profiles | Host-only | HCE |
| 17 | RAG | Embedding provider registry | Embedding backends | `EmbeddingProvider` | Internal register | IEP |
| 18 | RAG | Document handler registry | Format-specific loaders | `BaseDocumentHandler` | Shipped register | IEP |
| 19 | Integrations | Registry v2 metadata | Contract-aware provider metadata | `IntegrationRegistration` | Internal | IEP |
| 20 | LLM adapters | Model catalog | Model metadata / context windows | YAML `ModelCatalog` | Config overlay | IEP |
| 21 | Observability | Extension SDK | Diagnostic/signal payload schemas | `register_payload_schema` | Host/agent | HCE |
| 22 | Token optimization | Plugin descriptor | Optimizer capability declaration | `TokenOptimizationPluginDescriptor` | Contract only | IEP |
| 23 | Queueing | Task execution registry | Background task handlers | callable registration | Host-only | HCE |
| 24 | Runtime | Hook registry | Ordered hook handlers | `HookRegistry.register` | Internal + plugins | IEP |

**INFERENCE:** Surfaces 1–13 are the primary **third-party extension** candidates. Surfaces 14–16 are **application author** composition. Surfaces 17–24 are **runtime/internal** unless PLATFORM-PLUGIN-2 opens them.

### Per-surface detail (condensed)

#### 1 — Integration catalog (`intergrax.integrations`)

| Attribute | Value |
|-----------|-------|
| Discovery | `importlib.metadata` EP; `bootstrap_catalogs(discover_entry_points=…)` |
| Registration | `register_integration_plugin` → integration catalog slug |
| Config | `IntegrationManifest.env_prefix`; resolved via `IntegrationProfile` |
| DI | Factory `create_integration(**kwargs)`; host passes bindings |
| Lifecycle | Process-scoped catalog; no unload |
| Duplicate ID | `ConflictPolicy` + `catalog_registration_override` — error/override/skip |
| Trust | Trusted pip package; runs in process |
| Third-party | Yes — examples: `integrations/examples/custom_memory_kv`; tests: `test_external_integration_entry_point.py` |
| Docs | `EXTENSION_AUTHOR_GUIDE.md` §2 |

#### 2 — Shipped integration manifests

| Attribute | Value |
|-----------|-------|
| Discovery | `register_default_integrations` preset bundles |
| Registration | `register_from_manifest(manifest, factory)` |
| Third-party | No — first-party tree only |
| Docs | `EXTENSION_AUTHOR_GUIDE.md` dual-model note |

#### 3–4 — Tools & Skills (`intergrax.tools`, `intergrax.skills`)

| Attribute | Value |
|-----------|-------|
| Discovery | EP via `core/plugins/discovery.py` |
| Materialization | `build_registry_from_profile(ToolProfile/SkillProfile, ctx)` |
| Duplicate ID | bundle_id / tool_id / skill_id catalog policies |
| Third-party | Yes — `tools/examples/custom_echo`, `skills/examples/custom_pack` |
| Docs | `EXTENSION_AUTHOR_GUIDE.md` §3–4 |

#### 5 — Context (`intergrax.context`)

| Attribute | Value |
|-----------|-------|
| Discovery | EP + `bootstrap_context_catalog` |
| Registration | `register_context_plugin` |
| Opt-in | `INTERGRAX_DISCOVER_PLUGINS` |
| Docs gap | Guide table marks "Planned"; code ships `BuiltinContextPlugin` EP in root `pyproject.toml` |

#### 6 — Memory stores (`intergrax.memory_stores`)

| Attribute | Value |
|-----------|-------|
| Discovery | EP; duck-typed `create_user_profile_store` / `create_session_storage` / `create_session_turn_index` |
| Bootstrap | `bootstrap_memory_stores` counts plugins; host selects implementation |
| Docs | `EXTENSION_AUTHOR_GUIDE.md` §9 |

#### 7–9 — RAG components (chunkers, retrievers, rerankers)

| Attribute | Value |
|-----------|-------|
| EP groups | `intergrax.rag.chunkers`, `.retrievers`, `.rerankers` |
| Registration | Per-component bootstrap (`create_default_*`) calls `register_plugins` |
| DI | Retriever plugins receive vector store, embedding manager, graph store, profile |
| Tests | `tests/unit/rag/test_rag_plugin_discovery.py` |
| Docs | `EXTENSION_AUTHOR_GUIDE.md`; `architecture/RAG.md` |

#### 10 — Vendor Knowledge (`intergrax.vendor_knowledge.providers`)

| Attribute | Value |
|-----------|-------|
| Discovery | Separate loader in `contribution_catalog.py` |
| Composition | `VendorKnowledgeContributionCatalog` — instance-local, publication snapshot |
| Opt-in | `discover_entry_points` on composition builders; not global env alone |
| Conflict | `VendorKnowledgePluginConflict` on duplicate EP names |
| Reference | `tests/reference_plugins/vendor_knowledge/acme_reference/` |
| Docs | `VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md` |

#### 11 — Security defenses (`intergrax.security_defenses`)

| Attribute | Value |
|-----------|-------|
| Loader | `defense_plugin_loader.py` — always attempts EP when enabled |
| Registration | `register_security_defense_plugin(..., override=True)` |
| Integration | UAEP `HookPoint`s; `SecurityDefensePlugin` protocol |
| Trust | Fail-open/closed per plugin; no isolation |
| Docs | `EXTENSION_AUTHOR_GUIDE.md` (security section via SEC phases) |

#### 12 — Policy rules (`intergrax.policy_rules`)

| Attribute | Value |
|-----------|-------|
| Loader | `policy/rules/plugin_loader.py` |
| Composition | Merges with YAML via `policy_wiring.py` |
| Docs | `EXTENSION_AUTHOR_GUIDE.md` §10 |

#### 13 — Tool invocation patterns (`intergrax.tool_invocation_patterns`)

| Attribute | Value |
|-----------|-------|
| Loader | `tool_invocation_registry.py` — load-by-id |
| Scope | Nexus tool execution modes |
| Docs | Limited — code + tests |

#### 14 — RuntimePlugin

| Attribute | Value |
|-----------|-------|
| Discovery | Explicit list in host; **no** setuptools EP |
| Lifecycle | `bootstrap_runtime_plugins`; `on_shutdown` callbacks |
| Boundary | Must not import Tier-2 agents |
| Docs | `EXTENSION_AUTHOR_GUIDE.md` §8 |

#### 15 — AgentRegistry

| Attribute | Value |
|-----------|-------|
| Registration | `AgentRegistry.register(agent, contract=…)` |
| Validation | `assert_agent_assembly_valid`; skill/tool resolution |
| Third-party | Agents as packages in `agents/`; host wires — no EP |
| Docs | `architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` |

#### 16 — Application environment wiring

| Attribute | Value |
|-----------|-------|
| Mechanism | `applications/_shared/*_wiring.py`, manifests, profiles |
| Extension | Host selects presets, bundles, discover flags |
| Docs | `APPLICATION_HOSTING.md`, `EXTENSION_AUTHOR_GUIDE.md` §0–1 |

#### 17–18 — RAG embedding & document handlers

| Attribute | Value |
|-----------|-------|
| Registration | `EmbeddingProviderRegistry.register`; `DocumentHandlerRegistry.register` |
| Third-party | No EP — host or code modification |
| Taxonomy | IEP |

#### 19 — Integration registry v2

| Attribute | Value |
|-----------|-------|
| Purpose | Metadata for `(provider_id, category)` — INTEGRATIONS-3A |
| Runtime binding | **Not** performed — FACT per `registry_v2.py` docstring |
| Third-party | No |

#### 20 — LLM model catalog

| Attribute | Value |
|-----------|-------|
| Source | Bundled YAML + `INTERGRAX_LLM_MODEL_CATALOG_PATH` overlay |
| Third-party | Config file only |

#### 21 — Observability extension SDK

| Attribute | Value |
|-----------|-------|
| API | `extension_sdk.py` — schema_id namespaces for agents/applications |
| Registration | `register_payload_schema` |
| Not | A plugin loader |

#### 22 — Token optimization descriptor

| Attribute | Value |
|-----------|-------|
| Contract | `TokenOptimizationPluginDescriptor` in `contracts.py` |
| Loader | **None found** — fixture only in tests |
| Status | Descriptor-only / planned |

#### 23 — Task execution registry

| Attribute | Value |
|-----------|-------|
| API | `TaskExecutionRegistry.register(task_name, handler)` |
| Scope | Background workers — host composed |

#### 24 — Hook registry

| Attribute | Value |
|-----------|-------|
| API | `HookRegistry.register(point, handler, priority=…)` |
| Consumers | Runtime plugins, security defenses, middleware |

---

## C. Existing discovery models

**FACT:** **6 materially distinct discovery models** identified:

| ID | Model | Evidence |
|----|-------|----------|
| D1 | Unified setuptools EP loader | `intergrax/core/plugins/discovery.py` |
| D2 | Bespoke setuptools EP loaders | VK, security, policy, tool-invocation |
| D3 | Shipped first-party bootstrap | `register_default_*`, RAG defaults, context builtin |
| D4 | Explicit class/tuple at host call | `bootstrap_catalogs(integration_plugins=(…))` |
| D5 | Host manual registry | `AgentRegistry`, `TaskExecutionRegistry`, embedding registry |
| D6 | Static catalog / YAML | `model_catalog.yaml`, integration manifests |

**INFERENCE:** D1 and D2 are the primary unification candidates; D3–D6 should remain domain-specific per DO-NOT-UNIFY analysis.

---

## D. Existing registration/composition models

**FACT:** **5 materially distinct registration/composition models**:

| ID | Model | Evidence |
|----|-------|----------|
| R1 | Tier-0 catalog slug registration | integration/tool/skill/context catalogs |
| R2 | Profile-gated materialization | `IntegrationProfile`, `ToolProfile`, `SkillProfile` |
| R3 | Bootstrap composition pipelines | RAG stack, VK `*\_composition.py`, policy wiring |
| R4 | Runtime hook/event attachment | `RuntimePlugin`, `SecurityDefensePlugin`, `HookRegistry` |
| R5 | Instance-local contribution catalog | `VendorKnowledgeContributionCatalog` |

---

## E. Package-level entry-point mechanisms

**FACT:** Entry-point groups evidenced in code:

| Group | Loader | Conflict policy |
|-------|--------|-----------------|
| `intergrax.integrations` | `core/plugins/discovery.py` | Configurable `ConflictPolicy` |
| `intergrax.tools` | same | same |
| `intergrax.skills` | same | same |
| `intergrax.context` | same | same |
| `intergrax.memory_stores` | `load_entry_point_plugins` | same |
| `intergrax.rag.chunkers` | RAG bootstrap | same |
| `intergrax.rag.retrievers` | RAG bootstrap | same |
| `intergrax.rag.rerankers` | RAG bootstrap | same |
| `intergrax.vendor_knowledge.providers` | `contribution_catalog.py` | Error on duplicate name |
| `intergrax.security_defenses` | `defense_plugin_loader.py` | override=True always |
| `intergrax.policy_rules` | `plugin_loader.py` | registry.register |
| `intergrax.tool_invocation_patterns` | `tool_invocation_registry.py` | load by name |

**FACT:** Root `pyproject.toml` ships only `intergrax.context` builtin EP. Other groups are documented for third-party packages.

**FACT:** Opt-in gate: `INTERGRAX_DISCOVER_PLUGINS` (`intergrax/core/plugin_env.py`) — default **off** for Tier-0 wiring helpers.

---

## F. Manifests / capability metadata

| Mechanism | Metadata type | Location |
|-----------|---------------|----------|
| Integrations | `IntegrationManifest` | `integrations/core/manifest.py` |
| Tools | `ToolBundleManifest`, `ToolContract` | `tools/core/manifest.py` |
| Skills | `SkillBundleManifest`, `SkillManifest` | `skills/core/manifest.py` |
| Context | `ContextPlugin.plugin_id()` | `context/plugin.py` |
| VK | `VendorKnowledgeProviderContribution` | `runtime/vendor_knowledge/contribution.py` |
| Security | plugin_id, version, hook_points on instance | `defense_plugin.py` |
| Runtime | plugin_id, version, compatible_runtime | `RuntimePlugin` |
| Token opt | `TokenOptimizationPluginDescriptor` | `token_optimization/contracts.py` |
| Integrations v2 | `IntegrationRegistration` capabilities | `registry_v2.py` |

**INFERENCE:** No cross-domain manifest schema exists.

---

## G. Duplicate mechanisms

| Duplication | Type | Assessment |
|-------------|------|------------|
| EP loading: `discovery.py` vs bespoke loaders | Loader code | **INFERENCE:** Accidental — harmonization candidate for PLUGIN-4 if PLUGIN-2 approves |
| Integration manifest vs `IntegrationPlugin` | Registration API | **INFERENCE:** Intentional dual model for shipped vs external |
| Multiple catalog registries (integration/tool/skill/context) | Data structure | **INFERENCE:** Intentional domain separation |
| Integration catalog vs registry v2 | Metadata | **INFERENCE:** Transitional — v2 additive until INTEGRATIONS-3B |
| `RuntimePlugin` vs Tier-0 plugins | Concept name "plugin" | **INFERENCE:** Intentional — different tier and lifecycle |
| RAG per-type EP groups vs single RAG plugin group | Packaging | **INFERENCE:** Intentional — component injection differs |

---

## H. Architectural inconsistencies

1. **FACT:** Context EP implemented; author guide status "Planned" — documentation drift.
2. **FACT:** Security defense loader uses `override=True` always; Tier-0 catalogs use configurable conflict policy.
3. **FACT:** VK discovery requires explicit `discover_entry_points` on composition; Tier-0 uses env flag via wiring helpers.
4. **INFERENCE:** "Plugin" term overload — Tier-0 catalog, RuntimePlugin, VK contribution, security defense, policy handler.
5. **FACT:** Memory bootstrap counts EP plugins but does not auto-register into a global catalog — selection is host responsibility.
6. **FACT:** Token optimization descriptor without loader — contract ahead of mechanism.

---

## I. Public API gaps

- No single documented index of all EP groups (partially in `EXTENSION_AUTHOR_GUIDE`).
- No unified third-party package template covering multi-surface packages.
- Tool invocation patterns lack author-guide section.
- Context plugin public status unclear in docs vs code.
- No platform-level compatibility/version matrix across surfaces.
- `INTERGRAX_DISCOVER_PLUGINS` behavior not uniform across VK vs Tier-0.

---

## J. Security / trust gaps

**FACT:** All extensions are **trusted in-process Python** after pip install.

| Gap | Detail |
|-----|--------|
| No sandbox | No evidence of isolation for any surface |
| No code signing | EP trust = package install trust |
| Qualification split | discoverable ≠ production-qualified — qualification scattered per domain |
| Secret access | Integrations use env_prefix; no platform-wide secret scope model |
| Network/FS | Provider implementations have full process privileges |
| Defense plugins | fail_open/fail_closed per plugin — not centrally governed |

---

## K. Documentation gaps

- No `architecture/PLATFORM_PLUGINS.md` (deferred).
- `EXTENSION_AUTHOR_GUIDE` Context status stale.
- Historical paths (`docs/guides/`, `docs/architecture/`) may appear in GitHub search — **not** current canonical layout under `docs/project/`.
- No PLATFORM_PLUGIN audit slice in `audit_slices/` (optional future).
- Registry v2 not reflected in extension author guide.

---

## L. Third-party developer experience gaps

- Must know which EP group(s) to use per capability.
- Multi-capability package possible (fixture: `intergrax_catalog_fixture`) but no canonical guidance.
- Scaffold exists for integration/tool/skill — not for VK, security, policy, RAG components.
- Default discovery off — easy to believe EP "does not work".
- No reference "platform plugin" package demonstrating cross-surface authoring.

---

## M. Proposed taxonomy

| Code | Name | Definition |
|------|------|------------|
| **PEP** | PUBLIC_EXTERNAL_PLUGIN | setuptools EP or documented registration API; third-party pip install intended |
| **IP** | INTEGRATION_PROVIDER | PEP specialized for `IntegrationCategory` backends |
| **HCE** | HOST_COMPOSED_EXTENSION | Application host wires explicitly; no setuptools discovery |
| **IEP** | INTERNAL_EXTENSION_POINT | Registry exists; third-party extension not supported or not documented |
| **NE** | NOT_EXTENSIBLE | Closed implementation |

**INFERENCE:** Force-fitting VK contributions or security defenses into Tier-0 `IntegrationPlugin` would blur trust and lifecycle boundaries.

---

## N. Architectural questions for PLATFORM-PLUGIN-2

1. Should one external wheel expose integrations + tools + skills + RAG components + VK?
2. Is a shared manifest required, or are setuptools EP groups sufficient?
3. Should `core/plugins/discovery.py` become the only EP loader?
4. Unified `ConflictPolicy` defaults across all surfaces?
5. Single env flag vs per-domain opt-in for discovery?
6. Where does qualification gate sit — platform or per domain?
7. Should `RuntimePlugin` gain EP discovery or stay host-only?
8. Status of token optimization plugin loader?
9. Final role of integration registry v2 in extension story?
10. Should Context be promoted to fully public (doc + qualification)?
11. Agent registration — remain host-only forever?
12. Document handler / embedding registries — open EP or stay internal?

---

## O. Proposed target-direction options

**Option A — Coordination only (minimal):** Document EP groups, shared author guide, unified conflict/env flags; **no** new runtime.

**Option B — Packaging manifest (moderate):** Optional `pyproject` metadata or sidecar manifest listing capabilities; loaders unchanged.

**Option C — Platform Plugin Contract (broad):** New wrapper type registering multiple domain contributions with shared lifecycle — **only if** PLUGIN-2 proves domain loaders cannot stay separate.

**PROPOSAL:** Start from **Option A**; evaluate B if multi-surface packages become common; reject C unless evidence shows domain loaders cannot interoperate.

---

## P. DO-NOT-UNIFY findings

| Mechanism | Reason |
|-----------|--------|
| Vendor Knowledge contribution catalog | Publication snapshot, LKW qualification, tenant semantics — domain-specific |
| Security defense plugins | Hook-point security model; override policy; fail modes |
| RuntimePlugin | Tier-3 lifecycle; event bus — not catalog discovery |
| AgentRegistry | Tier-2 assembly; contracts; no third-party EP by design |
| RAG component registries | Different DI (vector store, embeddings) per component type |
| Integration registry v2 | Metadata-only transitional layer — not author surface |
| Policy YAML + EP handlers | Declarative + imperative merge — policy domain owns |
| Observability extension SDK | Schema registration ≠ plugin loading |
| Task execution registry | Worker-local handlers |
| Shipped integration manifest path | First-party scale (167 slugs) — performance and ownership |

---

## Q. Roadmap recommendations

See [`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md). **PLATFORM-PLUGIN-2** should decide taxonomy and whether `architecture/PLATFORM_PLUGINS.md` adopts Option A/B/C.

**Do not** implement unified global registry in PLUGIN-1 follow-up without PLUGIN-2 exit criteria.

---

## R. Evidence matrix

| Claim | Paths |
|-------|-------|
| Unified EP loader | `intergrax/core/plugins/discovery.py` |
| Tier-0 bootstrap | `intergrax/core/catalog_bootstrap.py` |
| Conflict policy | `intergrax/core/catalog_conflict.py` |
| Discover env flag | `intergrax/core/plugin_env.py` |
| Integration plugin register | `intergrax/integrations/registry/plugin_register.py` |
| Tool/Skill protocols | `intergrax/tools/core/plugin.py`, `intergrax/skills/core/plugin.py` |
| Context bootstrap | `intergrax/context/bootstrap.py` |
| Memory bootstrap | `intergrax/core/memory_bootstrap.py` |
| RAG EP bootstrap | `intergrax/rag/retrievers/bootstrap/retriever_bootstrap.py`, `tests/unit/rag/test_rag_plugin_discovery.py` |
| VK catalog | `intergrax/runtime/vendor_knowledge/contribution_catalog.py`, `tests/unit/runtime/vendor_knowledge/test_contribution_catalog.py` |
| Security loader | `intergrax/runtime/security/defense_plugin_loader.py`, `defense_plugin.py` |
| Policy loader | `intergrax/runtime/policy/rules/plugin_loader.py` |
| Tool invocation EP | `intergrax/runtime/nexus/tools/tool_invocation_registry.py` |
| RuntimePlugin | `intergrax/runtime/plugins/contract.py`, `bootstrap.py` |
| Agent registry | `intergrax/runtime/registry/agent_registry.py` |
| Integration registry v2 | `intergrax/runtime/integrations/registry_v2.py` |
| Extension author guide | `docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md` |
| VK author guide | `docs/project/technical/guides/VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md` |
| EP fixture package | `tests/fixtures/plugin_packages/intergrax_catalog_fixture/pyproject.toml` |
| VK reference plugin | `tests/reference_plugins/vendor_knowledge/acme_reference/` |
| Root EP | `pyproject.toml` `[project.entry-points."intergrax.context"]` |
| Application wiring | `intergrax/applications/_shared/integration_wiring.py` |
| Observability SDK | `intergrax/runtime/observability/extension_sdk.py` |
| Token opt contract | `intergrax/runtime/token_optimization/contracts.py` |

---

## S. Open uncertainties

| Uncertainty | Minimal additional evidence |
|-------------|----------------------------|
| Runtime signals EP group — if any beyond extension SDK | Grep `intergrax.runtime_signals` in `development` |
| Full list of shipped integration slugs count | Run `check_plugin_catalog.py` or catalog snapshot test |
| Whether any production host enables `INTERGRAX_DISCOVER_PLUGINS` by default | Inspect application host factories |
| MCP as extension surface vs tool export | Read `applications/local_workspace_application/mcp/server.py` host wiring only if PLUGIN-2 scopes MCP |

**STOP condition:** Further repo-wide reading not required for PLUGIN-2 planning; uncertainties are bounded and optional.

---

## Answers to key PLATFORM-PLUGIN-1 questions

| # | Answer |
|---|--------|
| 1 | **22+** materially distinct extension models (table §B) |
| 2 | VK, security, policy, RAG components, agents, runtime plugins — **intentionally domain-specific** |
| 3 | EP loader duplication, doc drift — **accidental** |
| 4 | PEP surfaces: Tier-0 groups 1–13 |
| 5 | HCE: RuntimePlugin, AgentRegistry, app wiring, observability SDK |
| 6 | Embedding, document handlers, registry v2 — still require core/host code or internal register |
| 7 | Yes — 12 setuptools groups evidenced |
| 8 | Yes — shipped bootstrap, manual register, profiles |
| 9 | Integration Library = Tier-0 integration catalog + profiles |
| 10 | VK contribution catalog; embedding/registry v2 internal registries |
| 11 | **INFERENCE:** Shared discovery **helper** yes; shared **catalog** no |
| 12 | **PROPOSAL:** Optional manifest — not proven required |
| 13 | **FACT:** Yes — fixture `intergrax_catalog_fixture` |
| 14 | Per-domain: env_prefix, profiles, YAML, inline on `ApplicationEnvironmentProfile` |
| 15 | Integration env_prefix; host secret stores; no platform-wide model |
| 16 | Host injects vector stores, wiring contexts, event bus, policy engine |
| 17 | **INFERENCE:** Some globals (catalog snapshots, shipped bootstrap flags) |
| 18 | Process-scoped register; shutdown on RuntimePlugin only |
| 19 | VK lifecycle, qualification — stay domain-specific |
| 20 | `compatible_runtime` on RuntimePlugin only; otherwise ad hoc |
| 21 | Trusted pip install / host code |
| 22 | Varies: error, override, skip per `ConflictPolicy`; security always override |
| 23 | Per-domain qualification (VK live, RAG qual tests, harness checks) |
| 24 | See §P |
| 25 | **Conditional yes** — coordination/packaging layer only |

---

## Validation notes

- Claims verified against executable code and cited tests on `development`.
- No `docs/project/architecture/PLATFORM_PLUGINS.md` created (intentional).
- Historical doc paths distinguished from current `docs/project/` layout.
- **Evidence location:** retained under `maintainers/plans/` as program-specific audit evidence. [`docs/audit_results/`](../../audit_results/README.md) holds **orchestrated** harness domain audit runs (`YYYY-MM-DD/`, `progress.json`, `RUN_SUMMARY.md`, `<DOMAIN>.md`) — a different workflow from this cross-cutting PLATFORM-PLUGIN inventory; not relocated to `audit_results/` without an orchestrated run.
