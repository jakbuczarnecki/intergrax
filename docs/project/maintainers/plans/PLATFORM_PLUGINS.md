# Platform Plugins - Maintainer Roadmap

**Program:** Platform Plugin architecture
**Status:** PLATFORM-PLUGIN-1 **Done** · PLATFORM-PLUGIN-2 **Done** · PLATFORM-PLUGIN-3 **Done** · PLATFORM-PLUGIN-4 **Done** · PLATFORM-PLUGIN-5 **Done** · PLATFORM-PLUGIN-6 **Done** · PLATFORM-PLUGIN-7 **Done** · PLATFORM-PLUGIN-8 **Done** · PLATFORM-PLUGIN-9 **Done** - program **CLOSED**
**Audit evidence:** [`PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md`](PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md)
**Canonical architecture:** [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md)

**Last updated:** 2026-08-12

**Post-program production audit:** [`PLATFORM_PLUGIN_PRODUCTION_AUDIT.md`](PLATFORM_PLUGIN_PRODUCTION_AUDIT.md) (`PLATFORM-PLUGIN-AUDIT-1`, verdict `APPROVED_WITH_GAPS`, SHA `f7b6eedf354d43b1459b8077a56f8acd3fdaaa3d`).

---

## Purpose

Establish a **platform-wide** extension model for Intergrax without prematurely unifying domain-specific mechanisms that differ for valid reasons.

This program answers:

- what extension surfaces exist today;
- which are public, host-composed, or internal;
- where duplication and inconsistency are accidental vs intentional;
- whether a canonical Platform Plugin Contract is justified.

**Out of scope for this program:** implementing RAG, Vendor Knowledge, LKW, Integrations, Tools, or Skills features. Those domains remain **audit evidence** and **consumers** of extension mechanisms.

---

## Current-state summary (PLATFORM-PLUGIN-1)

**FACT:** Intergrax today has **no single global plugin framework**. Extension is implemented through **multiple coexisting models**:

| Class | Examples |
|-------|----------|
| Tier-0 setuptools entry-point catalogs | integrations, tools, skills, context, memory stores, RAG chunkers/retrievers/rerankers |
| Domain-specific entry-point catalogs | vendor knowledge providers, security defenses, policy rules, tool invocation patterns |
| Shipped first-party bootstrap | `register_default_integrations`, `register_default_tools`, `register_default_skills`, RAG defaults |
| Host-composed wiring | `ApplicationEnvironmentProfile`, profiles, `RuntimePlugin`, `AgentRegistry` |
| Internal registries without third-party EP | embedding providers, document handlers, integration registry v2 (metadata), task execution registry |
| Descriptor-only / planned contracts | token optimization plugins |

**FACT:** Unified Tier-0 discovery loader exists (`intergrax/core/plugins/discovery.py`) but **not all surfaces use it**. Conflict policies, opt-in flags, and registration targets differ per domain.

**FACT:** Third-party installation is **discoverable and loadable** for several entry-point groups when `INTERGRAX_DISCOVER_PLUGINS=true` (or domain-specific opt-in). This is **not** production qualification.

See audit document for inventory, taxonomy proposal, gaps, and evidence matrix.

---

## Relationship to existing extension systems

| Domain program | Relationship to Platform Plugin |
|----------------|------------------------------|
| **Integrations** | Largest Tier-0 catalog; dual shipped manifest vs `IntegrationPlugin` model; registry v2 metadata track |
| **Tools / Skills** | Tier-0 catalog plugins; profile-gated materialization |
| **Context Engineering** | `ContextPlugin` public EP catalog; qualification domain-owned - see [EXTENSION_AUTHOR_GUIDE](guides/EXTENSION_AUTHOR_GUIDE.md) and [CONTEXT_ENGINEERING architecture](../../architecture/CONTEXT_ENGINEERING.md) |
| **RAG** | Separate EP groups per component type; bootstrap-time registry, not global catalog |
| **Vendor Knowledge** | Separate contribution catalog and EP group; host composition required |
| **Security / Policy** | Separate EP groups; hook/policy integration |
| **Runtime (Nexus)** | `RuntimePlugin` host-composed; distinct from Tier-0 catalogs |
| **Observability** | Extension SDK for payload schemas; not a plugin loader |
| **Agents** | `AgentRegistry` - host registration only; no setuptools discovery |

**Principle (frozen in PLATFORM-PLUGIN-2):** Platform Plugin program **coordinates** cross-cutting discovery, trust, lifecycle, and author experience at the **package boundary**. It **must not** replace domain contracts (integration category contracts, tool contracts, VK contributions, etc.). See [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) §6–§7.

---

## Roadmap stages

| Stage | Name | Status | Depends on | Exit criteria (summary) |
|-------|------|--------|------------|-------------------------|
| **PLATFORM-PLUGIN-1** | Global extension surface inventory & architecture audit | **Done** | - | Audit/evidence doc; this roadmap; no production code changes |
| **PLATFORM-PLUGIN-2** | Architecture decision & canonical domain doc | **Done** | PLUGIN-1 | [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) - taxonomy, platform/domain boundary, DO-NOT-UNIFY, contract scope |
| **PLATFORM-PLUGIN-3** | Author contract & packaging model | **Done** | PLUGIN-2 | Package-level Platform Plugin contract in `intergrax/core/plugins/`; optional `[tool.intergrax.plugin]` manifest parsing; multi-capability metadata rules; entry points remain required for discovery |
| **PLATFORM-PLUGIN-4** | Discovery & registration harmonization (where approved) | **Done** | PLUGIN-2, PLUGIN-3 | Shared `discovery.py` scan/load primitives adopted by security, policy, tool-invocation loaders; RAG already on shared utility; VK composition unchanged |
| **PLATFORM-PLUGIN-5** | Configuration, secrets & DI conventions | **Done** | PLUGIN-2 | Cross-surface config matrix (architecture §12.3); canonical flow §12.4; host-resolved config and credential bindings documented; integration `env_prefix` preserved; no global DI/secrets API |
| **PLATFORM-PLUGIN-6** | Lifecycle, compatibility & conflict policy | **Done** | PLUGIN-2 | `platform_semantics.py`: `check_platform_compatibility` / `PlatformCompatibilityResult`; `PlatformPluginLifecycleState`; `PlatformPluginConflictKind`; EP conflict classification; domain `ConflictPolicy` unchanged |
| **PLATFORM-PLUGIN-7** | Trust, qualification & production gates | **Done** | PLUGIN-2, PLUGIN-6 | `platform_qualification.py`: qualification contracts + production gates; trust model; external + host-embedded delivery; no sandbox/signing |
| **PLATFORM-PLUGIN-8** | Third-party developer experience & executable E2E proof | **Done** | PLUGIN-3, PLUGIN-7 | Reference external wheel (`examples/platform_plugins/intergrax_reference_tool_plugin/`); host-embedded example (`examples/platform_plugins/local_embedded_tool_extension/`); scaffold `extensions/` hook; E2E `tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py`; author guide §16; architecture §20.3–§20.4 |
| **PLATFORM-PLUGIN-9** | Qualification, rollout, deprecation & program closeout | **Done** | PLUGIN-4–8 | Contract tests, CI gates, deprecation audit, final closeout evidence - [`PLATFORM_PLUGIN_9_CLOSEOUT.md`](PLATFORM_PLUGIN_9_CLOSEOUT.md) |

### PLATFORM-PLUGIN-2 output (complete)

Canonical architecture hub: [`docs/project/architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md)

**Frozen decisions (summary):**

- Taxonomy: PEP, IP, HCE, IEP, NE - unchanged from audit.
- Platform Plugin = **package-level coordination** + shared vocabulary - **not** a universal runtime wrapper.
- **Canonical Platform Plugin Contract:** **yes** - package metadata scope only; domain contracts unchanged.
- Multi-capability packages: **allowed**; capabilities separately discoverable via domain EP groups.
- Discovery: shared `discovery.py` utility **yes**; single global EP group **no**; domain groups retained.
- Manifest: **optional** coordination layer; domain manifests + entry points remain authoritative.
- DO-NOT-UNIFY list: frozen in architecture §23.

Implementation stages PLUGIN-3..9 must conform to this document.

---

## Dependencies between stages

```text
PLUGIN-1 (audit)
    → PLUGIN-2 (architecture decision)
        → PLUGIN-3 (author contract)
        → PLUGIN-5 (config/secrets)
        → PLUGIN-6 (lifecycle/conflicts)
            → PLUGIN-4 (harmonization - only approved items)
            → PLUGIN-7 (trust/qualification)
                → PLUGIN-8 (DX)
                    → PLUGIN-9 (rollout)
```

PLUGIN-4 is intentionally **after** architecture decision: harmonization without PLUGIN-2 risks accidental centralization.

---

## Audit evidence placement

**FACT:** [`docs/audit_results/`](../../audit_results/README.md) exists on `development`. It stores **dated outputs from orchestrated harness architecture audits** ([`AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md)): `YYYY-MM-DD/` folders with `legacy campaign README`, `RUN_SUMMARY.md`, and per-domain `<DOMAIN>.md` results, initialized and validated via `scripts/docs/` tooling.

**FACT:** PLATFORM-PLUGIN-1 evidence lives in [`PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md`](PLATFORM_PLUGIN_1_EXTENSION_SURFACE_AUDIT.md) under `maintainers/plans/` - **retained here** because it is **program-specific** extension-surface inventory and architecture audit evidence, not an orchestrated per-domain run in the harness audit workflow. Domain programs persist Mode A2 results under `docs/audit_results/`; the Platform Plugin program coordinates cross-cutting extension architecture and keeps its audit alongside this roadmap.

---

## Program closeout criteria (PLATFORM-PLUGIN-9)

Before the Platform Plugin program can be marked **CLOSED**, a final audit must confirm:

1. Canonical [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) matches implementation.
2. Maintainer and author documentation matches executable behavior.
3. The **third-party reference package** (PLATFORM-PLUGIN-8) installs, discovers, and runs successfully.
4. The full **install → discovery → configuration → DI → runtime → cleanup** extension path works without core changes.
5. Trust, compatibility, and conflict rules are represented correctly in docs and runtime.
6. No accidental competing plugin architectures were introduced.
7. Approved **DO-NOT-UNIFY** boundaries from PLATFORM-PLUGIN-1 remain preserved.

---

## Changes discovered during PLATFORM-PLUGIN-1

1. **Unified discovery loader is partial** - `core/plugins/discovery.py` covers Tier-0 groups; VK, security, policy, tool-invocation patterns use separate loaders.
2. **Integrations dual registration model** - shipped `manifest.py` + factory vs third-party `IntegrationPlugin` is intentional but increases author cognitive load.
3. **Context entry points exist in code** - `EXTENSION_AUTHOR_GUIDE` still lists Context as "Planned"; doc drift recorded in audit.
4. **Token optimization plugin descriptor** - contract exists; no setuptools loader or production registration path found.
5. **Integration registry v2** - additive metadata registry; not an extension surface for third parties.
6. **`docs/audit_results/` exists** for orchestrated harness domain audits - PLATFORM-PLUGIN-1 program evidence correctly placed under `maintainers/plans/` (see § Audit evidence placement).

---

## PROVIDER-QUAL track (post PLUGIN-9)

Extends PLUGIN-7 qualification for **provider-scoped** evidence without a new qualification engine. Architecture: [`satellites/PLATFORM_PLUGINS_provider_qualification.md`](../../architecture/satellites/PLATFORM_PLUGINS_provider_qualification.md).

| ID | Task | Status | Purpose |
|----|------|--------|---------|
| **PROVIDER-QUAL-0** | Architecture decision | **Done** | `EXTEND_EXISTING` - reuse core qualification + platform coordination + ProofReceipt |
| **PROVIDER-QUAL-1** | Architecture freeze + contract design | **READY_FOR_REVIEW** | Subject/run/status/evidence/admission/CI boundary freeze; PostgreSQL template; Oracle extensibility proof |
| **PROVIDER-QUAL-2** | Typed contracts | **Done** | `ProviderQualificationSubject`, `ProviderQualificationRun`, evidence kinds; unit/contract tests; vendor-neutrality invariants |
| **PROVIDER-QUAL-3A** | Provider binding audit | **Done** | Initial audit; superseded by **PROVIDER-QUAL-3A-R1** correction |
| **PROVIDER-QUAL-3A-R1** | Provider binding correction | **Done** | **EXTEND_EXISTING_PROVIDER_BINDING** - reuse Integrations resolution; extend with typed domain-provider bridge; satellite §15 freeze; INV-1..INV-8 |
| **PROVIDER-QUAL-3B** | Typed domain-provider binding | **READY_FOR_REVIEW** | Initial `CollaborativeWorkPersistenceProvider` bridge; superseded by **PROVIDER-QUAL-3B-R1** lifecycle correction |
| **PROVIDER-QUAL-3B-R1** | Lifecycle-safe provider materialization | **READY_FOR_REVIEW** | Single provider lifecycle; `_collaborative_work_materialization` catalog path; no abandoned generic relational runtime; connection_factory preserved |
| **PROVIDER-QUAL-3B-R2** | Explicit typed materialization factory | **READY_FOR_REVIEW** | `CollaborativeWorkPersistenceFactory` + `CollaborativeWorkMaterializationBinding`; no magic keyword protocol; no `TypeError` capability probing; pre-built SQLite fail-closed |
| **PROVIDER-QUAL-3B-R3** | Provider-owned typed configuration materialization | **READY_FOR_REVIEW** | `CollaborativeWorkPersistenceFactory.materialize_collaborative_work_repositories()` only; provider `bind_collaborative_work_materialization(options)` owns typed config; no CW vendor config bag; Oracle adds no CW config fields |
| **PROVIDER-QUAL-3B-R4** | Separate binder from configured materializer | **READY_FOR_REVIEW** | `CollaborativeWorkMaterializationBinder` (unbound catalog factory) → `bind_collaborative_work_materialization(options)` → `CollaborativeWorkPersistenceFactory` (configured materializer) → `CollaborativeWorkRepositories`; unbound PostgreSQL/SQLite factories are binders only; no fake unbound `materialize` |
| **PROVIDER-QUAL-3C** | Evidence persistence/integration | **READY_FOR_REVIEW** | `ProviderQualificationRun` → `ProofReceipt` / `DocumentStore` durable projection; lookup by `qualification_run_id`; idempotent persist + conflict fail-closed; no parallel qualification store |
| **PROVIDER-QUAL-3C-R1** | Evidence durability + safe persistence hardening | **READY_FOR_REVIEW** | Reuses `ProofReceipt` / `DocumentStore`; adapter reconstruction over shared fake Mongo collection (unit contract only); unsafe credential-bearing evidence rejected before write via `intergrax.core.security.secret_safety`; no new qualification storage backend |
| **PROVIDER-QUAL-3C-R2** | Real durable DocumentStore reopen proof | **READY_FOR_REVIEW** | Real persistent MongoDB `DocumentStore` reopen proof: store A persist → close → independent store B recovers same `qualification_run_id`; qualification persistence remains generic `DocumentStore`-based; no process restart required |
| **PROVIDER-QUAL-4** | Provider qualification discovery/index | **READY_FOR_REVIEW** | Reuses `ProofReceipt` / `DocumentStore` bounded partition scan; exact-match filters (`provider_id`, version, capability, suite, domain, environment, status); returns `ProviderQualificationRun`; deterministic ordering; no parallel index; no validity/staleness |
| **PROVIDER-QUAL-4-R1** | Scalable qualification discovery | **READY_FOR_REVIEW** | Storage narrows candidate set via generic `DocumentStore` data-path equality queries before qualification receipts are decoded; storage-backed cursor pagination; no parallel qualification index |
| **PROVIDER-QUAL-5** | Evidence validity lifecycle | **READY_FOR_REVIEW** | Append-only QualificationValidityRecord via ProofReceipt/DocumentStore; explicit CURRENT/STALE/REVOKED evaluation; immutable ProviderQualificationRun; latest validity view; no requalification runner or admission policy |
| **PROVIDER-QUAL-5-R1** | Terminal revocation + bounded current view | **READY_FOR_REVIEW** | REVOKED terminal per qualification_run_id; later CURRENT/STALE cannot reactivate; current view uses bounded DocumentStore queries (revocation existence + latest evaluation); history API remains full append-only |
| **PROVIDER-QUAL-5-R2** | Canonical validity time + consistent latest record resolution | **READY_FOR_REVIEW** | REVOKED current-view query selects newest REVOKED deterministically; validity `evaluated_at` persisted canonically in UTC; storage current view matches pure resolver semantics; no migration (existing PROVIDER-QUAL-5 records already UTC in all construction paths) |
| **PROVIDER-QUAL-6** | Requalification decision + new-run identity | **READY_FOR_REVIEW** | Derived `ProviderRequalificationDecision` from `QualificationValidityInterpretation`; CURRENT not required; STALE/REVOKED required with preserved reason; immutable prior run; `prepare_provider_requalification_run_identity` mints new `qualification_run_id` via existing helper; no scheduler/runner/store; shared execution runner = PROVIDER-QUAL-7 |
| **PROVIDER-QUAL-4-R2** | DocumentStore query contract hardening | **READY_FOR_REVIEW** | Generic `DocumentStore` query semantics consistent across InMemory and MongoDB: per-field asc/desc keyset pagination, query-bound authenticated v2 cursors (equalities, sort, upper bound), legacy v1 cursors for simple partition scans; storage-side filtering/keyset pagination - not a claim of physical secondary indexes |

**Explicit out of scope (PROVIDER-QUAL-1):** runtime Python changes, GHA vendor jobs, LKW integration, MP-2, admission policy engine, automatic staleness engine.

**Explicit out of scope (PROVIDER-QUAL-2):** persistence/index integration, `ProofReceipt` persistence, recording PostgreSQL 16.6 evidence, qualification runner/harness, admission policy engine, automatic staleness engine, GitHub Actions vendor execution.

---

## Explicit boundaries

| Track | Boundary |
|-------|----------|
| RAG | RAG component EP groups remain domain-owned; Platform Plugin may harmonize discovery flags only if PLUGIN-2 approves |
| Vendor Knowledge | Contribution catalog and LKW qualification remain VK program; not subsumed by Tier-0 catalog |
| LKW | Application binding and live capability rollout are not generalized plugin loading |
| Integrations | Category contracts and provider packages remain authoritative in INTEGRATIONS domain |
| Tools / Skills | ToolContract / SkillManifest contracts remain authoritative |
| Security / Policy | Fail-open/closed semantics stay security-domain owned |

---

## Protocol v2 platform extensibility remediation (2026-08-18 audit)

Accepted audit unit [`PLATFORM_EXTENSIBILITY`](../../audit_results/2026-08-18/PLATFORM_EXTENSIBILITY.md) (**FAIL**, 6 ACCEPTED findings, `audited_sha` `70c947c889f40222e5efb191241bdd8fa9035b17`, operator accepted 2026-08-21). Canonical architecture target: [`architecture/PLATFORM_PLUGINS.md`](../../architecture/PLATFORM_PLUGINS.md) - [Protocol v2 platform extensibility target invariants (2026-08-18)](../../architecture/PLATFORM_PLUGINS.md#protocol-v2-platform-extensibility-target-invariants-2026-08-18).

**Status rule:** all blocks below are **ACCEPTED / PLANNED** only. Do **not** mark IMPLEMENTED, VERIFIED, CLOSED, or DONE in this section. Do **not** reopen historical PLATFORM-PLUGIN-1..9 rows. Cross-link **PROVIDER-QUAL** (§ PROVIDER-QUAL track) for provider-scoped evidence - do not duplicate or overwrite ongoing PROVIDER-QUAL work.

### PLATFORM-EXTENSIBILITY-QUALIFICATION-AUTHORITY-INTEGRITY

**Priority:** P0  
**Findings:** `AUDIT-20260818-PLATFORM_EXTENSIBILITY-01`, `03`, `04`  
**Status:** ACCEPTED / PLANNED

Production qualification becomes evidence-derived and bound to the exact package + capability being admitted. Reuse `intergrax.core.qualification` and domain-specific qualification; **no** second qualification engine. Package qualification may remain a prerequisite; capability/domain admission binds distribution + domain + exact EP + policy + evidence. Production admission binds distribution + manifest identity/hash + capability descriptor/EP + qualification result.

### PLATFORM-EXTENSIBILITY-ADMISSION-COVERAGE-INTEGRITY

**Priority:** P0/P1  
**Findings:** `AUDIT-20260818-PLATFORM_EXTENSIBILITY-02`  
**Status:** ACCEPTED / PLANNED

All supported public PEP domains consume the common production-admission boundary in strict/product profiles while retaining domain loaders/contracts/registries. Extend Policy loader pattern; **no** global runtime plugin loader.

### PLATFORM-EXTENSIBILITY-LIFECYCLE-EVIDENCE-INTEGRITY

**Priority:** P1/P2  
**Findings:** `AUDIT-20260818-PLATFORM_EXTENSIBILITY-05`, `06`  
**Status:** ACCEPTED / PLANNED

Manifest failures remain diagnosable (VALID / ABSENT / INVALID / UNREADABLE + safe reason codes). Installed-plugin lifecycle/discovery cache semantics become explicit (immutable process lifetime **or** controlled rediscovery) - not incidental cache behavior.

---

## References

- Extension author guide: [`technical/guides/EXTENSION_AUTHOR_GUIDE.md`](../../technical/guides/EXTENSION_AUTHOR_GUIDE.md)
- Vendor Knowledge plugin guide: [`technical/guides/VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md`](../../technical/guides/VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md)
- Provider category contracts plan: [`PROVIDER_CATEGORY_CONTRACTS.md`](PROVIDER_CATEGORY_CONTRACTS.md)
- Platform foundation: [`architecture/PLATFORM_FOUNDATION.md`](../../architecture/PLATFORM_FOUNDATION.md)
