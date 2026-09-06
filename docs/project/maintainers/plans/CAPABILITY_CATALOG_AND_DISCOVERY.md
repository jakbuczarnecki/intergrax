# Capability Catalog and Discovery - Plan

**Status:** Active (architecture frozen — CAPABILITY-CATALOG-1)  
**Architecture (1:1):** [`architecture/CAPABILITY_CATALOG_AND_DISCOVERY.md`](../../architecture/CAPABILITY_CATALOG_AND_DISCOVERY.md)  
**Last updated:** 2026-09-05

---

## Goal

Deliver a **cross-domain federated discovery plane** for V1 capability types (Agent, Skill, Tool) that aggregates read-only catalog sources, supports scoped query/filter/rank/recommendation, integrates with governance, and hands off to **existing domain lifecycle authorities** — without a universal capability engine, merged registry, or discovery-driven runtime mutation.

### Program boundaries (frozen)

| Boundary | Rule |
|----------|------|
| Capability Catalog | Pure federating consumer — read, aggregate, query, rank, recommend |
| AC-4 | Agent acquisition — remains under Agent Distribution; not merged |
| AW-7A | Worker capability recovery — remains under Autonomous Work; not merged |
| Platform Plugins | Canonical package/plugin coordination — not replaced |
| Tier-3 composition | `wire_application_environment()` — not bypassed |
| V1 types | Agent, Skill, Tool only |

---

## Program non-goals

- `UniversalCapabilityEngine` or `UniversalRegistry`
- Merging `AgentRegistry`, `SkillRegistry`, `ToolRegistry`
- Discovery or Marketplace mutating registries or installing into live runtime
- Marketplace billing/settlement as part of catalog core
- Third-party sandbox / remote execution engine (assessed in Stage 12 only)
- Integrations, Memory, RAG, Context, Policy, Models as V1 catalog types (boundary notes only)

---

## Dependency graph (stages)

```text
Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5
                              ↘
Stage 6 (Skill correctness) ───→ Stage 7 → Stage 8 → Stage 9
Stage 10 (bootstrap evidence) ──↗          ↘
Stage 11 (Marketplace surface)              Stage 14 (full AW integration)
Stage 12 (isolation maturity)               Stage 13 (metering)
```

Stages 11–13 may proceed in parallel after Stage 5 where dependencies allow; Stage 14 requires Stages 5, 8, 9, and 10.

---

## Stage 1 — Contracts and frozen boundaries

| Field | Value |
|-------|-------|
| **User goal** | One canonical vocabulary so Agent, Skill, and Tool discovery integrations do not fork terminology or bypass invariants. |
| **Scope** | Capability discovery terminology; minimal shared value contracts **only where cross-domain reuse is proven**; explicit mapping to domain-owned types; forbidden-flow documentation in tests/contracts. |
| **Reuse** | AC-4 terminology (`AgentDiscoveryCandidate`, source-qualified identity); Platform Plugins disposition vocabulary; TOOLS/SKILLS selection vocabulary. |
| **Non-goals** | Universal discovery port; merged registry interfaces; runtime generalization; new execution abstractions. |
| **Hard contracts** | Normative state vocabulary (`AVAILABLE` … `EXECUTABLE`); `DISCOVERY ≠ SELECTION ≠ ENABLEMENT …`; capability type enum frozen to Agent/Skill/Tool for V1; source identity + provenance fields on any shared candidate view. |
| **Required tests** | Contract tests asserting forbidden merges (no `UniversalRegistry`); terminology snapshot tests; tier-boundary checks (Tier-0 catalog must not import `applications/`). |
| **Regression gates** | Existing AC-4 unit/integration suite unchanged; Platform Plugin contract suite unchanged. |
| **Completion criteria** | Architecture invariants traceable to typed contracts; no new runtime mutation APIs; peer review sign-off on boundary doc ↔ contract alignment. |
| **Depends on** | Frozen architecture document (this program). |
| **Maturity** | **Implemented** — Tier-0 contracts at `intergrax/contracts/capability_catalog/` (`CapabilityKind`, `CapabilitySourceIdentity`, `CapabilityLogicalIdentity`, `CapabilityDiscoveryIdentity`, `CapabilityProvenance`, `CapabilityStageVocabulary`); V1 `schema_version` fail-closed via `Literal[...]` on all versioned Stage-1 models; contract tests `tests/unit/contracts/capability_catalog/test_capability_catalog_contracts.py` (including schema-version enforcement); architecture gates `tests/unit/contracts/capability_catalog/test_capability_catalog_architecture_gates.py` (recursive AST symbol inspection for forbidden registry classes and runtime-mutation APIs; AST import gate). |

---

## Stage 2 — Federated catalog read model

| Field | Value |
|-------|-------|
| **User goal** | Operators and runtimes can query a unified **read-only** view of Agent, Skill, and Tool catalog entries without a central database of truth. |
| **Scope** | Domain adapters/providers feeding federated read model; source identity; provenance preservation; read-only federation; deterministic conflict handling (fail closed). |
| **Reuse** | `CatalogSourceProvider` / AC-4 federation patterns; Skill catalog bundles; Tool catalog bundles; Platform Plugins discovery loader for package metadata. |
| **Non-goals** | Write-through cache to registries; central SQL catalog of record; replacing Agent Distribution catalog authority. |
| **Hard contracts** | `CapabilityCatalogSource` (read-only); `CapabilityCatalogEntry` with `source_id`, `capability_type`, domain-native identity projection; federation merge rules (deterministic ordering, conflict = reject both). |
| **Required tests** | Multi-source federation tests; conflict fail-closed tests; provenance round-trip tests; adapter contract tests per domain. |
| **Regression gates** | AC-4 federated discovery tests; no change to AC-3 lifecycle behavior. |
| **Completion criteria** | Three domain adapters (Agent, Skill, Tool) produce federated read results; conflicts documented and tested; zero registry mutation paths. |
| **Depends on** | Stage 1. |
| **Maturity** | **Implemented** — Tier-0 federated read model at `intergrax/capability_catalog/` (`CapabilityCatalogSource`, `CapabilityCatalogEntry`, `CapabilityCatalogSnapshot`, `FederatedCapabilityCatalog`, `merge_capability_catalog_entries`); domain adapters at `intergrax/capability_catalog/adapters/` (`AgentCatalogCapabilitySource`, `SkillBundleCatalogSource`, `ToolBundleCatalogSource`); `CapabilityCatalogEntry` enforces `identity.source == provenance.source` (full `CapabilitySourceIdentity`, including `source_kind`) at validation — fail-closed, no auto-normalization; `FederatedCapabilityCatalog.snapshot()` enforces `source.source_id == entry.identity.source.source_id` per provider read — configuration/source-contract integrity (`CapabilityCatalogConfigurationError`); no private cross-module imports in Stage-2 core (deterministic entry ordering via public `entry.identity.sort_key`); conflict semantics reuse AC-4 exact-duplicate dedupe + source-qualified identity conflict fail-closed; provider read failure aborts snapshot (no partial output); `intergrax/core/catalog_snapshot.py` **not** reused (bootstrap/product in-memory inventory — distinct non-authoritative semantics); federation tests `tests/unit/capability_catalog/test_federation.py`; architecture gates `tests/unit/capability_catalog/test_architecture_gates.py`; adapter contract tests `tests/unit/capability_catalog/adapters/test_{agent,skill,tool}_adapter.py`; regression gates `tests/unit/agent_distribution/test_federated_discovery.py`, `tests/unit/agent_distribution/test_federated_catalog.py`, `tests/unit/agent_distribution/test_agent_discovery.py`, `tests/unit/tools/registry/test_catalog.py`, `tests/unit/contracts/capability_catalog/`. |

---

## Stage 3 — Query, filtering, and candidate model

| Field | Value |
|-------|-------|
| **User goal** | Callers express capability needs with typed queries and receive stable candidate sets scoped to tenant, application, and work context. |
| **Scope** | Typed discovery query; candidate output model; tenant/application scope; availability status surfacing (catalog vs host-wired vs policy-blocked). |
| **Reuse** | AC-4 `AgentDiscoveryRequest`/`Result` patterns; Tool/Skill profile enablement semantics; policy bundle narrowing hooks. |
| **Non-goals** | Permission grants via query API; implicit install/select; LLM-only query protocol (optional extension only). |
| **Hard contracts** | `CapabilityDiscoveryQuery` (typed fields, no `dict[str, Any]` substitute); `CapabilityDiscoveryCandidate` (source-qualified); `AvailabilityDisposition` enum aligned with governance narrowing. |
| **Required tests** | Scope filter tests (org/tenant/app); empty vs blocked vs available candidate sets; query determinism tests. |
| **Regression gates** | AC-4 discovery request compatibility preserved for agent slice. |
| **Completion criteria** | End-to-end read query returns typed candidates for all three V1 types; scope parameters mandatory for enterprise paths. |
| **Depends on** | Stage 2. |
| **Maturity** | **Implemented** — typed query contracts at `intergrax/contracts/capability_catalog/` (`CapabilityDiscoveryScope`, `CapabilityDiscoveryScopeMode`, `CapabilityDiscoveryQuery`, `LogicalIdentityFilter`, `SourceFilter`, `CapabilityIdentityKey`, `CapabilityDiscoveryAvailabilityEvidence`, `AvailabilityDisposition`); candidate projection at `intergrax/capability_catalog/candidate.py` (`CapabilityDiscoveryCandidate` preserves `CapabilityCatalogEntry`, source-qualified identity, provenance); deterministic filtering at `intergrax/capability_catalog/discovery.py` (`discover_capability_candidates` consumes Stage-2 `CapabilityCatalogSnapshot` only — read-only, no registry scan); enterprise scope fail-closed (`organization_id` + `tenant_id` + `application_id` mandatory; `scope_visible_keys` evidence required); explicit `GLOBAL` scope path; availability dispositions projected from caller evidence (`CATALOG_AVAILABLE`, `HOST_AVAILABLE`, `BLOCKED`, `UNAVAILABLE`, `SCOPE_UNAVAILABLE`); Stage 3 does not execute governance; default query surfaces all in-scope dispositions; empty ≠ blocked distinguishable via disposition + optional `availability_constraints`; ordering preserves Stage-2 snapshot order; contract tests `tests/unit/contracts/capability_catalog/test_capability_discovery_contracts.py`; filtering tests `tests/unit/capability_catalog/test_discovery.py`; architecture gates extended in `tests/unit/capability_catalog/test_architecture_gates.py` and `tests/unit/contracts/capability_catalog/test_capability_catalog_architecture_gates.py`; regression gates `tests/unit/agent_distribution/test_federated_discovery.py`, `tests/unit/agent_distribution/test_agent_discovery.py`, `tests/unit/tools/registry/test_catalog.py`, `tests/unit/contracts/capability_catalog/`, `tests/unit/capability_catalog/`. |

**Availability evidence consistency (Stage 3):**

- HOST_AVAILABLE / BLOCKED / UNAVAILABLE evidence sets are mutually exclusive per `CapabilityIdentityKey`.
- Contradictory evidence fails closed during contract validation.
- Stage 3 does not resolve conflicting availability facts by precedence.
- `scope_visible_keys` remains an orthogonal visibility dimension.

---

## Stage 4 — Ranking

| Field | Value |
|-------|-------|
| **User goal** | Discovery results are ordered by relevance, policy fit, and domain-specific signals — without duplicating existing selection engines. |
| **Scope** | Pluggable rankers; reuse hooks into semantic tool selection, hierarchical tool selection, AC-4 `AgentSelectionStrategy`; shared ranking utility **only if two domains need identical contract**. |
| **Reuse** | Existing semantic/hierarchical tool selection; `AgentSelectionStrategy`; federation ordering from AC-4. |
| **Non-goals** | Copy-paste ranker code across domains; single global ML ranker; ranking that mutates candidate identity. |
| **Hard contracts** | `CapabilityRanker` port; ranked output preserves provenance; rank metadata as evidence, not authority. |
| **Required tests** | Deterministic ranking tests; tie-break rules; ranker plugin registration tests; no-permission-elevation tests. |
| **Regression gates** | Tool selection regression suite; AC-4 selection tests. |
| **Completion criteria** | At least one ranker per capability type or justified shared ranker; documented tie-break; ranking separated from selection stage. |
| **Depends on** | Stage 3. |
| **Maturity** | **Implemented** — ranking port at `intergrax/capability_catalog/ranking.py` (`CapabilityRanker`, `StableIdentityRanker`, `rank_capability_candidates`, `identity_sort_key`); ranked output at `intergrax/capability_catalog/ranked_candidate.py` (`RankedCapabilityCandidate` delegates identity/provenance/availability to Stage-3 candidate); ranking evidence/context at `intergrax/contracts/capability_catalog/ranking.py` (`CapabilityRankingEvidence`, `CapabilityRankingContext`, `CapabilityRankingSignal`); fail-closed integrity validation at `intergrax/capability_catalog/ranking_validation.py` (`validate_ranked_output` — same identities, no duplicates, no mutation, contiguous 1..N positions); baseline `StableIdentityRanker` (`stable.identity`) — canonical `identity.sort_key` order with `original_stage3_position` evidence, no semantic scoring; Agent adapter `intergrax/capability_catalog/adapters/agent_ranking.py` (`AgentStableIdentityCapabilityRanker` — reuses AC-4 stable identity ordering primitive from `sorted_eligible_identities` / `DeterministicIdentitySelectionStrategy`, **not** `select()`); Tool adapter `intergrax/capability_catalog/adapters/tool_ranking.py` (`KeywordOverlapToolCapabilityRanker` — keyword overlap scoring via shared Tool-domain primitive `intergrax/tools/search/keyword_ranking.py`, ordering only); Tool keyword tokenization/scoring uses shared Tool-domain primitive consumed by both TOOL-ENG-5 selection and Stage-4 capability ranking. Selection semantics remain domain-owned; Skill uses shared `StableIdentityRanker` (no Skill domain ranker — `SkillResolver` has no ranking primitive); tie-break: primary signal → `identity.sort_key` → original Stage-3 position; pluginability via constructor injection (no global registry); `CapabilityRankingError` in `intergrax/capability_catalog/errors.py`; ranking ≠ selection (no `select`/`winner` API); ranking ≠ governance (availability read-only, no elevation); tests `tests/unit/capability_catalog/test_ranking.py`, `tests/unit/contracts/capability_catalog/test_capability_ranking_contracts.py`, `tests/unit/tools/search/test_keyword_ranking.py`; architecture gates updated in `tests/unit/capability_catalog/test_architecture_gates.py`, `tests/unit/contracts/capability_catalog/test_capability_catalog_architecture_gates.py` (allow `rank` only in ranking modules); regression gates `tests/unit/agent_distribution/test_agent_discovery.py`, `tests/unit/agent_distribution/test_federated_discovery.py`, `tests/unit/tools/registry/`, `tests/unit/skills/`, `tests/unit/contracts/capability_catalog/`. |

---

## Stage 5 — Governance integration

| Field | Value |
|-------|-------|
| **User goal** | Policy and authority checks narrow discovery output fail-closed before selection or downstream lifecycle. |
| **Scope** | Policy narrowing integration; authority checks; fail-closed dispositions; evidence on blocked candidates. |
| **Reuse** | Policy bundle tool access; Agent trust gates; Platform Plugins STRICT posture patterns; AW governance hooks. |
| **Non-goals** | Discovery granting new permissions; bypassing domain admission; global policy engine replacement. |
| **Hard contracts** | `GovernedDiscoveryResult` with `allowed` / `blocked` partitions; stable reason codes; no elevation on `SELECTED` alone. |
| **Required tests** | Policy deny tests; conflict fail-closed; STRICT vs non-STRICT behavior where applicable; audit evidence shape tests. |
| **Regression gates** | Security/Policy bootstrap STRICT tests; AC-4 trust handoff unchanged. |
| **Completion criteria** | Governed discovery path required for production STRICT hosts; blocked candidates carry typed reasons. |
| **Depends on** | Stage 4. |
| **Maturity** | **Implemented** — governance contracts at `intergrax/contracts/capability_catalog/governance.py` (`GovernanceDisposition`, `CapabilityGovernanceReasonCode`, `GovernanceDecisionEvidence`, `CapabilityGovernanceContext`, `CapabilityGovernancePosture`, `CapabilitySetConstraintMode`, domain evidence projections `CapabilityToolGovernanceEvidence`, `CapabilityAgentGovernanceEvidence`, `CapabilitySkillGovernanceEvidence`); governed output at `intergrax/capability_catalog/governed_candidate.py` (`GovernedCapabilityCandidate`, `BlockedCapabilityCandidate`) and `intergrax/capability_catalog/governed_result.py` (`GovernedDiscoveryResult`); governance port at `intergrax/capability_catalog/governance.py` (`CapabilityGovernanceEvaluator`, `AvailabilityPreservingGovernanceEvaluator`, `govern_capability_candidates` — ALL evaluators must allow, ANY block → BLOCKED); fail-closed integrity validation at `intergrax/capability_catalog/governance_validation.py` (`validate_governed_output` — total partition, no elevation, rank order preserved); **governance set constraints distinguish an unconstrained dimension from an explicitly configured set** — `UNCONSTRAINED` + empty set means no narrowing; `EXPLICIT_SET` + empty set means deny all candidates in that dimension; **production ToolProfile and SkillProfile projections always use `EXPLICIT_SET` semantics, including when the projected set is empty**; **STRICT governed discovery requires an explicitly configured non-empty evaluator pipeline** — an empty STRICT evaluator pipeline is a governance configuration error and fails closed before candidate evaluation; **evaluator IDs in one governance pipeline must be unique** for unambiguous audit provenance; **evaluator runtime failure in STRICT** → candidate `BLOCKED` / `EVALUATOR_FAILURE`; **evaluator contract violation** → `CapabilityGovernanceError` / operation fails; Tool adapter `intergrax/capability_catalog/adapters/tool_governance.py` (`ToolPolicyGovernanceEvaluator` — projects caller-supplied tool access evidence, no execution); Agent adapter `intergrax/capability_catalog/adapters/agent_governance.py` (`AgentTrustGovernanceEvaluator` — projects trust/admission evidence, no trust verification); Skill adapter `intergrax/capability_catalog/adapters/skill_governance.py` (`SkillProfileGovernanceEvaluator` — profile enablement evidence only); STRICT missing evidence fail-closed; conflicting evidence fail-closed; `CapabilityGovernanceError` in `intergrax/capability_catalog/errors.py`; governance ≠ selection (no `select`/`winner` API); governance ≠ lifecycle (no install/activate/execute); tests `tests/unit/capability_catalog/test_governance.py`, adapter tests under `tests/unit/capability_catalog/adapters/`, `tests/unit/contracts/capability_catalog/test_capability_governance_contracts.py`; regression gates `tests/unit/runtime/nexus/tools/test_tool_access_policy_scope.py`, `tests/unit/agent_distribution/test_agent_distribution_package_trust.py`, `tests/unit/runtime/governance/test_runtime_execution_policy_admission.py`, `tests/integration/platform_plugins/test_plugin_engine_cross_flow.py`; **production STRICT enforcement** at composition boundary `intergrax/applications/_shared/production_capability_discovery_composition.py` (`discover_rank_and_govern_capabilities` — discovery → ranking → governance before downstream; `resolve_capability_governance_posture` maps host `ExecutionMode.STRICT` → `CapabilityGovernancePosture.STRICT`; `consume_governed_discovery_for_downstream` accepts only `GovernedDiscoveryResult`); authority evidence projection at `intergrax/applications/_shared/production_capability_governance_evidence.py` (Tool ← host `ToolProfile` via `available_tool_ids_for_profile`; Skill ← host `SkillProfile` via `enabled_skill_ids_for_profile`; Agent evidence supplied by registry-authority caller); architecture gates `tests/unit/applications/test_production_capability_discovery_architecture_gate.py`, integration tests `tests/unit/applications/test_production_capability_discovery_composition.py`. **Production STRICT hosts are required to cross the governed-discovery boundary before any capability selection or downstream lifecycle handoff.** |

---

## Stage 6 — Skill enterprise correctness

| Field | Value |
|-------|-------|
| **User goal** | Enterprise deployments can pin and audit **which skill versions** compose into resolved packs — closing the version pinning gap. |
| **Scope** | Architecture decision + implementation for skill version pinning; resolved skill provenance/snapshot; dependency evidence for transitive `requires_skills`. |
| **Reuse** | `SkillManifest`, `SkillResolver`, `ResolvedSkillPack`; existing catalog versioning. |
| **Non-goals** | Solving pinning inside Capability Catalog alone (Skill domain owns resolver truth); ad hoc discovery-side version overrides. |
| **Hard contracts** | Pinning decision ADR or architecture amendment in [`SKILLS.md`](../../architecture/SKILLS.md); immutable resolved snapshot identity where pinning enabled; discovery surfaces pinned vs floating disposition. |
| **Required tests** | Pin resolution tests; transitive dependency pin tests; discovery candidate version alignment tests. |
| **Regression gates** | SK-EXP skill composition suite; harness `wire_application_environment` skill checks. |
| **Completion criteria** | Documented pinning model shipped in Skill domain; discovery read model exposes version/disposition consistently. |
| **Depends on** | Stage 3 (candidate model); coordinates with Skills domain plan rows. |
| **Maturity** | **Implemented** — canonical version model in [`SKILLS.md`](../../architecture/SKILLS.md); root PINNED + transitive MATERIALIZED semantics in `intergrax/skills/resolver.py` (`ResolvedSkillRef`, `ResolvedSkillPack.snapshot_digest`); snapshot identity via collision-safe canonical binary framing in `intergrax/skills/snapshot_digest.py` (length-prefixed UTF-8 fields, schema `resolved_skill_pack.v1`, topological order, SHA-256); immutable agent snapshots in `intergrax/runtime/registry/agent_registry.py` (`get_resolved_skill_pack`); catalog version projection in `intergrax/capability_catalog/adapters/skill.py` (`version_label`, `SkillVersionBindingDisposition.MATERIALIZED`); contracts at `intergrax/contracts/capability_catalog/skill_version_binding.py` and `intergrax/skills/core/version_binding.py`; tests `tests/unit/skills/test_version_pinning.py`, `tests/unit/skills/test_snapshot_digest.py`, `tests/unit/skills/test_architecture_gates.py`, `tests/unit/runtime/registry/test_agent_registry_skills.py`, `tests/unit/capability_catalog/adapters/test_skill_adapter.py`; regression gates `tests/unit/skills/` (SK-EXP), `tests/unit/applications/test_skill_tool_profile.py`. |

---

## Stage 7 — Tool/Skill distribution and catalog maturity

| Field | Value |
|-------|-------|
| **User goal** | Enterprise and private deployments discover Tools and Skills from non-public catalog sources with versioned availability — without second registries. |
| **Scope** | Enterprise/private catalog source support for Tool and Skill; versioned availability metadata; lifecycle alignment with domain authority only. |
| **Reuse** | Platform Plugins external packages; Tool/Skill plugin paths; Agent `CatalogSourceProvider` pattern as reference. |
| **Non-goals** | `ToolRegistry`/`SkillRegistry` duplication; Marketplace backend requirement; runtime install from catalog query. |
| **Hard contracts** | Private `CapabilityCatalogSource` implementations; source-qualified tool/skill identities; install/enable remains `wire_application_environment()` + profile mutation paths. |
| **Required tests** | Private source adapter tests; air-gapped catalog fixture tests; no-registry-mutation tests. |
| **Regression gates** | Tool/Skill plugin E2E proofs; Tier-3 composition gate. |
| **Completion criteria** | At least one private catalog adapter each for Tool and Skill in proof; documented operator flow for profile update vs discovery. |
| **Depends on** | Stages 2, 5; Stage 6 recommended for Skill version display. |
| **Maturity** | **Implemented** — `PrivateToolCapabilityCatalogSource` and `PrivateSkillCapabilityCatalogSource` at `intergrax/capability_catalog/adapters/private_tool.py` and `private_skill.py`; enterprise-private `CapabilitySourceKind.ENTERPRISE_PRIVATE` source identity; versioned provenance (`version_label`, optional package/digest/publisher); federation with built-in Tool/Skill bundle sources; air-gapped in-memory fixtures; read-only discovery (`CATALOG_AVAILABLE` without registry/profile mutation); operator flow documented in architecture §Enterprise deployment; tests `tests/unit/capability_catalog/adapters/test_private_tool_adapter.py`, `test_private_skill_adapter.py`, `test_private_catalog_stage7.py`. |

---

## Stage 8 — Adaptive Unit-of-Work discovery

| Field | Value |
|-------|-------|
| **User goal** | Workers rediscover capabilities per work stage as goals and steps evolve — not once at bootstrap. |
| **Scope** | Goal / current step / capability need model; discovery per work stage; effective capability set computation; rediscovery between stages; deterministic evidence. |
| **Reuse** | Effective availability model from architecture; Unit of Work boundaries from runtime/work models; AW work-step semantics. |
| **Non-goals** | New global runtime inventory registry; automatic install on rediscovery. |
| **Hard contracts** | `WorkStageCapabilityNeed` (typed); `EffectiveCapabilitySet` as query result not registry; rediscovery evidence records. |
| **Required tests** | Stage transition rediscovery tests; effective set intersection tests; determinism under fixed inputs. |
| **Regression gates** | AW work orchestration tests unaffected unless explicitly integrated. |
| **Completion criteria** | Proof that capability need at step N can differ from step N+1 with evidence; effective set respects policy ∩ profile ∩ scope. |
| **Depends on** | Stage 5. |
| **Maturity** | **Implemented** — `WorkStageCapabilityNeed`, `EffectiveCapabilitySet`, `WorkStageCapabilityDiscoveryEvidence` at `intergrax/contracts/capability_catalog/work_stage.py` and `intergrax/capability_catalog/work_stage_discovery.py`; reuses Stage 3–5 pipeline; stage transition and policy ∩ profile ∩ scope proofs in `tests/unit/capability_catalog/test_work_stage_discovery.py`. |

---

## Stage 9 — Autonomous Work bridge

| Field | Value |
|-------|-------|
| **User goal** | AW-7A recovery path uses federated discovery and governance without direct registry mutation. |
| **Scope** | AW-7A decision → governance → existing domain authority; A0–A4 classification behavior; ordered Tool → Skill search integration with catalog plane. |
| **Reuse** | AW-7A policy (in progress); governed discovery from Stage 5; AC-4 remains separate for agent acquisition. |
| **Non-goals** | Merging AW-7A with AC-4; A4 self-authorization; silent ToolRegistry persistence (per AW-7D acceptance). |
| **Hard contracts** | AW disposition → catalog query mapping; durable mutation only via governance + domain APIs; evidence chain AW-8 compatible. |
| **Required tests** | A0–A4 classification tests; no direct registry mutation tests; CodeCraft not default path tests. |
| **Regression gates** | AW-7A acceptance criteria from [`AUTONOMOUS_WORK.md`](AUTONOMOUS_WORK.md) plan. |
| **Completion criteria** | AW-7A uses catalog/discovery plane for Tool/Skill; durable changes routed through governance; AC-4 path untouched. |
| **Depends on** | Stages 5, 8; AW-7A domain work. |
| **Maturity** | **In progress** (AW-7A) / federation **planned**. |

---

## Stage 10 — Bootstrap evidence

| Field | Value |
|-------|-------|
| **User goal** | Operators audit which Tools and Skills were discovered, admitted, and wired at application bootstrap — alongside existing Security/Policy/Context/Memory evidence. |
| **Scope** | Typed Tools load evidence; typed Skills load evidence; aggregation into `ApplicationPlatformPluginEvidence` (or successor aggregate); evidence-only semantics. |
| **Reuse** | `DomainPluginLoadReport` pattern; Tier-3 cross-flow evidence chain from Platform Plugins. |
| **Non-goals** | Evidence as registry authority; replacing domain admission. |
| **Hard contracts** | `ToolsPluginLoadReport`, `SkillsPluginLoadReport` (or domain-equivalent typed reports); aggregate fields on application evidence; `critical_bootstrap_acceptable` alignment with STRICT. |
| **Required tests** | STRICT fail-closed bootstrap tests; evidence aggregation tests; rejected plugin remains non-active. |
| **Regression gates** | `test_plugin_engine_cross_flow.py`; application composition architecture gate. |
| **Completion criteria** | Application evidence includes Tool and Skill typed reports; documented operator audit path. |
| **Depends on** | Stage 7 recommended; Platform Plugins evidence patterns. |
| **Maturity** | **Planned** (Tools/Skills reports noted as future in Platform Plugins architecture). |

---

## Stage 11 — Marketplace product surface

| Field | Value |
|-------|-------|
| **User goal** | Public and private marketplaces present discoverable capabilities with publisher and commercial metadata without becoming runtime. |
| **Scope** | Catalog product APIs/UI; public/private catalog registration; publisher metadata; discoverability; availability signals; commercial metadata fields (display only). |
| **Reuse** | Federated catalog from Stage 2; Agent Marketplace concept as `CatalogSourceProvider`; AC-4 optional marketplace source. |
| **Non-goals** | Runtime install; registry mutation; checkout/billing implementation (Stage 13). |
| **Hard contracts** | Marketplace → catalog source adapter only; no `AgentRegistry.register` or registry write APIs; pricing metadata non-authoritative for execution. |
| **Required tests** | Marketplace adapter read-only tests; forbidden flow regression tests (Forbidden 1, 3). |
| **Regression gates** | AC-4 marketplace-not-required tests; air-gapped deployment without marketplace. |
| **Completion criteria** | Marketplace lists capabilities via federated sources; lifecycle operations route to domain authorities only. |
| **Depends on** | Stages 2, 5. |
| **Maturity** | **Future product** — Agent Marketplace architecture already marks product as future. |

---

## Stage 12 — Isolation and external execution maturity

| Field | Value |
|-------|-------|
| **User goal** | Platform assesses trusted in-process limitations and defines a governed path toward stronger third-party isolation when required. |
| **Scope** | Threat model update; isolation provider interfaces as **future** work; remote execution as separately governed capability — assessment and ADR, not premature default implementation. |
| **Reuse** | Platform Plugins trusted in-process documentation; AW-7B sandbox prerequisites. |
| **Non-goals** | New execution engine in Capability Catalog; mandatory sandbox for all V1 deployments. |
| **Hard contracts** | Isolation decision record; discovery may tag `execution_posture` metadata without enforcing isolation. |
| **Required tests** | Documentation/traceability tests only until implementation approved; spike proofs gated separately. |
| **Regression gates** | No change to default in-process bootstrap without explicit host profile opt-in. |
| **Completion criteria** | Published assessment + go/no-go criteria for isolation providers; roadmap item with explicit dependencies. |
| **Depends on** | Stage 11 optional (public third-party growth). |
| **Maturity** | **Future** — assessment stage. |

---

## Stage 13 — Usage metering

| Field | Value |
|-------|-------|
| **User goal** | Usage can be metered for billing without embedding prices in registries. |
| **Scope** | Typed usage events; metering consumer; pricing/billing separation; publisher attribution from source-qualified identity. |
| **Reuse** | AC-4 selection evidence fields; runtime event spine; HOS/Observability patterns. |
| **Non-goals** | Price fields on `ToolRegistry`/`SkillRegistry`/`AgentRegistry`; Skill as direct metered execution unit. |
| **Hard contracts** | `CapabilityUsageEvent` (typed); separation from discovery APIs; attribution requires provenance from Stage 2. |
| **Required tests** | Event schema tests; no-registry-price tests; attribution round-trip from discovery evidence. |
| **Regression gates** | Runtime execution tests without billing side effects. |
| **Completion criteria** | Metering consumer can attribute usage to source-qualified capability identity; billing subsystem remains separate. |
| **Depends on** | Stages 2, 5; optional Stage 11 for publisher metadata richness. |
| **Maturity** | **Future** — AC-4 notes billing not implemented. |

---

## Stage 14 — Full autonomous worker integration

| Field | Value |
|-------|-------|
| **User goal** | End-to-end autonomous worker loop uses discovery repeatedly under governance through existing domain authorities. |
| **Scope** | Closed loop: Goal → Plan → Need → Discover → Rank → Govern → Select → Domain authority → Execute → Observe → Discover again. |
| **Reuse** | Stages 5, 8, 9, 10; AW orchestration; AC-4 for agent delegation needs only. |
| **Non-goals** | Single universal orchestrator replacing domain runtimes; discovery-driven auto-install without governance. |
| **Hard contracts** | Loop evidence chain; each iteration preserves invariant 1–20 from architecture; rediscovery triggers typed.need only. |
| **Required tests** | Integration proof across work stages; governance deny mid-loop; observe → rediscover determinism. |
| **Regression gates** | Full AW acceptance suite; AC-4 E2E; Tier-3 composition gate. |
| **Completion criteria** | Reference proof demonstrating full loop on harness with federated catalog; no forbidden flows in trace. |
| **Depends on** | Stages 5, 8, 9, 10 (minimum). |
| **Maturity** | **Planned** — target end-state for V1 discovery program. |

---

## Domain plan coordination

Implementation rows for domain-specific work remain in owning plans:

| Domain | Plan | Expected rows |
|--------|------|-----------------|
| Agent Distribution | [`AGENT_DISTRIBUTION.md`](AGENT_DISTRIBUTION.md) | AC-4 extensions via new `CatalogSourceProvider` only |
| Tools | [`TOOLS.md`](../../architecture/TOOLS.md) | Private catalog adapters, selection integration |
| Skills | [`SKILLS.md`](../../architecture/SKILLS.md) | Version pinning (Stage 6), private catalog |
| Autonomous Work | [`AUTONOMOUS_WORK.md`](AUTONOMOUS_WORK.md) | AW-7A–7D bridge (Stage 9) |
| Platform Plugins | [`PLATFORM_PLUGINS.md`](PLATFORM_PLUGINS.md) | Bootstrap evidence (Stage 10) |

This plan **coordinates** cross-domain delivery; it does not override domain architecture.

---

## Verification intent

- Federated read model never mutates registries (property tests + code review gate)
- AC-4 and AW-7A remain separately testable subsystems
- STRICT hosts fail closed on governance conflicts
- Air-gapped deployment operates without public Marketplace
- Skill version pinning decision recorded before enterprise discovery claims
- Tools/Skills bootstrap evidence visible in application aggregate
- Forbidden flows 1–6 covered by regression tests or architecture gate

---

## Quality bar (all stages)

Every implementation slice must be: enterprise-grade, plugin-extensible, modular, reusable, contract-driven, auditable, fail-closed, free of private bypasses, free of `getattr`/`setattr` contracts, free of `dict[str, Any]` as domain substitutes, and prefer reuse of existing components before new abstractions. Legacy surfaces without production consumers should be marked for removal — not compatibility-shimmed into the catalog model.

---

## Program status summary

| Stage | Name | Maturity |
|-------|------|----------|
| 1 | Contracts & frozen boundaries | Planned |
| 2 | Federated catalog read model | Planned |
| 3 | Query / filtering / candidate model | Planned |
| 4 | Ranking | Implemented |
| 5 | Governance integration | Implemented |
| 6 | Skill enterprise correctness | Planned |
| 7 | Tool/Skill catalog maturity | Planned |
| 8 | Adaptive Unit-of-Work discovery | Implemented |
| 9 | Autonomous Work bridge | AW-7A in progress |
| 10 | Bootstrap evidence | Planned |
| 11 | Marketplace product surface | Future |
| 12 | Isolation / external execution | Future assessment |
| 13 | Usage metering | Future |
| 14 | Full autonomous worker integration | Planned |

**Architecture delivery (CAPABILITY-CATALOG-1):** canonical architecture + plan pair — **done** (this commit).
