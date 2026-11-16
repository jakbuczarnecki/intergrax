# ADR-SFX-005: Scenario catalog as a Tier-2 sidecar (`ApplicationManifest` unchanged)

| Field | Value |
|-------|-------|
| **Status** | **Proposed** (T-IGX-037-C, SFX-P2-01; **not Accepted** - acceptance is OD-SFX-06, a separate operator review) |
| **Date** | 2026-11-12 |
| **Deciders** | Operator (OD-SFX-06). The exception path additionally needs a separate operator OD |
| **Code reference SHA** | `eadb1c2116fe` |
| **Related** | [`T-IGX-037-C`](../../../../../../governance/tickets/T-IGX-037-C.md) · SFX master plan R1 §6 (SFX-GAP-02), §7.3, §7.5, §7.6 (`governance/tickets/plans/SFX-MASTER-PLAN.md`, published by T-IGX-037-A) · [`ADR-SFX-001`](ADR-SFX-001.md) · [`AGENT_DISTRIBUTION.md`](../../../../architecture/AGENT_DISTRIBUTION.md) §6.3 · [`COLLABORATIVE_WORK.md`](../../../../architecture/COLLABORATIVE_WORK.md) · [`COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md`](../../../../capabilities/architecture/COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md) · [`EVOL_X_CHANGE_SAFETY.md`](../../../../capabilities/architecture/EVOL_X_CHANGE_SAFETY.md) · [`INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md`](../../../../maintainers/qualification/INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md) |

## Context

SFX subsystem 12 (Registry/Catalog/lifecycle/reuse) must let users find, compare and reuse scenarios: business domain, reference status, required integrations, qualification state and version history. No scenario catalog exists today (SFX-GAP-02, VERIFIED).

The obvious carriers are frozen or semantically wrong:

- `ApplicationManifest` (`intergrax/applications/contracts/manifest.py:319`) and `ApplicationPackage` (`intergrax/applications/contracts/application_package.py:56`) are `extra="forbid"` contracts (`manifest.py:326`, `application_package.py:59`). Adding any field changes an existing frozen public contract, which is automatically Class C (`INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md:139`, touch rule `:176`).
- `ApplicationGraphSpec` (`intergrax/applications/contracts/graph_spec.py:45`) describes runtime composition, not catalog metadata.
- `EnvironmentExtensions` (`intergrax/applications/contracts/environment_profile/bundles.py:411`) is a "typed escape hatch for product-specific policy slices", which carries `domain_policy_fragments` (`:416`). Catalog metadata placed there would sit in the policy path and could steer governance.

A working precedent exists: `AgentCatalogEntry` (`intergrax/agent_distribution/catalog.py:99`) is "Catalog metadata only - MUST NOT become execution truth" (`:100`), is frozen (`:102`), and points to packages by identity. The rule in `AGENT_DISTRIBUTION.md:564` is that catalog disappearance must not affect the reproducibility of digest-pinned installations.

Alternatives considered:

| Variant | Summary | Rejection rationale |
|---------|---------|---------------------|
| **A - Manifest field** | Add `catalog` / `scenario_meta` to `ApplicationManifest` or `ApplicationPackage` | Class C on a frozen public contract. All hosts must be requalified for a product-only need. |
| **B - `EnvironmentExtensions` slice** | Store catalog data as a domain policy fragment | Mixes discovery metadata into the policy surface. Wrong owner; the risk is that metadata steers governance. |
| **C - Platform `ScenarioRegistry`** | New Tier-0 registry for scenarios | A new platform authority and registry, forbidden by OD-SFX-01 and the "no second registry" rule. It would trigger ECAP-§16 (see ADR-SFX-001, C-ECAP-01). |
| **D - Repository files** | `applications/<slug>_scenario/catalog.json` | Not tenant-scoped, no CAS versioning, and hand-editable (the same weakness as spec frontmatter, R1 §14.2). |
| **E - Reuse `AgentCatalogEntry` type directly** | Store scenarios as agent catalog entries | Wrong semantics: `package_id_line` and `version_channel_refs` describe agent distribution packages. SFX reuses the **pattern**, not the type. |
| **F - Tier-2 sidecar in Collaborative Work (selected)** | `sfx.scenario_catalog_entry.v1`, owned by SFX, persisted as `WorkArtifactVersion` | No frozen-contract change, reuses the existing tenant-scoped versioned store, and matches the established catalog-entry pattern. |

## Decision

Adopt **Variant F**.

1. **Contract and owner.** `sfx.scenario_catalog_entry.v1` (SFX-CON-11 `ScenarioCatalogEntry`) is owned by Tier-2 `agents/scenario_factory`. It is metadata only and never execution truth. It is not a registry: nothing resolves, installs, enables or authorizes from it, and it creates no semantic authority. Its JSON Schema draft belongs to the T-IGX-037-C design set; the canonicalization rules are in ADR-SFX-002.
2. **Package key.** Each entry references exactly one package by `(package_id, version, package_checksum)`, taken from `ApplicationPackage.package_id` / `.version` (`application_package.py:61`, `:63`) and `ApplicationDistribution.checksum` (`application_package.py:52`).
   - `checksum` defaults to an empty string (`:52`). An entry whose package has an empty checksum is **rejected (fail closed)**; SFX never writes a checksum into the package.
   - Whether an SFX-computed canonical package digest may substitute for an empty checksum is an open item for ADR-SFX-002. Until then, the answer is no.
3. **Persistence (existing owner only).**
   - Each catalog entry is one `WorkArtifact` (`intergrax/contracts/collaborative_work.py:1238`). Each change is a new immutable `WorkArtifactVersion` (`:1283`), published through `WorkArtifactVersionRepository.publish_version` (`intergrax/collaborative_work/repository.py:1351`) with CAS on the current pointer.
   - Content is stored in `ArtifactContentStore` (`intergrax/collaborative_work/content_storage.py:105`). It is identified by an SFX media type in `ArtifactContentRef.media_type` and digest-bound by `integrity_digest` (`intergrax/contracts/collaborative_work.py:1220-1221`).
   - Scope is the version's `tenant_id` / `workspace_id` (`:1291-1292`).
   - `WorkArtifact` has no kind or lifecycle field. The proposed layout is therefore one workspace-scoped "SFX catalog" `WorkItem`, listed through `WorkArtifactRepository.list_for_work_item` (`repository.py:1300`), with history through `list_for_artifact` (`:1327`). Each entry also records its scenario `WorkItem` id. ADR-SFX-002 (persistence mapping, R1 §7.5) confirms or replaces this layout; it introduces no new store.
4. **Entry lifecycle.** `listed` → `deprecated` → `retired` lives **inside** the sidecar payload. It is display state only and changes no package, roster or policy state. Retiring an entry never affects digest-pinned installs (`AGENT_DISTRIBUTION.md:564`).
5. **Informational fields only.** Qualification and admission references (SFX qualification record ids, the admission dossier id, an external admission reference) are pointers. A catalog entry never states or implies production admission; see [`ADR-SFX-006`](ADR-SFX-006.md). Trust and quality labels are display hints, as in `AgentCatalogEntry.trust_labels`.
6. **Unchanged contracts.** `ApplicationManifest`, `ApplicationPackage` and `ApplicationGraphSpec` stay unchanged. `EnvironmentExtensions` must not be used for catalog data.

### Exception path (documented only; no WBS task depends on it)

A manifest/package field may be proposed only if a later requirement proves that the sidecar cannot meet it. All of the following are then required, in order:

1. A justification ADR that supersedes or amends this one.
2. The COMPAT-X additive policy: additive compatibility is allowed "only when an explicit family compatibility policy and evidence say so" (`COMPAT_X_CONTRACT_SCHEMA_EVOLUTION.md:41`). Pydantic defaults alone are not evidence.
3. EVOL-X change-impact selection, covering consumer closure and regression selection (`EVOL_X_CHANGE_SAFETY.md:10`).
4. ECAP-§2.D D1-D6 with **Class C** classification, plus an Architecture Reopen Record (`INTEGRAX_POST_FREEZE_EVOLUTION_GOVERNANCE.md:448-466`).
5. Requalification of the impacted targets.
6. A separate operator OD. Approval of this ADR does not pre-authorize it.

## Consequences

### Positive

- Zero frozen-contract change. Catalog work is Class A domain-module work (ADR-SFX-001, ME-1).
- Tenant scoping, immutable versions, CAS and content integrity come from Collaborative Work for free.
- Reproducibility is independent of the catalog, following the agent-distribution rule.

### Negative

- Catalog queries are limited to what CW listing provides (by work item, by artifact). Rich search such as facets or full text needs an SFX read model, which must stay a derived, rebuildable projection and never an authority.
- Collaborative Work qualification is REQUALIFICATION_REQUIRED (SFX-P1-03); catalog durability claims cannot exceed it.
- Packages with an empty `distribution.checksum` cannot be listed until ADR-SFX-002 decides the digest rule.

## Compliance

- Tier boundaries: the sidecar lives in `agents/`, and `intergrax/` gains no catalog code.
- No new registry, no new store, no change to policy fragments.
- Stays **Proposed** until OD-SFX-06. The exception path needs its own OD.

## Implementation notes

- **Not in scope at Proposed:** sidecar code (SFX-P11-03, post-freeze), the read model, and any manifest change.
- **Revision triggers:** any proposal to resolve, install or authorize from the catalog; any write to `ApplicationPackage` / `ApplicationManifest`; adoption of an SFX-computed package digest.
- **Verification (when implemented):** schema conformance for `sfx.scenario_catalog_entry.v1`; tests that a manifest/package byte digest is unchanged by catalog operations; a fail-closed test for an empty checksum; a test that retiring an entry leaves a pinned install reproducible.
