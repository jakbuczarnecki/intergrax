# Vendor Knowledge Facade — Implementation Plan

**Status:** `ACCEPTED / CLOSED`
**Branch:** `development`  
**Architecture:** [`../../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md)
**Reuse audit:** [`../audit/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../audit/KNOWLEDGE_SOURCE_INTEGRATIONS.md)  
**LKW intake discovery:** [`../../docs/project/technical/applications/local_workspace_application/KNOWLEDGE_INTAKE_DISCOVERY.md`](../../technical/applications/local_workspace_application/KNOWLEDGE_INTAKE_DISCOVERY.md)

---

## 1. Objective

Build one provider-neutral, plugin-based Vendor Knowledge platform above existing
category-specific vendor integrations so applications can consume external
enterprise knowledge through **three universal data-access modes**:

```text
1. Indexed / RAG
2. Durable / Storage / Materialization
3. Live / Realtime
```

Synchronization is a lifecycle mechanism of Durable / Storage / Materialization,
not a separate fourth mode. Storage technology is an implementation detail.

```text
existing provider/category integration
        |
        v
shared provider read primitives
        |
        +----------------------------------+
        |                                  |
        v                                  v
durable knowledge path                live capability path
        |                                  |
        v                                  v
Vendor Knowledge Adapter           Live Capability Adapter
        |                                  |
        v                                  v
Vendor Knowledge Facade            Validated Executor
        |                                  |
        v                                  v
Sync / Materialization Runtime      ephemeral Live Evidence
        |
        v
injected durable sink
├── DocumentStore
├── relational / NoSQL database
├── object storage
├── application repository
└── optional LKW Knowledge Intake → RAG
```

Existing integrations remain low-level and authoritative. Vendor Knowledge Facade
and Sync Coordinator cover the **Durable / Storage / Materialization** path today.
Accepted Live / Realtime capability
families and Slack's bounded configured-channel Ask path execute through the
shared validated live boundary; Indexed / RAG and Durable / Storage /
Materialization remain separate. The facade is not an integration category.

---

## 1A. Canonical three-mode platform roadmap — current session

This section is the current execution order for Vendor Knowledge platform work.
The historical execution order and detailed task history below are retained for
traceability, but any future sequencing that conflicts with this section is
**SUPERSEDED**.

**CURRENT:**
`PROVIDER-PROD-6-CROSS-PROVIDER-APPLICATION-CLOSEOUT` — `ACCEPTED / CLOSED`

**NEXT:**
`NONE — PROVIDER PRODUCTIONIZATION TRACK CLOSED`

### Provider productionization final closeout

```text
PROVIDER-PROD-6-CROSS-PROVIDER-APPLICATION-CLOSEOUT
  ACCEPTED / CLOSED

PROVIDER PRODUCTIONIZATION TRACK
  ACCEPTED / CLOSED

NEXT:
  NONE — PROVIDER PRODUCTIONIZATION TRACK CLOSED
```

The implemented Slack, Microsoft Graph, Google Workspace, Jira and
Confluence source identities coexist behind provider-neutral application
discovery, binding, Search and Ask paths. Databricks remains a connection
foundation only; it has no Vendor Knowledge source contract or plugin.

### VENDOR KNOWLEDGE EXTENSION READINESS

Extension readiness is a separate follow-up track from the closed provider
productionization track above. Its canonical roadmap/status is:

```text
VK-EXT-1  Unified Vendor Plugin Contribution Contract   ACCEPTED / CLOSED
VK-EXT-2  Plugin Discovery / Composition                READY_FOR_FINAL_CLOSEOUT
VK-EXT-3  Reference External Provider Proof             PLANNED
VK-EXT-4  Vendor Knowledge Plugin Author Guide          PLANNED
VK-EXT-5  Extension Readiness Closeout                  PLANNED
```

The `VendorKnowledgeProviderContribution` is the canonical extension ABI:
built-ins and optional entry-point providers feed the
`VendorKnowledgeContributionCatalog`, which feeds generic runtime and
application registries. Generic composition contains no provider-specific
business branches, and application-owned discovery/materializer hooks are
contribution-driven. Full external provider qualification remains VK-EXT-3;
the complete authoring guide remains VK-EXT-4. See
[`Vendor Knowledge contribution composition`](../../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md#19-vendor-knowledge-contribution-composition)
for the detailed architecture boundary.

Partial capabilities remain partial: Graph Drive remains
`FOUNDATION_ONLY` with the exact
`REQUIRES_GENERIC_BINARY_CONTENT_EXTRACTION_CAPABILITY` boundary, and Google
Drive preserves the same binary-content extraction requirement. Atlan and
Power BI remain `NOT IMPLEMENTED`, `DEFERRED` and `UNSUPPORTED`.

This closeout makes no universal mode-support claim, no complete ACL-coverage
claim (`ACL completeness = UNPROVEN`), and no commercial GA/SLA claim. The
security closeout is accepted for tenant, workspace, connection, binding,
credential, opaque-reference, mismatch, scoped-cleanup, Search-evidence and
Ask-evidence isolation boundaries.

VK-8 is closed by the focused product-level proof for Slack
`slack_conversation` and Microsoft Graph `teams_chat` coexisting in the same
tenant and workspace through generic discovery/binding, Durable sync, Indexed
materialization and canonical `KnowledgeDocument` indexing, Search, Ask, Live,
normalized inspection/lifecycle operations, tenant isolation and failure
isolation. Slack deletion propagation and complete per-user ACL semantics remain
unproven; other providers retain their VK-6 classifications, and this closeout
does not imply commercial or GA support.

The Microsoft Graph provider family closeout is also **ACCEPTED / CLOSED** for
its current bounded source scopes. `ms365_graph` is the sole canonical
provider identity in `COLLABORATION_SUITE`; Drive, Mail, Teams Channel, Teams
Chat and Calendar reuse one durable TenantConnection, `credential_ref`,
`Ms365GraphTenantConnectionIntegrationFactory`, restart rehydration and the
KnowledgeConnectionRegistry. Mailbox, team and calendar scope configuration is
source/discovery configuration, not a separate credential lifecycle.

```text
teams_chat       DURABLE ACCEPTED  INDEXED ACCEPTED  LIVE ACCEPTED  LKW_READY
mail             DURABLE ACCEPTED  INDEXED ACCEPTED  LIVE ACCEPTED  LKW_READY
teams_channel    DURABLE ACCEPTED  INDEXED ACCEPTED  LIVE ACCEPTED  LKW_READY
calendar         DURABLE ACCEPTED  INDEXED ACCEPTED  LIVE ACCEPTED  LKW_READY
drive            DURABLE FOUNDATION_ONLY  INDEXED FOUNDATION_ONLY
                 LIVE ACCEPTED  FOUNDATION_ONLY
```

The Drive blocker remains
`REQUIRES_GENERIC_BINARY_CONTENT_EXTRACTION_CAPABILITY`: files are `BINARY`,
while folders/packages have no canonical textual content representation. No
fake metadata-only Indexed claim is made. `LKW_READY` is a bounded technical
acceptance label, not commercial GA, complete ACL coverage or exhaustive
Microsoft Graph feature coverage. Generic LKW remains provider-neutral; Graph
credentials, direct Graph execution and Graph-specific Search/Ask stay at the
approved provider composition boundary.

### VK-9 final platform closeout

`VENDOR-KNOWLEDGE-PLATFORM-CLOSEOUT-1` is **ACCEPTED / CLOSED**.

The final audit was performed against current runtime code, composition roots,
focused VK-2 through VK-8 tests and the accepted Slack plus Microsoft Graph
family proofs. No production-code correction was required.

#### Final platform verdict

```text
plugin/capability contract       CLOSED
Durable                          CLOSED
Indexed                          CLOSED
Live                             CLOSED
provider coverage                CLOSED — truthful, source-kind specific
frontend neutrality              CLOSED
cross-provider E2E               CLOSED
identity/ownership               ACCEPTED / CLOSED
removal/revocation               PLATFORM CLOSED; provider limitations remain
public/internal platform surface ACCEPTED / INTERNAL PLATFORM SURFACE
PLATFORM_BLOCKER                 0
```

The three-mode architecture is complete and internally consistent. Durable,
Indexed and Live are independently composable; a plugin declares only the modes
that its source kind actually supports. The accepted architecture is:

```text
Frontend / Bot / Application
        |
        v
provider-neutral application contracts
        |
        v
VendorKnowledgeSourcePlugin + VendorKnowledgeSourceIdentity
        |
        +-----------------------------+
        |             |               |
        v             v               v
     DURABLE        INDEXED          LIVE
        |             |               |
        v             v               v
adapter/sync      materializer     capability
coordinator       registry         executor
sink              KnowledgeDoc     registration
        |             |               |
        +-------------+---------------+
                      |
                      v
                provider adapter
```

#### New-provider final path

A new provider/source kind adds provider-owned adapter/discovery composition,
one plugin declaration and only the mode-specific extension points it supports:

```text
DURABLE  adapter + plugin DURABLE declaration; reuse coordinator and sink
INDEXED  plugin INDEXED declaration + KnowledgeDocument materializer registration
LIVE     plugin LIVE declaration + provider capability handler/registration bundle
```

The generic route, inspection, lifecycle operations, Ask strategy port and
discovery service do not require provider-specific edits. Provider
authentication and setup remain provider-owned. `RemoteResourceDiscoveryStrategyRegistry`,
Slack strategy and Graph Teams Chat strategy are the active discovery route;
`WorkspaceRemoteResourceDiscoveryService` resolves a strategy and contains no
concrete Slack or Graph behavior.

#### Identity, ownership and lifecycle reconciliation

Tenant, workspace, connection, source identity, binding, remote item,
canonical document, materialization ownership, revision and Live capability
are fenced by explicit identifiers and validation. Document provenance includes
stable tenant/workspace/provider/source-kind/binding/remote-item identity;
revision and CAS/publication fences prevent stale ownership from being applied.
Cross-provider keys are distinct, and tenant/source/binding mismatches fail
closed. Live identity is resolved from the same source-kind and binding
contract as the plugin declaration.

`VK1-GAP-08` is therefore **ACCEPTED / CLOSED**. Provider item ACL coverage and
complete per-user visibility projection remain provider limitations, not an
identity architecture defect.

Removal/revocation is **PLATFORM_CLOSED_PROVIDER_LIMITATIONS_REMAIN**. The
lifecycle/index platform consumes authoritative deletion evidence when a
provider supplies it and does not infer deletion from snapshot absence.
Microsoft Graph Teams Chat proves authoritative `DELETED` materialization;
Slack deletion propagation remains `DELETION_UNPROVEN` because its accepted
adapter emits no authoritative tombstones. This is truthful provider behavior,
not a platform blocker, so `VK1-GAP-09` is closed at the platform level.

#### Public/internal platform surface

The reusable internal platform surface is complete: plugin contracts and
registry, adapter composition, Durable coordinator/sink, Indexed materializer
registry and canonical `KnowledgeDocument` bridge, Live registration/catalog/
executor, and provider-neutral application ports are available for reuse by
another Intergrax application. This is **ACCEPTED / INTERNAL PLATFORM SURFACE**.
There is no standalone pip-facing SDK, formal public REST facade, commercial
GA package or public SLA; those are productization/commercial decisions, not
architecture blockers.

#### Security closeout

Accepted boundaries preserve tenant isolation, workspace/connection/source/
binding ownership, revision/CAS checks, opaque candidate integrity, exact
provider/source/capability matching, credential secrecy, normalized errors,
bounded Live execution and provider-neutral exception handling. Known
limitations are provider item ACL completeness, per-user query ACL
propagation, provider deletion semantics, and bounded content/history scope.
They are not tenant-isolation failures.

#### Final limitations register

```text
PLATFORM_BLOCKER
  none

PROVIDER_LIMITATION
  selective provider/mode coverage; provider item and per-user ACL scope;
  Slack deletion tombstones; bounded content/history and source-specific reads

PRODUCTIZATION_LIMITATION
  no standalone public SDK or formal public REST facade; UX is not identical
  across providers

COMMERCIAL_LIMITATION
  no broad commercial/GA/SLA claim for Vendor Knowledge modes
```

“Platform complete” means that the provider-neutral architecture, lifecycle
contracts, extension model and representative cross-provider execution proof
are complete and production-grade for their documented scope. It does not mean
that every provider is implemented, every provider supports every mode, every
provider supplies authoritative deletion, complete per-user ACL propagation,
commercial GA status, a public SLA, standalone SDK packaging or identical UX.

#### Final VK-1 gap register

```text
VK1-GAP-01 plugin model                         CLOSED
VK1-GAP-02 capability model                     CLOSED
VK1-GAP-03 Durable lifecycle                    CLOSED
VK1-GAP-04 generic Indexed bridge               CLOSED
VK1-GAP-05 Live capability platform             CLOSED
VK1-GAP-06 frontend neutrality                  CLOSED
VK1-GAP-07 provider coverage                    CLOSED
VK1-GAP-08 identity/ownership                   CLOSED
VK1-GAP-09 lifecycle removal/revocation         CLOSED at platform level
VK1-GAP-10 documentation                        CLOSED
```

#### Final roadmap state

```text
VENDOR KNOWLEDGE ROADMAP
VK-1 through VK-9 — COMPLETE
```

Future Google Indexed expansion, Jira/Confluence Live work, Atlan/Power BI
implementation, Databricks source-kind selection, full ACL propagation and
commercial API packaging are separate provider/product tasks. They are not
`VK-10` and do not reopen this architecture roadmap.

#### PROVIDER-PROD-5C — Atlan / Power BI selection gate

The repository-grounded selection gate result is:

```text
Atlan     → DEFERRED
Power BI  → DEFERRED
```

Both providers remain **`UNSUPPORTED / NOT IMPLEMENTED`**. Neither has an
existing provider runtime or a repository-proven bounded Vendor Knowledge
source contract. External provider-contract research is therefore deferred
until one provider is deliberately reopened; external API semantics are
required before implementation selection. This is an intentional roadmap
deferral, not a claim that either provider is impossible and not a blocked
implementation task.

#### Authoritative additional-provider current state

```text
Databricks
  provider_id:        databricks
  category:           RELATIONAL_STORE
  relational runtime: EXISTS
  TenantConnection:   restart-safe foundation ACCEPTED
  VK adapter:         NOT IMPLEMENTED
  source kind:        UNRESOLVED / DEFERRED
  source contract:    DEFERRED
  VK plugin:          none
  Durable:            FOUNDATION_ONLY / NOT IMPLEMENTED
  Indexed:            UNSUPPORTED
  Live:               UNSUPPORTED
  readiness:          FOUNDATION_ONLY

Atlan
  status:             UNSUPPORTED / NOT IMPLEMENTED
  provider selection: DEFERRED
  external research:  DEFERRED; required before reopening

Power BI
  status:             UNSUPPORTED / NOT IMPLEMENTED
  provider selection: DEFERRED
  external research:  DEFERRED; required before reopening
```

#### PROVIDER-PROD-5D — Additional Providers closeout

`PROVIDER-PROD-5D-ADDITIONAL-PROVIDERS-CLOSEOUT` is **`ACCEPTED / CLOSED`**.
This closes the Additional Providers track without claiming that all three
providers are productionized:

```text
Databricks
  connection foundation: ACCEPTED
  VK source contract:    DEFERRED
  readiness:              FOUNDATION_ONLY

Atlan
  implementation:         NOT IMPLEMENTED
  selection:              DEFERRED
  external research:      DEFERRED

Power BI
  implementation:         NOT IMPLEMENTED
  selection:              DEFERRED
  external research:      DEFERRED

PROVIDER-PROD-5: ACCEPTED / CLOSED
NEXT (superseded by final closeout): PROVIDER-PROD-6-CROSS-PROVIDER-APPLICATION-CLOSEOUT
```

The Atlan and Power BI research deferral means that no repository-proven
provider source contract exists yet; it does not claim either provider is
impossible or permanently unsupported. The Databricks connection factory
registry is a connection foundation, not a Vendor Knowledge plugin registry.

### VK-4 indexed bridge acceptance proof

The accepted VK-4 proof covers:

- provider-neutral Indexed materializer/runtime resolution through VK-2 source identity;
- Slack `slack_conversation` and Microsoft Graph `teams_chat` on the same generic
  connected-source execution boundary;
- canonical `KnowledgeDocument` intake and deterministic tenant/workspace/provider/
  source-kind/binding/remote-item document identity;
- idempotent replay, revision/stale protection and authoritative removal where
  the provider emits tombstones;
- fail-closed registration, identity, document validation and index-write errors,
  with durable receipt/manifest state preserved for retry convergence;
- existing generic index/Search/Ask proof for the representative Slack path and
  canonical Graph Teams Chat materialization/index proof.

Full provider coverage remains VK-6, frontend neutrality remains VK-7, and
complete cross-provider product E2E remains VK-8.
Slack deletion remains unsupported/unproven while its adapter emits no
authoritative tombstones.

### VK-5 Live / Realtime capability closeout

`VENDOR-KNOWLEDGE-LIVE-CAPABILITY-CLOSEOUT-1` is **ACCEPTED / CLOSED**.

Accepted proof:

- provider-neutral Live registration/bootstrap through
  `VendorKnowledgeLiveRegistrationRegistry`;
- VK-2 provider/category/source-kind identity resolution and declared LIVE
  capability matching;
- tenant descriptor publication with existing active-connection and binding
  authorization boundaries;
- the existing `LiveCapabilityExecutorV1` remains the sole execution runtime;
- representative Slack `slack_conversation` and Microsoft Graph `teams_chat`
  discovery/execution, plus smoke coverage for Graph Drive, Mail, Teams Channel
  and Calendar;
- registration-driven test-only provider addition without LKW provider
  switches;
- duplicate/conflicting registration, source mismatch, schema, budget,
  authorization and safe-error behavior remains fail-closed;
- normalized evidence remains ephemeral unless an explicit receipt retention
  contract is selected; Live does not write Durable or Indexed state.

`VK1-GAP-05` (provider-specific Live registration/bootstrap leakage) is
**CLOSED**. Provider/source-kind capability subsets remain valid and
provider-owned limits remain outside generic bootstrap. Full provider/source
coverage remains VK-6, full frontend neutrality remains VK-7, and complete
cross-provider product E2E remains VK-8. Slack-specific Ask evidence
orchestration remains application-owned until VK-7.

### VK-2 plugin/capability contract status

The canonical platform discovery boundary is
`intergrax.runtime.vendor_knowledge.plugin`:

- `VendorKnowledgeSourceIdentity` uses the existing provider ID,
  `IntegrationCategory` and explicit source kind.
- `VendorKnowledgeMode` declares `INDEXED`, `DURABLE` and `LIVE`; each
  `VendorKnowledgeModeCapability` carries mode-scoped operations, constraints,
  version and an opaque runtime registration reference.
- `VendorKnowledgeSourcePluginRegistry` is the authoritative catalog for
  deterministic registration, lookup and discovery. Existing adapter,
  materialization/sync and live registries remain execution registries.
- Slack `slack_conversation` proves all three modes. Microsoft Graph
  `teams_chat` now proves Durable + Indexed + Live.

Descriptors are immutable, reject tenant/credential/connection state and do not
execute any mode lifecycle. Generic Indexed bridging is accepted and closed
through VK-4; application-specific migration, broader provider coverage and
LKW/frontend decoupling remain deferred to their roadmap tasks.

Validation closeout confirmed plugin contract, registry compatibility,
representative Slack/Graph proofs, and Slack Connected Source contract/E2E.
The previous pytest basetemp WinError 5 was resolved/avoided by a validated
clean task-specific test temp root. Six pre-existing Slack sync unit failures
remain baseline/stale and do not block VK-2.

### VK-3 durable lifecycle closeout status

`VENDOR-KNOWLEDGE-DURABLE-LIFECYCLE-CLOSEOUT-1` is **ACCEPTED / CLOSED**.

Accepted:

- provider-neutral durable coordinator (existing
  `VendorKnowledgeSyncCoordinator` + reconciliation/leases/publication);
- provider-neutral application materialization port (`KnowledgeSyncSink`) with
  production DocumentStore implementation
  (`DocumentStoreDurableKnowledgeSyncSink`);
- crash-safe recovery, replay/idempotency, checkpoint/receipt ordering;
- revision/update and tenant/source ownership isolation;
- representative cross-provider proof: Slack structured-record durable
  materialization + Microsoft Graph `teams_chat` full adapter→coordinator→sink
  path;
- durable operation independent from indexing (`DURABLE=YES` / `INDEXED=NO`
  Teams Chat works without any indexing service).

Conservative / deferred:

- provider coverage not audited → VK-6;
- generic Indexed bridge accepted and closed → VK-4;
- Live bootstrap accepted and closed → VK-5;
- frontend neutrality not fully proven → VK-7;
- Slack adapter sync unit suite remains 6 baseline/stale failures (unchanged);
- Slack deletion remains `DELETION_UNSUPPORTED` / `UNPROVEN` (`tombstones=False`);
  Teams Chat proves authoritative `DELETED` materialization.

### Architecture frozen for this roadmap

```text
Frontend / Application
        |
        v
provider-neutral Vendor Knowledge contracts
        |
        +--> Indexed / RAG
        +--> Durable / Storage / Materialization
        +--> Live / Realtime
        |
        v
vendor plugin / adapter implementation
        |
        +--> Microsoft 365
        +--> Google Workspace
        +--> Jira
        +--> Confluence
        +--> Slack
        +--> Atlan
        +--> Power BI
        +--> Databricks
        +--> future vendors
```

Frontends consume platform contracts and capabilities. They do not contain
vendor-specific knowledge branches. Adding a vendor should primarily require
shared-contract implementation, capability registration and provider adapter
logic, not a redesign of LKW or another frontend.

The three modes mean:

- **Indexed / RAG:** vendor knowledge is synchronized and durably materialized,
  passed through Knowledge Intake and indexed for Search / Ask. Current
  provider-to-index E2E proof is not implied for every vendor; the accepted
  Slack connected-source slice is the strongest accepted proof of the current
  Indexed / RAG application path.
- **Durable / Storage / Materialization:** persistent synchronized vendor state
  governed by reconciliation, typed continuation, cursor ownership, receipts,
  checkpointing, replay, recovery, publication, revision, deletion/revocation,
  ownership isolation and tenant/concurrency controls. This is not merely
  database access.
- **Live / Realtime:** a provider-neutral, bounded query of current provider
  data without relying on durable indexed materialization. It may be a bounded
  request rather than streaming or websocket communication; results are
  normalized, provenance-bearing and not accidentally persisted.

### Session boundary

**This session — Vendor Knowledge platform**

Owns plugin contracts, provider capability modeling, Indexed / RAG platform
bridging, Durable / Storage / Materialization lifecycle, Live / Realtime
contracts, provider adapters and registration, source-kind capability proofs,
and frontend-neutral platform interfaces.

**Separate LKW session**

Owns Conversation Context, Hybrid Ask product orchestration, workspace UX,
bot/channel activation, Slack/Teams conversation frontend behavior, LKW
policies, and mobile/web frontend behavior. `LKW-CONVERSATION-CONTEXT-1` is not
the next task in this Vendor Knowledge session.

LKW is a consumer of Vendor Knowledge capabilities, not the owner of vendor
integrations. The same rule applies to web, mobile, Slack, Teams and other
agents/applications. Vendor-specific implementation remains behind platform
contracts.

### Required future execution order

#### VK-1 — Unified three-mode contract audit

`VENDOR-KNOWLEDGE-UNIFIED-THREE-MODE-CONTRACT-AUDIT-1`

Audit the repository and classify the existing provider-neutral architecture
for Indexed / RAG, Durable / Storage / Materialization, Live / Realtime,
plugin registration, capability discovery and frontend neutrality as
`ACCEPTED`, `PARTIAL`, `MISSING` or `CONTRADICTED`. This audit determines
implementation work; no speculative rewriting precedes it.

#### VK-2 — Plugin and capability contract finalization

`VENDOR-KNOWLEDGE-PLUGIN-CAPABILITY-CONTRACT-1`

Finalize one canonical representation of provider, source kind, supported
access modes, supported operations, constraints, capability registration and
capability discovery. A source kind may independently support Indexed / RAG,
Durable / Storage / Materialization and/or Live / Realtime; support is explicit,
not inferred.

#### VK-3 — Durable lifecycle platform closeout

`VENDOR-KNOWLEDGE-DURABLE-LIFECYCLE-CLOSEOUT-1` — **ACCEPTED / CLOSED**

Closed shared durable lifecycle primitives on the existing coordinator
foundation and introduced `DocumentStoreDurableKnowledgeSyncSink` as the
provider-neutral durable materialization implementation of `KnowledgeSyncSink`.
Indexing remains optional and out of scope for Durable mode (→ VK-4).

#### VK-4 — Generic Indexed / RAG bridge

`VENDOR-KNOWLEDGE-INDEXED-BRIDGE-1`

**ACCEPTED / CLOSED.**

Make the provider-to-index path reusable:

```text
Vendor Knowledge source
→ shared durable/materialization contract
→ shared Knowledge Intake boundary
→ platform index
→ generic Search/Ask consumer
```

The accepted `LKW-SLACK-CONNECTED-SOURCE-1` result remains architecture
evidence/reference implementation. VK-4 adds the provider-neutral source
identity/materializer registry, canonical `KnowledgeDocument` intake boundary,
stable document identity, replay/revision/removal semantics and a second
Microsoft Graph Teams Chat indexed proof through the same bridge and generic
index service.

Conservative / deferred:

- full provider capability coverage → VK-6;
- Live bootstrap and invocation closeout accepted/closed → VK-5;
- full LKW/frontend neutrality → VK-7;
- broader cross-provider product E2E → VK-8;
- Slack deletion remains unproven where the adapter does not emit authoritative
  tombstones;
- provider-specific ACL enforcement remains subject to each adapter's visibility
  contract.

#### VK-5 — Live / Realtime platform closeout

`VENDOR-KNOWLEDGE-LIVE-CAPABILITY-CLOSEOUT-1`

**ACCEPTED / CLOSED.**

Close one provider-neutral invocation and registration model preserving
authorization, bounded calls, normalized results, safe errors, provenance,
no accidental persistence and provider-specific limits behind the adapter
boundary. The accepted implementation is registration-driven; existing Slack
and Microsoft Graph live implementations remain evidence, not work to duplicate.

#### VK-6 — Provider capability coverage

`VENDOR-KNOWLEDGE-PROVIDER-COVERAGE-1`

**Status:** `ACCEPTED / CLOSED`

The canonical matrix audits every implemented source kind separately:
Microsoft Graph `drive`, `mail`, `teams_channel`, `teams_chat` and `calendar`;
Slack `slack_conversation`; Google Workspace `drive`, `docs`, `sheets` and
`calendar`; Jira `issues`; and Confluence `pages`. It records adapter status,
plugin declaration, runtime registration, mode status, deletion semantics, ACL
scope, proof level, limitations and evidence.

The default plugin composition covers all twelve implemented source kinds.
Microsoft Graph has five accepted bounded Live registrations. Drive declares
only `DURABLE` and `LIVE`, with both Indexed and readiness at
`FOUNDATION_ONLY`; Mail, Teams Channel, Teams Chat and Calendar each declare
`DURABLE`, `INDEXED` and `LIVE`, with readiness `LKW_READY`.
Google Workspace `docs`, `sheets` and `calendar` each declare `DURABLE` and
`INDEXED`, with both modes `ACCEPTED` and readiness `LKW_READY`; Live is
`UNSUPPORTED`. Google Workspace `drive` declares `DURABLE` only, with Durable
`ACCEPTED`, Indexed and readiness `FOUNDATION_ONLY`, and Live `UNSUPPORTED`.
Its blocker is `REQUIRES_GENERIC_BINARY_CONTENT_EXTRACTION_CAPABILITY`.
Jira `issues` and Confluence `pages` are **`LKW_READY`** with Durable and
Indexed both `ACCEPTED`; Live remains `UNSUPPORTED`. They retain separate
canonical provider connections (`jira` / `ISSUE_TRACKER` and `confluence` /
`WIKI_KNOWLEDGE`) and use bounded project/space discovery, provider-owned
materialization and the generic application Search/Ask path. Atlan and Power BI
have no Vendor Knowledge implementation. Databricks has a relational runtime
and an accepted restart-safe TenantConnection foundation, but no selected
Vendor Knowledge source kind, source contract or Vendor Knowledge plugin.

The exact new-provider requirements are intentionally small:

```text
DURABLE: adapter + plugin DURABLE + existing coordinator/sink
INDEXED: plugin INDEXED + registered materializer -> KnowledgeDocument
LIVE: plugin LIVE + registration bundles/handlers -> generic Live bootstrap
```

No generic platform, LKW or frontend change is required merely to add a
provider; only provider composition/registration and the provider adapter are
owned by that provider task. Full frontend neutrality remains VK-7 and
cross-provider product E2E remains VK-8.

`VK1-GAP-07` (provider coverage without complete accepted source-kind proof) is
**CLOSED** by the authoritative matrix. This closeout means the coverage is
truthful and platform-aligned; it does not claim every provider supports every
mode.

#### Google Workspace family closeout — current authoritative status

The current productionized Google Workspace family slice is
**ACCEPTED / CLOSED** for bounded `drive`, `docs`, `sheets` and `calendar`
source scopes. Its sole canonical Vendor Knowledge identity is
`provider_id=google_workspace` with integration category
`COLLABORATION_SUITE`; external aliases, if accepted at configuration
boundaries, must be normalized before entering Vendor Knowledge.

All four source kinds reuse one `TenantConnection` and `credential_ref`
through `GoogleWorkspaceTenantConnectionIntegrationFactory`,
`TenantConnectionRehydrator` and `KnowledgeConnectionRegistry`. Source kinds
are bindings/configuration over the shared provider connection, not separate
credential lifecycles. The generic LKW/application path contains zero Google
credentials, direct Google execution, Google-specific Search/Ask branches or
business-level provider switches; provider composition owns the factory,
discovery/resource strategy, adapter and materializer registrations.

```text
source      adapter     DURABLE          INDEXED          LIVE          readiness
calendar    ACCEPTED    ACCEPTED         ACCEPTED         UNSUPPORTED    LKW_READY
docs        ACCEPTED    ACCEPTED         ACCEPTED         UNSUPPORTED    LKW_READY
sheets      ACCEPTED    ACCEPTED         ACCEPTED         UNSUPPORTED    LKW_READY
drive       ACCEPTED    ACCEPTED         FOUNDATION_ONLY UNSUPPORTED    FOUNDATION_ONLY
```

`LKW_READY` is evaluated against declared/required modes. It does not imply
Live support, complete ACL coverage or commercial GA/SLA. Google item/user ACL
completeness remains `UNPROVEN`. Docs and Sheets remain known-resource,
bounded structured projections without broad discovery or authoritative
deletion from ordinary snapshot absence. Calendar remains a bounded event
projection with source-owned incremental token, cancellation/tombstone and
reconciliation semantics. Drive remains a reconciliation foundation with
`BINARY` blob/native-export content; folders and shortcuts do not provide
canonical textual content, ordinary inventory absence is not deletion, and no
metadata-only Indexed claim is made.

#### VK-7 — Frontend neutrality proof

`VENDOR-KNOWLEDGE-FRONTEND-NEUTRALITY-PROOF-1`

Prove the invocation path:

```text
consumer selects capability/source
→ platform resolves provider plugin
→ shared contract invocation
→ provider adapter executes
```

Classify vendor-specific branching in consumer/application layers. Legitimate
provider-specific code in adapter and registration layers remains valid.

`VENDOR-KNOWLEDGE-FRONTEND-NEUTRALITY-PROOF-1` is **ACCEPTED / CLOSED**.
The VK-7 proof audited the generic LKW inspection/operations, plugin
configuration, Indexed/Live route, and Hybrid Ask boundaries. Generic Ask now
accepts an optional provider strategy through a neutral contract; it does not
import or branch on Slack, Microsoft Graph, or another concrete provider.
Provider-owned Ask expansion remains an optional strategy extension.

The normalized inventory continues to source lifecycle state, revision/CAS
requirements, available actions, and safe error codes from backend lifecycle
contracts. The generic plugin configuration boundary supplies connection,
resource, and capability metadata. A test-only provider fixture discovers a
resource and capability through those same contracts without changing generic
route, inspection, operation, or query code. Static checks guard the bounded
generic application surfaces against concrete provider imports and provider
switch literals.

The existing Slack connected-source flow remains an explicit provider-owned
strategy/composition path; this closeout does not claim visually identical UX.

#### VK-8 — Cross-provider three-mode E2E

`VENDOR-KNOWLEDGE-CROSS-PROVIDER-E2E-1`

Prove the plugin architecture across structurally different providers using
the same platform contracts, without requiring every source kind to support
every mode. Select representative providers only after the capability audit.

Status: **ACCEPTED / CLOSED**. The representative proof covers Slack
`slack_conversation` and Microsoft Graph `teams_chat` in one tenant/workspace,
with distinct bindings and document identities, generic discovery/binding,
Durable, Indexed, Search, Ask, Live, inspection/lifecycle, tenant isolation and
failure isolation. Slack deletion propagation and complete per-user ACL
semantics remain unproven; other providers retain their VK-6 classifications;
commercial/GA support is not implied.

#### VK-9 — Platform closeout

`VENDOR-KNOWLEDGE-PLATFORM-CLOSEOUT-1`

Final-audit plugin extensibility, three-mode contracts, Durable / Storage /
Materialization lifecycle, Indexed / RAG bridge, Live / Realtime access,
capability discovery, provider coverage, frontend neutrality, cross-provider
proofs and documentation truthfulness. Only after VK-9 may Vendor Knowledge be
described as a completed reusable platform capability.

The accepted status of `LKW-SLACK-CONNECTED-SOURCE-1` remains
**ACCEPTED / CLOSED** and is not reopened. It proves the selected Slack
application/indexed vertical slice; it does not turn Slack-specific
implementation into the generic platform contract.

---

## 2. Historical status ledger (superseded)

This status ledger is retained for traceability only. The current authority is
the VK-9 platform closeout and the VK-6 provider/source-kind capability matrix;
entries below must not be interpreted as current roadmap sequencing.

```text
DONE:     VENDOR-KNOWLEDGE-FACADE-ARCH-1
DONE:     VENDOR-KNOWLEDGE-FACADE-PLAN-1
DONE:     VENDOR-KNOWLEDGE-FACADE-AUDIT-1
DONE:     VENDOR-KNOWLEDGE-FACADE-CONTRACT-1
DONE:     VENDOR-KNOWLEDGE-FACADE-CORE-1
DONE:     VENDOR-KNOWLEDGE-CONNECTION-1
DONE:     VENDOR-KNOWLEDGE-SYNC-1A
DONE:     PLATFORM-DOCUMENT-STORE-CONDITIONAL-1
DONE:     VENDOR-KNOWLEDGE-SYNC-1B
DONE:     JIRA-KNOWLEDGE-ADAPTER-1
DONE:     CONFLUENCE-KNOWLEDGE-ADAPTER-1
DONE:     MSGRAPH-KNOWLEDGE-READ-SURFACE-1
DONE:     VENDOR-KNOWLEDGE-THREE-MODE-REUSE-ARCH-1
DONE:     SLACK-KNOWLEDGE-THREE-MODE-ARCH-1
DONE:     SLACK-KNOWLEDGE-FOUNDATION-1
ACCEPTED:
LKW-CONVERSATION-CONTEXT-ARCH-1
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-2
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A
  through REVIEW-FIX-3
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B
  through REVIEW-FIX-5-REVIEW-CORRECTION-1
MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE
  ACCEPTED
MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL
  ACCEPTED
MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL
  ACCEPTED
MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT
  ACCEPTED through REVIEW-FIX-1
MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR
  ACCEPTED through REVIEW-FIX-1-REVIEW-CORRECTION-1
VENDOR-KNOWLEDGE-ADAPTER-FAMILY-AUDIT-1
  ACCEPTED
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-PLAN-1
  ACCEPTED / CLOSED
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1
  ACCEPTED / CLOSED
READY_FOR_REVIEW:
GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1
MSGRAPH-KNOWLEDGE-ADAPTERS-1
MSGRAPH-KNOWLEDGE-ADAPTERS-1-FAMILY-CLOSEOUT
VENDOR-KNOWLEDGE-THREE-MODE-CAPABILITY-MATRIX-1
  READY_FOR_REVIEW
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-ARCH-1
  READY_FOR_REVIEW
ACCEPTED / CLOSED:
LKW-SLACK-CONNECTED-SOURCE-1
CHANGES_REQUIRED:
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1
  correction under review
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-1
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-1
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-2
IN_PROGRESS:
PLANNED:
LKW-CONVERSATION-CONTEXT-1
LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
LKW-SLACK-KNOWLEDGE-PROOF-1
GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1
GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1
GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1
LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1
LKW-GOOGLE-WORKSPACE-PROOF-1
DEFERRED: LKW-CONNECTED-SOURCE-1
```

`GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1` freezes the complete Google Workspace knowledge architecture and proof-first roadmap so one existing `GoogleWorkspaceCollaborationSuiteIntegration` can support Drive, Docs, Sheets, Calendar, Slides, Mail and Chat through shared provider primitives, separate Vendor Knowledge adapters and provider-neutral LKW consumption immediately after the complete Slack Knowledge vertical. Status: **READY_FOR_REVIEW**. The current tree contains Google Drive/Docs/Sheets and Calendar adapter implementations, but Google application-owned materialization, indexed wiring and live wiring remain unproven; the remaining runtime and application tasks stay **PLANNED**.

`VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-ARCH-1` is the canonical
[live capability rollout architecture](../../architecture/VENDOR_KNOWLEDGE_LIVE_CAPABILITY_ROLLOUT.md).
It is **READY_FOR_REVIEW**; the Microsoft Graph Drive, Mail, Teams Channel,
Teams Chat and Calendar live list capabilities and the Slack conversation live
family are **ACCEPTED / CLOSED**. All other provider/source-kind live tasks plus
the Google readiness gate remain **PLANNED**.

`VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1` implements the shared
provider-neutral live contract boundary and is **ACCEPTED / CLOSED**.
`MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1A-DRIVE` is **ACCEPTED / CLOSED** with one
bounded Drive list/query capability; `MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1B-MAIL`
is **ACCEPTED / CLOSED** with one bounded mailbox-folder list/query capability.
`MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1C-TEAMS-CHANNEL` is
**ACCEPTED / CLOSED** with one bounded Teams Channel list capability. The v1
capability returns at most one root post; it does not list replies or all
channel messages.
`MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1D-TEAMS-CHAT` and
`MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1E-CALENDAR` are **ACCEPTED / CLOSED** with
one binding-scoped metadata-only list page each. Other provider live capability
tasks remain **PLANNED**.

The canonical current-state classification is
[`VENDOR_KNOWLEDGE_THREE_MODE_CAPABILITY_MATRIX.md`](VENDOR_KNOWLEDGE_THREE_MODE_CAPABILITY_MATRIX.md),
task `VENDOR-KNOWLEDGE-THREE-MODE-CAPABILITY-MATRIX-1`, status
**ACCEPTED / CLOSED**. It records the current Google `drive`, `docs` and
`sheets` and `calendar` adapter implementations, the current exact Google
Calendar read surface, and the remaining application-mode gaps without
activating work. Any older wording that says Google Calendar has no Vendor
Knowledge adapter is a documentation contradiction and is not capability
evidence.

### VK-1 — unified three-mode contract audit result

**Status:** `ACCEPTED / CLOSED`

| Area | Status | Current truth |
|---|---|---|
| Plugin model | `ACCEPTED / CLOSED` | Authoritative provider/source plugin registry; explicit provider/category/source-kind identity; deterministic registration/discovery; runtime execution registries remain separate intentionally. |
| Capability model | `ACCEPTED / CLOSED` | Each source plugin explicitly declares supported modes; mode-specific runtime refs/capability refs remain separate; unsupported modes remain valid. |
| Durable | `ACCEPTED / CLOSED` | Provider-neutral durable coordinator, `KnowledgeSyncSink` and `DocumentStoreDurableKnowledgeSyncSink` are accepted, with representative Slack + Microsoft Graph Teams Chat proof; broader provider coverage remains VK-6/VK-8. |
| Indexed / RAG | `ACCEPTED / CLOSED` | Provider-neutral generic Indexed bridge is accepted, with representative Slack + Microsoft Graph Teams Chat proof through canonical `KnowledgeDocument`; full provider coverage remains VK-6 and complete cross-provider product E2E remains VK-8. |
| Live / Realtime | `ACCEPTED / CLOSED` | Provider-neutral registration/bootstrap now feeds the existing validated executor and tenant catalog; capability coverage remains selective by design. |
| Frontend neutrality | `ACCEPTED / CLOSED` | Generic LKW inspection/operations, plugin configuration, route boundaries and Hybrid Ask routing are provider-neutral; optional provider-owned strategies remain behind explicit composition boundaries. |
| Identity / ownership | `ACCEPTED / CLOSED` | Tenant, workspace, connection, source, binding, document, materialization and Live identity fences are explicit and fail closed; provider ACL completeness remains a provider limitation. |
| Public platform surface | `ACCEPTED / INTERNAL PLATFORM SURFACE` | Reusable runtime contracts and provider extension/composition boundaries are complete for Intergrax applications; standalone SDK/API packaging is productization, not an architecture requirement. |

The remaining provider/source-kind expansion is covered by the truthful VK-6
matrix and is not a platform blocker. VK-2 through VK-8 are accepted/closed,
and VK-9 is the final roadmap closeout. The current Google Calendar adapter
contradicted older matrix wording and was corrected as documentation truth,
not as a roadmap reorder.

`VENDOR-KNOWLEDGE-THREE-MODE-REUSE-ARCH-1` is the architecture/plan correction that freezes reusable provider foundations and separate consumption lifecycles for Indexed / RAG, Durable / Storage / Materialization and Live / Realtime access. The architecture itself does not accept provider mode claims; the Slack live family is now separately **ACCEPTED / CLOSED** through its focused production proof.

`SLACK-KNOWLEDGE-THREE-MODE-ARCH-1` freezes Slack as a reusable three-mode platform knowledge provider built on the existing `SlackConversationChannelIntegration`, distinguishes Slack-as-frontend from Slack-as-knowledge-source, and reprioritizes the roadmap so the complete Slack Knowledge vertical slice precedes Google Workspace knowledge work. `SLACK-KNOWLEDGE-FOUNDATION-1` platform typed reads, Vendor Knowledge adapter and durable sync proof are **DONE** (membership-correct inventory, root-window scope v2, hardened provider validation). `LKW-CONVERSATION-CONTEXT-ARCH-1` is **ACCEPTED** — provider-neutral Conversation Context Binding with observed-audience validation, binding identity, workspace resolution, thread memory isolation, shared capability boundary and deterministic guards in the LKW application domain. LKW conversation context implementation and shared-channel runtime remain **not** implemented; the separate Slack live capability family is **ACCEPTED / CLOSED**.

`LKW-SLACK-CONNECTED-SOURCE-1` is **ACCEPTED / CLOSED**. The proof accepts
selected Slack conversation durable synchronization, root/reply traversal,
typed continuation and crash recovery, application-owned materialization,
workspace-scoped indexed Search/Ask, and replay without duplicate indexed
evidence. Complete per-user Slack ACL enforcement, organization-wide automatic
indexing, native Slack search, attachments/file bodies, a Slack conversation
frontend, and combined indexed-plus-live answers remain unproved or deferred.

**Historical execution order (superseded):**

The following record preserves the previously frozen parallel tracks and their
task history. It is no longer the current Vendor Knowledge execution order;
the canonical order is the VK-1–VK-9 roadmap in §1A. In particular, the LKW
tasks below remain LKW history and must not be promoted to the next Vendor
Knowledge task.

```text
DONE:
SLACK-CONVERSATION-RUNTIME-1
LKW Slack frontend and Ask workflow foundations
VENDOR-KNOWLEDGE-THREE-MODE-REUSE-ARCH-1
JIRA-KNOWLEDGE-ADAPTER-1
CONFLUENCE-KNOWLEDGE-ADAPTER-1
Microsoft Graph Drive / Mail / Teams Channel / Teams Chat adapters
SLACK-KNOWLEDGE-THREE-MODE-ARCH-1

DONE:
SLACK-KNOWLEDGE-FOUNDATION-1
LKW-CONVERSATION-CONTEXT-ARCH-1 — ACCEPTED
SLACK-KNOWLEDGE-LIVE-CAPABILITIES-1 — ACCEPTED / CLOSED
LKW-SLACK-CONNECTED-SOURCE-1 — ACCEPTED / CLOSED

IN_PROGRESS / CHANGES_REQUIRED:
none for the accepted Slack connected-source slice

THEN (Slack / LKW track):
LKW-CONVERSATION-CONTEXT-1
LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1

JOIN (final Slack proof):
LKW-CONVERSATION-CONTEXT-1
+ LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
+ SLACK-LIVE-CAPABILITY-1
+ LKW-HYBRID-ASK-1
→ LKW-SLACK-KNOWLEDGE-PROOF-1

CURRENT VENDOR KNOWLEDGE / MICROSOFT GRAPH TRACK (independent of Google Workspace):
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1 — CHANGES_REQUIRED, correction under review
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-1 — CHANGES_REQUIRED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-2 — ACCEPTED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A — CHANGES_REQUIRED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-1 — CHANGES_REQUIRED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-2 — READY_FOR_REVIEW
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B — ACCEPTED through REVIEW-FIX-5-REVIEW-CORRECTION-1
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-1 — CHANGES_REQUIRED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-2 — CHANGES_REQUIRED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-3 — CHANGES_REQUIRED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-4 — CHANGES_REQUIRED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-5 — CHANGES_REQUIRED
→ VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-5-REVIEW-CORRECTION-1-STATUS-TRUTH-AND-NONSEQUENCE-PROOF — ACCEPTED
→ MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR — ACCEPTED through REVIEW-FIX-1-REVIEW-CORRECTION-1
→ MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1 — CHANGES_REQUIRED
→ MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1-REVIEW-CORRECTION-1-NO-PROVIDER-REREAD-AND-STATUS-HISTORY — ACCEPTED
→ Microsoft Graph adapter-family audit

INDEPENDENT GOOGLE WORKSPACE TRACK (after LKW-SLACK-KNOWLEDGE-PROOF-1 ACCEPTED):
GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1
→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1A-DRIVE
→ GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1A-DRIVE
→ Drive contract/integration proof
→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1B-DOCS
→ GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1B-DOCS
→ Docs contract/integration proof
→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1C-SHEETS
→ GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1C-SHEETS
→ Sheets contract/integration proof
→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1D-CALENDAR
→ GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1D-CALENDAR
→ Calendar contract/integration proof
→ LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1
→ LKW-GOOGLE-WORKSPACE-PROOF-1
→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1E–1G + matching adapters

Google Workspace does not gate reconciliation finalization or Microsoft Calendar acceptance.
Microsoft Calendar work does not gate the independent Google Workspace workstream.

remaining Hybrid Ask and provider packs
```

Microsoft Graph Mail low-level knowledge-read support is complete.

Mailbox folder paging, per-folder immutable-ID message delta, text message
content, normalized participants, attachment inventory and bounded ordinary
file-attachment content reads are implemented.

Removed delta entries remain folder-scoped and are not treated as proof of
global mailbox deletion.

Item attachments, reference-attachment downloads, MIME, raw internet headers
and recursive attached-message expansion are intentionally not implemented.

No Microsoft Vendor Knowledge adapter is exposed yet except Drive, Mail, Teams Channel, Teams Chat and Calendar.

Microsoft Graph Drive Vendor Knowledge adapter (`MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE`)
is implemented.

Microsoft Graph Mail Vendor Knowledge adapter (`MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL`)
is implemented.

Microsoft Graph Teams Channel Vendor Knowledge adapter
(`MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL`) is implemented.

Microsoft Graph Teams Chat Vendor Knowledge adapter
(`MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT`) is implemented.

Microsoft Graph Calendar Vendor Knowledge adapter
(`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR`) is **ACCEPTED** through
`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1-REVIEW-CORRECTION-1-NO-PROVIDER-REREAD-AND-STATUS-HISTORY`;
the prior review fix remains **CHANGES_REQUIRED**. Durable reconciliation
finalization is accepted through Review Fix 5 correction; the Calendar proof
covers safe non-primary missing-item finalization.

Drive capability matrix:

```text
source_kind: drive
scope: one known Microsoft Graph drive ID
full_inventory: yes
incremental_changes: yes
reconciliation: yes
binary_content: yes
structured_content: no
permissions: no
tombstones: yes
remote_versions: yes
```

Drive files map to `BINARY` content.

Folders, packages and unknown non-file records are metadata-only descriptors.

Graph `NEXT_PAGE` and `DELTA` continuations are wrapped in adapter-owned
opaque `KnowledgeCursor` values.

The accepted Drive live operation matrix is:

```text
bounded search: UNSUPPORTED_BY_PROVIDER
bounded list/query: SUPPORTED through the existing read_drive_delta_page boundary
exact item read: UNSUPPORTED_BY_PROVIDER
child read: NOT_APPLICABLE
bounded content read: DEFERRED
```

Drive live list returns deterministic textual metadata only. Content read is
deferred because the current provider surface returns binary bytes, the shared
live result is textual, and the existing adapter does not propagate the live
per-item byte budget.

The accepted Mail live operation matrix is:

```text
bounded search: UNSUPPORTED_BY_PROVIDER
bounded list/query: SUPPORTED through read_mail_messages_delta_page
exact item read: UNSUPPORTED_BY_PROVIDER
thread read: UNSUPPORTED_BY_PROVIDER
child read / attachment inventory: DEFERRED
bounded content read: DEFERRED
```

Mail live list is one page over one binding-derived opaque mailbox-folder scope.
It returns deterministic metadata only; message bodies, threads, attachment
inventory/content and continuation persistence remain excluded.

Drive permission capability remains false because the current low-level
permission projection explicitly does not prove a complete ACL or complete
inheritance graph.

The existing permission read surface is preserved for a future ACL-contract
task and is not represented as authoritative `KnowledgePermissions`.

Mail capability matrix (`MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL`):

```text
source_kind: mail
scope: one known mailbox user ID plus one known folder ID
full_inventory: yes
incremental_changes: yes
reconciliation: yes
content_fetch: yes
binary_content: no
rich_text_content: no
structured_content: yes
permissions: no
tombstones: yes
remote_versions: yes
```

The initial per-folder Graph delta sequence supplies reconciliation inventory.
The final `DELTA` continuation is the durable incremental checkpoint.

Mail content is `msgraph.mail.message.knowledge.v1` structured JSON. Participants
live in content, not descriptor metadata.

`REMOVED` delta entries mean removed from the synchronized folder view, not
global mailbox deletion.

Attachment presence (`has_attachments`) is preserved in descriptor metadata and
structured content. Attachment inventory and binary bytes remain deferred.
Permissions remain false.

Teams Channel capability matrix (`MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL`):

```text
source_kind: teams_channel
scope: one known team ID plus one known channel ID
full_inventory: yes
incremental_changes: no
reconciliation: yes
content_fetch: yes
binary_content: no
rich_text_content: no
structured_content: yes
permissions: no
tombstones: yes, explicit deletedDateTime only
remote_versions: yes
```

Reconciliation traverses root posts one at a time. Replies for the current root
are read immediately before advancing to the next root page. The final
reconciliation checkpoint is a complete adapter-owned cursor.

Exact message content is materialized as `msgraph.teams-channel.message.knowledge.v1`
structured JSON. Tombstones apply only when Graph explicitly returns
`deletedDateTime`; absence from a page is not treated as deletion.

Attachment inventory is included in structured content. Attachment URLs,
embedded card payloads and hosted-content bytes are excluded. Channel member
inventory is not an authoritative ACL projection. No Graph delta, webhook,
subscription or LKW changes are introduced.

Teams Chat capability matrix (`MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT`):

```text
source_kind: teams_chat
scope: one known mailbox-visible chat plus one immutable lastModifiedDateTime window
full_inventory: yes, inside the fixed source window
incremental_changes: no
reconciliation: yes
content_fetch: yes
binary_content: no
rich_text_content: no
structured_content: yes
permissions: no
tombstones: yes, explicit deletedDateTime only
remote_versions: yes
```

Structured schema: `msgraph.teams-chat.message.knowledge.v1`

Flat message collection: yes

Delta: not implemented

Absence-based deletion: not permitted

Attachment inventory: included

Attachment URLs: excluded

Embedded attachment payloads: excluded

Hosted-content bytes: excluded

Authoritative ACL: not implemented

LKW connected-source bridge: generic Indexed bridge accepted through VK-4;
provider-neutral Search/Ask coverage beyond the representative proof remains
deferred.

Live capability layer: bounded list implemented; body/content reads deferred

Live search: unsupported by provider

Calendar capability matrix (`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR`):

```text
source_kind: calendar
scope: one known mailbox Calendar ID + immutable UTC time window
primary full inventory: yes
primary incremental changes: yes
non-primary full inventory: yes
non-primary incremental changes: no
reconciliation: yes
content: STRUCTURED_RECORD
safe attachment inventory: yes
binary attachment content: no
permissions: no
delta tombstones: primary only
remote versions: yes
```

Structured schema: `msgraph.calendar.event.knowledge.v1`

Primary calendar delta: yes

Non-primary snapshot reconciliation: yes

Attachment inventory: included (bounded)

Attachment bytes: excluded

Authoritative ACL: not implemented

Microsoft Graph Calendar low-level knowledge-read support is complete using
stable Graph v1.0 contracts.

Caller-visible user calendars can be enumerated.

The primary calendar supports full and incremental calendar-view delta
synchronization for an explicit fixed time window.

Every selected user calendar, including non-default and shared caller-visible
calendars, supports a complete paged calendar-view snapshot for an explicit
fixed time window.

The Vendor Knowledge adapter uses delta for the primary calendar and
full-snapshot reconciliation for other calendars.

Event content, recurring occurrences and exceptions, participants, locations,
recurrence, attachment inventory and bounded ordinary file-attachment content
reads are implemented.

Removed delta entries apply only to the primary calendar view and are not
treated as proof of global event deletion.

No beta Graph endpoint, group calendar, recursive item attachment or reference-attachment download is implemented.

The Microsoft Graph Calendar Vendor Knowledge adapter is **ACCEPTED**
(`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR`) through
`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1-REVIEW-CORRECTION-1-NO-PROVIDER-REREAD-AND-STATUS-HISTORY`;
its prior review fix remains **CHANGES_REQUIRED**. Non-primary missing-item
detection is implemented and proven only after the final snapshot page.
Calendar ACL is not implemented.

The Calendar live capability is one initial metadata-only page over the
binding-selected opaque scope. Primary-delta and non-primary-snapshot strategy
selection remains authoritative in the adapter; continuation replay and
complete traversal are deferred. Event bodies, attendees and attachment
content are excluded.

Microsoft Graph Teams Chat low-level knowledge-read support is complete using
stable Graph v1.0 contracts.

Caller-visible chats can be enumerated and complete member rosters can be read.

Each selected chat supports complete paged message snapshots for explicit
lastModifiedDateTime windows. Reference-based fixed-window snapshot paging and
exact revision-bound active message reads are implemented for stateless adapter
use. Stable Graph v1.0 chat-message delta is not used.

Deleted messages are recognized only when Graph explicitly returns
deletedDateTime. Absence from a window snapshot is not treated as proof of
global deletion.

Message bodies, senders, mentions, reactions, safe attachment references,
forwarded-message metadata and bounded Teams-hosted content reads are
implemented.

Chat messages are a flat collection. No synthetic chat-replies endpoint is
implemented.

The Microsoft Graph Teams Chat Vendor Knowledge adapter and its review correction
are implemented.

The Teams Chat live capability is one binding-scoped metadata-only list page.
Provider-neutral live search remains unsupported. The LKW connected-source
bridge is implemented through the generic VK-4 Indexed path; message bodies,
mentions, reactions, attachment inventory/bytes and hosted content remain
excluded from this live capability.

File attachment URLs are retained only as hidden provider references and are
not downloaded directly. A later Microsoft Vendor Knowledge adapter can resolve
supported SharePoint and OneDrive references through the existing Drive
surface.

No beta Graph endpoint, webhook,
subscription, rich-card semantic renderer or direct external attachment
download is implemented.

Microsoft Graph Teams Channel low-level knowledge-read support is complete
using stable Graph v1.0 contracts.

Caller-visible channels can be enumerated per team and complete effective
member rosters can be read through allMembers.

Each selected channel supports complete paged root-message inventory and
threaded reply reads for explicit root posts. Stable Graph v1.0 channel
message delta is not used.

Deleted root posts and replies are recognized only when Graph explicitly
returns deletedDateTime. Absence from a page is not treated as proof of
global deletion.

Message bodies, senders, mentions, reactions, safe attachment references,
forwarded-message metadata and bounded Teams-hosted content reads are
implemented.

Channel messages use the dedicated replies endpoint for threaded replies.
File attachment URLs are retained only as hidden provider references and are
not downloaded directly.

No beta Graph endpoint, Teams Channel webhook,
subscription, rich-card semantic renderer or direct external attachment
download is implemented.

Current runtime state:

```text
Facade contracts implemented
Adapter registry implemented
IntegrationProfile resolver implemented
Connection-aware resolver implemented
Stateless facade core implemented
Tenant-scoped source bindings implemented
DocumentStore binding repository implemented
Platform-neutral synchronization coordinator implemented
Conditional DocumentStore capability implemented
DocumentStore sync lease repository implemented
DocumentStore checkpoint repository implemented
DocumentStore remote-item state repository implemented
DocumentStoreTaskQueue/Worker wiring implemented
Delivery-ID continuation scheduling implemented
Interrupted-task recovery implemented
Bounded sync-handler retry/backoff implemented
Jira issues knowledge adapter implemented
Confluence pages knowledge adapter implemented
First real vendor facade/coordinator proof implemented
Jira Issues, Confluence Pages, Microsoft Graph Drive, Microsoft Graph Mail, Microsoft Graph Teams Channel and Microsoft Graph Teams Chat Vendor Knowledge adapters implemented.

Microsoft Graph Teams Chat Vendor Knowledge adapter is implemented.
Microsoft Graph Calendar Vendor Knowledge adapter is **ACCEPTED** through
`REVIEW-FIX-1-REVIEW-CORRECTION-1`; its prior review fix remains
**CHANGES_REQUIRED**.
The generic VK-4 Indexed connected-source bridge is implemented; this historical
status block does not claim broader provider coverage.
```

Notes after `VENDOR-KNOWLEDGE-SYNC-1B`:

- generic delayed queue scheduling was **not** added;
- retry/backoff is scoped to the Vendor Knowledge sync handler only;
- the sink remains an injected port (`KnowledgeSyncSink`);
- LKW intake remains application-owned; the generic connected-source Indexed
  bridge is accepted through VK-4.

---

## 3. Frozen rules

1. No `knowledge_source` integration category.
2. No duplicate public vendor integrations.
3. Existing provider/category integrations remain the only vendor entrypoints.
4. Vendor integrations own API transport, authentication handoff, provider mapping and category operations.
5. Vendor integrations do not import LKW, RAG or workspace code.
6. The facade is a platform service above integrations.
7. Source adapters are thin mappings over already resolved integration instances.
8. Adapters do not own clients, credentials, persistence or checkpoints.
9. LKW and other knowledge-consuming applications use Vendor Knowledge Facade for durable operations and the validated live capability boundary for live operations; they do not call vendor SDKs or provider-specific integration methods directly.
10. Reuse `IntegrationProfile` and the existing integration catalog for integration resolution.
11. Reuse `SecretsStore` for secret material; durable bindings contain opaque references only.
12. Reuse `DocumentStoreTaskQueue`, `DocumentStoreTaskWorker` and `TaskExecutionRegistry` for later asynchronous sync.
13. Reuse provider-neutral `DocumentStore` for later facade persistence.
14. Do not import `ManagedWorkspaceRepository` into platform facade code.
15. Stable remote identity is separate from version, ETag and content hash.
16. One shared parser/chunk/embedding/indexing pipeline remains downstream.
17. ACL must be enforceable before model access.
18. One existing provider/category integration may expose multiple knowledge `source_kind` values through separate thin adapters.
19. All work remains on branch `development`.
20. Every future vendor task description and `READY_FOR_REVIEW` report must include a **THREE-MODE REUSE ASSESSMENT** (documentation evidence; does not require implementing all three modes):

```text
THREE-MODE REUSE ASSESSMENT:
- shared integration:
- shared client/transport:
- shared provider references:
- shared exact-read primitives:
- Indexed / RAG readiness:
- Durable / Storage / Materialization readiness:
- Live / Realtime exact-read readiness:
- Live / Realtime search/query readiness:
- provider primitive gaps:
- durable adapter gaps:
- live capability gaps:
- duplicate client/integration introduced: no
```

---

## 4. Current provider readiness matrix

Status-honest readiness across the three consumption modes. Architecture is not proof of implementation.

### Jira

```text
durable inventory/content foundation: implemented
RAG downstream bridge: not yet connected to LKW
live search/read primitives: existing integration operations available
provider-neutral live capability/executor: not implemented
```

### Confluence

```text
durable inventory/content foundation: implemented
RAG downstream bridge: not yet connected to LKW
live search/read primitives: existing integration operations available
provider-neutral live capability/executor: not implemented
```

### Microsoft Graph Drive

```text
durable adapter: implemented
bounded live list/query: ACCEPTED / CLOSED
bounded live search: unsupported by provider
exact live item read: unsupported by provider
child live read: not applicable
bounded live content read: deferred
LKW bridge: not implemented
```

### Microsoft Graph Mail

```text
durable adapter: implemented
exact live read foundation: available where an exact message is known
bounded provider-neutral live list: ACCEPTED / CLOSED
bounded provider-neutral live search: unsupported by provider
message body/thread/attachment reads: deferred or unsupported
LKW bridge: not implemented
```

### Microsoft Graph Teams Channel

```text
durable reconciliation adapter: implemented
exact message/thread read foundation: available
provider-neutral live bounded list: ACCEPTED / CLOSED
provider-neutral live search/discovery capability: unsupported by provider
LKW bridge: not implemented
```

### Microsoft Graph Teams Chat

```text
durable reconciliation adapter: implemented
exact message read foundation: implemented
provider-neutral live bounded list: ACCEPTED / CLOSED
provider-neutral live search/discovery capability: unsupported by provider
LKW Indexed bridge: generic VK-4 bridge implemented for `teams_chat`
provider-specific Search/Ask E2E: deferred to VK-8
```

### Microsoft Graph Calendar

```text
low-level read foundation: implemented
Vendor Knowledge adapter: ACCEPTED through REVIEW-FIX-1-REVIEW-CORRECTION-1
non-primary missing-item detection: implemented; final-page proof accepted
provider-neutral live bounded list: ACCEPTED / CLOSED
continuation/delta replay: deferred
LKW bridge: not implemented
```

---

## 5. Reuse decisions

| Area | Decision |
|---|---|
| Integration resolution | Reuse `IntegrationProfile.resolve()` / `resolve_from_profile()` through an injected resolver port. |
| Integration catalog | Reuse unchanged. Do not create another vendor catalog. |
| Adapter resolution | Add one minimal source-adapter registry keyed by provider, category and source kind. |
| Multi-surface vendors | One provider/category integration may serve several source adapters, for example Microsoft Graph `drive`, `mail`, `calendar`, `teams_chat` and `teams_channel`. |
| Multiple connections | Add tenant-scoped facade bindings above `IntegrationProfile`; the profile itself remains application composition. |
| Secrets | Reuse `SecretsStore`; persist only `connection_ref` / `credential_ref`. |
| Durable work | Reuse DocumentStore-backed queue and worker. |
| Durable state | Add later facade-owned repositories over `DocumentStore`. |
| Errors | Reuse integration errors as causes; expose a safe normalized facade error envelope. |
| LKW repository/runtime | Use as a proven pattern and later convergence point, not as a platform dependency. |

---

## 6. Ownership during parallel work

### Vendor facade track

Owns:

- vendor-neutral contracts;
- integration resolver port;
- source adapter port and registry;
- facade core;
- connection/source binding boundary;
- platform-neutral synchronization coordinator;
- vendor-specific adapters over existing integrations;
- focused contract and unit tests.

### LKW ingest track

Owns:

- Knowledge Intake;
- managed uploads and snapshots;
- Object Storage and staging;
- application operations and workers;
- parser/chunk/embedding/indexing invocation;
- Source → Document ownership;
- Slack file-intake UX.

### Deferred convergence

Deferred until both tracks are stable:

- `WorkspaceSource(CONNECTED_SOURCE)` binding;
- `SOURCE_CANDIDATE` resolution;
- connected-source ingestion processor;
- Slack source-management UX;
- retrieval-time ACL integration.

---

## 7. Historical implementation roadmap (superseded sequencing)

This detailed task history is retained to preserve prior statuses and evidence.
Its future sequencing is superseded by the canonical VK-1–VK-9 roadmap in §1A.
No status below is changed by this reframe.

### Phase 0 — Architecture, plan and reuse audit

#### `VENDOR-KNOWLEDGE-FACADE-ARCH-1`

**Status:** `DONE`

Corrected the architecture:

- rejected a generic knowledge-source category;
- rejected duplicate vendor integrations;
- placed the facade above existing integrations.

#### `VENDOR-KNOWLEDGE-FACADE-PLAN-1`

**Status:** `DONE`

Established ordered phases and the convergence point with LKW.

#### `VENDOR-KNOWLEDGE-FACADE-AUDIT-1`

**Status:** `DONE`

Confirmed reuse of:

- IntegrationProfile/factory/catalog;
- SecretsStore;
- DocumentStore;
- DocumentStoreTaskQueue/Worker;
- TaskExecutionRegistry.

Confirmed gaps:

- vendor-neutral contracts;
- source-adapter registry;
- tenant-scoped connection/source bindings;
- normalized facade errors;
- later checkpoint/lease/item state;
- missing vendor read/change methods.

---

### Phase 1 — Facade contracts

#### `VENDOR-KNOWLEDGE-FACADE-CONTRACT-1`

**Status:** `DONE`

**Purpose:** Define the minimum stable vocabulary and ports without implementing runtime behavior.

**Allowed scope:**

```text
intergrax/runtime/vendor_knowledge/__init__.py
intergrax/runtime/vendor_knowledge/models.py
intergrax/runtime/vendor_knowledge/contracts.py
intergrax/runtime/vendor_knowledge/errors.py
tests/unit/runtime/vendor_knowledge/
```

**Deliverables:**

- tenant-aware source binding reference;
- source scope and capabilities;
- stable remote item identity;
- separate revision/version state;
- page and opaque cursor result;
- binary, rich-text and structured-record content envelope;
- provenance/deep-link data;
- ACL/permission envelope;
- normalized facade error;
- `VendorIntegrationResolver` protocol;
- `VendorKnowledgeAdapter` protocol;
- `VendorKnowledgeFacade` protocol.

**Out of scope:**

- new integration category;
- registry implementation;
- facade implementation;
- vendor code;
- secrets lookup implementation;
- persistence;
- queues/workers;
- checkpoints/retries/leases;
- LKW/RAG changes.

**Acceptance:**

- strict models with `extra="forbid"` or equivalent;
- mandatory tenant identity where state crosses boundaries;
- no secret-bearing fields;
- remote identity separated from revision/content hash;
- explicit content modes;
- deterministic validation;
- focused tests green.

---

### Phase 2 — Facade core and adapter registry

#### `VENDOR-KNOWLEDGE-FACADE-CORE-1`

**Status:** `DONE`

**Dependency:** Phase 1

Implement:

```text
integration resolver adapter over IntegrationProfile
source adapter registry
facade core
fake integration + fake adapter proof
```

Expected flow:

```text
request
→ validate tenant/binding reference
→ resolve existing integration
→ resolve source adapter
→ invoke adapter
→ normalize result/error
```

Acceptance:

- no provider `if/elif` chain;
- no duplicate integration construction;
- duplicate adapter registration rejected;
- unknown adapter fails deterministically;
- cross-tenant request fails closed;
- no network or persistence required for proof.

---

### Phase 3 — Connection and source binding

#### `VENDOR-KNOWLEDGE-CONNECTION-1`

**Status:** `DONE`

Add tenant-scoped binding semantics:

```text
binding_id
tenant_id
provider_id
integration_kind
source_kind
integration reference
connection_ref / credential_ref
validated scope
safe display metadata
status
configuration version
```

Rules:

- no raw tokens or secrets;
- multiple connections/scopes per tenant supported;
- binding resolves exactly one existing integration and one source adapter;
- one Microsoft 365 connection may expose several independently configured source bindings;
- revocation/expiry represented explicitly;
- broad scopes require explicit policy approval.

---

### Phase 4 — Shared synchronization coordinator

#### `VENDOR-KNOWLEDGE-SYNC-1A`

**Status:** `DONE`

Implement platform-neutral orchestration with fake adapters and repository ports:

- source-level lease;
- checkpoint read;
- bounded page read;
- at-least-once replay;
- remote-item revision state;
- tombstone handling;
- checkpoint commit after durable page completion;
- retry classification;
- reconciliation entrypoint.

The coordinator outputs normalized items to a sink port. It does not parse, chunk, embed or write LKW documents.

#### `VENDOR-KNOWLEDGE-SYNC-1B`

**Status:** `DONE`

**Prerequisites:**

```text
DONE:     VENDOR-KNOWLEDGE-SYNC-1A
DONE:     PLATFORM-DOCUMENT-STORE-CONDITIONAL-1
DONE:     VENDOR-KNOWLEDGE-SYNC-1B
NEXT:     JIRA-KNOWLEDGE-ADAPTER-1
```

Wired the coordinator onto:

- `DocumentStoreTaskQueue`;
- `DocumentStoreTaskWorker`;
- `TaskExecutionRegistry`;
- facade-owned `DocumentStore` repositories.

Conditional write requirements (from `PLATFORM-DOCUMENT-STORE-CONDITIONAL-1`):

- source lease requires atomic `put_if_absent` / CAS;
- checkpoint commit requires CAS;
- remote delivery marker requires conditional write;
- implementation must fail closed when the resolved `DocumentStore` does not satisfy `ConditionalDocumentStore`;
- do not emulate CAS with ordinary `get()` + `put()`.

Retry/backoff is a bounded Vendor Knowledge handler policy only — not a generic delayed queue scheduler. The sync sink remains an injected port; LKW intake remains separate.

#### Composition modes

```text
Platform owns:
- durable job schema;
- scheduler and canonical worker handler;
- checkpoint/lease/item-state repositories;
- optional standalone tenant-scoped runtime.

Applications own:
- application operation lifecycle;
- queue/worker composition;
- user-facing recovery and status.

LKW must use the application-composition adapter from sync_task
inside its existing runtime and must not start a second
VendorKnowledgeSyncRuntime.
```

#### `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1`

**Status:** `CHANGES_REQUIRED` — correction under review

**Review fix:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-1` — `CHANGES_REQUIRED`

**Review fix 2:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-2` — `READY_FOR_REVIEW`

**Purpose:** Freeze provider-neutral durable reconciliation-finalization semantics so completed snapshot reconciliation can emit deterministic synthetic tombstones for items absent from the synchronized inventory without the checkpoint-failure / `restart=True` nondeterminism present on HEAD.

**Architecture:** [`VENDOR_KNOWLEDGE_RECONCILIATION_FINALIZATION.md`](VENDOR_KNOWLEDGE_RECONCILIATION_FINALIZATION.md)

**Frozen decisions (review-fix-2):**

- separate reconciliation-run state machine (`COLLECTING` → `PAGE_PREPARED` → `FINALIZING` → `COMPLETED`) with fail-closed `RECOVERY_REQUIRED` and controlled terminal `ABORTED`;
- single active slot keyed by `(tenant_id, binding_id)` with `binding_configuration_version` inside the run;
- `PAGE_PREPARED` intent before sink or remote-item state mutation;
- non-circular identity pipeline: `prepared_state_mutation_templates` → `prepared_state_mutations_fingerprint` → `prepared_batch_payload_fingerprint` (excluding `delivery_id`) → `delivery_id`;
- `applied_page_count` and `last_applied_delivery_id` as durable applied-page evidence; `effects_started := applied_page_count > 0`;
- exact private cursor objects in run state (`current_input_cursor`, `prepared_input_cursor`, `prepared_next_cursor`, `prepared_proposed_checkpoint`) with matching fingerprints;
- bounded prepared-intent payload bytes and state-mutation count checked before CAS;
- inspectable sink and item-state delivery receipts with receipt-driven retry (provider reread only when both receipts `ABSENT`);
- `ABORT_PRISTINE` requires `applied_page_count == 0` and pristine receipt proof; `COLLECTING` does not imply pristine;
- sink receipt `UNKNOWN` maps to `DEPENDENCY_UNAVAILABLE`, not retryable, → `RECOVERY_REQUIRED`;
- `FINALIZING` checkpoint idempotency including crash-after-checkpoint-commit recovery;
- operator-safe recovery interface (`RESUME_EXACT`, `FINALIZE_ALREADY_COMMITTED`, `ABORT_PRISTINE`, `REPAIR_REQUIRED`);
- bounded active candidate inventory with count and byte limits;
- incremental sync blocked while an active reconciliation run exists;
- reconciliation tombstones mean `absent_from_completed_synchronized_source_inventory` only.

**Implementation outcome:** `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A` is **ACCEPTED** through `REVIEW-FIX-3`, and `VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B` is **ACCEPTED** through `REVIEW-FIX-5-REVIEW-CORRECTION-1`.

**Blocks:** none in the Microsoft Graph adapter family; Calendar missing-item
detection is implemented and covered by the accepted Calendar proof.

---

### Phase 5 — Vendor proofs

#### `JIRA-KNOWLEDGE-ADAPTER-1`

**Status:** `DONE`

Content mode: `STRUCTURED_RECORD`.

Extend the existing Jira integration only where required, then map bounded issue data through a Jira source adapter.

Capability matrix (Jira `issues`):

```text
source_kind: issues
scope: one Jira project
content: STRUCTURED_RECORD
full_inventory: yes
reconciliation: yes
incremental_changes: no
permissions: no
tombstones: no
remote_versions: yes
```

Deferred / not implemented for Jira issues adapter:

```text
comments deferred
attachments deferred
custom-field projection deferred
end-user ACL projection deferred
deletion/revocation projection deferred
incremental change-feed deferred
general IssueTracker.search_issues endpoint migration deferred
```

#### `CONFLUENCE-KNOWLEDGE-ADAPTER-1`

**Status:** `DONE`

Content mode: `RICH_TEXT`.

Extend the existing Confluence integration only where required, then map pages, versions and visibility through a Confluence adapter.

Capability matrix (Confluence `pages`):

```text
source_kind: pages
scope: one Confluence space ID
content: RICH_TEXT / Confluence Storage Format
full_inventory: yes
reconciliation: yes
incremental_changes: no
permissions: no
tombstones: no
remote_versions: yes
```

Deferred / not implemented for Confluence pages adapter:

```text
blog posts deferred
attachments deferred
comments deferred
labels deferred
custom content deferred
end-user ACL projection deferred
deletion/revocation projection deferred
incremental change-feed deferred
legacy WikiKnowledge search migration deferred
```

#### `MSGRAPH-KNOWLEDGE-READ-SURFACE-1`

**Status:** `DONE`
Drive metadata, delta, tombstones, bounded binary content and caller-visible
sharing-permission reads are implemented.
The permission response is explicitly not treated as a proven complete
end-user ACL.
Sharing URLs, share IDs and invitation email addresses are not retained.
Download URLs are never persisted.
File conversion is not implemented yet.
Microsoft Graph Drive low-level read support is complete.
Mailbox root and child folder paging is implemented.
Mail folder paging and per-folder message metadata delta are implemented.
Message IDs are requested in ImmutableId format on every delta request.
A removed delta entry means removed from the synchronized folder and is not
treated as proof of global mailbox deletion.
Mail text bodies, participants and attachments are implemented.
Microsoft Graph Calendar low-level read support is complete using stable
Graph v1.0 contracts: primary-calendar incremental delta, per-calendar full
snapshots, bounded event content, participants, locations, recurrence and
bounded file attachments.
Removed primary-calendar delta entries are not treated as proof of global
event deletion.

Extend the single existing Microsoft Graph collaboration-suite integration/private client boundary with the low-level read behavior required by all approved Microsoft 365 knowledge surfaces.

Approved source kinds:

```text
drive
mail
calendar
teams_chat
teams_channel
```

Shared responsibilities:

- bounded inventory and pagination;
- delta/cursor support where Microsoft Graph provides it;
- stable object identity separated from revision;
- ETag/cTag or equivalent revision information;
- tombstones, deletions and revocations;
- attachment inventory and content retrieval;
- safe provider error and throttling mapping;
- permission and visibility reads where available;
- no LKW, RAG, parser, chunker or embedding imports.

Surface-specific low-level behavior:

- `drive`: SharePoint sites, document libraries, OneDrive drives/folders/files, delta, binary content and permissions;
- `mail`: Outlook folders, messages, conversation/thread metadata, bodies and attachments;
- `calendar`: calendars, events, organizers, attendees, recurrence and online-meeting metadata;
- `teams_chat`: one-to-one and group chats, messages, replies, edits, deletions, attachments and links;
- `teams_channel`: teams, channels, posts, threaded replies, mentions, edits, deletions and attachments.

This task must not create separate public Microsoft integrations for Drive, mail, calendar or Teams. The existing Microsoft Graph integration remains the single provider/category entrypoint.

#### `MSGRAPH-KNOWLEDGE-ADAPTERS-1`

**Status:** `ACCEPTED / CLOSED`

`MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE` is **ACCEPTED**.

`MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL` is **ACCEPTED**.

`MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL` is **ACCEPTED**.

`MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT` is **ACCEPTED** through
`REVIEW-FIX-1`.

`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR` is **ACCEPTED** through
`REVIEW-FIX-1-REVIEW-CORRECTION-1`; its prior review fix remains
**CHANGES_REQUIRED**. Shared reconciliation finalization is accepted through
Review Fix 5 correction.

#### `MSGRAPH-KNOWLEDGE-ADAPTERS-1-FAMILY-CLOSEOUT`

**Status:** `ACCEPTED / CLOSED`

The Drive, Mail, Teams Channel, Teams Chat and Calendar adapters are verified
as one Microsoft Graph integration family. Shared durable reconciliation,
provider-specific paging/content boundaries, deterministic delivery identity
and secret-safe public outputs are covered by the provider and focused
regression suites. Unsupported capabilities remain explicitly unsupported.

**Vendor Knowledge / Microsoft Graph track (independent of Google Workspace):**

The detailed entries below are historical review traceability. They are
preserved for audit history and do not override the current matrix or the
accepted family closeout above.

```text
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1 — CHANGES_REQUIRED, correction under review
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-1 — CHANGES_REQUIRED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-ARCH-1-REVIEW-FIX-2 — ACCEPTED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A — CHANGES_REQUIRED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-1 — CHANGES_REQUIRED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1A-REVIEW-FIX-2 — READY_FOR_REVIEW
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B — ACCEPTED through REVIEW-FIX-5-REVIEW-CORRECTION-1
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-1 — CHANGES_REQUIRED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-2 — CHANGES_REQUIRED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-3 — CHANGES_REQUIRED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-4 — CHANGES_REQUIRED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-5 — CHANGES_REQUIRED
VENDOR-KNOWLEDGE-RECONCILIATION-FINALIZATION-1B-REVIEW-FIX-5-REVIEW-CORRECTION-1-STATUS-TRUTH-AND-NONSEQUENCE-PROOF — ACCEPTED
MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR — ACCEPTED through REVIEW-FIX-1-REVIEW-CORRECTION-1
MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1 — CHANGES_REQUIRED
MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR-REVIEW-FIX-1-REVIEW-CORRECTION-1-NO-PROVIDER-REREAD-AND-STATUS-HISTORY — ACCEPTED
MSGRAPH-KNOWLEDGE-ADAPTERS-1 — READY_FOR_REVIEW
MSGRAPH-KNOWLEDGE-ADAPTERS-1-FAMILY-CLOSEOUT — READY_FOR_REVIEW
```

Google Workspace does not gate reconciliation finalization or Microsoft Calendar acceptance. Microsoft Calendar work does not gate the independent Google Workspace workstream.

Add separate thin adapters over the same resolved Microsoft Graph integration:

```text
MsGraphDriveKnowledgeAdapter
MsGraphMailKnowledgeAdapter
MsGraphCalendarKnowledgeAdapter
MsGraphTeamsChatKnowledgeAdapter
MsGraphTeamsChannelKnowledgeAdapter
```

Registry keys:

```text
(ms365_graph, collaboration_suite, drive)
(ms365_graph, collaboration_suite, mail)
(ms365_graph, collaboration_suite, calendar)
(ms365_graph, collaboration_suite, teams_chat)
(ms365_graph, collaboration_suite, teams_channel)
```

Content mapping:

- `drive` → `BINARY` for files; metadata-only non-file records for folders and inventory records;
- `mail` → `STRUCTURED_RECORD`; attachment binary content deferred;
- `teams_channel` → `STRUCTURED_RECORD` only; safe attachment inventory included; attachment URLs, embedded payloads, hosted-content bytes and binary attachment materialization excluded;
- `teams_chat` → `STRUCTURED_RECORD` only; safe attachment inventory included; attachment URLs, embedded payloads, hosted-content bytes and binary attachment materialization excluded;
- `calendar` → `STRUCTURED_RECORD`; safe attachment inventory included; attachment bytes deferred;

Each adapter:

- declares only its own capabilities;
- maps provider records into the canonical facade models;
- receives the already resolved Microsoft Graph integration;
- owns no client, credentials, persistence, checkpoint or retry runtime;
- uses the shared synchronization coordinator;
- remains independent from LKW.

Recommended implementation/proof order inside the Microsoft scope:

```text
1. drive / SharePoint
2. mail
3. teams_channel
4. teams_chat
5. calendar — ACCEPTED through `REVIEW-FIX-1-REVIEW-CORRECTION-1` (prior review fix CHANGES_REQUIRED)
```

Google Workspace remains an independent workstream. Google Workspace does not gate reconciliation finalization or Microsoft Calendar acceptance. Microsoft Calendar work does not gate the independent Google Workspace workstream.

The Microsoft Graph adapter family is complete through
`MSGRAPH-KNOWLEDGE-ADAPTERS-1-FAMILY-CLOSEOUT`. The next documented
cross-provider Vendor Knowledge task is
`VENDOR-KNOWLEDGE-ADAPTER-FAMILY-AUDIT-1`.

The complete Slack Knowledge vertical slice (`SLACK-KNOWLEDGE-FOUNDATION-1` → `LKW-CONVERSATION-CONTEXT-ARCH-1` → implementation tracks through `SLACK-LIVE-CAPABILITY-1`; final proof joins `LKW-HYBRID-ASK-1` at `LKW-SLACK-KNOWLEDGE-PROOF-1`) precedes the Google Workspace proof-critical path.

The task is grouped as one Microsoft Graph adapter family, but implementation and verification must preserve independent `source_kind`, scope, cursor and ACL semantics for every surface.

Drive adapter capability matrix (`MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE`):

```text
source_kind: drive
scope: one known Microsoft Graph drive ID
full_inventory: yes
incremental_changes: yes
reconciliation: yes
binary_content: yes
structured_content: no
permissions: no
tombstones: yes
remote_versions: yes
```

Drive files map to `BINARY` content.

Folders, packages and unknown non-file records are metadata-only descriptors.

Graph `NEXT_PAGE` and `DELTA` continuations are wrapped in adapter-owned
opaque `KnowledgeCursor` values.

Drive permission capability remains false because the current low-level
permission projection explicitly does not prove a complete ACL or complete
inheritance graph.

The existing permission read surface is preserved for a future ACL-contract
task and is not represented as authoritative `KnowledgePermissions`.

#### `DATABRICKS-KNOWLEDGE-ADAPTER-1`

**Status:** `DEFERRED`

First select one precise source kind: Unity Catalog metadata, workspace tree, volume files or an approved query snapshot.

---

### Historical Phase 6 — Post-adapter roadmap (superseded)

After the Microsoft Graph adapter-family audit (`MSGRAPH-KNOWLEDGE-ADAPTERS-1`), platform work splits into durable and live branches converging at Hybrid Ask.

#### COMMON PROVIDER FOUNDATION

```text
- existing integrations
- connections
- remote resources
- typed provider references
- exact reads
- inventory/change reads
- safe validation
- normalized errors
- provider capability matrix
```

Planned tasks:

| Task | Purpose |
|---|---|
| [`VENDOR-KNOWLEDGE-ADAPTER-FAMILY-AUDIT-1`](VENDOR_KNOWLEDGE_ADAPTER_FAMILY_AUDIT.md) | Audit adapter-family completeness and gap classification |
| [`VENDOR-KNOWLEDGE-THREE-MODE-CAPABILITY-MATRIX-1`](VENDOR_KNOWLEDGE_THREE_MODE_CAPABILITY_MATRIX.md) | Explicit per-provider, per-source-kind, per-mode capability matrix |

#### DURABLE BRANCH

```text
- source bindings
- Vendor Knowledge adapters
- Sync / Materialization Runtime
- generic durable sink contract
- application/database materialization
- LKW Connected Source bridge
- optional RAG ingestion
```

Planned tasks:

| Task | Purpose | LKW mapping |
|---|---|---|
| `VENDOR-MATERIALIZATION-SINK-CONTRACT-1` | Generic injected durable sink contract beyond DocumentStore proof | `LKW-KNOWLEDGE-LIFECYCLE-1` |
| `LKW-CONNECTED-SOURCE-1` | LKW bridge from Connected Source to facade sync runtime | `LKW-KNOWLEDGE-ACCESS-1`, `LKW-VENDOR-ACCESS-COLLABORATION-1` |

#### LIVE BRANCH

```text
- Live Access Bindings
- typed live capability contracts
- live capability registry
- validated read-only executor
- normalized Live Evidence
- execution receipts
- result/count/byte/time limits
- ephemeral retention default
```

Planned tasks:

| Task | Purpose | LKW mapping |
|---|---|---|
| `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1` | Implement the provider-neutral shared live contract delta, atomic registration boundary, effective budgets, safe provenance/locators, receipts and contract tests | `LKW-KNOWLEDGE-ACCESS-1`, `LKW-HYBRID-ASK-1` |

#### CONVERGENCE

```text
- Knowledge Query Orchestrator
- indexed + live evidence normalization
- Hybrid Ask
- unified provenance
```

Maps to existing LKW product blocks — do not create duplicate LKW roadmap blocks:

```text
LKW-KNOWLEDGE-ACCESS-1
LKW-HYBRID-ASK-1
LKW-VENDOR-ACCESS-COLLABORATION-1
LKW-VENDOR-ACCESS-DATA-1
LKW-KNOWLEDGE-LIFECYCLE-1
```

---

### Historical Phase 7 — LKW convergence (separate session)

#### `LKW-CONNECTED-SOURCE-1`

**Status:** `DEFERRED`

Dependency:

- facade core stable;
- connection/source binding stable;
- synchronization coordinator stable;
- at least one vendor proof stable;
- LKW managed-file intake stable.

Target flow:

```text
WorkspaceSource(CONNECTED_SOURCE)
→ connected-source binding
→ facade sync coordinator
→ normalized item/content
→ existing LKW ingestion pipeline
→ Document Store + Vector Store
```

No duplicate parsing or embedding path is allowed.

---

### Phase 8 — Slack Knowledge vertical (`SLACK-KNOWLEDGE-THREE-MODE-ARCH-1`)

**Classification:** architecture frozen; platform foundation **IMPLEMENTED**; Slack live family **ACCEPTED / CLOSED**; bounded configured-channel Ask readiness **ACCEPTED / CLOSED**; generic Indexed bridge **ACCEPTED / CLOSED** through VK-4; Slack deletion remains unproven.

One existing `SlackConversationChannelIntegration` is reused across indexed RAG, durable materialization without RAG and bounded live access. LKW application tasks remain outside platform ownership.

#### `SLACK-KNOWLEDGE-FOUNDATION-1`

**Status:** `DONE` (platform)

**Classification:** `IMPLEMENTED` — platform foundation; not the LKW bridge. The separate Slack live family is accepted below.

Delivered scope includes same-integration bot-token credential model on one `AsyncWebClient`, bot-membership inventory via single `users.conversations` stream, `conversation_kind`-explicit history/reply/exact reads, strict descriptor validation, `thread_broadcast` exclusion from history materialization, and durable sync retry proof for multi-page reply traversal followed by history resumption.

Target scope:

```text
existing Slack integration reuse
typed bounded Slack knowledge read surface
safe conversation inventory
history paging
thread/reply paging
exact message/version reads
edits and explicit deletion semantics
safe attachment inventory
provider-safe references and cursors
Slack Vendor Knowledge Adapter
capabilities
Facade registration
Sync Coordinator durable proof
injected database/store sink proof
three-mode reuse assessment
```

Do not freeze implementation signatures before the repository audit. Do not claim full ACL support unless an authoritative Slack authorization projection is proved.

**User-facing meaning after completion:** The platform can safely read and durably synchronize selected Slack conversations for any Intergrax application. No new Slack command or LKW feature is implied yet.

#### Slack three-mode matrix (live accepted; indexed/durable claims remain conservative)

| Concern | Indexed RAG | Durable materialization | Live access |
|---|---:|---:|---:|
| `SlackConversationChannelIntegration` | reused | reused | reused |
| Slack client, transport, credentials | reused | reused | reused |
| Shared Slack read primitives | reused | reused | reused |
| Slack Vendor Knowledge Adapter | implemented | implemented | not used |
| Slack Live Capability Adapter | not used | not used | implemented |
| LKW Knowledge Intake / RAG | optional consumer | not required | not automatic |
| Ephemeral evidence | not primary | not primary | required |
| Automatic persistence of live results | no | n/a | forbidden by default |

#### `LKW-SLACK-CONNECTED-SOURCE-1`

**Status:** `ACCEPTED / CLOSED` (LKW application — selected Slack
conversation synchronization, root/reply traversal, typed continuation,
crash-safe recovery, application-owned materialization and indexed Search/Ask
proof accepted)

Application-only use of the completed platform foundation:

```text
workspace Connection
→ approved Slack Remote Resource
→ Indexed Source
→ Vendor Knowledge synchronization
→ LKW Knowledge Intake
→ RAG
```

No new Slack client or provider adapter in LKW.

**User-facing meaning:** The user can attach an approved Slack conversation to
an LKW workspace, synchronize it and ask questions about its indexed history.
Replay and restart preserve the same corpus without duplicate indexed evidence.
This does not prove complete per-user Slack ACLs, organization-wide automatic
indexing, native Slack search, attachments/file bodies or combined indexed-plus-
live answers.

Connecting a Slack conversation as an Indexed Source does **not** activate the bot in that channel. Activating the bot in a channel does **not** automatically index channel history.

#### `LKW-CONVERSATION-CONTEXT-1`

**Status:** `PLANNED` (LKW application — LKW-wide, not platform Slack ownership)

Provider-neutral durable Conversation Context Bindings, observed-audience validation, workspace audience policy, conversation-level state versus thread-level memory, evidence guards and shared `READ_ONLY_ASK` capability boundary. Canonical architecture: [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](../../technical/applications/local_workspace_application/CONVERSATION_CONTEXT_ARCHITECTURE.md).

#### `LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1`

**Status:** `PLANNED` (LKW application — first provider adapter over generic context layer)

Slack channel/private-channel mention handling (`MENTION_ONLY` activation) over `LKW-CONVERSATION-CONTEXT-1`. Slack-specific event terms (`app_mention`, `message.channels`, etc.) remain in the adapter — not in the LKW core contract.

#### `SLACK-LIVE-CAPABILITY-1`

**Status:** `ACCEPTED / CLOSED` (platform)

Platform live path using the same integration and shared read primitives:

```text
validated capability
→ bounded Slack read
→ normalized ephemeral evidence
```

The accepted family publishes exactly three operations:

```text
vendor.slack.slack_conversation.list
  one recent-history call, at most 15 roots, bounded validated text
vendor.slack.slack_conversation.thread.read
  one reply page, at most 15 replies, bounded validated text
vendor.slack.slack_conversation.read
  one exact message, bounded typed text only when safe
```

Native workspace search, exhaustive history, full conversation/thread traversal,
authoritative permissions and file/attachment reads are unsupported or deferred
in live v1. No automatic durable persistence, indexing or provider cursor
persistence occurs.

**User-facing meaning after completion:** Authorized applications can read bounded current Slack information at request time without waiting for a complete durable synchronization.

#### `SLACK-LIVE-DISCOVERY-AND-ASK-READINESS-1`

**Status:** `ACCEPTED / CLOSED` (LKW application)

This closeout covers only bounded recent configured-channel Ask. Execution is
proven as two stages: list calls for resolved active bindings execute first;
normalized Slack list evidence is then filtered/ranked and at most three
thread roots are selected globally before binding-owned `thread.read` calls
execute through the shared live executor. Useful thread expansion does not
require callers to pre-supply timestamps; explicit timestamps remain priority
references only.

Resolution uses only active Slack live access bindings in the current
tenant/workspace and fails closed on unknown or ambiguous names. The shared
executor remains authoritative for every call, capability validation, budgets
and deadlines. Coverage is derived from actual attempted successful, failed and
truncated calls and exposes queried/skipped bindings, inspected roots,
thread/reply counts, provider calls, truncation and deterministic partial
reasons. Selected root and reply bodies remain transient normalized live
evidence and are not indexed or durably persisted.

Native Slack workspace search, exhaustive history, arbitrary token-accessible
channels and files/attachments are not implemented. Indexed and durable Slack
lifecycle claims remain separate and unchanged.

This is not native Slack workspace search and does not imply exhaustive history,
arbitrary token-accessible channel discovery, attachments/files, indexing or
durable live-result persistence. Commercial Slack support remains gated by the
existing commercial policy.

#### `LKW-SLACK-KNOWLEDGE-PROOF-1`

**Status:** `PLANNED` (LKW application)

**Prerequisites (join):** `LKW-SLACK-CONNECTED-SOURCE-1` + `LKW-CONVERSATION-CONTEXT-1` + `LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1` + `SLACK-LIVE-CAPABILITY-1` + `LKW-HYBRID-ASK-1`. Cannot claim indexed + live combined evidence before Hybrid Ask exists.

Required user proof:

```text
approved user selects a Slack conversation
→ binds it to an LKW workspace
→ synchronization and indexing complete
→ user asks through Slack
→ answer uses indexed Slack evidence
→ bounded live Slack evidence may be included when authorized
→ citations identify safe Slack message/thread provenance
```

**User-facing meaning after completion:** A user asking through Slack can receive one grounded answer combining Slack history, current authorized Slack evidence and other workspace sources — with strict personal/shared audience isolation per [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](../../technical/applications/local_workspace_application/CONVERSATION_CONTEXT_ARCHITECTURE.md).

---

### Phase 9 — Slack source management (frontend)

#### `LKW-SLACK-CONNECTED-SOURCES-1`

**Status:** `DEFERRED`

Add safe source discovery, selection, sync request and status through Slack. Slack remains a replaceable frontend and never receives credentials or unsafe provider locators.

---

### Phase 10 — Google Workspace Knowledge vertical (`GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1`)

**Architecture:** [`../../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md) §13.8. **Provider usage:** [`../../intergrax/integrations/providers/collaboration_suite/google_workspace/USAGE.md`](../../../../intergrax/integrations/providers/collaboration_suite/google_workspace/USAGE.md).

`GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1` is **READY_FOR_REVIEW**. All runtime tasks below are **PLANNED** — no Google knowledge read surface, Vendor Knowledge adapter, live capability or LKW Connected Source is implemented.

One existing `GoogleWorkspaceCollaborationSuiteIntegration` (`provider_id: google_workspace`, category: `collaboration_suite`) is reused across indexed RAG, durable materialization without RAG and bounded live access. Seven independently scoped source kinds: `drive`, `docs`, `sheets`, `calendar`, `slides`, `mail`, `chat`. Drive may discover all Drive-hosted resources; discovery does not determine durable `source_kind` — the platform derives canonical binding kind server-side. The frontend must not choose or override `source_kind`.

**Canonical durable resource ownership:**

| Google resource class | Canonical durable `source_kind` |
|---|---|
| Google-native document (Docs) | `docs` |
| Google-native spreadsheet (Sheets) | `sheets` |
| Google-native presentation (Slides) | `slides` |
| Ordinary uploaded/stored file | `drive` |
| Drive folder / My Drive / Shared Drive scope | `drive` |
| Google Calendar / calendar-event scope | `calendar` |
| Gmail scope | `mail` |
| Google Chat space / conversation scope | `chat` |

**Drive discovery flow:** Drive inventory → inspect authoritative Google resource type → derive canonical `source_kind` server-side → issue provider-neutral Remote Resource candidate → create only the canonical `KnowledgeSourceBinding`.

**Stable resource identity:** `provider_id = google_workspace` + `connection_ref` + canonical Google resource type + stable Google resource ID. Rename/move (where ID preserved) do not change identity. Export/download URL is never identity. The same native Google file must not become unrelated `drive` and `docs`/`sheets`/`slides` durable objects.

**Overlapping-binding policy (first proof):** explicit selected resources only; broad Drive/folder synchronization deferred. Future broad scopes require Option A (reject overlapping binding) or Option B (canonical deduplication record) — Option B not chosen until Vendor Knowledge and LKW ownership models support it safely; until then broad overlapping scopes fail closed or remain deferred.

**Product rationale:** Microsoft 365 proves enterprise-oriented collaboration and document access. Google Workspace lowers the entry barrier for individual testers, small teams and design partners who can authorize their own account. Supporting both proves that the LKW Connected Source architecture is provider-neutral rather than Microsoft-specific. The goal is not connector count — it is one convincing proof over different real-world source shapes and provider ecosystems. Google Workspace is the second strategic collaboration/document ecosystem, not an open-ended commitment to add every available SaaS provider.

#### `GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1`

**Status:** `PLANNED`

**Prerequisites (activation gates — not satisfied):**

```text
GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1 becomes ACCEPTED (currently READY_FOR_REVIEW)
Google Workspace runtime implementation starts only after LKW-SLACK-KNOWLEDGE-PROOF-1 becomes ACCEPTED (complete Slack Knowledge proof — currently PLANNED)
canonical Tenant Connection / credential-reference boundary available
SecretsStore-owned credential persistence available
runtime integration rehydration/resolution boundary available
Vendor Knowledge binding, registry and synchronization contracts available
```

Canonical owners: durable tenant Connection Catalog and runtime integration rehydration/resolution — `LKW-KNOWLEDGE-ACCESS-1`. Google Foundation must not introduce another tenant Connection catalog, a Google-only credential database, or OAuth tokens in provider config / bindings / LKW state.

Typed Google Workspace integration configuration; credential-reference resolution; least-privilege credential modes; one shared provider client family; provider request execution boundary; pagination token normalization; provider error normalization; rate-limit and retry classification; stable provider resource references; safe timestamps and revisions; safe display labels; bounded request limits; capability declaration; no LKW imports; no RAG imports; no application workspace concepts.

Credential routes (conceptually separated): individual-user OAuth (preferred for first proof); organization/admin-approved Google Workspace access; service-account or delegated organizational access when justified. Exact OAuth scopes and Google SDK signatures are verified in implementation against current official Google documentation — not frozen here. Secrets remain in Connection/SecretsStore; never in bindings, Remote Resources, LKW Sources, citations or cursors.

#### `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1`

**Status:** `PLANNED`

Extend the single existing `GoogleWorkspaceCollaborationSuiteIntegration` client boundary with typed read behavior per source kind. Substeps:

| Substep | Source kind | Status |
|---|---|---|
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1A-DRIVE` | `drive` | `PLANNED` |
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1B-DOCS` | `docs` | `PLANNED` |
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1C-SHEETS` | `sheets` | `PLANNED` |
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1D-CALENDAR` | `calendar` | `PLANNED` |
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1E-SLIDES` | `slides` | `PLANNED` (post-proof) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1F-MAIL` | `mail` | `PLANNED` (post-proof) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1G-CHAT` | `chat` | `PLANNED` (post-proof) |

Every read-surface task must define: resource scope; stable identity; revision/change identity; full inventory support; incremental/change-feed support; reconciliation support; exact read support; content mode; attachment behavior; permission capability; tombstone/deletion semantics; pagination/cursor behavior; bounded limits; safe provider errors. Do not claim a capability merely because Google exposes some related API method.

#### `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1`

**Status:** `PLANNED`

Each adapter substep pairs with its read surface and must be independently reviewed before the next surface begins:

| Substep | Source kind | Status |
|---|---|---|
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1A-DRIVE` | `drive` | `PLANNED` (proof-critical) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1B-DOCS` | `docs` | `PLANNED` (proof-critical) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1C-SHEETS` | `sheets` | `PLANNED` (proof-critical) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1D-CALENDAR` | `calendar` | `PLANNED` (proof-critical) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1E-SLIDES` | `slides` | `PLANNED` (post-proof) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1F-MAIL` | `mail` | `PLANNED` (post-proof) |
| `GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1G-CHAT` | `chat` | `PLANNED` (post-proof) |

Thin adapters over the same resolved `GoogleWorkspaceCollaborationSuiteIntegration`:

```text
GoogleWorkspaceDriveKnowledgeAdapter      (1A — proof-critical)
GoogleWorkspaceDocsKnowledgeAdapter       (1B — proof-critical)
GoogleWorkspaceSheetsKnowledgeAdapter     (1C — proof-critical)
GoogleWorkspaceCalendarKnowledgeAdapter   (1D — proof-critical)
GoogleWorkspaceSlidesKnowledgeAdapter     (1E — post-proof)
GoogleWorkspaceMailKnowledgeAdapter       (1F — post-proof)
GoogleWorkspaceChatKnowledgeAdapter       (1G — post-proof)
```

Registry keys:

```text
(google_workspace, collaboration_suite, drive)
(google_workspace, collaboration_suite, docs)
(google_workspace, collaboration_suite, sheets)
(google_workspace, collaboration_suite, calendar)
(google_workspace, collaboration_suite, slides)
(google_workspace, collaboration_suite, mail)
(google_workspace, collaboration_suite, chat)
```

Each adapter receives the resolved integration; owns no client, credentials, persistence, checkpoint or retry runtime; uses the shared synchronization coordinator; remains independent from LKW.

#### `LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1`

**Status:** `PLANNED`

**Prerequisites:** proof-critical Google read surfaces and adapters (Drive, Docs, Sheets, Calendar).

Target flow:

```text
workspace Connection
→ Google Workspace Remote Resource discovery
→ selected Drive / Docs / Sheets / Calendar resource
→ tenant KnowledgeSourceBinding
→ WorkspaceIndexedSourceBinding
→ Vendor Knowledge synchronization
→ existing LKW materialization/indexing pipeline
→ Search → Ask → citations
```

Reuses the generic Connected Source implementation proved by Slack. No Google-specific LKW configuration aggregate, mutation engine, indexing pipeline, vector database access or Source table. First Google LKW sources default to `PERSONAL_ONLY`.

#### `LKW-GOOGLE-WORKSPACE-PROOF-1`

**Status:** `PLANNED`

**Prerequisites:** `LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1` + proof-critical adapters + LKW indexed Search/Ask path.

Required first proof:

```text
user connects one Google account
→ selects approved Google resources
→ one Google Doc synchronized
→ one Google Sheet synchronized
→ one Google Calendar resource/event set synchronized
→ optionally one ordinary Drive file synchronized
→ LKW indexes the selected resources
→ Search retrieves provider-derived evidence
→ Ask produces one grounded answer
→ citations identify the correct Google source and resource
→ no Google API call is made by Ask after durable synchronization
```

User-oriented proof demonstrating narrative document, structured spreadsheet, calendar/event data and ordinary stored file — not merely adapter unit tests.

**Proof-first gate (binding — vertically incremental):**

```text
GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1

→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1A-DRIVE
→ GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1A-DRIVE
→ Drive contract/integration proof

→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1B-DOCS
→ GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1B-DOCS
→ Docs contract/integration proof

→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1C-SHEETS
→ GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1C-SHEETS
→ Sheets contract/integration proof

→ GOOGLE-WORKSPACE-KNOWLEDGE-READ-SURFACE-1D-CALENDAR
→ GOOGLE-WORKSPACE-KNOWLEDGE-ADAPTERS-1D-CALENDAR
→ Calendar contract/integration proof

→ LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1
→ LKW-GOOGLE-WORKSPACE-PROOF-1
→ read surfaces 1E–1G + adapters 1E–1G
```

Google Workspace does not gate reconciliation finalization or Microsoft Calendar acceptance. Microsoft Calendar work does not gate the independent Google Workspace workstream.

Each read surface and its adapter must be independently reviewable before proceeding. The final Google LKW proof may combine Docs, Sheets, Calendar and an optional ordinary Drive file.

---

## Phase 9 — Unified Vendor Knowledge live capability rollout

### `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-PLAN-1`

**Status:** `ACCEPTED / CLOSED`
**Scope:** planning only; no runtime implementation is activated by this task.

This rollout makes one shared execution boundary authoritative for all Vendor
Knowledge live capabilities:

```text
LiveCapabilityDescriptorV1
→ tenant-safe capability catalog
→ durable Live Access Binding lifecycle
→ exact immutable LiveCapabilityHandlerV1 registry
→ validated provider-neutral executor
→ bounded normalized live evidence
→ receipt-only retention
```

The rollout covers Microsoft Graph, Slack, Jira, Confluence and Google
Workspace. All source kinds without a production handler remain
`FOUNDATION_ONLY`; the existing generic live foundation is not provider
production proof.

#### Binding ownership split

The Google Workspace core workstream owns:

```text
provider integration primitives
source-specific read surfaces
provider pagination and cursors
source-specific typed models
Vendor Knowledge durable adapters
source-specific adapter tests
Google family implementation closeout
```

This Vendor Knowledge live rollout owns:

```text
LiveCapabilityDescriptorV1 declarations
LiveCapabilityHandlerV1 implementations
provider-neutral request/result contracts
handler registration
capability catalog registration
effective budgets
timeouts
provider error normalization
normalized live evidence
safe locators
receipt behavior
retention behavior
cross-provider contract tests
live family closeout
```

The Google core workstream must not build a second live executor, a
Google-specific live registry, a Google-specific receipt mechanism or a direct
LKW-to-Google execution path. This rollout must not reimplement Google
clients, authentication, provider transport, pagination, source-specific read
primitives or durable adapters.

#### Shared foundation status

```text
current production foundation:
  LiveCapabilityDescriptorV1
  tenant-safe capability catalog
  durable Live Access Binding lifecycle
  evidence-plan validation
  LiveCapabilityHandlerV1 protocol
  exact handler registry
  provider-neutral executor
  connection integration resolver
  basic item/byte budgets
  normalized result/evidence models
  receipt-only retention

ARCH-1 frozen shared delta: implemented as the FOUNDATION-1 review boundary
  canonical source_kind assertion and validation
  contract_version across descriptor/handler/request/result/binding
  strict capability-specific request models
  request/result schema resolution
  ValidatedLiveCapabilityCallV1 or equivalent typed call
  atomic descriptor-handler-schema registration
  missing-pair and duplicate-pair validation
  provider page/request/upstream/content budgets
  expanded provider-neutral error taxonomy
  source-kind-aware result/evidence provenance
  ordered item-identity-aware receipt hashing
  safe-locator validation/filtering
  shared registration bootstrap
  shared contract test suite

provider-specific production handlers:
  implemented for Microsoft Graph Drive, Mail, Teams Channel, Teams Chat and Calendar list

provider-specific production registrations:
  implemented for Microsoft Graph Drive, Mail, Teams Channel, Teams Chat and Calendar list

all other provider/source-kind live handlers and registrations:
  not implemented

cross-provider production proof: not implemented
```

#### Source-kind coverage

```text
Microsoft Graph: drive, mail, teams_channel, teams_chat, calendar
Slack:          slack_conversation
Jira:           issues
Confluence:     pages
Google Workspace: drive, docs, sheets, calendar, slides, mail, chat
```

There are 15 planned source kinds. Databricks remains excluded until one exact
source kind is selected. Power BI and Atlan remain outside this rollout unless
separately activated.

### `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-ARCH-1`

**Status:** `READY_FOR_REVIEW`
**Activation:** only after this plan is accepted and external review is
complete.

This architecture task freezes one reusable pattern for every provider,
including Google Workspace:

```text
capability naming
search/list versus exact read
request schemas
result schemas
connection scope
remote-resource scope
audience enforcement
effective budgets
timeouts
provider errors
normalized evidence
safe locators
receipts
retention
descriptor registration
handler registration
contract tests
family closeouts
readiness gates
```

It prohibits provider-specific live frameworks, direct application-to-provider
calls, duplicate credential resolution, duplicate clients, unbounded search,
write/admin capabilities and raw provider payload exposure. Every provider
uses the same executor, registry, normalized evidence and receipt boundary.

### `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1`

**Status:** `ACCEPTED / CLOSED`

**Dependency:** `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-ARCH-1` remains
`READY_FOR_REVIEW`; this foundation task is complete and its boundary is used
by the accepted five-capability Microsoft Graph family.

This task implements only the provider-neutral runtime delta frozen by
`ARCH-1`:

```text
descriptor identity/version/source-kind support
strict schema registry/resolution
typed validated call envelope
descriptor-handler-schema registration bundle
atomic registration validation
effective provider-call budgets
shared error taxonomy
normalized provenance additions
safe-locator enforcement
receipt contract and ordered identity-aware hashing
shared contract tests
bootstrap integration needed by all provider families
```

It must not implement Microsoft Graph, Slack, Jira, Confluence or Google
Workspace handlers, provider SDK calls, application UI/LKW behavior, durable
synchronization, indexing, or provider credentials/clients in handlers.

**Foundation acceptance gates:**

```text
canonical capability identity parser/validator; source_kind agreement across
ID/descriptor/handler; immutable contract version; strict pre-invocation
request validation with unknown-field rejection; resolved request/result
schemas; typed validated call; atomic publication; missing-pair, schema
mismatch and duplicate rejection; finite page/request/upstream-item/content
budgets; executor-owned timeout/output limits; stable secret-free errors;
provider/source/capability/version/binding provenance; unsafe-locator
removal/rejection; ordered item-identity-aware receipt hashing; only
EPHEMERAL and RECEIPT_ONLY retention; no automatic indexed-corpus writes; no
provider clients/credentials in handlers; focused shared contract tests
without external credentials.
```

### Provider task boundary

After `FOUNDATION-1` is accepted, each provider/source-kind task owns only:

```text
operation availability matrix
source-specific strict request models
canonical descriptors
source-specific handlers
provider read primitive mapping
bounded provider invocation
provider error mapping
normalized result mapping
safe locator mapping
provider-focused tests
registration into the shared boundary
```

Provider tasks must not redefine or privately alter shared identity,
versioning, schema, registration, budget, error, provenance, locator, receipt,
retention or executor semantics. No provider task is activated by this review
fix.

### Canonical rollout order

```text
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-PLAN-1
→ VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-ARCH-1
→ VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1

→ MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1A-DRIVE  # ACCEPTED / CLOSED
→ MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1B-MAIL  # ACCEPTED / CLOSED
→ MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1C-TEAMS-CHANNEL  # ACCEPTED / CLOSED
→ MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1D-TEAMS-CHAT  # ACCEPTED / CLOSED
→ MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1E-CALENDAR  # ACCEPTED / CLOSED
→ MSGRAPH-KNOWLEDGE-LIVE-CAPABILITIES-1-FAMILY-CLOSEOUT  # ACCEPTED / CLOSED

→ SLACK-KNOWLEDGE-LIVE-CAPABILITY-1
→ JIRA-KNOWLEDGE-LIVE-CAPABILITY-1
→ CONFLUENCE-KNOWLEDGE-LIVE-CAPABILITY-1

→ GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-READINESS-GATE-1

→ GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-CAPABILITY-1A-DRIVE
→ GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-CAPABILITY-1B-DOCS
→ GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-CAPABILITY-1C-SHEETS
→ GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-CAPABILITY-1D-CALENDAR
→ GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-CAPABILITY-1E-SLIDES
→ GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-CAPABILITY-1F-MAIL
→ GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-CAPABILITY-1G-CHAT
→ GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-CAPABILITIES-1-FAMILY-CLOSEOUT

→ VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FAMILY-AUDIT-1
```

These are planned tasks, not activation claims. No implementation task may be
activated before external review of this plan.

### `GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-READINESS-GATE-1`

**Status:** `PLANNED`
**Rule:** readiness is evaluated independently for each exact source kind.

The gate inspects `drive`, `docs`, `sheets`, `calendar`, `slides`, `mail` and
`chat` separately. Each source kind requires evidence of:

```text
stable source_kind identity
shared Google Workspace integration reuse
typed source-specific read surface
bounded provider read
safe provider references
stable remote item identity
provider error normalization
no secret-bearing public models
Vendor Knowledge adapter availability where required
focused source-specific tests
current task/review status
```

Possible per-source outcomes:

```text
READY_FOR_LIVE_ROLLOUT
BLOCKED_BY_CORE_READ_SURFACE
BLOCKED_BY_ADAPTER
BLOCKED_BY_PROVIDER_SEMANTICS
BLOCKED_BY_REVIEW
```

One ready source kind never proves readiness for the Google family. A Google
live task is active only when its exact source kind is
`READY_FOR_LIVE_ROLLOUT`; unfinished source kinds do not block ready ones.
The Google family closeout remains blocked until every source kind in the
accepted Google core family scope has either an accepted live implementation
or an explicit deferred/excluded decision.

### Provider live tasks and family closeouts

Each provider task implements only its source-specific handler and registration
over the frozen shared boundary. Each family closeout verifies:

```text
one shared provider integration
no duplicate provider clients
tenant-safe connection resolution
exact source-kind isolation
read-only enforcement
resource-scope enforcement
bounded request and result behavior
timeout behavior
provider error normalization
normalized evidence
safe locator behavior
receipt privacy
credential non-disclosure
contract tests
production proof
```

The Google family closeout additionally verifies reuse of the existing shared
`GoogleWorkspaceCollaborationSuiteIntegration`, with no separate Drive, Docs,
Sheets, Calendar, Slides, Mail or Chat clients.

### `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FAMILY-AUDIT-1`

**Status:** `PLANNED`

The final audit produces a matrix for every gated source kind and every
explicitly deferred source kind with its reason. Required columns are:

```text
provider
source_kind
capability_id
search/list support
exact-read support
resource scope
request schema
result schema
timeout
item budget
byte budget
evidence mapping
safe locator
receipt behavior
retention
descriptor registration
handler registration
proof status
commercial status
```

### Status after this task

```text
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-PLAN-1: ACCEPTED / CLOSED
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-ARCH-1: READY_FOR_REVIEW
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1: ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1A-DRIVE: ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1B-MAIL: ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1C-TEAMS-CHANNEL: ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1D-TEAMS-CHAT: ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1E-CALENDAR: ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITIES-1-FAMILY-CLOSEOUT: ACCEPTED / CLOSED
other Microsoft Graph live tasks: PLANNED
Slack live family: ACCEPTED / CLOSED
Jira live task: PLANNED
Confluence live task: PLANNED
GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-READINESS-GATE-1: PLANNED
Google source-kind live tasks: PLANNED / GATED_BY_CORE_READINESS
Google live family closeout: PLANNED
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FAMILY-AUDIT-1: PLANNED
```
