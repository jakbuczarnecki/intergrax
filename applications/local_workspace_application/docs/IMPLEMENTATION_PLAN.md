# Local Workspace Application — Implementation Plan

**Status:** Product-first MVP roadmap (2026-07-31)  
**Governing product rule:** [`PRODUCT_FIRST_MVP.md`](../../../docs/project/maintainers/plans/PRODUCT_FIRST_MVP.md)
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Ask Workspace discovery:** [`ASK_WORKSPACE_DISCOVERY.md`](ASK_WORKSPACE_DISCOVERY.md)  
**Slack MVP discovery:** [`SLACK_MVP_DISCOVERY.md`](SLACK_MVP_DISCOVERY.md)  
**Conversation context architecture:** [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](CONVERSATION_CONTEXT_ARCHITECTURE.md)
**Knowledge Intake discovery:** [`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md)  
**Hybrid knowledge access:** [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md)
**Hybrid Ask architecture:** [`HYBRID_ASK_ARCHITECTURE.md`](HYBRID_ASK_ARCHITECTURE.md)
**External verification:** [`LKW_PLATFORM_PROOF.md`](proof/LKW_PLATFORM_PROOF.md)
**Historical full plan:** [`archive/IMPLEMENTATION_PLAN_2026-07-22.md`](archive/IMPLEMENTATION_PLAN_2026-07-22.md)

## AUTHORITATIVE CURRENT STATUS

```text
CURRENT PRODUCT LEVEL
Backend Product Alpha — LKW MVP / Hybrid Knowledge Workspace

LAST ACCEPTED PRODUCT BLOCK
LKW-CONVERSATION-CONTEXT-1C — ACCEPTED / CLOSED
LKW-CONVERSATIONAL-FRONTEND-1 — ACCEPTED / CLOSED
Supporting accepted blocks:
  LKW-KNOWLEDGE-ACCESS-1 — ACCEPTED / CLOSED
  LKW-HYBRID-ASK-1 — ACCEPTED / CLOSED
  LKW-HYBRID-ASK-1A — ACCEPTED / CLOSED
  LKW-HYBRID-ASK-1B — ACCEPTED / CLOSED
  LKW-HYBRID-ASK-1C — ACCEPTED / CLOSED
  LKW-CONVERSATIONAL-INTERACTION-1A — ACCEPTED / CLOSED
  LKW-CONVERSATIONAL-INTERACTION-1B — ACCEPTED / CLOSED
  LKW-CONVERSATIONAL-INTERACTION-1C — ACCEPTED / CLOSED
  LKW-CONVERSATION-CONTEXT-1B2 — ACCEPTED / CLOSED
  LKW-CONVERSATION-CONTEXT-1B3 — ACCEPTED / CLOSED

CURRENT DIRECT LKW TASK
LKW-PLUGIN-CAPABILITY-CONFIGURATION-1 — READY_FOR_REVIEW

NEXT DIRECT LKW TASKS
LKW-INDEXED-SOURCE-LIFECYCLE-1 — PLANNED

PARALLEL NON-BLOCKING PLATFORM/PLUGIN TRACKS
Vendor Knowledge core session; individual vendor-plugin sessions;
Slack knowledge-source and Slack live-capability tracks; future integrations.

FINAL LKW 1.0 TARGET
LKW product completion plus one plugin-neutral Intergrax platform proof
and a bounded problem radar. The full vendor catalog is not required.
```

## LKW PRODUCT OWNERSHIP

LKW owns the product meaning and user-visible lifecycle for:

- workspaces;
- product-level knowledge configuration;
- natural-language operation;
- Conversation Context consumption;
- Indexed Source attachment and lifecycle from the product perspective;
- Live Access Binding configuration;
- Query Policy;
- Hybrid Ask orchestration;
- user-visible synchronization and operation states;
- freshness;
- provenance inspection;
- safe detach and local removal;
- product recovery behavior;
- frontend behavior;
- deployment and product hardening.

LKW consumes registered provider-neutral capabilities. It does not own the
implementation details of any particular external vendor.

## VENDOR KNOWLEDGE CORE OWNERSHIP

The shared Vendor Knowledge engine owns:

- provider-neutral plugin registration;
- capability discovery;
- canonical Connections;
- canonical Remote Resources;
- provider-neutral descriptors;
- provider dispatch;
- normalized durable delivery;
- normalized live capability execution;
- reconciliation contracts;
- checkpoints and provider-neutral errors;
- communication usable by LKW or any other Intergrax application.

## VENDOR PLUGIN OWNERSHIP

Individual vendor sessions own:

- provider authentication;
- provider APIs;
- rate limits;
- pagination;
- token refresh;
- vendor-specific cursors;
- vendor-specific resource discovery;
- vendor-specific change semantics;
- mapping into Vendor Knowledge canonical contracts.

Examples include Slack, Google Workspace, Microsoft 365, Jira, Confluence,
Databricks, Power BI, Atlan and future plugins. This is an open example set,
not an exhaustive or hard-coded catalog.

## PLUGIN-NEUTRAL LKW INVARIANTS

```text
LKW must not branch on vendor identity.
LKW must not contain a fixed catalog of vendor implementations.
LKW must consume only provider-neutral registered capabilities.
A missing individual vendor plugin is not an LKW blocker.
LKW may use any conforming real, proof or deterministic fixture plugin.
Vendor-specific behavior must terminate at the Vendor Knowledge boundary.
Adding a new vendor must not require changes to the LKW domain model.
Removing a vendor plugin must not invalidate unrelated LKW functionality.
Different plugins may expose different supported capability subsets.
LKW must discover supported capabilities instead of assuming them.
```

A genuine LKW blocker exists only when the required provider-neutral contract
is missing, contradictory or unsafe. When that happens, the missing boundary
belongs to the Vendor Knowledge core roadmap, not to an improvised
vendor-specific implementation inside LKW.

---

## CURRENT PRODUCT CAPABILITY

The currently accepted product description is:

- Slack personal DM uses Conversation Context and bounded durable thread memory.
- Natural-language messages use the accepted planner and deterministic executor.
- Workspace selection is durable.
- The Slack DM path uses one planner call, one executor call and one bounded deterministic response.
- Event processing has durable CAS-backed idempotency.
- A completed interaction appends one safe user/assistant exchange through the
  shared durable memory lifecycle; retries reuse the event receipt.
- Files, trusted attachments, local sources and Web URLs use existing LKW-owned intake boundaries.
- Hybrid Ask provider-neutral orchestration is accepted.
- Shared Slack channels remain outside the currently accepted frontend scope.

Slack conversational frontend acceptance does not claim that Slack Knowledge
indexing or Slack live evidence is complete. Those are separate integration
tracks.

## Protocol v2 LKW product proof remediation (2026-08-18)

Accepted [`LKW_PRODUCT_PROOF`](../../../docs/audit_results/2026-08-18/LKW_PRODUCT_PROOF.md) findings **01–06** (2026-08-21). All blocks below are **ACCEPTED / PLANNED** — not IMPLEMENTED, VERIFIED, or CLOSED.

### LKW-PROOF-EXECUTION-QUALIFICATION-INTEGRITY — P0/P1

**Findings:** LKW-PROOF-04, LKW-PROOF-05  
**Owner:** shared proof manifest/runner (`scripts/proof/`); LKW consumes manifest authority  

- Explicit required vs optional proof membership in canonical manifest.
- `--profile live` must not return shell success when required live proofs were blocked (`PASS_WITH_BLOCKED` limitation until remediated).
- Fold Governed Evidence Decision Proof (`advanced_flagship_proof`) into canonical `ProofManifestEntry` and public reference governance — no second flagship proof path.

### LKW-PROOF-REVIEWER-SEMANTICS-INTEGRITY — P2

**Findings:** LKW-PROOF-06  
**Owner:** LKW plan + [`PROOFS.md`](../../../docs/project/proofs/PROOFS.md)  

- Make proof profile semantics explicit: `--profile quick` does **not** execute Product Quick Start today.
- Either add bounded flagship LKW product smoke to QUICK or rename/document profiles so QUICK cannot be read as Product Quick Start coverage.

### Cross-link — LKW-PROOF-SOURCE-PROVENANCE-INTEGRITY — P0

**Findings:** LKW-PROOF-01, LKW-PROOF-02, LKW-PROOF-03  
**Primary owner:** [`docs/project/maintainers/plans/PROOF_RECEIPTS.md`](../../../docs/project/maintainers/plans/PROOF_RECEIPTS.md) — LKW consumes shared receipt/provenance authority; does not invent private receipt contract.

This section does **not** change the current direct LKW roadmap task (`LKW-PLUGIN-CAPABILITY-CONFIGURATION-1`) or falsely mark Product 1.0 work complete.

## DIRECT ROADMAP TO LKW 1.0

This is the one authoritative active execution order. It contains only work
directly required for the first complete LKW product and the final Intergrax
platform proof. `PLANNED` means not accepted; it does not claim that
implementation has started.

### A. Conversation Context completion

1. `LKW-CONVERSATION-CONTEXT-1B2` — **ACCEPTED / CLOSED**.
2. `LKW-CONVERSATION-CONTEXT-1C` — **ACCEPTED / CLOSED**: integrate bounded
   durable thread-memory reconstruction and exactly-once exchange persistence.

Required outcome:

- durable reconstruction;
- bounded thread context with empty-memory behavior when no valid snapshot exists;
- personal/shared isolation;
- fail-closed unsupported audiences;
- no vendor dependency.

### B. Generic plugin capability consumption

`LKW-PLUGIN-CAPABILITY-CONFIGURATION-1` — **READY_FOR_REVIEW**.

The product must be able to:

- list available provider-neutral connection capabilities;
- list configured Connections;
- discover Remote Resources;
- expose supported capability subsets;
- create Indexed Sources;
- create Live Access Bindings;
- configure Query Policy;
- disable or detach configuration;
- perform all of the above without vendor branching.

This extends the accepted `LKW-KNOWLEDGE-ACCESS-1` foundation where the
remaining product behavior is not already proven. It must not be marked
implemented unless the complete outcome is independently accepted.

Implementation status: **READY_FOR_REVIEW** for the read-only configuration
discovery slice. LKW consumes the registered provider-neutral Tenant Connection,
Remote Resource discovery and capability catalog boundaries. Discovery is
dynamic: adding or removing a conforming plugin changes the safe snapshot
without LKW domain changes or a named-vendor branch.

This slice exposes bounded personal conversational inspection only:
connections, remote resources and registered capabilities. It does not create
Indexed Sources, execute Live Access, mutate Query Policy or persist discovered
resources. Indexed Source lifecycle is the next direct task
(`LKW-INDEXED-SOURCE-LIFECYCLE-1`); Live Access execution remains a later task.
The shared core currently does not expose a generic durable/indexed eligibility
descriptor, so LKW reports that dimension as `UNKNOWN` rather than inferring it
from capability IDs or source kinds. This is a bounded Vendor Knowledge
problem-radar finding, not an LKW vendor blocker.

### C. Generic Indexed Source lifecycle

`LKW-INDEXED-SOURCE-LIFECYCLE-1` — **PLANNED**.

The direct product lifecycle covers:

- attach;
- initial synchronization;
- incremental synchronization;
- progress and status;
- retry;
- crash recovery;
- freshness;
- disable;
- detach;
- safe local removal;
- no deletion of upstream data.

Provider-specific cursors and API behavior belong to Vendor Knowledge or the
plugin. The LKW lifecycle consumes normalized operations and statuses.

### D. Generic Live Access lifecycle

`LKW-LIVE-ACCESS-LIFECYCLE-1` — **PLANNED**.

The direct product lifecycle covers:

- capability discovery;
- policy validation;
- bounded live invocation;
- timeout;
- cancellation;
- safe error normalization;
- provenance;
- no automatic persistence of live evidence;
- compatibility with Hybrid Ask.

A specific Google, Microsoft, Jira or other implementation is not required.
Live execution is selected from discovered provider-neutral capabilities.

### E. Unified configuration and inspection

`LKW-KNOWLEDGE-INSPECTION-AND-OPERATIONS-1` — **PLANNED**.

The product must answer and operate on questions such as:

```text
What sources does this workspace have?
Which are indexed, live or both?
Which are fresh?
Which failed?
When were they last synchronized?
Which policies apply?
Can this source be disabled, retried or detached?
```

Inspection must expose safe status, freshness and provenance without secrets,
private locators or provider-specific business logic.

### F. Natural-language administration completion

`LKW-NATURAL-LANGUAGE-ADMINISTRATION-1` — **PLANNED**.

Extend the accepted planner/executor vocabulary to generic plugin operations:

```text
list integrations
list connections
discover resources
attach resource as indexed
enable live access
configure query policy
show operation status
retry operation
disable synchronization
detach source
```

The planner and executor operate on provider-neutral contracts only. They do
not infer a fixed vendor catalog or implement vendor-specific actions.

### G. Product hardening

`LKW-PRODUCT-HARDENING-1` — **PLANNED**.

Minimum scope:

- bounded concurrency;
- timeouts;
- cancellation;
- idempotency;
- retries;
- restart recovery;
- migrations;
- health and readiness;
- observability;
- safe logging;
- auditability;
- retention;
- backup/restore boundary;
- secure defaults;
- model-runtime failure handling.

### H. Deployment and onboarding

`LKW-DEPLOYMENT-AND-OPERATIONS-1` — **PLANNED**.

Minimum scope:

- canonical self-hosted deployment;
- required stores;
- Ollama/vLLM configuration;
- Slack application configuration;
- secret handling;
- first-tenant bootstrap;
- first workspace;
- first source;
- upgrade;
- rollback;
- operator documentation.

### I. Product acceptance and platform proof

`LKW-PRODUCT-ACCEPTANCE-PROOF-1` — **PLANNED**.

Required proof:

```text
one tenant
→ Slack personal DM
→ durable Conversation Context
→ create and select workspace
→ add file or trusted attachment
→ add Web URL
→ discover one conforming plugin capability
→ configure one durable Indexed Source
→ complete and reconstruct synchronization after restart
→ execute one bounded live capability
→ Hybrid Ask combines indexed and live evidence
→ unified provenance and citations
→ natural-language administration
→ duplicate events do not duplicate mutations
→ plugin replacement does not alter LKW domain behavior
→ system exposes useful platform problems through safe diagnostics
```

The proof may use one accepted real plugin, one controlled proof plugin or one
deterministic conforming fixture. It must not require every planned vendor.

### J. Release gate

`LKW-1.0-RELEASE-GATE` — **PLANNED**.

The gate requires:

- all direct product blocks accepted;
- deployment documentation;
- restart/recovery proof;
- security and isolation proof;
- no hard-coded vendor dependencies;
- known platform gaps recorded as problem-radar findings;
- no requirement that the full vendor catalog be complete.

## PARALLEL PLATFORM AND PLUGIN ROADMAPS — NOT LKW RELEASE BLOCKERS

These tracks remain valid and useful, but they are outside the direct LKW 1.0
execution order. They run in separate sessions at two levels:

```text
Vendor Knowledge core session
individual vendor-plugin sessions
```

### Vendor Knowledge core track

The core track owns provider-neutral registration, capability discovery,
canonical Connections and Remote Resources, descriptors, dispatch, normalized
durable delivery, normalized live execution, reconciliation, checkpoints and
provider-neutral errors.

### Slack vertical and proof track

Slack has three distinct roles:

```text
Slack conversational frontend
Slack durable/indexed knowledge source
Slack bounded live capability
```

The Slack DM conversational frontend is `LKW-CONVERSATIONAL-FRONTEND-1`,
**ACCEPTED / CLOSED**. `SLACK-KNOWLEDGE-FOUNDATION-1`,
`LKW-SLACK-CONNECTED-SOURCE-1`, `SLACK-LIVE-CAPABILITY-1` and
`LKW-SLACK-KNOWLEDGE-PROOF-1` concern the separate knowledge-source and live
capability tracks. `LKW-SLACK-CONNECTED-SOURCE-1` is not an unavoidable
prerequisite for every LKW product task. It may be the preferred real plugin
proof when available, but any conforming plugin can satisfy the direct proof.

### Other plugin tracks

The following remain parallel and non-blocking:

- `GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1`;
- Google Drive / Docs / Sheets / Calendar adapters;
- `LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1` and
  `LKW-GOOGLE-WORKSPACE-PROOF-1`;
- Microsoft Graph Mail / Teams / OneDrive / SharePoint / Calendar work;
- Jira;
- Confluence;
- Databricks;
- Power BI;
- Atlan;
- other vendor-specific implementations.

These tracks may strengthen product proofs. The absence of a particular
vendor does not block LKW 1.0, and vendor-specific work must not be moved into
the LKW domain model.

## LKW AS INTERGRAX PLATFORM PROOF AND PROBLEM RADAR

LKW serves two purposes:

1. first usable Intergrax product;
2. real application that reveals missing or weak platform boundaries.

```text
LKW may expose a missing platform capability through a bounded finding.
It must not absorb platform or vendor responsibilities merely to avoid the finding.
```

Problem-radar findings should record:

- missing provider-neutral contracts;
- unsafe lifecycle gaps;
- missing composition support;
- insufficient recovery;
- leaky vendor abstractions;
- deployment friction;
- observability gaps;
- token/runtime inefficiencies.

Findings are routed to the appropriate Vendor Knowledge core, plugin or
platform roadmap/session. They do not automatically expand the current LKW
task.

---

## HISTORICAL IMPLEMENTATION DETAIL — NOT CURRENT STATUS

The sections below preserve implementation evidence, earlier decomposition and
historical task identity for traceability. They do not define the current
execution order or current product status. The authoritative status and
roadmap are the sections above.

## 1. Historical document role and source of truth

At the time, this file recorded:

- the active LKW execution order;
- the current next implementation block;
- the Workspace Contents source-expansion roadmap;
- the integration boundary between LKW and VENDOR-KNOWLEDGE;
- the planned synchronization, inspection and removal lifecycle;
- the post-MVP direction toward LKW 1.0.

This historical record does not override the authoritative current status or
direct roadmap above. The archived plan separately preserves the earlier full
product brief, implementation gates, proof-portability notes and milestones.

### Governing rule

```text
Deliver the smallest real product experience that demonstrates user value.
Use implementation of that product to discover and improve Intergrax.
Do not build the platform first and hope a useful product appears later.
```

Every implementation slice is preceded by a bounded architecture review. Cursor receives the accepted architecture and a precise implementation scope, not an open-ended platform audit.

---

## 2. Historical product direction

LKW is a **private-by-default, tenant-scoped, deployment-neutral Hybrid Knowledge Workspace**. It lets a user attach controlled knowledge sources (indexed), authorize bounded live provider reads, process indexed knowledge durably, ask natural-language questions and receive grounded answers with inspectable provenance from indexed evidence, live evidence, or both.

Binding architecture: [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md).

“Local” means user-controlled deployment and configuration, including first-class self-hosted topology. It does not mean that every source must be a local folder or that all upstream data must physically live on one device.

The knowledge portfolio grows through one LKW-owned lifecycle for **indexed** knowledge and a separate, governed path for **live** access:

```text
managed files and channel attachments
→ preconfigured local folders
→ explicit Web URLs
→ organizational vendor systems (indexed and/or live)
→ Workspace Knowledge Configuration (Indexed Sources + Live Access Bindings + Query Policy)
→ grounded Hybrid Ask with unified provenance
```

Target vendor-backed knowledge includes, without making all providers one implementation task:

- Microsoft 365, Outlook and Teams knowledge;
- OneDrive and SharePoint files;
- Jira issues and Confluence pages;
- Databricks notebooks, datasets and related knowledge assets;
- Atlan catalog assets;
- Power BI reports, dashboards and semantic knowledge;
- e-mail, chats, documents, calendars, meeting material and future organizational sources.

Microsoft Teams in vendor-access blocks means Teams-hosted organizational knowledge. It does not automatically add Microsoft Teams as a second conversational frontend.

---

## 3. Superseded product roadmap (historical)

Canonical execution order:

```text
COMPLETED / ACCEPTED:
LKW-WORKSPACE-CONTENTS-1B-5-2
→ END-TO-END WEB_URL KNOWLEDGE INTAKE
→ including accepted C1 and C2 corrections

LKW-MODEL-RUNTIME-1
→ OLLAMA / vLLM END-TO-END PORTABILITY
→ accepted with full canonical proof and evidence v2

COMPLETED / ACCEPTED:
LKW-KNOWLEDGE-ACCESS-1
→ WORKSPACE CONNECTIONS, INDEXED SOURCES AND LIVE ACCESS CONFIGURATION

CURRENT ARCHITECTURE REVIEW:
LKW-HYBRID-ASK-ARCH-1
→ ACCEPTED / CLOSED

NEXT IMPLEMENTATION:
LKW-HYBRID-ASK-1
→ IN_PROGRESS
→ RAG + LIVE KNOWLEDGE QUERY WITH UNIFIED PROVENANCE
  1A — provider-neutral core contracts, Query Policy V2, Evidence Plan validation — ACCEPTED / CLOSED
  1B — provider-neutral live execution + Knowledge Query orchestration — ACCEPTED / CLOSED
  1C — Workspace Ask integration, HTTP V2, bounded acceptance proof — READY_FOR_REVIEW

LKW-CONVERSATIONAL-FRONTEND-1
→ NATURAL-LANGUAGE EXECUTION AND LIVE SLACK CUTOVER
    ├── internal: LKW-CONVERSATIONAL-INTERACTION-1B (resolver + executor)
    └── internal: LKW-CONVERSATIONAL-INTERACTION-1C (Slack mixed-message cutover)

LKW-VENDOR-ACCESS-COLLABORATION-1
→ MICROSOFT 365 + JIRA + CONFLUENCE INDEXED AND LIVE ACCESS

LKW-VENDOR-ACCESS-DATA-1
→ DATABRICKS + POWER BI + ATLAN READ-ONLY ACCESS

LKW-KNOWLEDGE-LIFECYCLE-1
→ SHARED SYNCHRONIZATION, FRESHNESS, PERMISSIONS AND REMOVAL
    ├── internal: 1C synchronization lifecycle
    ├── internal: 1D provenance inspection
    └── internal: 1E safe removal

LKW-LIVE-PLATFORM-PROOF-1
→ COMPLETE DEMONSTRABLE PLATFORM PROOF
```

### 3.1 Recently accepted

| Task | User/product outcome | Status |
|---|---|---|
| `1B-5-2` | A trusted client can attach an allowed public HTTPS URL to a workspace, after which LKW durably registers, securely captures, indexes and exposes the resulting knowledge through grounded Ask using the existing Knowledge Intake lifecycle | **ACCEPTED** |
| `LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1` | Hybrid Knowledge Workspace vocabulary, indexed/live/hybrid modes, security model and product roadmap frozen for review | **ACCEPTED** |
| `LKW-MODEL-RUNTIME-1` | The same LKW workspace, document and vector index pass generation, structured planning, validated tool calling, public HTTP Ask, citations and persisted runs on Ollama `qwen2.5:14b` and vLLM `Qwen/Qwen2.5-3B-Instruct` without reindexing | **ACCEPTED** |
| `LKW-CONVERSATION-CONTEXT-ARCH-1` | Provider-neutral Conversation Context Binding with observed-audience validation, binding identity, workspace resolution, thread memory isolation, shared capability boundary and deterministic guards | **ACCEPTED** |

### 3.2 Next and planned product blocks

| Block | One-sentence outcome | Status |
|---|---|---|
| `LKW-MODEL-RUNTIME-1` | The same LKW workflows run on Ollama or vLLM through configuration, and both runtimes pass planner, tool-calling and grounded-Ask proof gates | **ACCEPTED** |
| `LKW-KNOWLEDGE-ACCESS-1` | A workspace can be configured with provider Connections, discoverable Remote Resources, Indexed Sources, Live Access Bindings and bounded Query Policies without exposing credentials | **ACCEPTED / CLOSED** |
| `LKW-HYBRID-ASK-ARCH-1` | Unified evidence, query orchestration and read-only live execution contract | **ACCEPTED / CLOSED** |
| `LKW-HYBRID-ASK-1` | One workspace question can combine indexed RAG evidence with authorized live provider evidence and return one grounded answer with unified provenance | **ACCEPTED / CLOSED** |
| `LKW-CONVERSATIONAL-FRONTEND-1` | A user can operate LKW naturally through Slack or another frontend while the planner, resolver and validated executor invoke real LKW capabilities | **READY_FOR_REVIEW** |
| `LKW-VENDOR-ACCESS-COLLABORATION-1` | LKW supports indexed and controlled live knowledge access across Microsoft 365, Google Workspace, Jira and Confluence through provider-neutral contracts | **PLANNED** |
| `LKW-VENDOR-ACCESS-DATA-1` | LKW provides governed read-only access to Databricks, Power BI and Atlan, allowing live analytical and metadata evidence to participate in Hybrid Ask | **PLANNED** |
| `LKW-KNOWLEDGE-LIFECYCLE-1` | Indexed and live workspace knowledge share coherent freshness, permission, operation, provenance and safe-removal semantics without deleting upstream data | **PLANNED** |
| `LKW-LIVE-PLATFORM-PROOF-1` | A live demonstration shows Slack conversations, Google Docs/Sheets/Calendar (when implemented), Microsoft 365 sources, local files, Web URLs, indexed vendor knowledge, live vendor queries, unified citations and Ollama/vLLM portability in one LKW workspace | **PLANNED** |

### 3.2.1 Conversation context and Slack vertical (platform + LKW)

Provider-neutral personal/shared conversation context architecture precedes shared-channel runtime and Slack connected-source work. Canonical contract: [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](CONVERSATION_CONTEXT_ARCHITECTURE.md).

| Task | Owner | User outcome | Status |
|---|---|---|---|
| `LKW-CONVERSATION-CONTEXT-ARCH-1` | LKW application (docs) | Provider-neutral Conversation Context Binding with observed-audience validation, binding identity, workspace resolution, thread memory isolation, shared capability boundary and deterministic guards | **ACCEPTED** |
| `SLACK-KNOWLEDGE-THREE-MODE-ARCH-1` | Platform (docs) | Architecture frozen: one Slack integration reused across indexed RAG, durable materialization and live access; frontend and knowledge-source roles separated | **DONE** |
| `SLACK-KNOWLEDGE-FOUNDATION-1` | Platform | Platform can safely read and durably synchronize selected Slack conversations (bot token + bot-membership inventory for public/private/IM/MPIM) for any Intergrax application; no new Slack command or LKW feature implied yet | **DONE** |
| `LKW-SLACK-CONNECTED-SOURCE-1` | LKW application | User can attach an approved Slack conversation to an LKW workspace, synchronize it and ask questions about its indexed history | **IN_PROGRESS / CHANGES_REQUIRED** (`REVIEW-FIX-2` — **CHANGES_REQUIRED**; `REVIEW-FIX-3` not accepted) |
| `LKW-CONVERSATION-CONTEXT-1` | LKW application | Durable Conversation Context Bindings, workspace audience policy, memory partitioning and evidence guards | **PLANNED** |
| `LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1` | LKW application | Slack channel/private-channel mention handling over the generic LKW context layer | **PLANNED** |
| `SLACK-LIVE-CAPABILITY-1` | Platform | Authorized applications can read bounded current Slack information at request time without waiting for complete durable synchronization | **PLANNED** |
| `LKW-SLACK-KNOWLEDGE-PROOF-1` | LKW application | User asking through Slack receives one grounded answer combining indexed Slack history, authorized live Slack evidence and other workspace sources with strict audience isolation — requires Hybrid Ask | **PLANNED** |

**`LKW-CONVERSATION-CONTEXT-1` implementation slices:**

| Slice | Status |
|---|---|
| `LKW-CONVERSATION-CONTEXT-1A` | **ACCEPTED** |
| `LKW-CONVERSATION-CONTEXT-1B1` | **ACCEPTED** |
| `LKW-CONVERSATION-CONTEXT-1B2` | **ACCEPTED / CLOSED** |
| `LKW-CONVERSATION-CONTEXT-1B3` | **ACCEPTED / CLOSED** |
| `LKW-CONVERSATION-CONTEXT-1C` | **ACCEPTED / CLOSED** |

**Required dependency (implementation tracks):**

```text
SLACK-KNOWLEDGE-FOUNDATION-1
→ LKW-CONVERSATION-CONTEXT-ARCH-1 — ACCEPTED
→ LKW-SLACK-CONNECTED-SOURCE-1          # independent from conversational activation
→ LKW-CONVERSATION-CONTEXT-1            # LKW-wide prerequisite for shared adapters
→ LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
→ SLACK-LIVE-CAPABILITY-1
```

**Final proof join (all prerequisites):**

```text
LKW-SLACK-CONNECTED-SOURCE-1
+ LKW-CONVERSATION-CONTEXT-1
+ LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1
+ SLACK-LIVE-CAPABILITY-1
+ LKW-HYBRID-ASK-1
→ LKW-SLACK-KNOWLEDGE-PROOF-1
```

`LKW-CONVERSATION-CONTEXT-1` is a prerequisite/supporting block of the wider conversational frontend execution path — not a competing planner/executor. Final Slack proof cannot claim indexed + live combined evidence before Hybrid Ask exists. Google Workspace runtime implementation (`GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1` and below) starts only after `LKW-SLACK-KNOWLEDGE-PROOF-1` becomes **ACCEPTED** (currently **PLANNED**); `GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1` remains **READY_FOR_REVIEW**. `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR` follows the first accepted Google Workspace LKW proof (`LKW-GOOGLE-WORKSPACE-PROOF-1`).

**Available today:** The user can operate LKW through Slack DM, continue a
canonical thread with bounded durable recent turns, and ask about knowledge
already present in the resolved personal workspace. The legacy exact-command
fallback still has temporary in-memory selection state. Durable shared-channel
transport, shared capability enforcement, shared source eligibility,
mention/thread-continuation transport, Slack history indexing and live Slack Ask
remain **not** implemented.

### 3.3 Internal implementation slices (historical identity preserved)

| Slice | Placement | Status |
|---|---|---|
| `CONV-1A` | Planner contract under `LKW-CONVERSATIONAL-FRONTEND-1` | **ACCEPTED** (sufficient to continue) |
| `CONV-1B` | Resolver + executor under `LKW-CONVERSATIONAL-FRONTEND-1` | **ACCEPTED / CLOSED** |
| `CONV-1C` | Slack natural-language cutover under `LKW-CONVERSATIONAL-FRONTEND-1` | **READY_FOR_REVIEW** |
| `1B-5-3` | Web URL ingestion, indexing and Ask proof | **MERGED INTO 1B-5-2** |
| `1B-6-0` | LKW / VENDOR-KNOWLEDGE ownership contract | **REPLANNED** → architecture in `LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1`; implementation in `LKW-KNOWLEDGE-ACCESS-1` |
| `1B-6-1` | Connection and Remote Resource discovery | **MAPPED INTO** `LKW-KNOWLEDGE-ACCESS-1` |
| `1B-6-2` | First real provider vertical slice | **MAPPED INTO** `LKW-HYBRID-ASK-1` |
| `1B-6-3` | Additional vendor packs | **MAPPED INTO** `LKW-VENDOR-ACCESS-COLLABORATION-1` and `LKW-VENDOR-ACCESS-DATA-1` |
| `1C` | Shared synchronization lifecycle | **MAPPED INTO** `LKW-KNOWLEDGE-LIFECYCLE-1` |
| `1D` | Provenance inspection | **MAPPED INTO** `LKW-KNOWLEDGE-LIFECYCLE-1` |
| `1E` | Safe removal | **MAPPED INTO** `LKW-KNOWLEDGE-LIFECYCLE-1` |

### 3.4 Old-to-new task mapping

| Previous task | New roadmap placement | Mapping |
|---|---|---|
| `1B-5-2` + former `1B-5-3` | Expanded end-to-end `1B-5-2` | **MERGED** |
| `CONV-1B` | Internal slice of `LKW-CONVERSATIONAL-FRONTEND-1` | **MAPPED INTO** |
| `CONV-1C` | Internal slice of `LKW-CONVERSATIONAL-FRONTEND-1` | **MAPPED INTO** |
| `1B-6-0` | `LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1` + `LKW-KNOWLEDGE-ACCESS-1` | **REPLANNED** |
| `1B-6-1` | Connection and Remote Resource discovery in `LKW-KNOWLEDGE-ACCESS-1` | **MAPPED INTO** |
| `1B-6-2` | First real provider vertical slice in `LKW-HYBRID-ASK-1` | **MAPPED INTO** |
| `1B-6-3` | Collaboration and Data connector packs | **MAPPED INTO** |
| `1C` | Internal lifecycle slice of `LKW-KNOWLEDGE-LIFECYCLE-1` | **MAPPED INTO** |
| `1D` | Internal provenance slice of `LKW-KNOWLEDGE-LIFECYCLE-1` | **MAPPED INTO** |
| `1E` | Internal removal slice of `LKW-KNOWLEDGE-LIFECYCLE-1` | **MAPPED INTO** |

---

## 4. Legacy Workspace Contents execution detail (accepted foundations)

The following table preserves accepted intake foundations and the accepted WEB_URL slice. It does not override the functional block order in §3.

### Completed and accepted intake foundations

| Task | Result | Status |
|---|---|---|
| `LKW-WORKSPACE-CONTENTS-1B-0` | Channel-neutral Knowledge Intake contract | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-1` | Durable Knowledge Input → Source → operation → queue/worker foundation | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-2` | Managed-file HTTP intake | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-3` | Slack attachment and multi-attachment adapter over managed-file intake | ACCEPTED |

### Accepted intake expansions

| Task | Result | Status |
|---|---|---|
| `LKW-WORKSPACE-CONTENTS-1B-4-1` | HTTP listing and acceptance of opaque preconfigured `LOCAL_FOLDER` Source Candidates | ACCEPTED (with corrections) |
| `LKW-WORKSPACE-CONTENTS-1B-4-1-C1` | Harden public safety, candidate identity and operation evidence | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-4-2` | Slack safe numbered Source Candidate selection | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-5-1` | Shared secure Web Content Capture contract and HTTPS backend | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-5-2` | End-to-end WEB_URL Knowledge Intake | ACCEPTED |
| `LKW-WORKSPACE-CONTENTS-1B-5-2-C1` | Harden production index path and real RAG proof | ACCEPTED (correction to `1B-5-2`) |
| `LKW-WORKSPACE-CONTENTS-1B-5-2-C2` | Close retrieval evidence and regression review | ACCEPTED (correction to `1B-5-2`) |

`LKW-CONVERSATIONAL-INTERACTION-1A` — channel-neutral structured interaction plan contract and provider-neutral LLM planner (**ACCEPTED / CLOSED**).

---

## 5. `LKW-WORKSPACE-CONTENTS-1B-4-2` — Slack Source Candidate selection

**One-sentence outcome:** An approved Slack user can list and select a numbered preconfigured folder for the active workspace while Slack receives only safe candidate metadata and opaque identity.

Expected flow:

```text
approved Slack DM
→ effective active workspace
→ public LKW Source Candidate list endpoint
→ safe numbered labels
→ user selects a number
→ Slack sends opaque candidate_id to the public acceptance endpoint
→ existing Knowledge Intake and KNOWLEDGE_INGESTION lifecycle
→ safe acknowledgement
```

Slack must never receive, persist or infer:

- the local path;
- allowlist or shadow roots;
- candidate fingerprint;
- provider locator or configuration path.

Slack remains a replaceable frontend and must not own folder discovery or indexing.

---

## 6. `LKW-WORKSPACE-CONTENTS-1B-5` — explicit Web URL intake

**One-sentence outcome:** A trusted client can attach an allowed Web URL as workspace knowledge through the same durable Knowledge Intake and operation lifecycle.

The slice must reuse:

- `KnowledgeInput` and the accepted intake boundary;
- durable Source ownership;
- idempotent operation creation;
- the existing queue, worker and recovery path;
- the existing document indexing and provenance model.

It must not create a second URL-specific queue, worker or document pipeline. URL access policy, redirects, private-network protection, size limits and safe error behavior must be frozen before implementation.

### 6.1 Bounded decomposition (`1B-5`)

| Task | Outcome | Status |
|------|---------|--------|
| `1B-5-1` | Shared secure Web Content Capture contract and HTTPS backend | ACCEPTED |
| `1B-5-2` | End-to-end WEB_URL Knowledge Intake | **ACCEPTED** |
| `1B-5-2-C1` | Harden production index path and real RAG proof | **ACCEPTED** (correction to `1B-5-2`) |
| `1B-5-2-C2` | Close retrieval evidence and regression review | **ACCEPTED** (correction to `1B-5-2`) |
| `1B-5-3` | Web URL ingestion, indexing and Ask proof | **MERGED INTO 1B-5-2** |
| `1B-5-4` | Slack explicit Web URL intake | **SUPERSEDED** by `LKW-CONVERSATIONAL-INTERACTION-1C` |

**Why `1B-5-4` was superseded:** the target frontend will not use a separate strict URL command; one natural message can contain multiple source types and actions (attachments, URLs, local references, workspace targets). See [`CONVERSATIONAL_INTERACTION.md`](CONVERSATIONAL_INTERACTION.md).

Platform prerequisite: [`docs/project/architecture/WEB_CONTENT_CAPTURE.md`](../../../docs/project/architecture/WEB_CONTENT_CAPTURE.md). `1B-5-1` is accepted; `LKW-CONVERSATIONAL-INTERACTION-1A` (planner contract) precedes `1B-5-2` so URL intake is designed as a planner action, not another terminal command.

---

## 7. Product block definitions

Binding architecture: [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md). The sections below summarize one-sentence outcomes. Detailed `1B-6` / `1C`–`1E` material is preserved in §8–§10 as implementation reference mapped into the blocks above.

### 7.1 `LKW-MODEL-RUNTIME-1` — Ollama / vLLM end-to-end portability

**One-sentence outcome:** The same LKW workflows run on Ollama or vLLM through configuration, and both runtimes pass planner, tool-calling and grounded-Ask proof gates.

**Status:** **ACCEPTED**.

Accepted proof pair:

```text
Ollama qwen2.5:14b
vLLM Qwen/Qwen2.5-3B-Instruct
```

The proof covers basic generation, structured Conversation Interaction planning, deterministic plan validation, native tool calling, real `local.workspace.search` execution, public HTTP Ask, verified citations, persisted Ask runs and full shared-index invariance without reindexing. Canonical evidence: [`LKW_MODEL_RUNTIME_PORTABILITY.md`](evidence/LKW_MODEL_RUNTIME_PORTABILITY.md).

Provider switch may require application restart. Conversation LLM and embedding provider remain separate — switching chat runtime must not silently reindex. Runtime hot swapping and universal model compatibility are not claimed.

### 7.2 `LKW-KNOWLEDGE-ACCESS-1` — Workspace Knowledge Configuration

**One-sentence outcome:** A workspace can be configured with provider Connections, discoverable Remote Resources, Indexed Sources, Live Access Bindings and bounded Query Policies without exposing credentials. All user-managed product configuration that must survive restart is durable; tenant Connections are stored in a platform-owned durable Connection Catalog with `SecretsStore`-owned credentials and runtime registry rehydration.

**Status:** **ACCEPTED / CLOSED**.

Expected capabilities: durable tenant Connection catalog with restart-safe rehydration; connection listing and safe inspection; remote-resource discovery; workspace binding; indexed vs live selection; capability allowlists; query policy; safe connection health.

**Internal decomposition** (historical slice identity — all slices **ACCEPTED** as part of closed parent `LKW-KNOWLEDGE-ACCESS-1`):

| Slice | Title | Status |
|-------|-------|--------|
| `1A` | Implementation contract freeze | **ACCEPTED** (including C3 mutation semantics; C4 persistence boundary) |
| `1B` | Provider-neutral durable workspace authorization foundation | **ACCEPTED** (included in closed parent) |
| `1C-1` | Durable tenant Connection catalog and restart rehydration | **ACCEPTED** (included in closed parent) |
| `1C-2` | Safe Connection / Remote Resource discovery and typed capability catalog | **ACCEPTED** (included in closed parent) |
| `1D` | HTTP create/disable for bindings with server-derived metadata | **ACCEPTED** (included in closed parent) |
| `1E` | Query Policy and complete configuration projection | **ACCEPTED** (included in closed parent) |
| `1F` | One-connection indexed/live reuse proof (with restart reconstruction) | **ACCEPTED** (included in closed parent) |

**Acceptance gate:** Must prove that **one durable Connection** reconstructed after restart can support both Indexed Source and Live Access Binding without duplicating credentials or integrations.

#### 7.2.1 User-visible roadmap

**Available today:**

- create/select an LKW workspace;
- add already supported indexed inputs such as files and approved Web URLs;
- ask questions against indexed knowledge;
- use the current Slack/HTTP indexed workflow where already implemented.

**After `1B`:** backend-only durable workspace authorization and configuration state; no new connector-management UI yet.

**After `1C-1`:** an administrator can configure a connector once; the connector survives restart; LKW remembers that the Connection exists; secrets remain outside LKW. This may initially be backend/admin capability rather than Slack UI.

**After `1C-2` and `1D`:** users can list available Connections; inspect discoverable resources; select which resources are indexed; separately authorize bounded live access.

**Later frontend work:** natural-language configuration and use through Slack or another frontend.

Do not claim that connector-management actions are already implemented.

### 7.3 `LKW-HYBRID-ASK-ARCH-1` — Hybrid Ask architecture contract

**One-sentence outcome:** Freeze the production-grade architecture and implementation contract for combining indexed RAG evidence with authorized read-only live provider evidence in one grounded Workspace Ask response with unified provenance, strict policy enforcement and no automatic persistence of live result bodies.

**Status:** **ACCEPTED / CLOSED**.

Canonical contract: [`HYBRID_ASK_ARCHITECTURE.md`](HYBRID_ASK_ARCHITECTURE.md).

### 7.4 `LKW-HYBRID-ASK-1` — Hybrid Ask with unified provenance

**One-sentence outcome:** One workspace question can combine indexed RAG evidence with authorized live provider evidence and return one grounded answer with unified provenance.

**Status:** **IN_PROGRESS**.

Architecture: [`HYBRID_ASK_ARCHITECTURE.md`](HYBRID_ASK_ARCHITECTURE.md). Provider-specific live handlers are **Vendor Knowledge** scope; LKW owns provider-neutral contracts, policy, plan validation, orchestration and Ask integration. `LKW-HYBRID-ASK-1C` requires at least one accepted Vendor Knowledge `LiveCapabilityHandler` — provider selection is not an LKW roadmap decision.

Initial live access is read-only.

**Internal decomposition (three slices):**

| Slice | Title | Status |
|-------|-------|--------|
| `1A` | Provider-neutral core contracts, durable Query Policy V2 and Evidence Plan validation | **READY_FOR_REVIEW** |
| `1B` | Provider-neutral Live Capability execution and Knowledge Query orchestration | **PLANNED** |
| `1C` | Workspace Ask integration, HTTP V2 and bounded product acceptance proof | **PLANNED** |

**Dependency chain:** `1A → 1B → 1C`.

**Acceptance gate:** Must prove that indexed evidence and live evidence can be combined; live evidence is **not** automatically persisted; each evidence item retains mode and provenance; at least one indexed and one live citation in one grounded answer (at `1C` with Vendor Knowledge handler).

### 7.5 `LKW-CONVERSATIONAL-FRONTEND-1` — Natural-language execution and Slack cutover

**One-sentence outcome:** A user can operate LKW naturally through Slack or another frontend while the planner, resolver and validated executor invoke real LKW capabilities.

**Status:** **READY_FOR_REVIEW**. `CONV-1B` (resolver + executor) is **ACCEPTED / CLOSED**; `CONV-1C` (Slack mixed-message cutover) is **READY_FOR_REVIEW**. See [`CONVERSATIONAL_INTERACTION.md`](CONVERSATIONAL_INTERACTION.md).

### 7.6 `LKW-VENDOR-ACCESS-COLLABORATION-1`

**One-sentence outcome:** LKW supports indexed and controlled live knowledge access across Microsoft 365, Google Workspace, Jira and Confluence through provider-neutral contracts.

**Status:** **PLANNED**. Scope: OneDrive/SharePoint files; Microsoft mail and Teams-hosted organizational knowledge; Google Drive, Docs, Sheets, Calendar and remaining Google surfaces when implemented; Jira issue discovery/search/state and selected project sync; Confluence space/page discovery, search/read and selected space sync.

**Acceptance gate:** Must prove at least one provider vertical slice in both paths:

```text
same vendor integration
├── durable normalized delivery toward LKW/RAG
└── bounded live query/read toward Live Evidence
```

Jira or Confluence is the preferred first provider-neutral live search/read capability proof because the existing integrations already expose operational search and exact-read methods.

A Microsoft Graph live search capability requires a separate bounded contract and must not be simulated through delta, reconciliation or full inventory.

**Execution order:**

```text
1. conversation context architecture accepted (`LKW-CONVERSATION-CONTEXT-ARCH-1` — **ACCEPTED**);
2. complete Slack Knowledge vertical (implementation tracks through SLACK-LIVE-CAPABILITY-1; final proof joins LKW-HYBRID-ASK-1 at LKW-SLACK-KNOWLEDGE-PROOF-1);
3. complete Google Workspace proof-critical path (`GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1` → vertically incremental Drive/Docs/Sheets/Calendar read surfaces and adapters → `LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1` → `LKW-GOOGLE-WORKSPACE-PROOF-1`);
4. complete Teams Chat and Calendar durable adapters (Calendar after first accepted Google LKW proof);
5. perform the adapter-family and three-mode capability audit;
6. use Jira or Confluence for the first bounded live search/read capability proof;
7. add bounded Microsoft Graph live capabilities separately;
8. converge at Hybrid Ask.
```

The immediate current platform task in the Slack vertical is complete (`SLACK-KNOWLEDGE-FOUNDATION-1` **DONE**). `LKW-SLACK-CONNECTED-SOURCE-1` is **IN_PROGRESS / CHANGES_REQUIRED** (`LKW-SLACK-CONNECTED-SOURCE-1-REVIEW-FIX-2` — **CHANGES_REQUIRED**; `REVIEW-FIX-3` not accepted; final crash-safe recovery and real indexed Search/Ask proof remain under correction). Next Slack-vertical implementation task is `LKW-CONVERSATION-CONTEXT-1`. Global LKW product next implementation block is `LKW-HYBRID-ASK-1` (**IN_PROGRESS**; `LKW-HYBRID-ASK-1A` **READY_FOR_REVIEW**). Google Workspace knowledge architecture is frozen (`GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1` **READY_FOR_REVIEW**); Google runtime (`GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1` and below) is **PLANNED** and gated on `LKW-SLACK-KNOWLEDGE-PROOF-1` becoming **ACCEPTED** (currently **PLANNED**). Microsoft Graph Teams Chat adapter is **DONE**; Calendar adapter (`MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR`) is **PLANNED** after the first accepted Google Workspace LKW proof.

### 7.7 `LKW-VENDOR-ACCESS-DATA-1`

**One-sentence outcome:** LKW provides governed read-only access to Databricks, Power BI and Atlan, allowing live analytical and metadata evidence to participate in Hybrid Ask.

**Status:** **PLANNED**. All queries pass capability-specific validation and policy — no unrestricted user SQL/DAX.

**Acceptance gate:** Must state per resource whether it supports durable materialization, RAG indexing, live query, or a documented subset. Do not assume Power BI, Databricks and Atlan should all be copied into RAG.

### 7.8 `LKW-KNOWLEDGE-LIFECYCLE-1`

**One-sentence outcome:** Indexed and live workspace knowledge share coherent freshness, permission, operation, provenance and safe-removal semantics without deleting upstream data.

**Status:** **PLANNED**. Consolidates former `1C`, `1D`, `1E` slices (§8–§10).

### 7.9 `LKW-LIVE-PLATFORM-PROOF-1` — Complete demonstrable platform proof

**One-sentence outcome:** A live demonstration shows Slack conversations, Google Docs/Sheets/Calendar (when implemented), Microsoft 365 sources, local files, Web URLs, indexed vendor knowledge, live vendor queries, unified citations and Ollama/vLLM portability in one LKW workspace.

**Status:** **PLANNED**.

Target scenario: start with Ollama → create/select workspace → upload files → add Web URL → configure MS365 and Google Workspace (when implemented), Jira, Confluence, Databricks, Power BI, Atlan → Hybrid Ask with indexed + live evidence → restart with vLLM → repeat without changing LKW domain behavior. Public claims must distinguish real provider proof, controlled integration proof and deterministic fixture proof.

**Final platform proof requirement:**

```text
one durable vendor connection (reconstructed after restart)
→ one durable/indexed use
→ one live use
→ one hybrid answer
→ no duplicated vendor client
→ explicit provenance and freshness
→ no automatic persistence of live results
```

---

## 8. `LKW-WORKSPACE-CONTENTS-1B-6` — VENDOR-KNOWLEDGE integration (historical reference)

**Status:** **REPLANNED** — scope distributed across `LKW-KNOWLEDGE-ACCESS-1`, `LKW-HYBRID-ASK-1`, `LKW-VENDOR-ACCESS-COLLABORATION-1`, `LKW-VENDOR-ACCESS-DATA-1` and `LKW-KNOWLEDGE-LIFECYCLE-1`. Retained for ownership-boundary detail.

### 8.1 Ordering rule

`1B-6` is placed after explicit Web URL intake and before the shared `1C` lifecycle:

```text
1B-5 Web URL intake
→ 1B-6 VENDOR-KNOWLEDGE integration
→ 1C shared synchronization and completion lifecycle
→ 1D document inspection and provenance
→ 1E safe source-owned knowledge removal
```

This ensures synchronization, operation inspection, completion notification, provenance and removal are designed once for local folders, Web URLs and vendor-backed Sources.

### 8.2 Ownership boundary

```text
VENDOR-KNOWLEDGE owns:
provider authentication and credential handling
→ provider APIs and provider-specific errors
→ pagination, rate limits and token refresh
→ provider-specific resource discovery
→ incremental cursors, checkpoints and reconciliation semantics
→ canonical provider-neutral resource and knowledge-item projection

LKW owns:
tenant/workspace authorization and source attachment
→ KnowledgeInput and durable Source identity
→ idempotency and durable operation lifecycle
→ queue/worker dispatch and recovery
→ Documents, Chunks and Vectors
→ Ask Workspace and safe provenance projection
→ local retention and deletion semantics
```

VENDOR-KNOWLEDGE expands the provider catalog. It does not become a second knowledge workspace or a second LKW ingestion runtime.

### 8.3 Architectural invariants

LKW must not implement separate pipelines such as:

```text
Confluence ingestion pipeline
Teams ingestion pipeline
Power BI ingestion pipeline
Databricks ingestion pipeline
Jira ingestion pipeline
Atlan ingestion pipeline
```

Provider-specific differences end at the VENDOR-KNOWLEDGE boundary. Downstream LKW processing remains provider-neutral.

Public LKW contracts may expose only safe opaque fields such as:

```text
connection_id
resource_id
provider_id or vendor
resource_type
label
description
available
safe provenance
```

They must not expose:

- access or refresh tokens;
- client secrets;
- connection strings;
- private provider locators or endpoints;
- raw credentials;
- internal checkpoint or cursor material unless explicitly converted into a safe public status.

### 8.4 `1B-6-0` — integration contract and ownership gate

Freeze, from the real VENDOR-KNOWLEDGE implementation rather than assumptions:

- vendor connection identity;
- vendor resource identity;
- canonical knowledge item shape;
- credential ownership and access boundary;
- checkpoint/cursor ownership;
- LKW Source correlation and provenance;
- idempotency and configuration-version behavior;
- deletion and upstream-unavailability semantics;
- error normalization;
- the exact public capability LKW consumes.

Do not freeze a provider-specific LKW Source architecture. Exact names of new `KnowledgeInputKind` or Source representation remain an architecture decision for this gate.

### 8.5 `1B-6-1` — safe connection and resource discovery

Expose tenant-scoped, authorized and safely labelled vendor connections/resources. Discovery must support unavailable states without leaking credentials or technical locator details.

Representative resource families:

| Family | Examples |
|---|---|
| Files and documents | OneDrive, SharePoint |
| Wiki and technical knowledge | Confluence |
| Communication | Teams chats/channels, Outlook e-mail |
| Work management | Jira |
| Calendars and meetings | Outlook Calendar, Teams meeting material |
| Data and notebooks | Databricks |
| Data catalog | Atlan |
| Reports and analytics | Power BI |

### 8.6 `1B-6-2` — vendor resource Knowledge Intake

Attach one real vendor resource end to end:

```text
safe opaque vendor resource selection
→ public LKW Knowledge Intake capability
→ durable KnowledgeInput
→ durable Source
→ KNOWLEDGE_INGESTION operation
→ existing queue and worker
→ provider-neutral canonical items
→ existing document indexing
→ workspace Documents with safe provenance
```

This task proves one representative provider. It must not attempt to implement every provider listed in the roadmap.

### 8.7 `1B-6-3` — provider-neutral verification

Demonstrate that materially different provider families can reuse the same LKW contracts and lifecycle. The verification may use focused adapters/fakes where full production integrations are not yet justified, but it must prove that adding another vendor does not require:

- a new LKW queue or worker;
- a new operation model;
- a new Source ownership system;
- a new document indexing pipeline;
- provider credentials inside LKW public APIs;
- provider-specific business logic in Slack.

---

## 9. `LKW-WORKSPACE-CONTENTS-1C` — shared synchronization and completion lifecycle

**Mapped into:** `LKW-KNOWLEDGE-LIFECYCLE-1` (**PLANNED**).

**One-sentence outcome:** A user can synchronize and inspect asynchronous processing for local-folder, Web URL and vendor-backed Sources through one operation model and receive channel-neutral completion information.

The lifecycle must cover:

- manual synchronization;
- durable accepted, queued, processing, completed and failed states;
- reliable operation counters and safe errors;
- restart recovery;
- provider cursors/checkpoints through the VENDOR-KNOWLEDGE boundary;
- added, changed and removed upstream items according to frozen policy;
- channel-neutral completion events consumed by Slack or future surfaces.

Completion notification must not be implemented as a Slack-only ingestion mechanism.

---

## 10. `LKW-WORKSPACE-CONTENTS-1D` — document inspection and provenance

**Mapped into:** `LKW-KNOWLEDGE-LIFECYCLE-1` (**PLANNED**).

**One-sentence outcome:** A user can inspect which documents are indexed in a workspace, their Source ownership, status and safe origin without exposure of private paths, credentials or provider locators.

Safe provenance may include:

- Source label and safe source type;
- vendor/provider and resource type;
- original item type such as file, e-mail, chat message, thread, calendar event, wiki page, issue, dashboard, dataset, notebook or catalog asset;
- last synchronization time;
- safe document status and indexing metadata.

Every persisted Document remains owned by exactly one durable Source.

---

## 11. `LKW-WORKSPACE-CONTENTS-1E` — safe source-owned knowledge removal

**Mapped into:** `LKW-KNOWLEDGE-LIFECYCLE-1` (**PLANNED**).

**One-sentence outcome:** A user can detach a Source and remove its LKW-owned knowledge safely and idempotently without deleting original data in the upstream system.

Removal may include:

```text
Documents
→ Chunks
→ Vectors
→ local cached or managed objects where applicable
→ correlation state
→ LKW-owned synchronization/checkpoint state according to the 1B-6 contract
```

Removal must not automatically delete:

- Outlook e-mail or calendar events;
- Teams messages or meeting data;
- SharePoint or OneDrive files;
- Confluence pages or Jira issues;
- Power BI reports;
- Databricks data or notebooks;
- Atlan catalog assets;
- any other upstream vendor originals.

The operation means “detach from LKW and delete the local knowledge representation,” not “delete the source system record.”

---

## 12. Cross-cutting acceptance rules

Every source-expansion slice must preserve:

- tenant and workspace isolation;
- private-by-default behavior;
- opaque public identities;
- no private path or credential disclosure;
- idempotent acceptance and operation identity;
- one durable Source per attached logical origin according to its frozen identity contract;
- one existing Knowledge Intake runtime;
- one queue/worker lifecycle;
- shared document indexing and provenance;
- recovery without a parallel recovery table unless a separately accepted architecture gap requires it;
- Slack and future conversational surfaces as API/capability clients only.

Parallel work rules:

- work from the current repository state;
- preserve unrelated changes from other sessions;
- do not block solely because HEAD differs from an earlier instruction;
- do not use destructive Git commands;
- commit only the task’s own scope.

---

## 13. LKW-PF6 — Token Optimization product proof (planned)

**Status:** **LKW-PF6-0** Done / Closed (proof design only). **LKW-PF6-A**–**C** planned after universal platform proof.

Token Optimization is a platform capability under `intergrax/runtime/token_optimization`. LKW integrates through public contracts only — no duplication of router, pipeline, cache-aware gate, or proof harness.

| Phase | Depends on | Scope |
|-------|------------|-------|
| **LKW-PF6-A** | TOKEN-10G | Baseline measurement on product workflows |
| **LKW-PF6-B** | LKW-PF6-A, TOKEN-10D | Integrate public runtime contract |
| **LKW-PF6-C** | LKW-PF6-B | Baseline vs optimized product proof |

Canonical: [`docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md`](../../../docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md) (Token Optimization subsection) · [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md) §11.

---

## 14. Historical post-MVP direction (superseded)

LKW 1.0 aims to be installable, restart-safe, auditable, daily-usable and supportable for its declared source portfolio. Broad provider breadth is not automatically required for 1.0; the bounded `1B-6` contract, one end-to-end provider proof and provider-neutral verification establish the expansion path.

Future work remains product-pulled and may include:

- broader vendor coverage selected by user or commercial value;
- richer document reconciliation;
- workspace outputs and history;
- a local setup/diagnostics companion;
- another conversational frontend;
- organization membership and controlled sharing;
- production security, operations and release hardening.

A second conversational adapter and broad provider matrix do not delay the current Workspace Contents sequence unless a documented product blocker requires them.

---

## Appendix A — Historical task summary

```text
ARCHITECTURE:
LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1
HYBRID KNOWLEDGE ACCESS AND LIVE PLATFORM PROOF ROADMAP
→ ACCEPTED

LKW-HYBRID-ASK-ARCH-1
UNIFIED EVIDENCE, QUERY ORCHESTRATION AND READ-ONLY LIVE EXECUTION
→ READY_FOR_REVIEW

LAST ACCEPTED IMPLEMENTATION:
LKW-MODEL-RUNTIME-1
OLLAMA / vLLM END-TO-END PORTABILITY
→ ACCEPTED (including C1–C4 corrections and evidence v2)

LKW-KNOWLEDGE-ACCESS-1
→ Connections, Remote Resources, Indexed Sources, Live Access Bindings and Query Policy
→ ACCEPTED / CLOSED

NEXT:
LKW-HYBRID-ASK-1 → RAG + live with unified provenance
→ IN_PROGRESS (1A READY_FOR_REVIEW)
→ internal slices 1A–1C (see §7.4)
→ LKW-CONVERSATIONAL-FRONTEND-1 (CONV-1B + CONV-1C internal)
→ LKW-VENDOR-ACCESS-COLLABORATION-1
→ LKW-VENDOR-ACCESS-DATA-1
→ LKW-KNOWLEDGE-LIFECYCLE-1 (1C + 1D + 1E internal)
→ LKW-LIVE-PLATFORM-PROOF-1

FINAL TARGET:
Live Slack platform proof — indexed + live + Hybrid Ask + Ollama/vLLM portability

ARCHITECTURE DOC:
KNOWLEDGE_ACCESS_ARCHITECTURE.md — indexed/live/hybrid, Connections,
Live Access Bindings, Workspace Knowledge Configuration, Query Policy,
Knowledge Query Orchestrator, Evidence Items, MCP boundary, security.

HYBRID_ASK_ARCHITECTURE.md — unified evidence, orchestrator, live executor,
retention, Ask Run V2, implementation slices 1A–1C.
→ READY_FOR_REVIEW

Conversational planning (CONV-1A):
- LLM produces a strict semantic draft (`ConversationInteractionDraft`).
- The adapter automatically derives the draft JSON Schema from its Pydantic class.
- Application code deterministically compiles technical IDs and exact evidence spans (`interaction_plan_compiler`).
- The unchanged `ConversationInteractionPlan` remains the canonical plan contract.
- The existing deterministic request validator remains the final grounding boundary.
```

## Appendix B — Status vocabulary

| Label | Meaning |
|---|---|
| `ACCEPTED` | Implementation or architecture was independently audited and accepted |
| `NEXT` | First bounded implementation block to be started next |
| `CURRENT` | Task actively being implemented |
| `PLANNED` | Accepted roadmap block not yet started |
| `DOCUMENTED / READY_FOR_REVIEW` | Architecture or contract documented; implementation not claimed |
| `DEFERRED` | Intentionally outside the active execution path |
| `HISTORICAL` | Retained record that does not define current status |
