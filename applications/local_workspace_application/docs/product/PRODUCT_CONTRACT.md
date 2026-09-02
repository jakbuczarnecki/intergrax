# LKW Daily-Use Product Contract

**Task:** LKW-PRODUCT-1
**Status:** READY_FOR_REVIEW
**Authority:** This document freezes the LKW 1.0 daily-use product contract. It
defines the product outcome, boundaries, and acceptance gates; it does not
certify current implementation readiness.

## One-sentence product definition

Local Knowledge Workspace (LKW) is a private AI knowledge workspace that
connects selected work sources, keeps required knowledge synchronized, and lets
people ask grounded questions with visible sources and user-controlled access.

## Product promise

LKW helps a person:

```text
connect where my work knowledge lives
→ ask questions
→ receive grounded answers
→ inspect exact sources
→ understand freshness and state
→ keep using it every day
```

The user value is knowledge access and trustworthy understanding. RAG, agents,
Mongo, Qdrant, Nexus, and Intergrax internals are implementation means, not
the primary product proposition.

## Canonical interface decisions (LKW 1.0)

**Slack is the primary and reference daily-use conversational interface for
LKW 1.0.** This is frozen product contract, not an implementation preference.

HTTP and application APIs remain the canonical reusable backend/application
boundary. They are **not** the primary end-user surface. Slack is a thin
client over reusable LKW capabilities.

Future clients - Teams, web, mobile, CLI - may reuse the same application/API
boundaries but are **not** the current primary LKW 1.0 user experience. MCP
remains an independent technical/client surface.

### Slack dual role

Slack has two separate responsibilities that must not be conflated:

1. **Slack as user interface to LKW** - conversational daily-use UX:
   workspace selection, Ask, citations, source state, sync, and recovery.
2. **Slack as knowledge source / vendor content** - connected-source
   architecture for indexed Slack conversation content.

Productization of Slack conversational UX is separate from Slack connected-source
/vendor architecture.

### Thin-client invariant

Slack and all other clients must **not** own independent:

- Ask business logic;
- knowledge lifecycle;
- onboarding truth;
- source lifecycle;
- configuration truth;
- authorization semantics.

Clients consume reusable application/API/service boundaries. No Slack-only (or
client-only) lifecycle mutation is permitted where a canonical reusable backend
capability exists.

### First-run state

First-run onboarding state is derived from durable backend state. The accepted
`WorkspaceSetupSnapshotService` and
`GET /v1/local_workspace/workspaces/{workspace_id}/setup-snapshot` remain
canonical reusable capabilities consumed by Slack PRODUCT-3D.

There is **no** persisted `onboarding_step`, `wizard_state`, or
`onboarding_complete` truth.

### Citation inspect / open (resolved)

The citation inspect/open architecture decision is **resolved**, not open.

Chosen architecture: **host-mediated provider-neutral document inspect/open
boundary**. Conceptual endpoint:
`GET /.../documents/{document_id}` (exact final route may be established during
PRODUCT-3E).

Safe contract concept includes: document/source identity, display name, source
type, source label, logical source location, provenance, page/location, bounded
preview/metadata, and optional `external_url`/provider deep-link target where
capability exists.

**Forbidden:** UI/Slack direct Qdrant reads; arbitrary host filesystem exposure;
arbitrary raw local path exposure; vendor-specific citation routing in Slack;
creation of a separate document subsystem solely for citation opening.

PRODUCT-3E will implement/prove this already-chosen architecture.

## Primary users

LKW 1.0 is bounded to these first-public-product personas:

1. **Software engineer / technical lead** - reconstruct decisions and
   technical context spread across work sources.
2. **Project, product, or knowledge worker** - locate decisions, project
   context, and information distributed across communication and documents.
3. **Technical evaluator / CTO** - evaluate trustworthiness and product
   boundaries; this persona must not drive everyday UX.

LKW 1.0 does not expand its primary UX to HR, sales, support, enterprise
administration, consumers, or other broad markets.

## Jobs to be done

The product must help users:

- find everything relevant about a topic, project, customer, or decision;
- explain what was decided;
- show how and why a decision changed;
- show exact supporting sources;
- determine whether information is fresh;
- scope a question to selected sources;
- refresh selected knowledge;
- report insufficient evidence clearly instead of pretending certainty.

## Top daily workflows

The core daily-use set is intentionally limited to:

1. Open or use an existing workspace.
2. Ask a grounded question.
3. Inspect citations and open the original source.
4. Restrict a question to selected knowledge sources.
5. Check source/data freshness.
6. Connect, disable, re-enable, or detach a source through safe product
   controls.
7. Request synchronization or refresh.
8. See and recover a source that needs user attention.

## First-public-product vendor target

The target vendor family for the first public product is:

- Slack;
- Google Workspace;
- Microsoft 365.

This is a product target, not a claim that every adapter is production-ready.
PRODUCT-5 must verify each vendor end to end before public support is claimed.
Jira and Confluence are not first-public-product commitments.

## Installation persona

The first installation persona is a technically comfortable user who can
install and run a local or self-hosted application, but is not expected to
know Intergrax internals, Python/FastAPI, MongoDB, Qdrant schemas, internal
operation IDs, or internal APIs.

The final product journey must not require manual database changes, internal
scripts, manual curl/JSON as normal UX, knowledge of Intergrax structures, or
assistance from the project author.

## Minimum daily-use UX

LKW 1.0 daily-use UX is **Slack-first**. The product must expose
capability-level paths for:

- workspace selection and management;
- knowledge sources and connections;
- source state and synchronization;
- Ask;
- citations and opening the original source;
- freshness;
- activity and problems requiring attention;
- basic settings.

Slack, future Teams/web/mobile/CLI surfaces, and MCP are thin interfaces over
the same reusable application/API boundaries. No client-only lifecycle or
mutation path is permitted. Product controls must preserve user-controlled
access and must not silently modify original source content.

Installation/evaluation paths (quickstart scripts, curl/API proof, Docker
bootstrap) prove infrastructure and backend capability. They do **not** define
the primary daily-use client.

## Product architecture invariants

- Product language leads with user outcomes, not platform internals.
- Answers are grounded in selected knowledge and expose usable provenance.
- The product distinguishes indexed knowledge, synchronization state, and
  source freshness; it does not imply live or current knowledge without
  evidence.
- A lack of sufficient grounding is a visible product result.
- Source connection, disablement, re-enablement, detachment, refresh, and
  recovery are product lifecycle actions, not client-specific tricks.
- Durable configuration and knowledge state survive restart and supported
  upgrades.
- Provider failures must not silently destroy previously valid indexed
  knowledge.
- Generated content, where offered, is separate from original sources and
  does not write back by default.

## Success metrics / acceptance gates

LKW 1.0 is product-acceptable only when all of the following are demonstrated:

1. A new supported installation reaches a running LKW without repository-
   internal knowledge.
2. A user can connect or provide their own data and obtain a grounded answer.
3. The answer exposes usable source evidence/citations.
4. Restart preserves required configuration and durable state.
5. Normal daily synchronization does not require repetitive manual
   intervention.
6. The user can see last/next synchronization or freshness state where
   relevant.
7. A provider outage does not silently destroy previously valid indexed
   knowledge.
8. Typical authentication and synchronization failures explain what happened
   and what action is available.
9. Product updates preserve supported persistent data/configuration according
   to the later upgrade contract.
10. Insufficient grounding is reported instead of fabricated confidence.
11. No normal product flow requires direct Mongo/Qdrant or manual internal
   operation manipulation.
12. A real-user acceptance session completes without assistance from the
   author.

## Explicit non-goals for the first public product

LKW 1.0 does not commit to:

- enterprise HA;
- multi-instance worker coordination;
- a SaaS multi-tenant control plane;
- dozens of vendors;
- broad write-back into connected systems;
- mobile application implementation;
- a plugin marketplace;
- enterprise compliance certification;
- speculative platform abstractions without an LKW use case;
- replacing accepted backend mechanisms merely to make them “newer”.

## Current starting-point evidence

The bounded evidence pass for this contract found:

- the supported quickstart describes a one-command-per-OS local evaluation
  path using managed sample intake, indexed Ask, a grounded answer, a source
  citation, and persisted Ask-run verification
  ([`QUICKSTART.md`](QUICKSTART.md));
- the quickstart explicitly limits that path to indexed behavior and does not
  claim Hybrid Ask, live-provider access, production readiness, or compliance;
- the application README describes the current local/Docker evaluation path
  and states that its documented platform proof is separate from the product
  quickstart
  ([`applications/local_workspace_application/README.md`](../../../../applications/local_workspace_application/README.md));
- the user journey labels several connection, freshness, and hybrid-access
  capabilities as target or planned, while documenting indexed citations and
  the bounded quickstart as the current evidence
  ([`USER_JOURNEY.md`](USER_JOURNEY.md)).

These observations are starting-point evidence only. Historical
`PLANNED`, `IMPLEMENTED`, `PARTIAL`, or `ACCEPTED` labels are not copied into
this contract as production-readiness claims.

## Roadmap implications: PRODUCT-2…PRODUCT-12

From the current Slack-first contract correction point, the near-term sequence
is frozen as:

1. **SLACK-FIRST CONTRACT CORRECTION** - restore canonical product contract
   (this document and aligned product docs).
2. **LKW-PRODUCT-4 - SLACK DAILY-USE PRODUCT EXPERIENCE** - daily Slack UX for
   workspace selection, knowledge inventory, source state, sync/refresh,
   disable/enable/detach, Ask, citations/open-source, freshness,
   attention/problems, and basic settings/configuration using shared backend
   capabilities. Does **not** require a web frontend for LKW 1.0.

Do not expand the roadmap beyond PRODUCT-4 in current planning.

### Closed / accepted milestones (do not reopen)

- **PRODUCT-2 - ZERO-TO-VALUE INSTALLATION:** **CLOSED.** Proves
  installation/zero-to-value infrastructure; does not redefine the primary
  daily-use client.
- **PRODUCT-3B - KNOWLEDGE SURFACE HTTP PROJECTION:** **CLOSED.** Accepted
  reusable backend projection (`GET /knowledge/inventory`, knowledge operations
  execute, operation list, `error_code` projection).
- **PRODUCT-3C - SETUP SNAPSHOT & FIRST-RUN ORCHESTRATION CONTRACT:** **CLOSED.**
  Accepted reusable setup orchestration capability (`setup-snapshot` endpoint;
  derivation-only orchestration contract).
- **PRODUCT-3D - SLACK FIRST-RUN PRODUCT EXPERIENCE:** **CLOSED.** Slack
  conversational first-run over accepted backend capabilities; no web frontend;
  no new onboarding persistence.
- **PRODUCT-3E - CITATION INSPECT/OPEN + ERROR/RESUME ACCEPTANCE:** **CLOSED.**
  Host-mediated document inspect/open; bounded Slack error behavior;
  restart/resume acceptance.
- **PRODUCT-3 - FIRST-RUN ONBOARDING:** **CLOSED** (accepted closing commit
  `b35de405354c582d0f93847e29993c15887d3ad3`; required ancestor
  `580015167baa62868ed08623aed8d6d68f39001e`). Acceptance matrix recorded in
  `PRODUCT_3_FIRST_RUN_GAP_AUDIT.md` §13. Live Slack proof deferred to
  PRODUCT-11 (clean-machine gate).

The previously issued **LKW-PRODUCT-3D - FIRST-RUN PRODUCT UI AND WELCOME FLOW**
(web-first) is **CANCELLED / INVALID**. It produced no implementation and must
not be represented as an executed PRODUCT-3D task.

### PRODUCT-3D - SLACK FIRST-RUN PRODUCT EXPERIENCE (closed)

Conceptual journey (Slack conversational UX, not web):

```text
user contacts LKW in Slack
→ LKW identifies/resolves workspace state
→ welcome / workspace selection or creation
→ add/provide first knowledge using supported Slack-compatible path
→ snapshot-driven preparation/sync state
→ attention/recovery where needed
→ READY
→ suggested first question
→ Ask
→ grounded response + citation display
```

Consumes `WorkspaceSetupSnapshotService` and existing workspace/intake/ask
boundaries. No web frontend. No new onboarding persistence.

### Remaining PRODUCT-2…PRODUCT-12 context

Future product tasks beyond PRODUCT-4 must still prove the contract in this
canonical order:

- **PRODUCT-2 - ZERO-TO-VALUE INSTALLATION:** **CLOSED** (see above).
- **PRODUCT-3 - FIRST-RUN ONBOARDING:** **CLOSED** (see closed milestones).
- **PRODUCT-4 - SLACK DAILY-USE PRODUCT EXPERIENCE:** Daily Slack UX using
  shared backend capabilities (see frozen definition above). **Not** generic
  “real product UI” and **not** a web-frontend requirement for LKW 1.0.
- **PRODUCT-5 - REAL VENDOR EXPERIENCE:** Verify and productize Slack, Google
  Workspace and Microsoft 365 end to end before claiming public support.
- **PRODUCT-6 - DAILY KNOWLEDGE EXPERIENCE:** Prove the core daily Ask
  experience: grounded answers, citations/source opening, freshness, source
  scoping, refresh, Indexed/Live visibility where supported, and
  insufficient-evidence behavior.
- **PRODUCT-7 - BACKGROUND OPERATION:** Prove scheduled/incremental/on-demand
  synchronization, retry behavior, last/next sync and stale indication
  without repetitive manual intervention.
- **PRODUCT-8 - HUMAN FAILURE RECOVERY:** Map technical failures to
  understandable user-facing state, recommended actions, retryability and
  attention-required behavior.
- **PRODUCT-9 - UPDATE / MIGRATION / BACKUP EXPERIENCE:** Define and prove
  persistence across supported updates, schema/config migration,
  compatibility, backup guidance and rollback expectations.
- **PRODUCT-10 - HUMAN OBSERVABILITY:** Expose clear user-facing health/status
  for the application, dependencies, connections and source synchronization
  state.
- **PRODUCT-11 - REAL-USER ACCEPTANCE:** A real user unfamiliar with the
  repository must independently install, configure, connect own data, ask,
  inspect citations, operate source lifecycle, restart and continue.
- **PRODUCT-12 - VALUE PROOF:** Demonstrate strong real-world use cases
  showing that LKW is useful, not merely technically functional.

Exact implementation status and production readiness remain validation work.
When a gap is found, the next task must identify the smallest reusable
platform change required by a product requirement; it must not add mechanism
without a user need.

## Change control

Once accepted, this document is the canonical LKW daily-use product contract.
Later roadmap tasks may discover implementation gaps, but must not silently
expand or redefine the product. Material changes to the primary personas,
first-product vendor set, core workflows, product boundaries, or non-goals
require an explicit product/architecture decision.
