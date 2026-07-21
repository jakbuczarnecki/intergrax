# Local Workspace Application — Implementation Plan

**Status:** Working product roadmap (2026-07-21)
**Task:** LKW-PLAN-SURFACES
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
**External verification:** [`docs/public-adoption/LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md)

```text
Current product level: Backend Product Alpha
Current roadmap stage: Stage 1 — Trusted Ask Workspace
Current implementation focus: Architectural discovery for Ask Workspace

Primary goal:
Deliver an installable, daily-usable, auditable and operationally safe LKW 1.0.
```

---

## 1. Document role and source of truth

| Document | Role |
|----------|------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How LKW is built — ownership, boundaries, runtime shape |
| **This file (`IMPLEMENTATION_PLAN.md`)** | Where the product is going and what we execute now |
| [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) | How an external person verifies working capabilities |
| [`journal/`](journal/) | Chronological implementation notes (historical detail) |

This document is the **only** source of truth for:

- LKW development order
- current stage
- next vertical slice
- production gates
- deferred scope

Do **not** create a separate `PRODUCT_ROADMAP.md`.

### Governing rule

```text
product roadmap controls execution order
platform work is pulled by product needs
proof is part of product acceptance
```

Platform mechanisms, provider ports, observability vendors, and historical proof tracks are **not** an independent LKW execution queue. They enter the active plan only when a production stage requires them.

---

## 2. Product objective

LKW (Local Knowledge Workspace) is a **local system for working with a user's private documents**.

A target user must be able to:

1. Install LKW
2. Create a workspace
3. Point at folders
4. Synchronize documents
5. Ask questions
6. Receive answers with sources
7. Generate artifacts
8. Browse history
9. Diagnose problems
10. Update the system without data loss

LKW has three roles:

| Role | Meaning |
|------|---------|
| **Real product** | Primary — ship a usable local workspace product |
| **Platform proof** | Secondary — working product flows prove Intergrax capabilities |
| **Platform problems detector** | Secondary — product pressure surfaces reusable platform gaps |

The last two roles exist **because** we build real product functions. They do not define the product roadmap order.

### Ownership boundary

```text
LKW owns product capabilities and local execution.

Intergrax owns normalized interaction intake,
identity/context propagation, task execution and result delivery contracts.

Slack, Microsoft Teams, MCP, HTTP and a minimal local companion
are replaceable surfaces over the same LKW capabilities.
```

LKW is not a Slack bot. LKW is a local product built on Intergrax. Slack is the first familiar conversational surface — a replaceable frontend adapter over Intergrax interaction intake and LKW capabilities. Removing or disconnecting Slack does not disable LKW.

---

## 3. Replaceable interaction surfaces

### Product principle

**Replaceable interaction surfaces** means every familiar tool reaches LKW through the same normalized path:

```text
Known user tool
    ↓
surface adapter
    ↓
normalized Intergrax interaction envelope
    ↓
identity + tenant + workspace context
    ↓
LKW capability Task
    ↓
Nexus / agents / tools
    ↓
typed TaskResult
    ↓
surface-neutral response model
    ↓
surface renderer
```

```text
The surface translates interaction.

The surface does not implement the product capability.
```

Slack must not become:

- the LKW runtime
- the source of product business logic
- the owner of workspace state
- the owner of agent orchestration
- a required dependency of the LKW domain
- the only supported way to use LKW

### Surface-neutral product logic

The following must **not** be implemented independently inside Slack, Teams, or another surface adapter:

- workspace selection rules
- document search
- Ask Workspace orchestration
- synthesis
- citation creation
- artifact generation
- operation lifecycle
- authorization policy
- tenant isolation
- workspace isolation
- run persistence
- retry semantics

### Surface adapter responsibilities

A surface adapter may own only:

- receiving platform events
- validating platform authenticity
- mapping external identity
- acknowledging the incoming event
- mapping channel/thread identity to an Intergrax interaction session
- rendering typed results
- handling platform message limits
- buttons, menus and platform-specific presentation
- sending status and final responses

### Surface portfolio

#### Canonical product surfaces

| Surface | Status | Purpose |
|---------|--------|---------|
| **HTTP API** | implemented / technical canonical surface | Stable public product contract; application integration; testing; fallback independent of external communication vendors |
| **MCP** | implemented / technical user and developer surface | Access from MCP-compatible tools; developer workflows; demonstration that LKW is not tied to a chat vendor |

#### Planned familiar-tool surfaces

| Surface | Status | Purpose |
|---------|--------|---------|
| **Slack** | planned / first conversational reference surface | Direct messages with the LKW bot; workspace selection; Ask Workspace; answers with citations; operation status; artifact notifications; approval interactions; thread-based task continuity |
| **Microsoft Teams** | planned candidate / second business conversational surface | Prove the same LKW capabilities from a second widely known business tool; verify normalized interaction and response contracts; demonstrate that Slack concepts have not leaked into the product domain |
| **Local companion** | planned / configuration and operations surface | Installation; daemon status; filesystem folder picker; allowlist configuration; linking an external interaction surface; model and secret settings; diagnostics; updates; backup and restore |

**Slack preferred transport:** Slack Socket Mode — preferred for local LKW because the local daemon initiates an outbound connection and does not need a publicly exposed webhook endpoint.

**Teams transport:** Do not prescribe the final Teams transport or implementation framework in this roadmap. Resolve by focused discovery when the Teams adapter becomes active.

```text
The local companion is not required to duplicate the full conversational UI.
```

#### Possible later surfaces (non-committed candidates)

- e-mail intake and delivery
- other enterprise messaging tools
- a richer native chat UI
- mobile notifications

Do **not** promise WhatsApp, Discord, or other consumer platforms for LKW 1.0.

### 1.0 surface requirement

```text
At least one first-class conversational surface is required for LKW 1.0.

Slack Socket Mode is the reference conversational surface.

LKW capabilities remain usable through canonical HTTP and MCP surfaces.

A second recognized tool surface must demonstrate that the product
and Intergrax interaction layer do not depend on Slack.
```

Implementation of Teams remains later than Slack, but the production roadmap requires a **surface portability demonstration** before LKW 1.0.

This roadmap does **not** claim that Slack or Teams already works.

---

## 4. Current product state

```text
Current product level: Backend Product Alpha
Current roadmap stage: Stage 1 — Trusted Ask Workspace
Current implementation focus: architectural discovery for Ask Workspace
```

### Working today (implemented / live-verified)

| Capability | State |
|------------|-------|
| Local application host | implemented / live-verified |
| HTTP API | implemented / live-verified |
| Managed workspaces | implemented / live-verified |
| Folder sources | implemented / live-verified |
| Durable synchronization | implemented / live-verified |
| `DocumentStoreTaskQueue` | implemented / live-verified |
| Restart recovery for queued operations | implemented / live-verified |
| Idempotent ingest | implemented / live-verified |
| Workspace isolation | implemented / live-verified |
| Tenant isolation | implemented / live-verified |
| Structured search evidence | implemented / live-verified |
| Source provenance | implemented / live-verified |
| Persistent state | implemented / live-verified |
| Live proof + ProofReceipt | implemented / live-verified |
| MCP surface | implemented |
| Interaction intake baseline (platform/application) | implemented |

### Not yet a finished product

Honest gaps relative to LKW 1.0:

| Capability | State |
|------------|-------|
| Ask Workspace (public Q&A with stable citations) | planned (Stage 1) |
| Slack conversational surface | planned (Stage 2) |
| Normalized surface adapter contract | planned (Stage 2) |
| Second-surface portability demonstration | planned (Stage 2) |
| Minimal local companion | planned (Stage 5) |
| External identity mapping | planned (Stage 2) |
| Outbound data policy | planned (Stage 2) |
| Full document lifecycle | planned (Stage 3) |
| Outputs and history | planned (Stage 4) |
| Operations / security / release hardening | planned (Stage 6) |
| Token optimization runtime | deferred until Ask Workspace baseline exists |
| LKW 1.0 release | planned (Stage 6 gate) |

Backend Product Alpha means: a real host, real workspaces, real sync, and real search evidence exist — not that the product is complete.

---

## 5. Definition of LKW 1.0

LKW 1.0 is the first production version:

```text
local
single-user
installable
restart-safe
source-file-safe
auditable
daily-usable
```

### Required for LKW 1.0

- Canonical HTTP product API
- MCP compatibility
- One first-class familiar conversational surface
- Slack as the reference conversational surface
- A second-surface portability demonstration
- Minimal local companion for installation and configuration

### Not required for LKW 1.0

- SaaS
- Kubernetes
- Enterprise RBAC
- Multiple organizations
- Mobile application
- Every model / vector / storage provider
- Full matrix of every operating system
- Production support for every communication platform
- Feature parity of presentation across all surfaces
- A large proprietary desktop chat application

```text
Semantic capability parity is required.

Pixel-level or platform-widget parity is not.
```

### Minimal 1.0 promises

1. Source files are never modified
2. Answers are limited to the workspace
3. Answers cite sources
4. Data survives restart
5. Operation errors are visible
6. Artifacts land in the shadow workspace
7. Filesystem access is controlled
8. History of major actions is available
9. Installation does not require a repository checkout
10. Updates do not destroy data
11. Conversational surfaces invoke the same surface-neutral capabilities as HTTP/MCP
12. Disconnecting Slack does not disable the local product

---

## 6. Execution principles

### Vertical slices first

Every active task delivers a real user-facing capability (or a frozen architecture needed for that capability). Avoid platform-only waves that do not unlock a product outcome.

### Platform work is product-pulled

A platform mechanism is built only when it blocks a real product flow in the current stage.

### Surface neutrality

```text
Every user-facing LKW capability is implemented once.

Surfaces invoke the capability and render its typed result.

No surface owns a private version of the product workflow.
```

Before accepting a surface integration, verify:

- no Slack-specific field in LKW domain models
- no Teams-specific field in LKW agent contracts
- no duplicate prompt or orchestration logic
- no separate persistence model for the same run
- identical tenant/workspace isolation
- identical policy behavior
- identical ProofReceipt semantics where applicable

### One major platform gap per task

If a second major platform gap appears during a slice:

```text
BLOCKED_BY_PLATFORM_GAP
```

Stop. Record the gap. Do not open a parallel platform refactor inside the same task.

### Proof as acceptance

Proof is not a separate work program.

Every working vertical slice updates:

```text
docs/public-adoption/LKW_PLATFORM_PROOF.md
```

### Test order

```text
contract
→ unit
→ boundary integration
→ API
→ one live run
```

### Token budget

Ordinary task:

```text
soft limit: 2M
stop and review: 4M
```

Large vertical slice:

```text
soft limit: 4M
stop and review: 8M
```

After exceedance:

```text
TOKEN_BUDGET_EXCEEDED
```

Cursor must not run an open-ended fix loop past budget. Stop, report, and wait for operator review.

---

## 7. Production roadmap

Six production stages define LKW development order. Complete earlier stages before expanding later ones, unless a later stage item is explicitly documented as a blocker of the current stage.

```text
Stage 2 depends on the completed, typed and persisted Ask Workspace contract from Stage 1.
```

Slack work must not begin before the surface-neutral Ask Workspace capability exists.

---

### Stage 1 — Trusted Ask Workspace

**Status:** `CURRENT`

#### User outcome

The user asks a workspace question and receives a checkable answer grounded in documents.

#### Product flow

```text
question
→ workspace-scoped retrieval
→ evidence
→ synthesis
→ answer
→ stable citations
→ persisted run
```

#### Required capabilities

- Public Ask Workspace API
- Workspace-scoped retrieval
- Context assembly
- Synthesis
- Stable citations
- Insufficient-evidence behavior
- Persisted run
- Persisted question, evidence, and answer
- Read of a completed run
- Timeout and failure states
- **Surface-neutral** Ask Workspace capability and typed response — usable without Slack-specific fields

#### Completion gate

- Real question
- Real documents
- Answer
- Citations
- No hallucination when evidence is missing
- Tenant / workspace isolation
- Persistence across restart
- Public live proof
- Completion uses the **canonical HTTP path first**
- Public result contract has no Slack-specific fields

Do **not** make Slack part of the Stage 1 implementation. Stage 1 creates the surface-neutral capability that later surfaces expose.

#### Next action

```text
Focused architectural discovery:
local.workspace.search
→ typed evidence
→ local.workspace.synthesize
→ citations
→ persisted run result
```

---

### Stage 2 — Replaceable Conversational Surfaces

**Status:** planned

```text
Depends on: completed, typed and persisted Ask Workspace contract from Stage 1.
```

#### User outcome

The user can interact with LKW from a familiar communication tool without learning a new conversational application.

The same Ask Workspace capability works through multiple interchangeable surfaces.

#### Required capabilities

- Normalized interaction envelope
- Normalized external identity mapping
- Tenant and workspace context resolution
- Conversation/thread session mapping
- Typed surface-neutral response contract
- Asynchronous acknowledgement
- Durable long-operation delivery
- Duplicate-event protection
- External-user allowlist
- Fail-closed authorization
- Outbound data policy
- Message-size-aware rendering
- Offline/local-host status behavior
- Slack Socket Mode adapter
- Slack DM flow
- Slack thread continuity
- Workspace selection
- Ask Workspace answer with citations
- Operation status messages
- Approval interaction baseline
- Second-surface adapter demonstration using Microsoft Teams
- Canonical HTTP/MCP paths remain operational

#### Slack reference flow

```text
Slack user
→ DM or mention
→ Slack Socket Mode adapter
→ normalized interaction intake
→ identity and workspace resolution
→ local.workspace.ask
→ typed answer and citations
→ Slack renderer
→ thread reply
```

#### Surface portability flow

```text
same workspace
same question
same LKW capability
same policy
same run semantics
same citation model

→ invoked through Slack
→ invoked through Microsoft Teams or the selected second business surface
```

Surface formatting may differ. Product semantics must not.

#### Privacy requirement

Distinguish:

```text
local processing
```

from:

```text
data never leaves the device
```

When an answer, citation, snippet or artifact is sent to Slack or Teams, that content enters an external cloud service.

Required product controls:

- Explicit external-surface connection
- Approved external workspace/tenant
- Approved external user IDs
- Configurable outbound content policy
- Clear warning that delivered content leaves the local device
- No automatic artifact upload without policy or approval

#### Offline behavior

The local daemon may be unavailable because the computer is offline, asleep, shut down, or not running LKW.

Require a clear offline state rather than implying cloud-agent availability.

#### Completion gate

Stage 2 is complete only when:

1. Slack Socket Mode works with a real Slack workspace
2. A user can select a workspace
3. A user can ask a real workspace question
4. The answer contains citations
5. Long operations acknowledge quickly and complete asynchronously
6. Duplicate Slack events do not duplicate product operations
7. Unauthorized Slack users fail closed
8. Thread/session context does not leak across workspaces
9. Slack-specific data is absent from LKW domain and agent contracts
10. HTTP and MCP remain functional
11. The same product capability is exercised through a second recognized surface
12. The second adapter does not require duplication of Ask Workspace logic
13. A public proof demonstrates surface interchangeability
14. Cloud-channel data exposure is clearly documented and policy-controlled

---

### Stage 3 — Reliable Document Lifecycle

**Status:** planned

#### User outcome

The workspace automatically stays aligned with folder contents.

#### Required capabilities

- New files
- Modified files
- Deleted files
- Rename / move
- Stale chunk removal
- Retry
- Document status
- Partial sync result
- Operation recovery
- Continuous synchronization

#### Surface requirement

Relevant synchronization status and failures must be available through:

- the canonical API
- the connected conversational surface
- the local companion

Do **not** put document reconciliation logic inside Slack or Teams.

#### Completion gate

- File change changes search results
- Deleted file disappears from retrieval
- Rename does not create uncontrolled duplicates
- One bad file does not hide the rest
- Retry is explicit
- Restart does not lose lifecycle state
- Sources remain read-only
- Sync status / failures visible on API, conversational surface, and companion

---

### Stage 4 — Workspace Outputs and History

**Status:** planned

#### User outcome

The user generates durable work products from workspace knowledge.

#### Required capabilities

- Reports
- Summaries
- E-mails
- Timelines
- Fact tables
- Risk lists
- Shadow artifacts
- Artifact provenance
- Versioning
- Run history
- Explicit export approval

#### Multi-surface delivery

- Short status and summary may be delivered through Slack or Teams
- Complete artifact remains in the shadow workspace by default
- External artifact upload requires explicit policy or approval
- Artifact provenance remains surface-neutral
- Run history is shared across surfaces
- Asking through Slack and viewing through the local companion must reference the same run and artifact

#### Completion gate

- At least three artifact types
- Durable artifact
- Citations / provenance
- Restart persistence
- Shadow-only default
- Explicit export consent
- Readable failure status
- Multi-surface delivery rules above satisfied

---

### Stage 5 — Installable Local Companion

**Status:** planned

#### User outcome

A non-technical user installs the local LKW runtime, connects familiar interaction tools and manages local-only configuration without using the repository.

#### Required capabilities

- Windows installer first
- Local daemon lifecycle
- System startup
- Folder picker
- Workspace and source setup
- Filesystem allowlist
- Slack connection setup
- External surface connection status
- Approved user/workspace mapping
- Model configuration
- Secret storage
- Health
- Diagnostics
- Update
- Uninstall
- Backup/restore entry points

#### Scope boundary

```text
The companion does not need to duplicate Slack or Teams as a full chat client.
```

It may include a minimal local fallback interaction view, but this is not required to reproduce the complete conversational experience before 1.0.

#### Completion gate

- Clean Windows install
- No repository checkout
- Daemon starts
- User selects a folder
- User links Slack
- User sees connection health
- User can revoke the connection
- Ask Workspace works through the connected surface
- Canonical local API remains available
- Update preserves data and connection configuration
- Safe uninstall
- Diagnostics bundle

---

### Stage 6 — Operational, Security, Quality and Release Gate

**Status:** planned

#### User outcome

The system is predictable, diagnosable, safe for long-lived use, and correct / economical enough to be marked 1.0.

#### Required capabilities

- Component health
- Worker health
- Queue health
- Failed operation visibility
- Retry / recovery
- Backup / restore
- Migrations
- Log retention
- Secure localhost client access
- Secret storage
- File limits
- Parser safety
- Symlink policy
- Resource limits
- Audit
- Diagnostics bundle
- Versioned quality corpus
- Retrieval quality metrics
- Citation correctness
- Unsupported-answer measurement
- Leakage measurement
- Latency
- Token usage
- Model cost
- Storage usage
- Provider configuration
- One alternative model provider
- Soak testing
- Release checklist

#### Surface-specific readiness

- Slack token and app secrets stored securely
- External provider credentials excluded from logs and receipts
- External event replay protection
- Duplicate delivery idempotency
- Rate-limit handling
- Message length and file upload limits
- External identity audit
- Outbound data policy audit
- Connection revoke behavior
- External surface outage handling
- Slack unavailable does not block HTTP/MCP/local administration
- Second-surface outage does not corrupt LKW state

#### Token Optimization placement

```text
Token Optimization is not the current standalone roadmap.
It is applied after a stable Ask Workspace baseline exists.
```

Order:

```text
real Ask Workspace
→ baseline measurement
→ optimization
→ quality and cost comparison
```

#### LKW 1.0 release gate

1. Core local runtime works independently of Slack and Teams
2. Canonical HTTP API works
3. MCP remains available
4. Slack works as the first conversational surface
5. At least one second recognized surface demonstrates adapter interchangeability
6. Product logic is not duplicated between surfaces
7. Local companion handles setup and diagnostics
8. Disabling or disconnecting Slack does not disable LKW
9. External data delivery policy is enforced
10. Remaining product, security, quality and operational gates pass

Also retain:

- Install works
- Document lifecycle works
- Ask Workspace works
- Citations work
- Artifacts work
- History works
- Backup / restore works
- Security gate works
- Quality thresholds pass
- Cost baseline is known
- Soak test passes
- Public proof is complete

---

## 8. Current stage

```text
Current stage: Stage 1 — Trusted Ask Workspace
Current status: Discovery required before implementation
Next deliverable: frozen architecture and task definition for LKW-PRODUCT-2
```

### Discovery must establish

- Existing synthesize flow
- Existing citation model
- Current run persistence
- Structured output path
- Product / platform boundary
- One predicted platform blocker
- Focused boundary test

Do **not** start a full Ask Workspace implementation task until discovery freezes architecture and the LKW-PRODUCT-2 task definition.

Do **not** start Stage 2 Slack work before the surface-neutral Ask Workspace capability from Stage 1 exists.

```text
Stage 2 depends on the completed, typed and persisted Ask Workspace contract from Stage 1.
```

---

## 9. Production gates

Every stage must pass the applicable gates before it is closed.

### Product gate

Did the user receive a genuinely useful capability?

### Architecture gate

Does LKW use the platform without bypasses and without direct vendor calls from product code?

### Operational gate

Does the flow survive restart, failure, and retry?

### Audit gate

Can an external person verify the flow through:

```text
docs/public-adoption/LKW_PLATFORM_PROOF.md
```

### Surface portability gate

Applies to Stage 2 and every later stage exposing new user-facing capabilities:

```text
Can the capability be invoked through a different surface
without changing its domain logic, agent orchestration or persistence?
```

---

## 10. Platform problem classification

Classify every detected gap as one of:

```text
product-blocking
platform-reusable-nonblocking
production-hardening
```

| Class | Rule |
|-------|------|
| **product-blocking** | Resolve before closing the current stage |
| **platform-reusable-nonblocking** | Record it; do not auto-create the next task |
| **production-hardening** | Return in Stage 6 unless it already blocks the product |

```text
Not every detected pattern becomes an implementation task.
```

Historical platform backlog items (proof maturity waves, vendor observability rollouts, provider matrices, PostgreSQL/vector portability without a product need) remain reference material only. They are not the active product order.

---

## 11. Deferred scope

Explicitly deferred until a current production stage documents them as blockers:

- Support for every messaging provider
- macOS installer
- Mobile client
- SaaS
- Multi-organization support
- Enterprise RBAC
- Kubernetes
- PostgreSQL migration without a product need
- Vector-store portability without a product need
- Broad observability vendor rollout
- Prometheus / Grafana / Tempo / Sentry as standalone proofs
- Scaffold propagation as a standalone program
- All-provider matrices
- Autonomous agent actions
- Web search
- Cross-device sync

**Microsoft Teams** is a **planned second-surface candidate for Stage 2 portability acceptance** — not indefinite deferred scope.

```text
Additional communication surfaces beyond the Stage 2 portability pair
remain product-demand-driven.
```

```text
Deferred items may enter the active plan only when they become a documented blocker of a current production stage.
```

---

## 12. Completed milestones

Short register of closed product baselines. Detail lives in verification docs, journal, and git history — not in this roadmap.

| Milestone | Result (one line) | Status | Evidence |
|-----------|-------------------|--------|----------|
| **LKW.0** | Application host baseline and architecture scaffold | Done | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| **LKW.1** | Index / search / synthesize baseline with live product proof | Closed in scope | [`LKW_1_LIVE_VERIFICATION.md`](LKW_1_LIVE_VERIFICATION.md) |
| **LKW.2** | Multi-step pipeline baseline (`local.workspace.*`) | Closed | [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) |
| **LKW.3** | Serving and application composition (`filesystem.*` + allowlist) | Done | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| **LKW.5** | Persistence and restart proof (`LKW_DATA_HOME`, durable vectors) | Closed | [`LKW_5_PERSISTENCE_VERIFICATION.md`](LKW_5_PERSISTENCE_VERIFICATION.md) |
| **LKW.6** | Interaction intake baseline (OS daemon / intake router); future Slack and Teams adapters will consume this platform/application baseline | Closed | [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`journal/`](journal/) |
| **LKW.7** | File watcher and incremental indexing baseline | Closed | [`LKW_7_FILE_WATCHER_VERIFICATION.md`](LKW_7_FILE_WATCHER_VERIFICATION.md) |
| **LKW-PRODUCT-1** | Managed workspaces and folder sources (create / attach / sync / search) | Done | [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) |
| **LKW-PRODUCT-1-HARDENING** | Durable sync and structured search evidence handoff | Done | [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) |

Slack and Teams are **not** historically completed.

---

## 13. Historical references

Detailed historical narratives, micro-wave status tables, and former proof-first queues are **not** the active product roadmap. Consult:

| Location | Contents |
|----------|----------|
| [`journal/`](journal/) | Dated implementation notes |
| Application `*VERIFICATION*.md` docs | Live verification write-ups (e.g. [`LKW_1_LIVE_VERIFICATION.md`](LKW_1_LIVE_VERIFICATION.md), [`LKW_5_PERSISTENCE_VERIFICATION.md`](LKW_5_PERSISTENCE_VERIFICATION.md), [`LKW_7_FILE_WATCHER_VERIFICATION.md`](LKW_7_FILE_WATCHER_VERIFICATION.md)) |
| [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) | Public proof steps and receipts |
| Git history | Exact code and doc evolution |

Former items such as `LKW-PF0–LKW-PF7`, Token Optimization sequences (`TOKEN-1A` …), observability vendor proof packs, PostgreSQL / vector portability as standalone obligations, and scaffold propagation programs are classified as:

```text
historical platform backlog
```

They must not reappear as the active product execution order. Reintroduce only under §10 / §11 when a current production stage documents a blocker.

---

## Appendix A — Document map (quick)

```text
ARCHITECTURE.md
→ how LKW is built

IMPLEMENTATION_PLAN.md
→ where the product is going and what we execute now

LKW_PLATFORM_PROOF.md
→ how an external person verifies working capabilities
```

## Appendix B — Status vocabulary

Use these labels consistently in future updates:

| Label | Meaning |
|-------|---------|
| `implemented` | Code and tests exist in-repo |
| `live-verified` | Demonstrated in a live run / proof |
| `planned` | On the active six-stage roadmap |
| `candidate` | Recognized option pending discovery / portability acceptance |
| `deferred` | Explicitly out of active order until a documented blocker |

Do not mark Ask Workspace, Slack, Teams, surface portability, local companion, full document reconciliation, outputs/history, operations/security/release hardening, token optimization runtime, or LKW 1.0 as done until the matching stage gates pass.
