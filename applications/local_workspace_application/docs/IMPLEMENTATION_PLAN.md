# Local Workspace Application — Implementation Plan

**Status:** Working product roadmap (2026-07-21)
**Task:** LKW-PLAN-RESET
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

---

## 3. Current product state

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

### Not yet a finished product

Honest gaps relative to LKW 1.0:

| Capability | State |
|------------|-------|
| Ask Workspace (public Q&A with stable citations) | planned (Stage 1) |
| Citations in a final user-facing answer | planned |
| Full document reconciliation (delete/rename/stale) | planned (Stage 2) |
| Desktop UI / tray client | planned (Stage 4) |
| Installer | planned (Stage 4) |
| Backup / restore | planned (Stage 5) |
| Production security hardening | planned (Stage 5) |
| Token optimization runtime | deferred until Ask Workspace baseline exists |
| LKW 1.0 release | planned (Stage 6 gate) |

Backend Product Alpha means: a real host, real workspaces, real sync, and real search evidence exist — not that the product is complete.

---

## 4. Definition of LKW 1.0

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

### Not required for 1.0

- SaaS
- Kubernetes
- Enterprise RBAC
- Multiple organizations
- Slack
- Mobile application
- Every model / vector / storage provider
- Full matrix of every operating system

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

---

## 5. Execution principles

### Vertical slices first

Every active task delivers a real user-facing capability (or a frozen architecture needed for that capability). Avoid platform-only waves that do not unlock a product outcome.

### Platform work is product-pulled

A platform mechanism is built only when it blocks a real product flow in the current stage.

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

## 6. Production roadmap

Six production stages define LKW development order. Complete earlier stages before expanding later ones, unless a later stage item is explicitly documented as a blocker of the current stage.

---

### Stage 1 — Trusted Ask Workspace

**Status:** `CURRENT`

#### User outcome

The user asks a workspace question and receives a checkable answer grounded in documents.

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

#### Completion gate

- Real question
- Real documents
- Answer
- Citations
- No hallucination when evidence is missing
- Tenant / workspace isolation
- Persistence across restart
- Public live proof

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

### Stage 2 — Reliable Document Lifecycle

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

#### Completion gate

- File change changes search results
- Deleted file disappears from retrieval
- Rename does not create uncontrolled duplicates
- One bad file does not hide the rest
- Retry is explicit
- Restart does not lose lifecycle state
- Sources remain read-only

---

### Stage 3 — Workspace Outputs and History

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

#### Completion gate

- At least three artifact types
- Durable artifact
- Citations / provenance
- Restart persistence
- Shadow-only default
- Explicit export consent
- Readable failure status

---

### Stage 4 — Installable Local Application

**Status:** planned

#### User outcome

A non-technical user installs LKW and uses it without a repository checkout.

#### Required capabilities

- Windows installer as first target
- Daemon lifecycle
- Thin desktop / tray client
- Workspace management
- Folder picker
- Sync status
- Ask Workspace
- Citations
- History
- Settings
- Diagnostics
- Update
- Uninstall

#### Completion gate

- Clean Windows install
- Launch from system menu
- Automatic host start
- Complete flow from UI
- Data preserved on update
- Diagnostics bundle
- Safe uninstall

---

### Stage 5 — Operational and Security Readiness

**Status:** planned

#### User outcome

The system is predictable, diagnosable, and safe for long-lived use.

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

#### Completion gate

- Health visible
- Recovery passes
- Backup / restore passes
- Update migration passes
- Secrets do not appear in logs
- Foreign process does not get free API access
- Soak test passes

---

### Stage 6 — Cost, Quality and Release Gate

**Status:** planned

#### User outcome

LKW is correct, fast, and economical enough to be marked 1.0.

#### Required capabilities

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
- Release checklist

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

#### Release gate

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

## 7. Current stage

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

---

## 8. Production gates

Every stage must pass four gates before it is closed.

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

---

## 9. Platform problem classification

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
| **production-hardening** | Return in Stage 5 unless it already blocks the product |

```text
Not every detected pattern becomes an implementation task.
```

Historical platform backlog items (proof maturity waves, vendor observability rollouts, provider matrices, PostgreSQL/vector portability without a product need) remain reference material only. They are not the active product order.

---

## 10. Deferred scope

Explicitly deferred until a current production stage documents them as blockers:

- Slack Socket Mode
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

```text
Deferred items may enter the active plan only when they become a documented blocker of a current production stage.
```

---

## 11. Completed milestones

Short register of closed product baselines. Detail lives in verification docs, journal, and git history — not in this roadmap.

| Milestone | Result (one line) | Status | Evidence |
|-----------|-------------------|--------|----------|
| **LKW.0** | Application host baseline and architecture scaffold | Done | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| **LKW.1** | Index / search / synthesize baseline with live product proof | Closed in scope | [`LKW_1_LIVE_VERIFICATION.md`](LKW_1_LIVE_VERIFICATION.md) |
| **LKW.2** | Multi-step pipeline baseline (`local.workspace.*`) | Closed | [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) |
| **LKW.3** | Serving and application composition (`filesystem.*` + allowlist) | Done | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| **LKW.5** | Persistence and restart proof (`LKW_DATA_HOME`, durable vectors) | Closed | [`LKW_5_PERSISTENCE_VERIFICATION.md`](LKW_5_PERSISTENCE_VERIFICATION.md) |
| **LKW.6** | Interaction intake baseline (OS daemon / intake router) | Closed | [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`journal/`](journal/) |
| **LKW.7** | File watcher and incremental indexing baseline | Closed | [`LKW_7_FILE_WATCHER_VERIFICATION.md`](LKW_7_FILE_WATCHER_VERIFICATION.md) |
| **LKW-PRODUCT-1** | Managed workspaces and folder sources (create / attach / sync / search) | Done | [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) |
| **LKW-PRODUCT-1-HARDENING** | Durable sync and structured search evidence handoff | Done | [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) |

---

## 12. Historical references

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

They must not reappear as the active product execution order. Reintroduce only under §9 / §10 when a current production stage documents a blocker.

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
| `deferred` | Explicitly out of active order until a documented blocker |

Do not mark Ask Workspace, final-answer citations, full document reconciliation, UI, installer, backup/restore, production security, token optimization runtime, or LKW 1.0 as done until the matching stage gates pass.
