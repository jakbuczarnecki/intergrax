# Local Workspace Application — Implementation Plan

**Status:** Product-first MVP roadmap (2026-07-22)  
**Governing product rule:** [`PRODUCT_FIRST_MVP.md`](../../../docs/plan/PRODUCT_FIRST_MVP.md)  
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)  
**Ask Workspace discovery:** [`ASK_WORKSPACE_DISCOVERY.md`](ASK_WORKSPACE_DISCOVERY.md)
**External verification:** [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md)

```text
Current product level: Backend Product Alpha
Current milestone: LKW MVP
Current roadmap stage: Stage 1 — Trusted Ask Workspace
Current implementation focus: MVP-2 — Trusted Ask Workspace implementation

Immediate goal:
Deliver the smallest complete LKW experience that a real user can try and value:
local documents → Slack question → grounded answer with sources.

Longer-term goal:
Deliver an installable, daily-usable, auditable and operationally safe LKW 1.0.
```

---

## 1. Document role and source of truth

| Document | Role |
|----------|------|
| [`PRODUCT_FIRST_MVP.md`](../../../docs/plan/PRODUCT_FIRST_MVP.md) | Governing product-development rule for every Intergrax application and agent |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How LKW is built — ownership, boundaries and runtime shape |
| **This file (`IMPLEMENTATION_PLAN.md`)** | LKW product brief, MVP execution order, post-MVP direction and current task |
| [`ASK_WORKSPACE_DISCOVERY.md`](ASK_WORKSPACE_DISCOVERY.md) | Frozen Ask Workspace contract and exact MVP-2 implementation scope |
| [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) | How working capabilities are externally verified |
| [`journal/`](journal/) | Historical implementation notes |

This file is the only source of truth for:

- the LKW product brief,
- the MVP definition,
- the active execution order,
- the current vertical slice,
- the MVP validation gate,
- post-MVP product direction,
- the LKW 1.0 release direction,
- deferred scope.

### Governing rule

```text
Deliver the smallest real product experience that demonstrates user value.
Use implementation of that product to discover and improve Intergrax.
Do not build the platform first and hope a useful product appears later.
```

For LKW this means:

```text
real user problem
→ smallest valuable local-document workflow
→ working LKW MVP
→ real-user validation
→ feedback-driven next priority
→ only then broader product and platform hardening
```

Platform mechanisms, provider ports, observability vendors, portability matrices and historical proof tracks are not an independent LKW execution queue.

---

## 2. LKW product brief

### 2.1 What we are building

LKW (Local Knowledge Workspace) is a local product that lets a user work with private documents through natural-language questions and verifiable answers.

The user designates local folders. LKW indexes those files, keeps the source files read-only, retrieves relevant evidence and returns an answer with sources.

### 2.2 Why it should exist

People working with document-heavy matters lose time searching folders, opening many files, locating specific facts and verifying where information came from.

General-purpose assistants often require manual uploads or do not provide durable provenance, local filesystem boundaries or inspectable execution.

LKW should make this workflow faster while keeping the source material under the user's control.

### 2.3 First target user

The first MVP user is:

```text
one knowledge worker
using one local Windows machine
working with one or more document-heavy matters
using Slack in daily work
```

Representative users include:

- a lawyer or legal-support user reviewing case files,
- a consultant analysing project documents,
- a business owner reviewing contracts and correspondence,
- a project manager working across specifications, notes and reports.

The first validation does not target everyone. It targets one concrete design partner with a real folder and real questions.

### 2.4 Current pain

Today the user typically:

1. opens a folder,
2. searches filenames or text manually,
3. opens multiple documents,
4. compares fragments,
5. writes a summary,
6. checks again where every statement came from.

Time is lost in retrieval, comparison and source verification.

### 2.5 Value proposition

```text
For a document-heavy knowledge worker
who needs to find and verify information across local files,
LKW provides a grounded answer with inspectable sources
through a familiar conversational tool,
without requiring repeated manual document search or a separate chat application.
```

### 2.6 Primary user workflow

```text
user designates a local folder
→ LKW creates or uses a managed workspace
→ documents are synchronized
→ user opens Slack
→ user selects the workspace
→ user asks a question
→ LKW retrieves local evidence
→ LKW produces a grounded answer
→ Slack displays the answer and sources
→ user verifies and uses the result
```

### 2.7 Product roles

LKW has three roles:

| Role | Priority | Meaning |
|------|----------|---------|
| **Real product** | Primary | Solve a real local-document problem for a user |
| **Platform proof** | Secondary | Working LKW flows demonstrate Intergrax capabilities |
| **Platform problem detector** | Secondary | Product pressure exposes concrete reusable platform gaps |

The secondary roles exist because a real product is being built. They do not define implementation order independently of user value.

---

## 3. LKW MVP definition

### 3.1 MVP value statement

```text
A real user can ask a question in Slack about their local documents
and receive a useful, grounded answer with verifiable sources.
```

### 3.2 MVP scope

The MVP supports:

- one local LKW installation,
- one user,
- one approved Slack workspace,
- one approved Slack user,
- one or more managed LKW workspaces,
- local folder sources,
- repeatable synchronization,
- Ask Workspace,
- grounded answers,
- citations or source references,
- persisted run result,
- basic error states,
- minimal repeatable setup for a design partner.

### 3.3 MVP may remain intentionally limited

The MVP may use:

- Windows as the only supported operating system,
- manual or scripted installation,
- configuration files or a simple setup command,
- one model provider,
- one vector-store provider,
- a controlled design-partner environment,
- a limited file-type set,
- a simple Slack DM interaction,
- basic operational diagnostics.

### 3.4 Explicitly outside the MVP

The following must not delay first user validation unless a concrete MVP blocker is documented:

- Microsoft Teams,
- a second production conversational adapter,
- a complete local companion UI,
- automatic application updates,
- complete delete/rename/move reconciliation,
- broad artifact generation,
- enterprise RBAC,
- multiple organizations,
- SaaS,
- mobile clients,
- every model or storage provider,
- full observability vendor stack,
- broad portability matrices,
- PostgreSQL migration without product need,
- token optimization before a measured baseline,
- full LKW 1.0 security and operations hardening.

### 3.5 MVP gate

LKW MVP is reached only when:

1. one user can start the local LKW environment through a repeatable setup;
2. the user can create or use a workspace and attach a real local folder;
3. documents synchronize successfully;
4. Ask Workspace works through the canonical HTTP contract;
5. the same Ask Workspace capability is invoked from Slack;
6. one approved Slack user can select the workspace;
7. the user can ask a real question;
8. the answer is grounded in real local documents;
9. the answer includes verifiable sources;
10. insufficient evidence does not produce an invented answer;
11. the flow is repeatable across several questions;
12. the local source files remain unchanged;
13. the user can understand basic failure or offline states;
14. a real user can try the workflow and judge whether it saves time or improves confidence.

### 3.6 MVP value measurement

Technical proof is not enough. MVP validation must collect observable product evidence:

- whether the user completes the flow without developer intervention,
- whether the answer is useful,
- whether the cited sources are correct,
- whether the user finds information faster than manually,
- whether the user trusts the answer more because sources are visible,
- whether the user asks additional real questions,
- whether the user wants to continue using or testing LKW,
- which missing capability blocks repeated use.

A ProofReceipt confirms execution. It does not replace user-value validation.

### 3.7 MVP demonstration

```text
Starting state:
a real local folder is configured and synchronized.

User action:
the user sends a Slack DM, selects the LKW workspace and asks a real question.

Product behavior:
Slack invokes the surface-neutral Ask Workspace capability;
LKW searches local evidence, synthesizes a bounded answer and persists the run.

Visible result:
the user receives an answer with sources in Slack.

Value evidence:
the user verifies the sources and confirms whether the result was useful,
faster or easier than the previous manual workflow.
```

---

## 4. Current product state

```text
Current product level: Backend Product Alpha
Current milestone: LKW MVP
Current active slice: Trusted Ask Workspace
Current implementation focus: MVP-2 — Trusted Ask Workspace implementation
Discovery: ASK_WORKSPACE_DISCOVERY.md (MVP-1 complete)
```

### Working today

| Capability | State |
|------------|-------|
| Local application host | implemented / live-verified |
| HTTP API | implemented / live-verified |
| Managed workspaces | implemented / live-verified |
| Folder sources | implemented / live-verified |
| Durable synchronization | implemented / live-verified |
| `DocumentStoreTaskQueue` | implemented / live-verified |
| Queued-operation restart recovery | implemented / live-verified |
| Idempotent ingest | implemented / live-verified |
| Workspace isolation | implemented / live-verified |
| Tenant isolation | implemented / live-verified |
| Structured search evidence | implemented / live-verified |
| Source provenance | implemented / live-verified |
| Persistent state | implemented / live-verified |
| Live proof + ProofReceipt | implemented / live-verified |
| MCP surface | implemented |
| Interaction intake baseline | implemented |

### Missing before MVP validation

| Capability | State |
|------------|-------|
| Surface-neutral Ask Workspace result | planned — MVP-2 (contract frozen) |
| Final grounded answer with stable sources | planned — MVP-2 (contract frozen) |
| Persisted question/evidence/answer run | planned — MVP-2 (contract frozen) |
| Completed Ask-run read after restart | planned — MVP-2 (contract frozen) |
| Minimal Slack identity and workspace mapping | planned — MVP |
| Slack Socket Mode DM flow | planned — MVP |
| Basic outbound-data warning/policy | planned — MVP |
| Minimal repeatable design-partner setup | planned — MVP |
| Real-user MVP validation | planned — MVP gate |

### Post-MVP gaps

| Capability | State |
|------------|-------|
| Full document reconciliation | post-MVP candidate |
| Workspace outputs and history | post-MVP candidate |
| Full local companion | post-MVP / 1.0 |
| Second conversational adapter | post-MVP candidate |
| Broad operations and security hardening | post-MVP / 1.0 |
| Token optimization runtime | after Ask Workspace baseline and measured need |
| LKW 1.0 release | future release gate |

Backend Product Alpha means the product has a real host, real workspaces, real synchronization and real search evidence. It does not yet mean that a user can experience the complete MVP value.

---

## 5. Ownership and replaceable interaction surfaces

### 5.1 Ownership boundary

```text
LKW owns product capabilities and local execution.

Intergrax owns normalized interaction intake,
identity/context propagation, task execution and result-delivery contracts.

Slack, MCP, HTTP and future tools
are replaceable surfaces over the same LKW capabilities.
```

LKW is not a Slack bot. Slack is the first familiar conversational surface for the MVP.

Disconnecting Slack must not disable the core local product.

### 5.2 Normalized path

```text
known user tool
→ surface adapter
→ normalized Intergrax interaction
→ identity + tenant + workspace context
→ LKW capability Task
→ Nexus / agents / tools
→ typed TaskResult
→ surface-neutral response
→ surface renderer
```

```text
The surface translates interaction.
The surface does not implement the product capability.
```

### 5.3 Surface-neutral product logic

The following must not be reimplemented inside Slack or another adapter:

- document search,
- Ask Workspace orchestration,
- synthesis,
- citation creation,
- artifact generation,
- operation lifecycle,
- authorization policy,
- tenant isolation,
- workspace isolation,
- run persistence,
- retry semantics.

Workspace selection may have a surface-specific interaction, but the resolved workspace identity and access rules belong to LKW/Intergrax contracts, not Slack-only business logic.

### 5.4 Adapter responsibilities

A surface adapter may own:

- receiving platform events,
- validating platform authenticity,
- mapping external identity,
- acknowledging events,
- mapping channel/thread identity to an interaction session,
- rendering typed results,
- handling platform message limits,
- platform-specific buttons and menus,
- sending status and final responses.

### 5.5 Surface portfolio

| Surface | MVP role | Longer-term role |
|---------|----------|------------------|
| **HTTP API** | Canonical Ask Workspace contract and test surface | Stable product integration surface |
| **MCP** | Existing independent technical surface | Developer/tool integration surface |
| **Slack** | First user-facing conversational MVP surface | First-class optional conversational surface |
| **Microsoft Teams** | Not required for MVP | Candidate selected by user demand or commercial value |
| **Local companion** | Minimal setup may remain scripted/manual | Installation, configuration and diagnostics for 1.0 |

Slack Socket Mode remains the preferred MVP transport because the local daemon initiates an outbound connection and does not require a public webhook endpoint.

Before MVP, surface portability is protected by:

- a surface-neutral Ask Workspace contract,
- no Slack-specific fields in domain or agent models,
- canonical HTTP behavior,
- MCP remaining available,
- adapter boundary tests,
- no duplicated orchestration or persistence.

A second large adapter is not required merely to prove the abstraction before user value is validated.

---

## 6. Execution principles

### 6.1 Fastest-value rule

When several tasks are possible, choose the one that most directly reduces the distance to:

```text
local documents → Slack question → grounded answer with sources → real-user validation
```

Decision order:

1. Does the task complete a missing step in the MVP workflow?
2. Does it make the result useful, trustworthy or repeatable?
3. Does it remove a blocker preventing a user from trying the MVP?
4. Does it enable real design-partner validation?
5. Is it a reusable platform improvement required by one of the above?

If all answers are no, the task is not an active LKW MVP priority.

### 6.2 Vertical slices first

Every active implementation task must answer:

```text
What new valuable thing can the user do after this task?
```

A frozen discovery task is allowed only when it is necessary to avoid uncertain implementation of the immediately following vertical slice.

### 6.3 Platform work is product-pulled

A platform mechanism is implemented only when the active LKW workflow cannot be completed safely or correctly without it.

The required relationship is:

```text
real LKW capability
→ concrete platform gap
→ gap classification
→ reusable fix when justified
→ LKW consumes the fix
→ complete product flow is revalidated
```

### 6.4 One major platform gap per task

If a second major platform gap appears:

```text
BLOCKED_BY_PLATFORM_GAP
```

Stop and restore the task boundary.

### 6.5 Stop conditions

Stop and reassess when:

- the task no longer produces or protects an MVP capability,
- abstractions or providers are added without an MVP need,
- repeated fixes target symptoms rather than a frozen contract,
- live proof becomes the main debugging mechanism,
- no user can try the product after several completed tasks,
- implementation cost grows without reducing distance to MVP.

Use:

```text
BLOCKED_BY_PLATFORM_GAP
SCOPE_DRIFT_DETECTED
TOKEN_BUDGET_EXCEEDED
MVP_VALUE_UNCLEAR
```

### 6.6 Proof and validation

Proof is part of technical acceptance, not the product goal.

Working slices update:

```text
docs/public-adoption/LKW_PLATFORM_PROOF.md
```

Validation order:

```text
contract test
→ focused unit test
→ boundary integration test
→ application API test
→ one end-to-end live run
→ real-user or design-partner validation
```

### 6.7 Token discipline

Ordinary task:

```text
soft limit: 2M tokens
stop and review: 4M tokens
```

Large vertical slice:

```text
soft limit: 4M tokens
stop and review: 8M tokens
```

Do not continue an open-ended fix loop beyond the review threshold.

---

## 7. Active MVP execution path

This is the active task order. The later 1.0 roadmap does not override it.

### MVP-0 — Product brief alignment

**Status:** done

Result:

- product purpose defined,
- first user defined,
- primary workflow defined,
- MVP scope and exclusions defined,
- value measurement defined,
- MVP gate defined.

Canonical references:

- this document,
- [`PRODUCT_FIRST_MVP.md`](../../../docs/plan/PRODUCT_FIRST_MVP.md).

### MVP-1 — Trusted Ask Workspace discovery

**Status:** done

Result: frozen Ask Workspace contract and exact MVP-2 scope.

Canonical reference: [`ASK_WORKSPACE_DISCOVERY.md`](ASK_WORKSPACE_DISCOVERY.md).

Key findings:

- reuse managed `POST .../workspaces/{workspace_id}/search` evidence path (`local.workspace.search` → `search_summary.evidence` → `WorkspaceSearchHitV1`);
- do **not** use `local.workspace.synthesize` as the Ask answer engine (shadow-draft writer; ungrounded message fallback);
- LKW must own Ask orchestration, citation projection, Ask-run persistence and completed-run read;
- major blocker: `PRODUCT_BLOCKING` — missing product Ask orchestration/persistence (resolution stays in LKW; not a platform framework).

### MVP-2 — Trusted Ask Workspace

**Status:** `CURRENT`

**One-sentence summary:** Implement surface-neutral HTTP Ask Workspace that reuses managed search evidence, applies an insufficient-evidence gate, produces a grounded answer with projected citations, persists the run, and supports completed-run read after restart.

Frozen contract: [`ASK_WORKSPACE_DISCOVERY.md`](ASK_WORKSPACE_DISCOVERY.md) §4.

User-visible result:

```text
HTTP question
→ workspace-scoped retrieval
→ verified evidence
→ sufficiency gate
→ grounded answer (LKW-owned assembly)
→ stable citations
→ persisted Ask run
→ completed-run read
```

Minimum acceptance (see discovery §10):

- `POST /v1/local_workspace/workspaces/{workspace_id}/ask`,
- surface-neutral typed result (`run_id`, status, question, answer or insufficient-evidence, citations),
- reuse `local.workspace.search` + verified hit mapping,
- insufficient evidence does not invent a grounded answer,
- persist question/evidence/answer/citations/status,
- `GET /v1/local_workspace/asks/{run_id}` with tenant isolation,
- restart persistence,
- no Slack-specific fields; no provider-payload leakage in the Ask result.

Exact scope, forbidden work and required tests: discovery §9–§12.

No Slack-specific fields or logic are permitted in this slice.

### MVP-3 — Slack MVP discovery

**Status:** planned

Purpose: freeze only the architecture needed for the first familiar-tool workflow.

Discovery must establish:

- Socket Mode connection path,
- minimal event-to-interaction mapping,
- approved Slack workspace/user mapping,
- workspace selection interaction,
- asynchronous acknowledgement,
- duplicate-event boundary,
- basic outbound-data rule,
- offline/local-host behavior,
- typed response rendering,
- one expected major platform blocker.

Do not design Teams, a universal messaging framework or a full companion in this task.

### MVP-4 — Slack conversational MVP

**Status:** planned

User-visible result:

```text
approved Slack user
→ DM to LKW
→ select workspace
→ ask real question
→ same Ask Workspace capability
→ answer with sources
→ Slack thread reply
```

Minimum acceptance:

- real Slack workspace,
- Socket Mode,
- one approved user,
- one approved Slack workspace,
- workspace selection,
- Ask Workspace invocation,
- grounded answer with sources,
- quick acknowledgement for long work,
- duplicate-event protection,
- fail-closed unauthorized user behavior,
- no Slack-specific product logic,
- HTTP and MCP still work,
- clear warning that Slack delivery sends content to an external cloud service.

Not required in this slice:

- Microsoft Teams,
- broad approval workflows,
- artifact uploads,
- feature-complete Slack UI,
- multiple users,
- enterprise administration.

### MVP-5 — Minimal design-partner package

**Status:** planned

User-visible result: a design partner can start and configure the MVP through a documented repeatable path.

Minimum scope:

- Windows-first setup,
- one installation or bootstrap script,
- one configuration path,
- folder/workspace setup,
- Slack credentials setup,
- start/stop command,
- health check,
- short operator runbook,
- data location documented,
- uninstall/reset instructions.

This is not the full Stage 5 local companion.

### MVP-6 — Real-user validation

**Status:** planned

A real user performs the primary workflow with real documents and real questions.

Validation records:

- setup completion,
- successful task completion,
- answer usefulness,
- citation correctness,
- approximate time saved,
- trust/confidence feedback,
- reuse or follow-up questions,
- blockers to repeated use,
- most valuable next capability.

### MVP decision gate

After validation, do not automatically start the next technical stage.

Choose the next priority from:

```text
observed user need
→ measurable value
→ concrete blocker
→ cheapest next valuable workflow
```

Possible next priorities include:

- document lifecycle,
- outputs and artifacts,
- history,
- better setup/companion,
- retrieval quality,
- another familiar tool such as Microsoft Teams,
- security or operations work required by the real user environment.

---

## 8. Post-MVP roadmap to LKW 1.0

The following six stages describe product maturity toward LKW 1.0. They are not an automatic task queue before MVP validation.

### Stage 1 — Trusted Ask Workspace

**MVP-critical**

User outcome: the user asks a workspace question and receives a checkable answer grounded in documents.

Capabilities:

- workspace-scoped retrieval,
- evidence assembly,
- synthesis,
- stable citations,
- insufficient-evidence behavior,
- persisted run,
- timeout/failure states,
- surface-neutral result contract.

### Stage 2 — Familiar conversational access

**Slack MVP-critical; broader portability post-MVP**

User outcome: the user accesses the same LKW capability from a familiar communication tool.

MVP requirement:

- Slack DM,
- approved identity,
- workspace selection,
- answer with sources,
- basic async and duplicate handling,
- outbound-data warning.

Post-MVP expansion may include:

- richer Slack interactions,
- shared session/history UX,
- approvals,
- artifact notifications,
- Microsoft Teams or another surface selected by real demand,
- broader surface-portability proof.

### Stage 3 — Reliable document lifecycle

**Post-MVP candidate**

Capabilities may include:

- new and modified files,
- deleted files,
- rename/move,
- stale chunk removal,
- document status,
- partial sync result,
- retry and recovery,
- continuous synchronization.

Pull this work forward only when stale or changing documents block the intended MVP user.

### Stage 4 — Workspace outputs and history

**Post-MVP candidate**

Capabilities may include:

- reports,
- summaries,
- e-mails,
- timelines,
- fact tables,
- risk lists,
- shadow artifacts,
- artifact provenance,
- versioning,
- run history,
- explicit export approval.

The first output type should be selected from observed user value, not from a completeness checklist.

### Stage 5 — Installable local companion

**1.0 direction; minimal packaging occurs before MVP validation**

Capabilities may include:

- Windows installer,
- daemon lifecycle,
- system startup,
- folder picker,
- workspace/source setup,
- filesystem allowlist,
- Slack connection setup,
- identity mapping,
- model and secret settings,
- health and diagnostics,
- update,
- uninstall,
- backup/restore entry points.

The companion does not need to duplicate Slack as a full chat client.

### Stage 6 — Operational, security, quality and release readiness

**Post-MVP / 1.0 gate**

Capabilities may include:

- component, worker and queue health,
- recovery and failed-operation visibility,
- backup/restore,
- migrations,
- retention,
- secure localhost access,
- secret storage,
- parser/file/symlink/resource limits,
- audit and diagnostics,
- versioned quality corpus,
- citation and unsupported-answer metrics,
- leakage measurement,
- latency, token, cost and storage baselines,
- soak tests,
- release checklist.

Token optimization begins only after a stable measured Ask Workspace baseline and a demonstrated product or cost need.

---

## 9. LKW 1.0 definition

LKW 1.0 is intended to be:

```text
local
single-user first
installable
restart-safe
source-file-safe
auditable
daily-usable
supportable for its declared scope
```

### Expected 1.0 outcomes

- canonical HTTP API works,
- MCP remains available,
- Ask Workspace works with sources,
- one first-class familiar conversational surface works,
- the local core remains independent of external surfaces,
- local setup and diagnostics are usable,
- document lifecycle is reliable enough for the declared use,
- history/artifacts support validated user needs,
- backup/recovery and security gates match the declared use,
- quality and cost baselines are known,
- public proof reflects actual capabilities.

### Not automatically required for 1.0

- SaaS,
- Kubernetes,
- enterprise RBAC,
- multiple organizations,
- mobile application,
- every provider,
- every operating system,
- every messaging platform,
- identical presentation across surfaces,
- a large proprietary desktop chat UI.

A second conversational adapter is valuable when it demonstrates commercial or user value. It is not a mandatory pre-MVP deliverable.

---

## 10. Product and platform gates

### MVP value gate

Can a real user complete the primary workflow and judge its value?

### Product gate

Does the slice give the user a genuinely useful capability?

### Architecture gate

Does LKW use Intergrax without bypasses or duplicated surface-specific product logic?

### Operational gate

Is the capability reliable enough for the intended MVP or release user?

### Audit gate

Can the technical flow be verified through tests, live proof and `LKW_PLATFORM_PROOF.md`?

### Surface-neutrality gate

Can the capability remain independent of Slack-specific domain, orchestration and persistence semantics?

Before MVP, this is demonstrated through the HTTP/MCP/Slack boundaries and contract tests. A second production adapter is not required.

---

## 11. Platform-gap classification

Every detected gap is classified as:

| Class | Rule |
|-------|------|
| **product-blocking** | Solve before completing the active MVP workflow |
| **platform-reusable-nonblocking** | Record; do not create an automatic implementation task |
| **production-hardening** | Place after the MVP gate unless risk is unacceptable for the MVP user |
| **product-specific** | Keep in LKW; do not generalize prematurely |

```text
Not every detected pattern becomes a platform task.
```

Historical platform backlogs, provider matrices, vendor observability rollouts and portability programs remain reference material only until a real LKW need pulls them into scope.

---

## 12. Deferred scope

Deferred until MVP validation or a documented blocker:

- Microsoft Teams,
- support for every messaging provider,
- complete local companion,
- macOS installer,
- mobile client,
- SaaS,
- multi-organization support,
- enterprise RBAC,
- Kubernetes,
- PostgreSQL migration without a product need,
- vector-store portability without a product need,
- broad observability vendor rollout,
- Prometheus/Grafana/Tempo/Sentry as standalone proofs,
- scaffold propagation as a standalone program,
- all-provider matrices,
- autonomous external actions,
- web search,
- cross-device synchronization,
- optimization before a measured baseline.

```text
Deferred work enters the active plan only when it blocks the MVP,
protects the intended user, or is selected from real validation feedback.
```

---

## 13. Completed milestones

| Milestone | Result | Status | Evidence |
|-----------|--------|--------|----------|
| **LKW.0** | Application host baseline and architecture scaffold | Done | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| **LKW.1** | Index/search/synthesize baseline with live proof | Closed in scope | [`LKW_1_LIVE_VERIFICATION.md`](LKW_1_LIVE_VERIFICATION.md) |
| **LKW.2** | Multi-step `local.workspace.*` pipeline baseline | Closed | [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) |
| **LKW.3** | Serving and filesystem-policy application composition | Done | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| **LKW.5** | Persistent vectors and restart proof | Closed | [`LKW_5_PERSISTENCE_VERIFICATION.md`](LKW_5_PERSISTENCE_VERIFICATION.md) |
| **LKW.6** | Interaction intake baseline for future surface adapters | Closed | [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`journal/`](journal/) |
| **LKW.7** | File watcher and incremental indexing baseline | Closed | [`LKW_7_FILE_WATCHER_VERIFICATION.md`](LKW_7_FILE_WATCHER_VERIFICATION.md) |
| **LKW-PRODUCT-1** | Managed workspaces and folder sources | Done | [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) |
| **LKW-PRODUCT-1-HARDENING** | Durable sync and structured search evidence | Done | [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) |
| **LKW-MVP-BRIEF** | Product purpose, first user, MVP workflow, value and gate defined | Done | This document · [`PRODUCT_FIRST_MVP.md`](../../../docs/plan/PRODUCT_FIRST_MVP.md) |
| **MVP-1** | Trusted Ask Workspace discovery — frozen contract | Done | [`ASK_WORKSPACE_DISCOVERY.md`](ASK_WORKSPACE_DISCOVERY.md) |

Ask Workspace implementation (MVP-2), Slack MVP, design-partner packaging, user validation, Microsoft Teams, full document lifecycle, outputs/history, companion and LKW 1.0 are not completed.

---

## 14. Historical references

Detailed history is available in:

| Location | Contents |
|----------|----------|
| [`journal/`](journal/) | Dated implementation notes |
| Application `*VERIFICATION*.md` files | Live verification write-ups |
| [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) | Public technical proof paths and receipts |
| Git history | Exact code and documentation evolution |

Former proof-first queues, standalone Token Optimization sequences, vendor observability packs, PostgreSQL/vector portability obligations and scaffold propagation programs are historical platform backlog, not the active LKW product order.

---

## Appendix A — Current task summary

```text
Current milestone: LKW MVP
Current task: MVP-2 — Trusted Ask Workspace
Frozen contract: ASK_WORKSPACE_DISCOVERY.md
Completed discovery: MVP-1
Next user-surface task: MVP-3/MVP-4 — Slack discovery and Slack MVP
MVP gate follows minimal packaging and real-user validation.
```

## Appendix B — Status vocabulary

| Label | Meaning |
|-------|---------|
| `implemented` | Code and tests exist |
| `live-verified` | Demonstrated in a live run |
| `MVP-critical` | Required before first user-value validation |
| `post-MVP candidate` | Selected by validation feedback or a concrete blocker |
| `planned` | Intentionally on the active execution path |
| `deferred` | Outside active scope until justified by product need |

The active measure of progress is:

```text
How much closer is a real user to asking about local documents in Slack
and receiving a useful, grounded answer with sources?
```
