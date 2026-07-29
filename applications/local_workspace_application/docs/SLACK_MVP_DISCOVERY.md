# Slack conversational MVP discovery

```text
Status: FROZEN_FOR_IMPLEMENTATION — SLACK-CONVERSATION-RUNTIME-1 DONE / LIVE_VERIFIED
Next slice: MVP-4 — Slack conversational MVP
  LKW-SLACK-WORKFLOW-1A — DONE / LIVE_VERIFIED
    approved DM → configured active workspace → Ask HTTP → answer
  LKW-SLACK-WORKFLOW-1B-1 — IMPLEMENTED / OPERATOR_VERIFIED
    exact DM "workspaces" → tenant-scoped numbered active listing → same-thread reply
    (Ask count 0; no buttons / pending)
  LKW-SLACK-WORKFLOW-1B-2 — OPERATOR_VERIFIED
    exact DM "workspace <n>" → fresh tenant-scoped list → 1-based in-memory selection
    (configured workspace = default fallback; selected = effective active;
     `workspaces` always marks effective active; restart clears selection;
     Ask count 0 on selection; no pending / ACTION / persistence)
  LKW-WORKSPACE-MANAGEMENT-1 — IMPLEMENTED / READY_FOR_REVIEW
    workspace create <name> → HTTP create → in-memory select (Ask count 0)
    workspace delete <n> → pending deletion (TTL 5m; no delete yet)
    workspace delete confirm → DELETE /workspaces/{id} using stored id
    workspace delete cancel → clear pending only
    cleanup: workspace/sources/docs/ops/vectors + Ask runs (policy A);
    local source files never deleted; no source attachment in this task
  LKW-SLACK-COMMAND-CATALOG-1 — IMPLEMENTED / READY_FOR_REVIEW
    exact DM "help" → dynamic command list from decorated handler metadata
    (registry = parse/dispatch/help; opt-in discovery on workflow; Ask for non-commands)
  LKW-STORAGE-TENANCY-CONTRACT-1 — DOCUMENTED / READY_FOR_REVIEW
    Slack is API/capability client only; storage/tenancy canonical in ARCHITECTURE.md
  LKW-WORKSPACE-CONTENTS-1A — OPERATOR_VERIFIED
    exact DM "sources" → effective active workspace → public tenant-scoped HTTP
    source list → safe provider-neutral summaries → same-thread reply
    (Ask count 0; no source mutations; no full path disclosure;
     real Slack operator verification for help + sources; not LIVE_VERIFIED)
  LKW-WORKSPACE-CONTENTS-1B-0 — DOCUMENTED / READY_FOR_REVIEW
    channel-neutral Knowledge Intake + async ingestion contract
    Canonical: KNOWLEDGE_INTAKE_DISCOVERY.md
  LKW-WORKSPACE-CONTENTS-1B-3 — IMPLEMENTED / READY_FOR_REVIEW
    Slack DM message attachments → existing managed-file HTTP intake
    (ordinary message/files + compatible file_share; files:read required;
     provider downloads private files; one Slack event → one IntakeBatch;
     completion notification deferred to LKW-WORKSPACE-CONTENTS-1C;
     no separate file_shared event subscription)
  LKW-WORKSPACE-CONTENTS-1B-4-2 — IMPLEMENTED / CORRECTION REQUIRED
    exact DM "source candidates"
    → safe numbered candidate list
    exact DM "source add <n>"
    → fresh public candidate list
    → opaque candidate_id
    → existing public acceptance endpoint
    → existing Knowledge Intake lifecycle
    Ask count 0
    no path/fingerprint disclosure
    unavailable candidates are excluded from numbering and selection
    safe acceptance error codes are normalized
    POST candidate disappearance does not clear workspace selection
    review gate: audit LKW-WORKSPACE-CONTENTS-1B-4-2-C2
    not ACCEPTED
```

**Platform runtime status:**

```text
SLACK-CONVERSATION-RUNTIME-1 — DONE / LIVE_VERIFIED

SlackConversationChannelIntegration Socket Mode/Web API runtime verified against a real Slack workspace.
Evidence: [proof/SLACK_CONVERSATION_RUNTIME_LIVE_PROOF.md](proof/SLACK_CONVERSATION_RUNTIME_LIVE_PROOF.md)

verified against real Slack Socket Mode
verified DM MESSAGE mapping
verified outbound reply
verified interactive single choice
verified ACTION mapping
verified confirmation

Platform transport blocker for MVP-4 product work is closed.

LKW-SLACK-WORKFLOW-1A — DONE / LIVE_VERIFIED
(approved DM → configured tenant/active workspace → product dedupe → Ask HTTP → threaded safe answer)
Evidence: [proof/LKW_SLACK_ASK_WORKFLOW_1A_LIVE_PROOF.md](proof/LKW_SLACK_ASK_WORKFLOW_1A_LIVE_PROOF.md)
Real Slack happy path: LIVE_VERIFIED.
Duplicate-event suppression: DETERMINISTIC_CONCURRENCY_VERIFIED
(artificial same-event live redelivery not required for 1A completion;
 multi-process / HA dedupe remains out of scope for single-process MVP).
Preflight (historical / re-run):
uv run python applications/local_workspace_application/scripts/run-lkw-slack-ask-configuration-preflight.py
Proof checklist (historical / re-run):
uv run python applications/local_workspace_application/scripts/run-lkw-slack-ask-workflow-proof.py

Ownership:
SlackConversationChannelIntegration
→ Socket Mode
→ ack
→ Slack event mapping
→ Slack Web API send
→ Block Kit mapping
→ lifecycle/reconnect/health

LKW Slack companion (applications/local_workspace_application/slack_companion/)
→ authorization (1A DONE / LIVE_VERIFIED)
→ tenant + configured active workspace mapping (1A DONE / LIVE_VERIFIED)
→ product dedupe (1A DONE / LIVE_VERIFIED; DETERMINISTIC_CONCURRENCY_VERIFIED)
→ Ask HTTP (1A DONE / LIVE_VERIFIED)
→ answer/citation rendering (1A DONE / LIVE_VERIFIED)
→ workspace listing command (1B-1 IMPLEMENTED / OPERATOR_VERIFIED)
→ text workspace selection (1B-2 OPERATOR_VERIFIED; in-memory only)
→ workspace create / delete confirm (LKW-WORKSPACE-MANAGEMENT-1 IMPLEMENTED / READY_FOR_REVIEW)
→ dynamic command catalog + help (LKW-SLACK-COMMAND-CATALOG-1 IMPLEMENTED / READY_FOR_REVIEW)
→ storage/tenancy contract (LKW-STORAGE-TENANCY-CONTRACT-1 DOCUMENTED / READY_FOR_REVIEW)
→ inspect active workspace sources (LKW-WORKSPACE-CONTENTS-1A OPERATOR_VERIFIED)
→ Knowledge Intake contract (LKW-WORKSPACE-CONTENTS-1B-0 DOCUMENTED / READY_FOR_REVIEW)
→ Slack attachments → managed-file intake (LKW-WORKSPACE-CONTENTS-1B-3 IMPLEMENTED / READY_FOR_REVIEW)
→ Source Candidate selection (LKW-WORKSPACE-CONTENTS-1B-4-2 IMPLEMENTED / CORRECTION REQUIRED)
→ completion notification / source lifecycle remainder / pending question / ACTION / persistence (later; 1C+)

Live transport proof command:
uv sync --extra integrations-slack
uv run python scripts/proof/slack_conversation_channel_live_proof.py
```

**Task:** MVP-3  
**Classification:** docs-only product discovery  
**Base commit:** `6c9e1eab634852e42d45e086faa78aca71a77016`  
**Governing plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) · [`PRODUCT_FIRST_MVP.md`](../../../docs/plan/PRODUCT_FIRST_MVP.md)  
**Ask contract:** [`ASK_WORKSPACE_DISCOVERY.md`](ASK_WORKSPACE_DISCOVERY.md)  
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md) · **Knowledge Intake:** [`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md)

Discovery does not implement Slack connectivity, handlers, persistence, UI or tests.

---

## 1. Decision summary

| Decision | Classification | Frozen choice | Reason | MVP-4 consequence |
|----------|----------------|---------------|--------|-------------------|
| Slack transport | `FROZEN` | Socket Mode only (outbound WebSocket) | No public inbound HTTP; matches local daemon | `SlackConversationChannelIntegration` owns Socket Mode/Web API transport; no Events HTTP webhook |
| Supported conversation surface | `FROZEN` | Direct messages only; human text; threaded bot replies | Smallest usable private workflow | Ignore channels, mentions, slash, files, reactions |
| Approved Slack workspace count | `FROZEN` | Exactly one approved `team_id` | Fail-closed MVP boundary | Config allowlist; other teams denied |
| Approved user count | `FROZEN` | Exactly one approved Slack `user_id` | Fail-closed MVP boundary | Other users get generic denial; no Ask |
| Ask invocation boundary | `REUSE` | `POST /v1/local_workspace/workspaces/{workspace_id}/ask` | Existing Trusted Ask Workspace is canonical | Slack is HTTP client only; no second Ask stack |
| Workspace selection model | `FROZEN` / `PRODUCT-LOCAL` | One active workspace per approved user; Block Kit select when >1 | Smallest complete Slack interaction | Persist selection in DocumentStore; auto-select when exactly one |
| Acknowledgement behavior | `FROZEN` | Ack Socket Mode envelope immediately; visible “Checking…” after auth + workspace resolve | Prevent Slack retries; separate transport from product | Envelope ack never waits on Ask/LLM/Mongo/Qdrant |
| Duplicate-event key | `FROZEN` | `slack_team_id + ":" + slack_event_id` | Stable logical event identity across Socket Mode redelivery | Claim before Ask; duplicates skip Ask and final reply |
| Slack event identity | `FROZEN` | `payload.event_id`; dedupe key `payload.team_id + ":" + payload.event_id` | Canonical Events API logical identity | Missing `event_id` fails closed; `client_msg_id` never used as fallback |
| Slack app permissions | `FROZEN` | `connections:write`; `chat:write`; `im:history`; `files:read`; `message.im`; Socket Mode + Interactivity + App Home Messages | Minimum DM-only Socket Mode workflow including attachment download | No app mentions, slash commands, channel access, HTTP webhook, or separate `file_shared` subscription |
| Duplicate-event persistence | `REUSE` / `PRODUCT-LOCAL` | MongoDB `DocumentStore`, TTL 7 days | Same persistence style as Ask runs | Dedupe repository in Slack companion |
| Response location | `FROZEN` | Original DM thread (`thread_ts` = message `ts` when top-level) | Keeps answer next to question | `chat.postMessage` with channel + thread_ts |
| Outbound-data boundary | `FROZEN` | Slack is external cloud; export only question, ack, answer, safe citation labels | Privacy honesty for local product | No raw chunks, paths, prompts, vectors, tokens |
| Offline behavior | `FROZEN` | Slack unavailable ≠ LKW unavailable | Product core is local HTTP/MCP | Missing/invalid Slack tokens disable only companion |
| Existing Slack provider reuse | `DEFERRED` / reject for MVP-4 send path | Outbound webhook `notify(message)` + slash HTTP intake | Webhook loses channel/thread; no Socket Mode lifecycle | Companion uses bot-token Web API for threaded replies |
| Existing interaction runtime reuse | `DEFERRED` | Do not route Ask through intake → Task → Nexus | Distorts Ask HTTP product boundary; slash-oriented | Companion calls Ask HTTP; intake remains for other surfaces |
| Expected blocker classification | `PLATFORM_BLOCKED` | `conversation_channel` exists; Slack runtime binding missing | Do not implement MVP-4 until SlackConversationChannelIntegration has Socket Mode/Web API runtime |

---

## 2. Product purpose

**What is being built?**  
A minimal Slack Socket Mode adapter (LKW optional companion) that lets one approved Slack user operate a Hybrid Knowledge Workspace: select or create a workspace, add indexed knowledge (files, URLs, configured sources), request connection setup through backend capabilities, ask Hybrid Ask questions, and receive grounded answers with citations, freshness and operation status — all through LKW HTTP capabilities only.

**Target experience (planned — not all steps implemented):**

```text
Create a workspace for Project Orion.
Add these files and this website.
Connect the engineering Jira project and the Orion SharePoint site.
Use the latest Jira blockers and messages from the client to tell me
whether we are ready to deploy.
```

Slack remains a **thin client**. It must not own knowledge configuration, provider credentials, vendor clients, RAG, tool selection or operation state. Binding architecture: [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md).

**Current implemented slice:** select workspace, Ask over indexed knowledge, inspect sources, managed-file and URL intake via HTTP (planner execution not yet wired for natural language).

**Who is the first user?**  
One approved knowledge worker who already uses Slack and needs answers from local company documents.

**What task are they completing?**  
Ask a natural-language question about documents already synchronized into an LKW managed workspace and verify the answer against listed sources.

**Why Slack is valuable as the first familiar-tool surface?**  
The user stays in a tool they already open daily; they do not need a separate LKW UI to get value from local retrieval and grounded answers.

**What is the smallest useful result?**  
One DM question → acknowledgement → grounded answer with a short safe source list in the thread, backed by an existing persisted Ask run.

Product statement:

```text
For one approved knowledge worker who already uses Slack and needs answers from local
company documents, the Slack LKW adapter enables asking a question in a familiar DM
and receiving a grounded answer with verifiable sources, without opening a separate
LKW interface.
```

Do not expand the target to all organizations or all Slack users.

---

## 3. Primary user workflow

### 3.1 Happy path (`FROZEN`)

```text
1. Approved user opens a DM with the LKW Slack app.
2. User sends a question.
3. Slack event is acknowledged immediately (Socket Mode envelope ack).
4. Adapter verifies Slack workspace and user authorization.
5. Adapter resolves the active LKW knowledge workspace.
6. If no workspace is selected, adapter starts the minimal workspace-selection interaction.
7. Adapter sends a visible short acknowledgement in the original thread
   (“Checking the selected workspace…”) when an active workspace is already resolved.
8. Adapter invokes the existing Ask Workspace HTTP endpoint.
9. Ask Workspace returns completed or insufficient_evidence (or failed).
10. Adapter renders the typed response (plain text).
11. Adapter posts the final answer in the original Slack thread.
12. The persisted Ask run remains retrievable through existing LKW HTTP GET .../asks/{run_id}.
```

### 3.2 First-use / no active workspace (`FROZEN`)

```text
1–4. Same as happy path through authorization.
5. No active workspace for the approved user.
6. Adapter lists usable workspaces for the mapped LKW tenant
   via GET /v1/local_workspace/workspaces (X-Tenant-Id = mapped tenant).
7a. Exactly one workspace → auto-select, persist as active, continue with the pending question
    (visible ack + Ask).
7b. Multiple workspaces → post Block Kit selection prompt in the thread;
    retain at most one pending question; do not invoke Ask yet.
8. User selects a workspace (Block Kit interaction).
9. Adapter persists active workspace, resumes the pending question, then steps 7–12 of happy path.
```

### 3.3 Change workspace (`FROZEN`)

User may send the exact text command:

```text
workspaces
```

This posts the same Block Kit selection UI (or auto-selects if only one). Changing selection updates the persisted active workspace and does not invent an Ask answer.

### 3.4 Target live platform proof scenario (`PLANNED` — `LKW-LIVE-PLATFORM-PROOF-1`)

```text
1. Start LKW with Ollama.
2. Create or select a workspace via Slack or HTTP.
3. Upload files and add a Web URL (indexed knowledge).
4. Configure Microsoft 365, Jira, Confluence (collaboration pack).
5. Configure Databricks, Power BI, Atlan (data pack) — read-only live bindings.
6. Ask a Hybrid Ask question requiring indexed and live evidence.
7. Receive one answer with source-specific citations and live freshness markers.
8. Restart LKW with vLLM; repeat the same product scenario.
```

Public claims must distinguish **real provider proof**, **controlled integration proof** and **deterministic fixture proof**. See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7.8 and [`LKW_PLATFORM_PROOF.md`](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md).

---

## 4. Supported Slack surface

### 4.1 In scope (`FROZEN`)

```text
Socket Mode
direct messages only (channel_type = im)
human-authored text messages only
one approved Slack workspace (team_id)
one approved Slack user (user_id)
one LKW Slack app installation
threaded bot replies
Block Kit interactive payloads only for workspace selection
```

### 4.2 Explicitly rejected (`OUT-OF-SCOPE`)

```text
public channels
private channels
app mentions
slash commands
message shortcuts
file uploads   ← original MVP-3 / Ask-only slice only (see clarification below)
voice messages
reactions
edited messages
deleted messages
multi-party DMs (mpim)
multiple approved users
multiple Slack organizations
enterprise-grid behavior
Teams
```

**Clarification — file events:** The original MVP-3 / Ask-only slice excluded file events. **`LKW-WORKSPACE-CONTENTS-1B-3`** implements Slack DM message attachments (ordinary `message` with `files`, and compatible `subtype=file_share`) mapped through the Slack conversation provider → companion → existing public managed-file HTTP intake. The Slack provider downloads private files with the bot token (`files:read`); token and private URLs never enter LKW core. One Slack submission becomes one LKW `IntakeBatch`. Completion notification remains **`LKW-WORKSPACE-CONTENTS-1C`**. Do **not** subscribe to a separate top-level `file_shared` event for this slice.

### 4.3 Unsupported-event handling (`FROZEN`)

```text
ack Socket Mode envelope (after minimum envelope validation)
→ classify event
→ if unsupported or rejected by product rules: stop
→ do not invoke Ask
→ do not call search/model
→ do not post a product answer
```

Unsupported types are ignored silently except unauthorized team/user (see §7).

---

## 5. Socket Mode lifecycle contract

### 5.1 Transport invariants (`FROZEN`)

```text
Slack connects through an outbound Socket Mode WebSocket.
No public inbound HTTP endpoint is required for Slack.
Slack is optional and must not block core LKW startup.
HTTP and MCP remain usable when Slack is offline.
```

Required invariant:

```text
Slack unavailable
≠
LKW unavailable
```

### 5.2 Lifecycle owner (`PRODUCT-LOCAL` · `FROZEN`)

```text
LKW-owned optional companion component
```

Not:

```text
shared platform interaction host
```

**Evidence-based reason:** Existing platform Slack pieces provide outbound webhook notifications and slash-command HTTP intake (`POST /v1/interactions/intake` → Task → Nexus). `conversation_channel` + `SlackConversationChannelIntegration` now define the category boundary, but Socket Mode/Web API runtime binding is still missing. Routing Ask through interaction intake would bypass the frozen Ask HTTP product boundary. Transport ownership is platform (`SlackConversationChannelIntegration`); LKW owns only the product conversation handler/workflow.

### 5.3 Lifecycle states (`FROZEN`)

```text
disabled
connecting
ready
reconnecting
degraded
stopped
```

| Concern | Behavior |
|---------|----------|
| Startup | Core LKW starts normally. If Slack disabled or tokens missing → `disabled`; do not fail host readiness. If enabled with tokens → start companion → `connecting` → `ready` or `degraded`. |
| Shutdown | Companion stops WebSocket cleanly → `stopped`. In-flight Ask may complete in-process; Slack final delivery is best-effort (one attempt). |
| Reconnect | On disconnect → `reconnecting` with bounded exponential backoff (initial 1s, factor 2, max 60s, jitter). |
| Token revocation | Auth failures → `degraded`; stop product Ask for new events; do not crash host. |
| Health | Companion exposes Slack-specific health (`ready` / `degraded` / `disabled`); separate from core LKW health. |
| Readiness | Core LKW readiness ignores Slack state. |
| HA / clustering | `OUT-OF-SCOPE` — single local process only. |

---

## 6. Slack event-to-product mapping

### 6.1 Accepted inbound events (`FROZEN`)

Primary product event: Socket Mode envelope carrying Events API `message` for a DM.

Frozen Socket Mode envelope structure for Events API messages:

```text
Socket Mode envelope
├── envelope_id
└── payload
    ├── team_id
    ├── event_id
    └── event
        ├── type
        ├── channel_type
        ├── channel
        ├── user
        ├── text
        ├── ts
        ├── thread_ts optional
        ├── bot_id optional
        ├── subtype optional
        └── client_msg_id optional
```

Identity distinction (`FROZEN`):

```text
envelope_id
= transport acknowledgement identity

payload.event_id
= logical Events API event identity

payload.event.client_msg_id
= optional client message metadata only
```

Do not use `payload.event.event_id`. Do not describe `event_id` as nested inside the inner `event` object.

Minimum fields required for MVP-4:

| Field | Required | Role |
|-------|----------|------|
| Socket Mode `envelope_id` | yes | Transport ack identity only |
| `payload.team_id` | yes | Workspace authorization; source of `slack_team_id` |
| `payload.event_id` | yes for Events API messages | Logical event identity; source of `slack_event_id`; product dedupe |
| `payload.event.type` | yes | Must be `message` |
| `payload.event.channel_type` | yes for messages | Must be `im` |
| `payload.event.channel` | yes | Reply target |
| `payload.event.user` | yes | User authorization |
| `payload.event.text` | yes for ask candidates | Question text |
| `payload.event.ts` | yes | Thread anchor / reply target |
| `payload.event.thread_ts` | optional | If present, reply in that thread; else use `ts` |
| `payload.event.bot_id` | detect if present | Reject bot-authored |
| `payload.event.subtype` | detect if present | Reject unsupported subtypes (e.g. `message_changed`, `bot_message`) |
| `payload.event.client_msg_id` | optional | Diagnostic metadata only; never product identity or dedupe |

Do not require unused fields. Interactive workspace-selection payloads (`block_actions`) must supply: team, user, channel, action selected `workspace_id`, and correlation to the pending selection thread. They do **not** require an Events API `payload.event_id` (see §11.5).

### 6.2 Normalized product input (`FROZEN` contract — not implemented here)

```text
SlackInboundAskCandidate
- slack_team_id
- slack_user_id
- slack_channel_id
- slack_event_id
- slack_message_ts
- slack_thread_ts
- question
```

### 6.3 Rejection rules before Ask (`FROZEN`)

```text
bot-authored event
self-authored event (bot user)
unsupported subtype
non-DM event
empty / whitespace-only text
unauthorized team
unauthorized user
duplicate event (after successful claim fails as already seen)
```

### 6.4 Envelope ack vs product rejection (`FROZEN`)

```text
Acknowledge the Socket Mode envelope immediately after minimum envelope validation
(presence of envelope_id + parseable payload type).
```

Product-level rejection happens after ack. Ack prevents transport retries; product work remains fail-closed.

---

## 7. Identity and authorization contract

### 7.1 Separated identities (`FROZEN`)

| Identity | Meaning |
|----------|---------|
| Slack workspace identity | `slack_team_id` |
| LKW tenant identity | mapped `tenant_id` |
| Slack user identity | `slack_user_id` |
| LKW user identity | fixed approved local user label from config (for diagnostics only in MVP-4) |
| LKW knowledge workspace | managed `workspace_id` selected/persisted for the user |

### 7.2 Configuration-driven mapping (`FROZEN`)

```text
approved Slack team_id
→ fixed LKW tenant_id

approved Slack team_id + Slack user_id
→ fixed approved local user identity
```

### 7.3 Security rules (`FROZEN`)

```text
no tenant_id accepted from Slack message text
no user-selectable tenant
no default tenant fallback
no unknown-user fallback
no Ask invocation before authorization
no search/model execution for unauthorized events
```

Slack adapter must send `X-Tenant-Id` from the frozen mapping only when calling LKW HTTP.

### 7.4 Unauthorized behavior (`FROZEN`)

```text
Unauthorized team or unauthorized user:
acknowledge the Socket Mode envelope
→ do not invoke Ask
→ send the same generic denial message when a reply channel is safely available
→ expose no team, tenant, workspace or document details
```

Denial text:

```text
You are not authorized to use this LKW Slack app.
```

Unknown team and unknown user use the same generic text (no enumeration).

If a malformed or untrusted event does not contain a safe reply channel, silently stop after envelope ack and bounded, redacted diagnostic logging.

---

## 8. Knowledge-workspace selection contract

### 8.1 Listing (`REUSE` + `PRODUCT-LOCAL`)

```text
GET /v1/local_workspace/workspaces
Header: X-Tenant-Id = mapped tenant_id
```

Show only workspaces returned for that tenant. Prefer usable/active statuses already exposed by `WorkspaceResponseV1.status` (exclude clearly deleted/disabled if the API marks them; otherwise show API list as-is).

### 8.2 Selection UX (`FROZEN` — option A)

```text
A. Slack Block Kit buttons/select menu
```

Not option B (numbered text commands) for selection UI. Text command `workspaces` only opens the Block Kit selector.

| Rule | Value |
|------|-------|
| Max workspaces rendered | 25 options (Slack select practical bound) |
| More than display limit | Show first 25 + message: “Showing 25 workspaces. Narrow local workspaces in LKW HTTP, then retry.” |
| One workspace | Auto-select; persist; execute pending question |
| Multiple | Block Kit select; wait for interaction |
| Persistence | MongoDB `DocumentStore` (same host store as Ask runs) |
| Selection key | `tenant_id + slack_team_id + slack_user_id` |
| Persisted value | `workspace_id` |
| Survives restart | yes |
| Change workspace | `workspaces` command or re-select |
| Selected workspace deleted / 404 on Ask | Clear active selection; prompt selection again; retain pending question if still within TTL |
| Pending question | At most one per approved user |
| Pending retention TTL | 15 minutes |
| Resume after selection | yes, automatically once |
| Resume protection | atomic one-time `pending` → `resumed` transition; only the successful transition may invoke Ask |

Do not design a generic conversation-memory framework.

---

## 9. Ask invocation contract

### 9.1 Canonical boundary (`REUSE` · `FROZEN`)

```text
POST /v1/local_workspace/workspaces/{workspace_id}/ask
```

Invariant:

```text
Slack is a client of Ask Workspace,
not a second Ask implementation.
```

Forbidden for the Slack companion:

```text
direct Qdrant access
direct WorkspaceAskService import/call
direct LocalWorkspaceTaskExecutor use
direct LLM calls
local.workspace.synthesize
rebuilding search evidence
constructing citations from model text
```

### 9.2 HTTP request Slack must send

| Element | Source |
|---------|--------|
| Path `workspace_id` | Active persisted selection |
| Header `X-Tenant-Id` | Config-mapped tenant only |
| Body `question` | Slack message text (trimmed) |
| Body `limit` | default `10` (same as `WorkspaceAskRequestV1`) |

### 9.3 Response fields Slack needs (`REUSE`)

From `WorkspaceAskResponseV1`:

```text
run_id
workspace_id
status          # completed | insufficient_evidence | failed
question
answer          # nullable
citations[]     # evidence_id, file_name, source_path, excerpt, location.page optional
error           # optional code/message for failed
```

Statuses rendered by Slack:

```text
completed
insufficient_evidence
```

`failed` and transport/HTTP failures use the generic error message (§12).

### 9.4 Failure handling (`FROZEN`)

| Case | Slack behavior |
|------|----------------|
| HTTP unavailable / connection error | Generic error reply; no invented answer |
| Timeout | Generic timeout/error reply (Ask client timeout 60s) |
| Workspace not found (404) | Clear selection; prompt workspace selection; keep pending question if within TTL |
| Authorization failure on HTTP | Generic error (should not happen if mapping correct); no tenant leak |
| Malformed response | Generic error |
| Internal Ask `status=failed` | Generic error; optional bounded `run_id` suffix if present |
| `insufficient_evidence` | Fixed insufficient-evidence message; no model improvisation in Slack |

---

## 10. Acknowledgement and asynchronous execution

### 10.1 Transport acknowledgement (`FROZEN`)

```text
ack envelope immediately after minimum envelope validation
```

Must not wait for search, LLM, Ask completion, workspace selection UI completion, MongoDB, or Qdrant.

### 10.2 Visible acknowledgement (`FROZEN`)

After authorization and active-workspace resolution, post in the same thread:

```text
Checking the selected workspace…
```

If workspace selection is required, the selection prompt replaces the generic processing acknowledgement.

### 10.3 Execution mechanism (`FROZEN`)

```text
bounded in-process asynchronous task
```

Not Kafka / durable platform background queue for Slack Ask orchestration.

Product-first reason: one-user MVP; Ask HTTP already persists durable runs in MongoDB; Slack delivery is best-effort.

### 10.4 Delivery and durability (`FROZEN`)

```text
Ask run remains durable in MongoDB.
One bounded Slack delivery attempt is sufficient for MVP-4.
No durable outbound delivery queue in MVP-4.
```

| Concern | Rule |
|---------|------|
| Ask timeout | 60 seconds client-side |
| User-visible timeout | `I could not complete this request. Please try again.` |
| Process stop during Ask | Ask may finish and persist; Slack reply may be lost; user can retry with a new message / use HTTP GET by `run_id` if known |
| Slack delivery failure after Ask success | Ask run kept; no automatic Slack retry queue |
| Slack delivery retry | not required in MVP-4 |

---

## 11. Duplicate-event contract

### 11.1 Dedupe identity (`FROZEN`)

The only canonical product dedupe key for Events API message envelopes:

```text
dedupe_key =
payload.team_id + ":" + payload.event_id
```

Normalized field names used elsewhere in this contract:

```text
slack_team_id
comes from payload.team_id

slack_event_id
comes from payload.event_id

dedupe_key = slack_team_id + ":" + slack_event_id
```

Do not leave alternative extraction paths. Do not use only message text, bare timestamp, channel ID, or envelope ID as the product dedupe key.

```text
client_msg_id is optional diagnostic metadata only.
It is not the canonical event identity.
It is not a dedupe fallback.
```

Reason:

```text
event_id identifies the Slack Events API delivery;
client_msg_id identifies a client-originated message when present
and is not guaranteed for every supported event or interaction.
```

Do not create a secondary dedupe key.

### 11.2 Missing `payload.event_id` (`FROZEN`)

```text
events_api envelope with missing or blank payload.event_id
→ acknowledge envelope_id
→ classify product event as malformed
→ do not claim dedupe record
→ do not invoke Ask
→ do not call search
→ do not call the model
→ do not post a product answer
→ record only a bounded, redacted diagnostic
```

Do not fall back to:

```text
client_msg_id
message timestamp
channel ID
envelope ID
hash of message text
```

`envelope_id` may be used only to acknowledge the Socket Mode envelope.

### 11.3 Atomic processing boundary (`FROZEN`)

```text
authorized event
→ claim dedupe key (atomic create-if-absent → status=processing)
→ only claim owner may invoke Ask
```

Duplicate result:

```text
duplicate event
→ no second Ask invocation
→ no second LLM call
→ no second Slack final reply
```

### 11.4 Persisted record (`FROZEN`)

```text
dedupe_key
status: processing | completed | failed
first_seen_at
updated_at
ask_run_id optional
slack_reply_ts optional
expires_at
```

Storage: existing MongoDB `DocumentStore` (`PRODUCT-LOCAL` repository in Slack companion).

TTL: **7 days**.

### 11.5 Interactive Block Kit payloads (`FROZEN`)

Do not require an Events API `event_id` for Block Kit interactions.

```text
events_api message envelope:
product identity = payload.team_id + payload.event_id

interactive block_actions envelope:
transport identity = envelope_id
selection action correlation =
team.id + user.id + channel.id + action_ts/message_ts + selected workspace_id
```

Workspace-selection interactions must not invoke Ask more than once for the same pending question.

```text
pending-question record owns a one-time resume transition

pending
→ resumed

only the successful atomic transition may invoke Ask
```

Do not introduce a second generic dedupe framework for Block Kit. The existing pending-question repository contract is sufficient.

### 11.6 Failed-state retry (`FROZEN`)

```text
automatic Slack redelivery does not retry failed product work;
manual new user message creates a new event and may retry.
```

A record in `failed` stays claimed for its dedupe key until expiry; duplicates of that same `event_id` do not re-run Ask.

---

## 12. Response rendering contract

### 12.1 Format (`FROZEN`)

```text
Plain text for Ask results.
Block Kit only for workspace selection.
```

Thread placement: always reply in the original message thread.

### 12.2 Completed

```text
<answer>

Sources:
[1] <safe source label> — <safe location>
[2] <safe source label> — <safe location>
```

| Rule | Value |
|------|-------|
| Max answer length | 3000 characters; truncate with `…` |
| Max citations rendered | 5 |
| Extra citations | omit with note `(+N more sources in Ask run)` when `run_id` shown is optional |
| Message splitting | avoid; single truncated message preferred for MVP |
| Safe source label | `file_name` if present, else basename of `source_path` |
| Safe location | `page N` when `location.page` present; else omit location segment |
| Absolute paths | never post raw `source_path` if it looks absolute; sanitize to basename |

### 12.3 Insufficient evidence

```text
I could not find enough verified information in the selected workspace to answer reliably.
```

Do not use the raw question as an answer.

### 12.4 Error

```text
I could not complete this request. Please try again.
```

Optional bounded suffix when `run_id` is present and safe:

```text
 (ref: <run_id>)
```

### 12.5 Citation rules (`FROZEN`)

Slack may number citations, shorten labels, and show safe page values.

Slack must not invent citations, modify evidence identity, serialize raw evidence chunks, expose absolute local paths, prompts, Qdrant payloads, or MongoDB records.

Sanitization minimum:

```text
absolute local path
→ basename (file_name preferred)
```

---

## 13. Outbound-data and privacy boundary

### 13.1 Rule (`FROZEN`)

```text
Slack is an external cloud service.
Every question sent through Slack and every answer posted to Slack leaves the local LKW boundary.
```

The question already originates in Slack; retrieved document-derived answer content is newly exported to Slack.

### 13.2 Allowed outbound

```text
user’s Slack question
visible processing acknowledgement
final answer
bounded safe citation labels
bounded safe source locations
generic error status
optional run_id reference
workspace display names in selection UI (tenant-scoped list only)
```

### 13.3 Forbidden outbound

```text
full document content
raw evidence chunks
embedding vectors
model prompt
system prompt
internal policy state
Qdrant payload
MongoDB Ask record
absolute local file paths
stack traces
credentials
tokens
tenant configuration
unrelated workspace metadata
```

### 13.4 Setup warning for MVP-5 docs (`FROZEN` text)

```text
Using Slack delivery sends questions, answers and displayed source labels to Slack’s cloud service.
```

No approval workflows in MVP-4.

---

## 14. Secrets and configuration contract

### 14.1 Slack app manifest and permission contract (`FROZEN`)

Minimum required Slack app configuration for MVP-4.

#### App-level token

```text
Token type:
Slack app-level token

Required app-level scope:
connections:write

Purpose:
open and maintain the Socket Mode connection

Configuration variable:
LOCAL_WORKSPACE_SLACK_APP_TOKEN

Expected token prefix:
xapp-
```

Do not include a real token in documentation or source control.

#### Bot token

```text
Token type:
Slack bot token

Required variable:
LOCAL_WORKSPACE_SLACK_BOT_TOKEN

Expected token prefix:
xoxb-

Purpose:
Slack Web API calls, including chat.postMessage
```

Do not include a real token.

#### Bot token scopes (minimum)

```text
chat:write
→ publish processing, selection and final reply messages

im:history
→ receive/read direct-message events for the bot/app DM surface

files:read
→ files.info + private download of DM message attachments (LKW-WORKSPACE-CONTENTS-1B-3)
```

Do not add:

```text
app_mentions:read
commands
channels:history
groups:history
mpim:history
chat:write.public
files:write
users:read
```

The frozen DM-only workflow plus attachment intake does not require the excluded scopes.

#### Event subscription

```text
Bot event subscription:
message.im
```

This is the only message event required for the DM-only MVP.

Do not add:

```text
app_mention
message.channels
message.groups
message.mpim
file_shared
reaction_added
```

#### Slack app features/settings

```text
Socket Mode:
enabled

Interactivity:
enabled

App Home:
Messages tab enabled

Event Subscriptions:
enabled

Bot event:
message.im
```

```text
Interactivity does not require a public HTTP Request URL in the Socket Mode MVP.
Interactive payloads are delivered through Socket Mode.
```

Do not introduce an HTTP Events API endpoint.

### 14.2 Required secret/config classes (`FROZEN`)

```text
Slack app-level token (Socket Mode)
Slack bot token
approved Slack team ID
approved Slack user ID
mapped LKW tenant ID
optional default/active workspace configuration
enable flag
LKW HTTP base URL for Ask/list calls
```

### 14.3 Conventional variable names

Existing platform Slack naming uses `INTERGRAX_SLACK_*` for webhook/signing only. MVP-4 product companion uses LKW-local names (no values committed):

```text
LOCAL_WORKSPACE_SLACK_ENABLED
LOCAL_WORKSPACE_SLACK_APP_TOKEN      # xapp-… (never commit)
LOCAL_WORKSPACE_SLACK_BOT_TOKEN      # xoxb-… (never commit)
LOCAL_WORKSPACE_SLACK_APPROVED_TEAM_ID
LOCAL_WORKSPACE_SLACK_APPROVED_USER_ID
LOCAL_WORKSPACE_SLACK_TENANT_ID
LOCAL_WORKSPACE_SLACK_LKW_BASE_URL           # default http://127.0.0.1:<LKW port>
LOCAL_WORKSPACE_SLACK_DEFAULT_WORKSPACE_ID   # optional
```

Do not add OAuth client ID/secret, signing secret, redirect URL, installation database, enterprise ID, or multiple-team mappings.

### 14.4 Signing secret (`FROZEN`)

```text
Slack signing secret is not required by the Socket Mode-only MVP-4 transport.
```

```text
A signing secret is relevant only if a future HTTP-based Events API,
slash-command or interactivity endpoint is introduced.
Those paths are out of scope for MVP-4.
```

Do not add `LOCAL_WORKSPACE_SLACK_SIGNING_SECRET`.

`INTERGRAX_SLACK_SIGNING_SECRET` remains relevant only for HTTP Events/slash intake, which MVP-4 does not use.

### 14.5 Handling rules (`FROZEN`)

```text
secrets loaded from environment or local secret configuration
never logged
redacted in diagnostics
missing token disables Slack adapter (disabled)
invalid/revoked token degrades Slack adapter only
```

---

## 15. Offline and failure behavior

| Failure | Slack behavior | Core LKW behavior | Ask-run behavior | User-visible result | Retry behavior |
|---------|----------------|-------------------|------------------|---------------------|----------------|
| No internet at startup | Companion `degraded`/`disabled` | HTTP/MCP up | n/a | none in Slack | reconnect backoff when network returns |
| Socket Mode disconnect | `reconnecting` | unaffected | in-flight Ask may persist | possible missed Slack reply | reconnect; no durable Slack outbox |
| Token revoked | `degraded` | unaffected | no new Ask from Slack | none / prior denial if any | operator fixes tokens |
| Slack API send failure | log redacted error | unaffected | Ask may already be persisted | user may see ack without final reply | one attempt only |
| LKW HTTP unavailable | no Ask | depending on outage | none created | generic error | user retries later |
| Qdrant unavailable | Ask returns failed/insufficient via HTTP | HTTP up; Ask fails closed | failed/insufficient persisted per Ask rules | generic or insufficient message | new user message |
| MongoDB unavailable | selection/dedupe/Ask persist may fail | degraded product | Ask may 502 | generic error | user retries later |
| Ask timeout (60s) | stop waiting | Ask server may still finish | possibly persisted | generic error | new user message |
| App shutdown during Ask | companion stops | host stop | may persist | reply may be lost | new message after restart |
| Duplicate delivery | ignore product work | unaffected | no second run | no second reply | n/a |
| Selected workspace deleted | clear selection | unaffected | 404 path | selection prompt | select again |
| Unauthorized user | denial message | unaffected | none | generic denial | n/a |
| Malformed Slack event | ignore after ack | unaffected | none | none | n/a |

Invariant:

```text
Slack failure never disables HTTP or MCP.
```

Delivery semantics:

```text
at-most-one product execution per persisted dedupe key
```

Do not claim distributed exactly-once delivery.

---

## 16. Existing component reuse assessment

| Component | Current responsibility | Reusable for MVP-4? | Exact reused part | Why not sufficient alone? | Ownership after MVP-4 |
|-----------|------------------------|---------------------|-------------------|---------------------------|-----------------------|
| Slack notification provider (`notify` / webhook) | Outbound webhook notifications | No for threaded Ask replies | none required | `notify(message)` loses channel/thread_ts; webhook ≠ bot Web API | remains HITL/alerts; Slack companion owns `chat.postMessage` |
| `SlackInteractionAdapter` / slash adapter | Slash-command payload → `Task` | No | none | Slash/Commands + Task materialization; Slack MVP is DM events + Ask HTTP | unchanged platform adapter |
| `InboundInteraction` | Normalized intake before Task | `DEFERRED` | none | Forces Task/Nexus path; tenant defaults unsafe; not needed for Ask client | unchanged |
| Interaction router/factory / intake wiring | `POST /v1/interactions/intake` | `DEFERRED` | none | Would distort Ask ownership and Socket Mode lifecycle | remains for lab/OS intake |
| Slack HTTP signature verification | Validates `X-Slack-Signature` on HTTP | No for Socket Mode transport | none | Socket Mode uses app token on WebSocket, not request signing | unchanged for future HTTP intake |
| Ask HTTP schemas/routes/service | Trusted Ask Workspace | `REUSE` | request/response contract + persistence | Slack must not import service internals | LKW core |
| Managed workspace list HTTP | List tenant workspaces | `REUSE` | `GET /workspaces` | Does not select/persist Slack active workspace | LKW core + Slack selection repo |
| MongoDB `DocumentStore` | Durable product state | `REUSE` | store handle already on LKW host | Needs Slack-specific selection/dedupe repositories | LKW host provides store; companion repos are product-local |

Honest reuse summary:

```text
REUSE: Ask HTTP + workspace list HTTP + DocumentStore.
Do not reuse webhook notify or interaction intake for the Slack Ask DM path.
Socket Mode lifecycle belongs to `SlackConversationChannelIntegration` (platform runtime binding — still missing).
LKW owns the product Slack conversation handler/workflow only.
```

---

## 17. Ownership boundary

| Concern | Owner |
|---------|-------|
| Slack transport lifecycle | Slack adapter/companion (`PRODUCT-LOCAL`) |
| Event parsing | Slack adapter/companion |
| Slack authorization mapping | Slack adapter/companion (config) |
| Workspace selection interaction | Slack adapter/companion |
| Pending-question state | Slack adapter/companion (`DocumentStore`) |
| Dedupe state | Slack adapter/companion (`DocumentStore`) |
| Ask invocation | Slack adapter/companion as HTTP client |
| Search | LKW core (`local.workspace.search` via Ask) |
| Answer generation | LKW core (`AskAnswerAssembler`) |
| Citation projection | LKW core |
| Ask persistence | LKW core (`WorkspaceAskRepository`) |
| Slack rendering | Slack adapter/companion |
| Slack outbound send | Slack adapter/companion (bot Web API) |
| Core readiness | LKW host |
| Slack health | Slack adapter/companion (non-blocking) |

Platform owns only existing reusable contracts actually consumed (HTTP Ask/list + DocumentStore). No new generic messaging framework.

---

## 18. Major blocker classification

```text
PLATFORM_BLOCKED
```

```text
conversation_channel category and provider definitions exist;
Slack conversation-channel runtime remains missing.
```

Socket Mode inbound lifecycle remains **missing**, but the required transport owner is
**`SlackConversationChannelIntegration`**, not an LKW-owned Socket Mode companion.
LKW owns the product conversation handler that consumes typed inbound events and drives Ask.

| Field | Value |
|-------|-------|
| classification | `PLATFORM_BLOCKED` |
| remaining platform blocker | `SlackConversationChannelIntegration` has no Socket Mode/Web API runtime binding |
| Slack transport owner | `SlackConversationChannelIntegration` |
| LKW workflow owner | LKW Slack conversation handler |
| excluded from this blocker | Teams runtime; universal rich UI; Ask rewrite |

---

## 19. Exact MVP-4 implementation plan

### 19.1 Recommended module layout (`FROZEN`)

```text
applications/local_workspace_application/slack/
    __init__.py
    config.py
    models.py
    authorization.py
    workspace_selection.py
    dedupe_repository.py
    ask_client.py
    rendering.py
    socket_mode_companion.py

applications/local_workspace_application/host/
    (wire optional companion startup/shutdown only)
```

MVP-4 may add a small dependency on an official Slack Socket Mode client library in `pyproject.toml` / `uv.lock` during implementation (not in this discovery). Discovery forbids adding it now.

### 19.2 File responsibilities

#### `slack/config.py`

- **Responsibility:** load/validate env config; enable/disable decision.
- **Public:** `SlackCompanionConfig`, `load_slack_companion_config()`.
- **Dependencies:** env only.
- **Forbidden:** Ask logic; Slack network I/O.

#### `slack/models.py`

- **Responsibility:** discovery contracts as typed models (`SlackInboundAskCandidate`, selection/dedupe records).
- **Public:** pydantic models for companion state.
- **Dependencies:** pydantic.
- **Forbidden:** HTTP calls; Socket Mode I/O.

#### `slack/authorization.py`

- **Responsibility:** team/user allowlist checks; map to tenant/user.
- **Public:** `authorize_slack_actor(...) -> AuthorizedSlackActor | Rejected`.
- **Dependencies:** config.
- **Forbidden:** Ask; workspace listing side effects beyond pure mapping.

#### `slack/workspace_selection.py`

- **Responsibility:** list via HTTP; auto-select; Block Kit payload build/parse; pending question + active workspace persistence.
- **Public:** selection service + repository helpers.
- **Dependencies:** DocumentStore; workspace list HTTP client; config.
- **Forbidden:** citation projection; Qdrant; model calls.

#### `slack/dedupe_repository.py`

- **Responsibility:** claim/get/update dedupe records with TTL.
- **Public:** `claim`, `mark_completed`, `mark_failed`.
- **Dependencies:** DocumentStore.
- **Forbidden:** Slack send; Ask invocation.

#### `slack/ask_client.py`

- **Responsibility:** HTTP client for Ask endpoint only.
- **Public:** `ask_workspace(tenant_id, workspace_id, question, limit) -> WorkspaceAskResponseV1-compatible DTO`.
- **Dependencies:** HTTP; config base URL.
- **Forbidden:** importing `WorkspaceAskService`; RAG; synthesis.

#### `slack/rendering.py`

- **Responsibility:** plain-text render + path sanitization; Block Kit selection view.
- **Public:** `render_completed`, `render_insufficient`, `render_error`, `render_workspace_selector`.
- **Dependencies:** Ask response DTO fields only.
- **Forbidden:** inventing citations; reading files.

#### `slack/socket_mode_companion.py`

- **Responsibility:** Socket Mode lifecycle; envelope ack; event dispatch; in-process async Ask orchestration; bot Web API send.
- **Public:** `SlackSocketModeCompanion` with `start`/`stop`/health.
- **Dependencies:** Slack SDK (added in MVP-4); other slack modules.
- **Forbidden:** owning search/answer/citation persistence; blocking core readiness.

#### Host wiring

- Optional start when `LOCAL_WORKSPACE_SLACK_ENABLED=true` and tokens present.
- Register shutdown hook.
- Do not mount Slack as a required dependency of HTTP/MCP.
- **Ask workspace lookup (LKW-SLACK-ASK-404-FIX):** listing/`GET`/`POST …/ask` share
  `app.state.lkw_managed_workspace_service` (+ repository/document store). Ask HTTP maps
  only `WorkspaceAskLookupError` to public `404 not_found`; bare `LookupError` subclasses
  such as `KeyError` during search/assembly must not present as workspace-not-found.
  Internal logs may use safe reason codes (`workspace_lookup_failed`,
  `tenant_scope_mismatch`, `repository_inconsistency`) without tenant/workspace IDs.
- **Text workspace selection (LKW-SLACK-WORKFLOW-1B-2):** command `workspace <positive integer>`
  (trim; `workspace` case-insensitive; 1-based index into a fresh tenant-scoped
  `GET /v1/local_workspace/workspaces` list with the same active-first ordering as
  `workspaces`). Selection is process-local (`team_id` + `user_id`).
  Effective active workspace = in-memory selected workspace when present, else
  configured `LOCAL_WORKSPACE_SLACK_ACTIVE_WORKSPACE_ID` (default fallback only).
  Ask routing, `workspaces` ordering, and the `— active` marker all use the same
  effective workspace; `workspaces` always marks effective active. Restart clears
  selection. No pending question, no ACTION resume, no DocumentStore persistence.
  Real Ask `http_404` for an in-memory selection clears that selection without
  silent configured fallback retry (configured then becomes effective again).
- **Workspace lifecycle (LKW-WORKSPACE-MANAGEMENT-1):**
  - `workspace create <name>` → tenant-scoped `POST /v1/local_workspace/workspaces`
    → in-memory select created workspace (effective active immediately; Ask count 0).
    Name: trim; collapse whitespace; reject control characters; max 100 chars.
  - `workspace delete <n>` → fresh ordered list; creates actor-scoped pending deletion
    (workspace_id + safe name + expiry, TTL 5 minutes); **no DELETE yet**.
  - `workspace delete confirm` → `consume-valid` pending →
    `DELETE /v1/local_workspace/workspaces/{workspace_id}` (HTTP 204); uses stored id.
  - `workspace delete cancel` → clear pending only.
  - Deletion removes LKW-owned state (workspace, sources, document refs, operations,
    workspace-scoped vectors, Ask runs — **policy A: remove workspace-owned Ask history**).
    Local source files/directories are never deleted or modified.
  - Deleting selected clears selection; deleting configured with no selection suppresses
    configured fallback until create/select; never auto-selects another workspace.
  - No source attachment, rename, Block Kit, or persisted pending in this task.
- **Command catalog (LKW-SLACK-COMMAND-CATALOG-1):**
  - Formal Slack commands are `@slack_command`-annotated async methods on
    `SlackAskWorkflow`; discovery is opt-in on the workflow instance only
    (`__lkw_slack_command__`); no module/global scan, no OpenAPI/endpoint mapping,
    no global Intergrax command framework.
  - Immutable registry orders by `priority` then `command_id`; drives parse,
    first-match dispatch, and dynamic `help` from the same metadata
    (`syntax` / `description` / `example` / `visible_in_help`).
  - Exact DM `help` (trim; case-insensitive; no aliases): zero HTTP / zero Ask;
    no selection or pending-deletion mutation; marks dedupe completed.
  - Invalid formal attempts (`workspace`, invalid create/delete) stay on
    hidden registry entries (`visible_in_help=False`) and never fall through to Ask.
  - Non-matching DM still invokes the regular Ask flow after authorization + dedupe.
  - A future visible decorated command automatically appears in `help`.
- **Slack is only a client (LKW-STORAGE-TENANCY-CONTRACT-1):** Slack companion
  does not decide local vs remote deployment, does not select Document Store /
  Vector Store providers, does not write storage directly, and does not branch
  on storage topology. Commands call public LKW capabilities/API and must
  behave the same regardless of where storage is located. Canonical tenancy,
  storage location, private-by-default, source-connector, and future sharing
  contract: [`ARCHITECTURE.md` — Deployment, storage and tenancy model](ARCHITECTURE.md#deployment-storage-and-tenancy-model).
  LKW-WORKSPACE-CONTENTS-1A — OPERATOR_VERIFIED:
  exact DM `sources` → effective active workspace → public tenant-scoped HTTP
  source list → safe provider-neutral source summaries → same-thread reply
  (zero Ask; no source mutations; no full path disclosure; dynamic help;
   real Slack operator verification for `help` and `sources`; not LIVE_VERIFIED).
  LKW-WORKSPACE-CONTENTS-1B-0 — DOCUMENTED / READY_FOR_REVIEW:
  channel-neutral Knowledge Intake and asynchronous ingestion contract
  ([`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md)).
  LKW-WORKSPACE-CONTENTS-1B-3 — IMPLEMENTED / READY_FOR_REVIEW:
  Slack DM message attachments → existing managed-file multipart HTTP intake
  (`files:read`; `message.im` preserved; no separate `file_shared` subscription;
   provider-local private download; one event → one IntakeBatch;
   immediate acceptance summary only; completion notification = 1C).
  LKW-WORKSPACE-CONTENTS-1B-4-2 — IMPLEMENTED / CORRECTION REQUIRED:
  exact DM `source candidates` → safe numbered candidate list
  (unavailable candidates excluded from numbering and selection);
  exact DM `source add <n>` → fresh public candidate list → opaque candidate_id
  → existing public acceptance → existing Knowledge Intake lifecycle
  (Ask count 0; no path/fingerprint disclosure;
   safe acceptance error codes normalized;
   POST candidate disappearance does not clear workspace selection).
  Review gate: audit LKW-WORKSPACE-CONTENTS-1B-4-2-C2. Not ACCEPTED.

---

## 19a. Knowledge Intake extension (Slack adapter mapping)

**Status:** `FROZEN ARCHITECTURAL CONTRACT` for adapter rules. **`LKW-WORKSPACE-CONTENTS-1B-3`** implements DM message attachment → managed-file intake. **`LKW-WORKSPACE-CONTENTS-1B-4-2`** implements safe numbered Source Candidate selection over the public HTTP API (Ask count 0; no path/fingerprint disclosure; unavailable candidates excluded from numbering/selection; safe acceptance error codes normalized; POST candidate disappearance does not clear workspace selection). Status: **IMPLEMENTED / CORRECTION REQUIRED** — review gate `audit LKW-WORKSPACE-CONTENTS-1B-4-2-C2`; not ACCEPTED. Exact Block Kit design for other intake paths remains **DEFERRED**. URL intake remains **not** implemented yet. Completion notification remains **`LKW-WORKSPACE-CONTENTS-1C`**.

Binding contract: [`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md) · [`ARCHITECTURE.md` — Channel-neutral Knowledge Intake](ARCHITECTURE.md#channel-neutral-knowledge-intake-and-asynchronous-ingestion).

| Rule | Binding |
|------|---------|
| Slack remains a conversation-channel frontend | Collects channel-native input; invokes public LKW capabilities; displays status |
| Slack attachments | Map to item-level `managed_file` Knowledge Inputs after core capability exists; LKW resolves/creates managed-upload-backed Source |
| Multiple attachments | One Intake Batch → N `managed_file` Knowledge Inputs → N item-level Sources → N item-level Ingestion Operations; Slack shows safe aggregate summary only |
| Folder / archive / channel-exposed collection | Map to `uploaded_folder_snapshot` only (no live sync); one snapshot Source → many Documents |
| Connected local folder | Selected as safe Source Candidate (`candidate_id` + safe label); never full path; LKW resolves/creates connector-backed Source |
| Raw filesystem path command | **REJECTED** (not an approved Slack product input) |
| URL intake | Must be explicit; ordinary Ask messages containing URLs remain Ask |
| Transport | Acknowledge Slack envelope immediately; long transfer/ingestion does not keep the Slack request open |
| Adapter must not | Parse, chunk, embed, write Document/Vector Store, select providers, call Qdrant, run filesystem ops, call LLM for ingestion, own operation state, create Source identity, own batch state, control retry or partial-success state |
| LKW core | Does **not** call Slack; completion returns via channel-neutral lifecycle event + Conversation Correlation |
| Exact commands / Block Kit | **DEFERRED** — do not freeze syntax in this discovery |

Slack maps attachments into public LKW intake requests; it creates neither Source nor Intake Batch state itself. Aggregate summary wording is illustrative only (exact user-facing text **not** frozen).

---

## 20. Exact MVP-4 acceptance criteria

1. One real Slack workspace.
2. Socket Mode.
3. One approved Slack user.
4. DM text question.
5. Fail-closed unknown Slack workspace/team.
6. Fail-closed unknown user.
7. LKW workspace selection (auto or Block Kit).
8. Existing Ask Workspace HTTP invocation.
9. Qdrant-backed retrieval (via Ask).
10. Grounded answer.
11. Verified citations rendered safely.
12. Thread reply.
13. Immediate transport acknowledgement.
14. Visible processing acknowledgement when workspace already selected.
15. Duplicate event does not cause a second Ask.
16. Unauthorized event causes no search or model invocation.
17. Slack outage does not break HTTP/MCP.
18. Slack result contains no absolute local path.
19. No Slack-specific logic added to Ask Workspace.
20. No Teams implementation.
21. No universal messaging framework.
22. Real controlled Slack proof.
23. Socket Mode envelope uses `envelope_id` for transport ack.
24. Events API message dedupe uses `payload.event_id`.
25. Missing `payload.event_id` produces no Ask invocation.
26. `client_msg_id` is never used as the canonical dedupe identity.
27. Slack app uses an app-level token with `connections:write`.
28. Slack bot token has `chat:write` and `im:history`.
29. Slack app subscribes only to `message.im` for message intake.
30. Interactivity payloads arrive through Socket Mode.
31. No HTTP Slack webhook or signing-secret boundary is introduced.

### Controlled proof sequence

```text
start LKW + Qdrant + MongoDB
→ synchronize marker document
→ start Slack Socket Mode adapter
→ approved user selects workspace
→ approved user sends question
→ acknowledgement appears
→ grounded answer with source appears in thread
→ replay same event
→ no second Ask or reply
→ unauthorized user sends message
→ no Ask invocation
→ disconnect Slack
→ HTTP Ask still succeeds
```

Additional event-identity and app-configuration proof checks (MVP-4; not architecture alternatives):

```text
inspect one real events_api Socket Mode envelope
→ confirm envelope_id exists
→ confirm payload.team_id exists
→ confirm payload.event_id exists
→ confirm inner payload.event contains the DM message fields

replay same logical event with the same payload.event_id
→ no second Ask
→ no second final reply

submit malformed test fixture without payload.event_id
→ envelope acknowledged
→ no Ask
→ no dedupe fallback to client_msg_id

verify Slack app configuration
→ Socket Mode enabled
→ Interactivity enabled
→ App Home Messages enabled
→ connections:write present
→ chat:write present
→ im:history present
→ message.im subscribed
```

**Requires real Slack workspace and credentials:** Socket Mode connect, DM send/receive, Block Kit selection, threaded replies, unauthorized-user check, disconnect check.

**Can use local HTTP without Slack:** Ask run persistence read-back, marker sync, post-disconnect HTTP Ask.

---

## 21. MVP-4 exclusions (`OUT-OF-SCOPE`)

```text
Teams
channel conversations
multiple users
multiple Slack workspaces
OAuth installation flow
Slack Marketplace distribution
enterprise administration
broad approval workflow
document uploads from Slack
message history ingestion
Slack search
conversational memory beyond active workspace selection
follow-up semantic context
streaming token output
rich file previews
durable outbound delivery queue
Kafka unless proven mandatory
generic companion framework
generic messaging framework
generic identity platform
generic session framework
generic Block Kit abstraction
mobile or desktop Slack-specific behavior
HA and multi-instance Socket Mode ownership
slash-command Ask path
HTTP Events API webhook mode
```

---

## 22. Open questions

```text
Open implementation-blocking questions: none
```

### Live validation items

These are proof checks for MVP-4, not unresolved architecture decisions:

```text
confirm configured Slack app receives message.im events
confirm Block Kit actions arrive through Socket Mode
confirm threaded chat.postMessage behavior in the real test workspace
```

---

## 23. Inspected files

Governing:

- `docs/plan/PRODUCT_FIRST_MVP.md`
- `applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`
- `applications/local_workspace_application/docs/ARCHITECTURE.md`
- `applications/local_workspace_application/docs/ASK_WORKSPACE_DISCOVERY.md`

Ask contract:

- `applications/local_workspace_application/serving/workspace_schemas.py`
- `applications/local_workspace_application/serving/workspace_routes.py`
- `applications/local_workspace_application/workspaces/ask_models.py`
- `applications/local_workspace_application/workspaces/ask_service.py`

Interaction candidates:

- `intergrax/runtime/interactions/models.py`
- `intergrax/runtime/interactions/router.py`
- `intergrax/runtime/interactions/factory.py`
- `intergrax/applications/_shared/interaction_wiring.py`
- `intergrax/runtime/interactions/verification/slack_signature.py`
- `intergrax/runtime/interactions/adapters/slash_command_adapter.py` (boundary confirmation)

Slack integration:

- `intergrax/integrations/providers/notification_channel/slack/USAGE.md`
- `intergrax/integrations/providers/notification_channel/slack/integration.py`
- `intergrax/integrations/providers/notification_channel/slack/adapter.py`
- `intergrax/integrations/providers/notification_channel/slack/config.py`
- `intergrax/integrations/providers/notification_channel/slack/manifest.py`
- `intergrax/integrations/providers/notification_channel/slack/bundle.py`
- `intergrax/integrations/providers/notification_channel/slack/opens.py`

Dependency search: no `slack_sdk` / `slack_bolt` / `SocketMode` runtime dependency in `pyproject.toml`; references exist only as forbidden-vendor guards/tests.
