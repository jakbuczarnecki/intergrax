# Slack conversational MVP discovery

```text
Status: FROZEN_FOR_IMPLEMENTATION
Next slice: MVP-4 — Slack conversational MVP
```

**Task:** MVP-3  
**Classification:** docs-only product discovery  
**Base commit:** `6c9e1eab634852e42d45e086faa78aca71a77016`  
**Governing plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) · [`PRODUCT_FIRST_MVP.md`](../../../docs/plan/PRODUCT_FIRST_MVP.md)  
**Ask contract:** [`ASK_WORKSPACE_DISCOVERY.md`](ASK_WORKSPACE_DISCOVERY.md)  
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)

Discovery does not implement Slack connectivity, handlers, persistence, UI or tests.

---

## 1. Decision summary

| Decision | Classification | Frozen choice | Reason | MVP-4 consequence |
|----------|----------------|---------------|--------|-------------------|
| Slack transport | `FROZEN` | Socket Mode only (outbound WebSocket) | No public inbound HTTP; matches local daemon | Companion owns Socket Mode client; no Events HTTP webhook |
| Supported conversation surface | `FROZEN` | Direct messages only; human text; threaded bot replies | Smallest usable private workflow | Ignore channels, mentions, slash, files, reactions |
| Approved Slack workspace count | `FROZEN` | Exactly one approved `team_id` | Fail-closed MVP boundary | Config allowlist; other teams denied |
| Approved user count | `FROZEN` | Exactly one approved Slack `user_id` | Fail-closed MVP boundary | Other users get generic denial; no Ask |
| Ask invocation boundary | `REUSE` | `POST /v1/local_workspace/workspaces/{workspace_id}/ask` | Existing Trusted Ask Workspace is canonical | Slack is HTTP client only; no second Ask stack |
| Workspace selection model | `FROZEN` / `PRODUCT-LOCAL` | One active workspace per approved user; Block Kit select when >1 | Smallest complete Slack interaction | Persist selection in DocumentStore; auto-select when exactly one |
| Acknowledgement behavior | `FROZEN` | Ack Socket Mode envelope immediately; visible “Checking…” after auth + workspace resolve | Prevent Slack retries; separate transport from product | Envelope ack never waits on Ask/LLM/Mongo/Qdrant |
| Duplicate-event key | `FROZEN` | `slack_team_id + slack_event_id` | Stable logical event identity across Socket Mode redelivery | Claim before Ask; duplicates skip Ask and final reply |
| Duplicate-event persistence | `REUSE` / `PRODUCT-LOCAL` | MongoDB `DocumentStore`, TTL 7 days | Same persistence style as Ask runs | Dedupe repository in Slack companion |
| Response location | `FROZEN` | Original DM thread (`thread_ts` = message `ts` when top-level) | Keeps answer next to question | `chat.postMessage` with channel + thread_ts |
| Outbound-data boundary | `FROZEN` | Slack is external cloud; export only question, ack, answer, safe citation labels | Privacy honesty for local product | No raw chunks, paths, prompts, vectors, tokens |
| Offline behavior | `FROZEN` | Slack unavailable ≠ LKW unavailable | Product core is local HTTP/MCP | Missing/invalid Slack tokens disable only companion |
| Existing Slack provider reuse | `DEFERRED` / reject for MVP-4 send path | Outbound webhook `notify(message)` + slash HTTP intake | Webhook loses channel/thread; no Socket Mode lifecycle | Companion uses bot-token Web API for threaded replies |
| Existing interaction runtime reuse | `DEFERRED` | Do not route Ask through intake → Task → Nexus | Distorts Ask HTTP product boundary; slash-oriented | Companion calls Ask HTTP; intake remains for other surfaces |
| Expected blocker classification | `NO_PLATFORM_BLOCKER` | Socket Mode companion is product-local | Missing LKW companion ≠ platform gap | No new shared messaging framework in MVP-4 |

---

## 2. Product purpose

**What is being built?**  
A minimal Slack Socket Mode adapter (LKW optional companion) that lets one approved Slack user select a local LKW knowledge workspace, ask a question in a DM, and receive a grounded answer with verified sources in the same thread.

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
file uploads
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

**Evidence-based reason:** Existing platform Slack pieces provide outbound webhook notifications and slash-command HTTP intake (`POST /v1/interactions/intake` → Task → Nexus). There is no Socket Mode connection/lifecycle owner in code. Routing Ask through interaction intake would bypass the frozen Ask HTTP product boundary and reintroduce slash-command/`tenant` defaults unsuitable for fail-closed Slack MVP. Architecture already anticipated a product-local `host/slack_socket.py`-style companion; MVP-4 freezes that as an LKW Slack companion package (see §19).

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

Minimum fields required for MVP-4:

| Field | Required | Role |
|-------|----------|------|
| Socket Mode `envelope_id` | yes | Transport ack identity |
| `payload.team_id` (or equivalent team scope on envelope/payload) | yes | Workspace authorization |
| `payload.event.event_id` / top-level event id when present; else Events API `event_id` | yes | Dedupe key |
| `event.type` | yes | Must be `message` (or interactive `block_actions` for selection) |
| `event.channel_type` | yes for messages | Must be `im` |
| `event.channel` | yes | Reply target |
| `event.user` | yes | User authorization |
| `event.text` | yes for ask candidates | Question text |
| `event.ts` | yes | Thread anchor / reply target |
| `event.thread_ts` | optional | If present, reply in that thread; else use `ts` |
| `event.bot_id` | detect if present | Reject bot-authored |
| `event.subtype` | detect if present | Reject unsupported subtypes (e.g. `message_changed`, `bot_message`) |

Do not require unused fields. Interactive workspace-selection payloads must supply: team, user, channel, action selected `workspace_id`, and correlation to the pending selection thread.

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
acknowledge transport event
→ do not call LKW Ask
→ send one bounded generic denial message in the DM thread
→ do not expose workspace names, document names or tenant details
```

Denial text:

```text
You are not authorized to use this LKW Slack app.
```

Unknown team and unknown user use the same generic text (no enumeration).

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

```text
dedupe_key = slack_team_id + ":" + slack_event_id
```

Do not use only message text, bare timestamp, channel ID, or envelope ID as the sole product dedupe key (envelope IDs are transport-scoped and may not equal logical event identity across retries).

### 11.2 Atomic processing boundary (`FROZEN`)

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

### 11.3 Persisted record (`FROZEN`)

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

### 11.4 Failed-state retry (`FROZEN`)

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

### 14.1 Required secret/config classes (`FROZEN`)

```text
Slack app-level token (Socket Mode)
Slack bot token
approved Slack team ID
approved Slack user ID
mapped LKW tenant ID
optional default/active workspace configuration
enable flag
```

### 14.2 Conventional variable names

Existing platform Slack naming uses `INTERGRAX_SLACK_*` for webhook/signing only. MVP-4 product companion uses LKW-local names (no values committed):

```text
LOCAL_WORKSPACE_SLACK_ENABLED
LOCAL_WORKSPACE_SLACK_APP_TOKEN      # xapp-… (never commit)
LOCAL_WORKSPACE_SLACK_BOT_TOKEN      # xoxb-… (never commit)
LOCAL_WORKSPACE_SLACK_APPROVED_TEAM_ID
LOCAL_WORKSPACE_SLACK_APPROVED_USER_ID
LOCAL_WORKSPACE_SLACK_TENANT_ID
LOCAL_WORKSPACE_SLACK_DEFAULT_WORKSPACE_ID   # optional
LOCAL_WORKSPACE_SLACK_ASK_BASE_URL           # default http://127.0.0.1:<LKW port>
```

### 14.3 Signing secret (`FROZEN`)

```text
Not required for Socket Mode-only MVP-4.
```

`INTERGRAX_SLACK_SIGNING_SECRET` remains relevant only for HTTP Events/slash intake, which MVP-4 does not use.

### 14.4 Handling rules (`FROZEN`)

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
Socket Mode lifecycle is new PRODUCT-LOCAL work.
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
NO_PLATFORM_BLOCKER
```

Socket Mode inbound lifecycle is **missing**, but the required owner is an **LKW product-local companion**, not a reusable platform interaction host. Existing shared intake would distort the Ask HTTP boundary.

| Field | Value |
|-------|-------|
| classification | `NO_PLATFORM_BLOCKER` |
| missing Socket Mode host | classified `PRODUCT-LOCAL` |
| shared-platform work in MVP-4 | none required |
| excluded platform work | universal messaging framework; Socket Mode platform rewrite; Teams; identity platform |

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

Maximum three. Architecture decisions above are frozen; these need live Slack validation only.

### Q1 — Exact Events API event_id placement in Socket Mode envelopes

- **Why unresolved from repo:** No Socket Mode client/fixtures in repository; payload nesting must be confirmed against a live envelope.
- **Default if unanswered:** Prefer `payload.event_id` when present; else `payload.event.client_msg_id` only as fallback if `event_id` absent; still scope with `team_id`.
- **Blocks implementation:** No — default is sufficient to start; adjust mapping in live proof if needed.

### Q2 — Bot token scopes required for DM + Block Kit + threaded reply

- **Why unresolved from repo:** No Slack app manifest for Socket Mode MVP in repo; scopes are Slack-app configuration.
- **Default if unanswered:** `app_mentions:read` not required; require DM history/read + `chat:write` + interactive components as needed for `im` + Block Kit actions; document exact scopes during MVP-4 setup notes.
- **Blocks implementation:** No for coding; yes for live proof until app configured.

### Q3 — Whether unauthorized denial should be omitted for unknown team (to reduce probe signal)

- **Why unresolved from repo:** Product-owner preference about probe resistance vs UX; frozen above as generic denial for both.
- **Default if unanswered:** Keep generic denial for unauthorized team and user (already frozen).
- **Blocks implementation:** No.

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
