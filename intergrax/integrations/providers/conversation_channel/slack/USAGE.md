# Slack conversation channel (`slack`)

Category: `conversation_channel`

Public Integration class: `SlackConversationChannelIntegration`
Backend class: `SlackConversationChannelBackend`

## Current status

```text
BETA
DONE / LIVE_VERIFIED
Socket Mode inbound + Web API outbound runtime supported
verified against real Slack Socket Mode
verified DM MESSAGE mapping
verified outbound reply
verified interactive single choice
verified ACTION mapping
verified confirmation
default_enabled = false
runtime_binding_supported = true
```

Socket Mode inbound and Web API outbound runtime are supported.

The provider exposes platform conversation-channel capabilities for:
- inbound MESSAGE mapping
- outbound message delivery
- single-choice rendering
- inbound ACTION mapping
- thread-aware addressing
- controlled lifecycle start, health and stop

Application-specific live evidence and product workflow status are owned by
the consuming application and are intentionally not referenced here.

Product workflows, application authorization and application identity mapping
are outside the provider contract.

Disabled registry/factory construction performs no SDK init and no network I/O.

## Shared operations

- `start(handler)` — start Socket Mode; returns after the transport has connected and can receive envelopes
- `stop()` — stop delivery, cancel in-flight handler tasks, close Socket Mode resources
- `send(message)` — `chat.postMessage` (text, optional single-choice Block Kit, optional `thread_ts`)
- `health()` — provider lifecycle readiness only (no network probe on each call)

## Construction

```text
enabled=False
→ contract instance, no tokens required, no SDK

from_backend(backend)
→ test injection path

from_config(enabled config with tokens)
→ SlackConversationChannelBackend (clients constructed; connect only in start())
```

## Vendor transport

```text
Inbound:  slack_sdk.socket_mode.aiohttp.SocketModeClient (Socket Mode)
Outbound: slack_sdk.web.async_client.AsyncWebClient (chat.postMessage)
Reconnect: owned by slack_sdk auto_reconnect_enabled
```

## Required environment

```text
INTERGRAX_SLACK_APP_TOKEN   # xapp-…  (connections:write)
INTERGRAX_SLACK_BOT_TOKEN   # xoxb-…  (chat:write, im:history, knowledge reads)
INTERGRAX_SLACK_API_TIMEOUT_SECONDS   # optional
INTERGRAX_SLACK_CONVERSATION_ENABLED  # optional; or pass enabled=True
INTERGRAX_SLACK_PROOF_TIMEOUT_SECONDS # optional; live-proof wait budget
```

Configure the required tokens in the process environment or the consuming
application's local `.env` (never commit real tokens). The platform live-proof
harness `scripts/proof/slack_conversation_channel_live_proof.py` may load a
local `.env` when provided; process environment still overrides `.env`.

Optional dependency:

```text
uv sync --extra integrations-slack
```

## Required Slack app settings

```text
Socket Mode enabled
Interactivity enabled
App Home Messages tab enabled
Event Subscriptions enabled
Bot event: message.im
```

## Supported inbound

```text
events_api + message + channel_type=im  → MESSAGE
interactive + block_actions + one static_select → ACTION
```

Every envelope with a usable `envelope_id` is acknowledged immediately, before mapping or handler dispatch.

## Supported outbound

```text
plain text
text + one ConversationSingleChoice (static_select)
optional thread_id → thread_ts
```

## Explicitly unsupported

- slash commands, shortcuts, modals, view submissions
- app mentions, channel/group/mpim messages
- files, reactions, edits, deletions
- arbitrary Block Kit from the application
- OAuth / multi-install / Enterprise Grid
- product authorization, identity mapping, or application workflow logic

## Declared conversation features

```text
text
single_choice
```

---

## Implemented — Slack Knowledge foundation (`SLACK-KNOWLEDGE-FOUNDATION-1`)

**Classification:** `IMPLEMENTED` platform foundation · `NOT` LKW bridge · `NOT` live capability.

The existing `SlackConversationChannelIntegration` now exposes typed provider-specific knowledge-read operations through the same shared `AsyncWebClient` owned by `SlackConversationChannelBackend`:

```text
list_accessible_conversations_page(...)
read_conversation_history_page(...)
read_thread_replies_page(...)
read_exact_message(...)
read_file_info(...)   # safe file inventory only; no binary download
```

`SlackConversationKnowledgeAdapter` (`source_kind: slack_conversation`) maps these reads into Vendor Knowledge records and synchronizes through the shared Facade / Sync runtime into any injected durable sink.

### Credential model (one integration, one WebClient, one bot token)

```text
bot token (INTERGRAX_SLACK_BOT_TOKEN):
  conversational Socket Mode + Web API runtime
  bot-membership inventory via users.conversations
  public/private/IM/MPIM knowledge reads for conversations the bot can access
```

The bot must be a member of each conversation to inventory and read it. Token values never appear in errors, logs, health or public views.

### Supported durable source kinds

```text
public_channel
private_channel
im
mpim
```

### Required scopes (knowledge reads)

```text
channels:read
groups:read
im:read
mpim:read
channels:history
groups:history
im:history
mpim:history
files:read            # safe historical file inventory (metadata only)
```

Current DM conversational runtime scopes (`connections:write`, `chat:write`, `im:history`, `files:read`) remain separate. Enabling the chatbot does **not** authorize knowledge synchronization.

### Fixed root-window synchronization

One durable source = one approved `conversation_id` + immutable `root_oldest`/`root_latest` Slack timestamp window (`slack.conversation.scope.v2`).

**Root-window and reply interval (implemented):**

```text
- root message ts must lie inside [root_oldest, root_latest]
- reply thread_root_ts must lie inside the same root window
- reply own message ts must lie inside [root_oldest, root_latest] (same closed interval)
- thread_broadcast / thread-reply pointers from history are validated but not separately materialized
- replies are discovered only through traversal of their root thread
```

A reply whose own timestamp falls inside the window but whose thread root lies outside the configured root window is excluded. `full_inventory=true` means complete inventory **inside this explicit root-window scope only**.

Inventory uses bot-membership `users.conversations` with `types=public_channel,private_channel,im,mpim` on the shared bot-token `AsyncWebClient` (no `user=` or per-call token override). Provider-owned cursors paginate the single inventory stream. History and thread reads are cursor-paginated with a maximum page size of **15** messages per `conversations.history` / `conversations.replies` call.

### Structured message schema

```text
slack.conversation.message.knowledge.v1
```

### Safe file inventory

Historical files are projected as safe metadata records (`file_id`, name, mimetype, size, mode, `is_external`). Private URLs, download URLs and binary content are **not** exposed.

### Rate limits

Slack `ratelimited` / timeout / service failures normalize as retryable dependency failures. No sleeps inside provider or adapter code.

### Deletion semantics (provider gap)

Polling does **not** provide an authoritative durable deletion feed. The adapter sets `tombstones=false`. Absence from later reconciliation is **not** treated as provider-confirmed deletion. Events API / Socket Mode deletion durability is **not** implemented.

### Explicitly not implemented here

```text
LKW-SLACK-CONNECTED-SOURCE-1
Slack command / UI for attaching conversations
Knowledge Intake bridge
RAG / chunks / embeddings
Slack Live Capability Adapter
authoritative Slack ACL projection
binary historical file download
```

---

## PLANNED — NOT IMPLEMENTED

**Classification:** `PLANNED` implementation only.

```text
LKW-SLACK-CONNECTED-SOURCE-1
Knowledge Intake bridge / LKW connected-source wiring
Slack Live Capability Adapter
live/search capability
durable deletion feed
authoritative ACL projection
binary historical file download
RAG / chunks / embeddings
```

Slack remains one category-correct public `conversation_channel` integration. The existing `SlackConversationChannelIntegration`, its client, transport and credential resolution are reused for:

- conversational runtime (**implemented**);
- shared typed Slack knowledge reads (**implemented**);
- durable materialization (**implemented**);
- indexed RAG (**planned**);
- bounded live access (**planned**).

**Rejected duplicate integrations (do not create):**

```text
SlackKnowledgeIntegration
SlackRagIntegration
SlackDatabaseIntegration
SlackLiveIntegration
LkwSlackClient
```

No application-specific or consumption-mode-specific Slack client or public integration may be introduced. LKW must not construct Slack SDK clients or call Slack Web API history endpoints directly.

**Dual role independence:**

```text
Slack frontend enabled  does not imply  Slack knowledge access enabled
Slack Indexed Source    does not imply  Slack Live Access Binding
Slack Live Access Binding does not imply durable synchronization or RAG indexing
```

Enabling the Slack chatbot does not authorize indexing or querying Slack history. Conversation transport events do not automatically become durable knowledge.

Binding architecture: [`docs/project/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../../../../docs/project/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md) §13.7.
