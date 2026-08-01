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
INTERGRAX_SLACK_BOT_TOKEN   # xoxb-…  (chat:write, im:history)
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

## PLANNED — NOT IMPLEMENTED: Slack knowledge-provider reuse

**Classification:** `ARCHITECTURALLY FROZEN` direction · `PLANNED` implementation.

Slack remains one category-correct public `conversation_channel` integration. The existing `SlackConversationChannelIntegration`, its client, transport and credential resolution will be reused for:

- conversational runtime (implemented today);
- shared typed Slack knowledge reads (`PLANNED`);
- durable materialization (`PLANNED`);
- indexed RAG (`PLANNED`);
- bounded live access (`PLANNED`).

**Rejected duplicate integrations (do not create):**

```text
SlackKnowledgeIntegration
SlackRagIntegration
SlackDatabaseIntegration
SlackLiveIntegration
LkwSlackClient
```

No application-specific or consumption-mode-specific Slack client or public integration may be introduced. LKW must not construct Slack SDK clients or call Slack Web API history endpoints directly.

**Planned provider-specific read primitives (conceptual — not listed under current shared operations):**

```text
list accessible conversation inventory
read bounded conversation-history page
read bounded thread/reply page
read an exact message or exact revision
read safe attachment inventory
read explicit edit/deletion state
read bounded search result where Slack and policy support it
```

**Dual role independence:**

```text
Slack frontend enabled  does not imply  Slack knowledge access enabled
Slack Indexed Source    does not imply  Slack Live Access Binding
Slack Live Access Binding does not imply durable synchronization or RAG indexing
```

Enabling the Slack chatbot does not authorize indexing or querying Slack history. Conversation transport events do not automatically become durable knowledge.

**Scope audit (future implementation task):** Required Slack scopes for knowledge reads must be audited against official Slack documentation and preserve least privilege. Do not invent or freeze future scopes in this document.

Current required scopes above (`connections:write`, `chat:write`, `im:history`, `files:read`) apply to the **implemented** DM conversational runtime only.

Binding architecture: [`docs/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../../../docs/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md) §13.7.
