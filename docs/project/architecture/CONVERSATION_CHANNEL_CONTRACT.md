# Conversation Channel Contract

**Status:** Normative (CONVERSATION-CHANNEL-1)  
**Category:** `conversation_channel`  
**Contract:** `ConversationChannelIntegrationContract`  
**Schema ID:** `conversation_channel_integration_contract.v1`  
**Hub:** [`INTEGRATIONS.md`](INTEGRATIONS.md)

---

## 1. Category purpose

A conversation channel is an external near-real-time communication system
that delivers human-originated conversation events to an application
and allows the application to reply within the same addressable
conversation context.

Polish:

```text
Zewnętrzny system komunikacji czasu zbliżonego do rzeczywistego,
który dostarcza aplikacji zdarzenia rozmowy pochodzące od człowieka
i pozwala aplikacji odpowiadać w kontekście tej samej adresowalnej rozmowy.
```

---

## 2. Membership rule

A provider belongs to `conversation_channel` only when it supports:

```text
human actor
→ sends a conversation event
→ external vendor delivers it to the application
→ application can reply to the same conversation context
```

Not membership criteria: any UI, any inbound payload, any notification mechanism,
any collaboration vendor, or any API used by a human.

---

## 3. Distinction from `notification_channel`

| | `notification_channel` | `conversation_channel` |
|--|------------------------|------------------------|
| Direction | application → vendor → recipient | human ↔ vendor ↔ application |
| Primary op | `notify` | receive event / reply / action / lifecycle / health |
| Capabilities | CONNECT, WRITE, HEALTH_CHECK | CONNECT, READ, WRITE, HEALTH_CHECK |

Do not merge the categories. Do not add READ to notification contracts.

---

## 4. Providers (contract-defined)

| provider_id | Integration class | Runtime binding |
|-------------|-------------------|-----------------|
| `slack` | `SlackConversationChannelIntegration` | Socket Mode + Web API |
| `teams` | `TeamsConversationChannelIntegration` | unsupported |
| `discord` | `DiscordConversationChannelIntegration` | unsupported |
| `telegram` | `TelegramConversationChannelIntegration` | unsupported |
| `mattermost` | `MattermostConversationChannelIntegration` | unsupported |
| `rocket_chat` | `RocketChatConversationChannelIntegration` | unsupported |
| `google_chat` | `GoogleChatConversationChannelIntegration` | unsupported |

Same vendor may also exist as `notification_channel` via a **separate** Integration class
and `(provider_id, category)` registry identity.

---

## 5. Shared models

Module: `intergrax/integrations/contracts/conversation_channel.py`

- `ConversationAddress` — `installation_id`, `conversation_id`, optional `thread_id`
- `ConversationActor` — provider-scoped `actor_id`
- `ConversationEventKind` — `message` | `action`
- `InboundConversationEvent`
- `ConversationActionSelection` — `action_id` + `selected_value`
- `ConversationChoiceOption` / `ConversationSingleChoice` (max 25 options)
- `OutboundConversationMessage` — text + at most one single-choice component
- `ConversationDeliveryReceipt` — vendor accepted message (not human read)

Field names are vendor-neutral (no `team_id`, `thread_ts`, Block Kit, Adaptive Cards).

---

## 6. Runtime backend contract

`ConversationChannelBackend`:

- `start(handler)` — begin vendor event delivery; one handler per instance
- `stop()` — stop delivery; close resources
- `send(message)` — outbound text (+ optional single choice)
- `health()` — provider readiness only

`ConversationEventHandler = Callable[[InboundConversationEvent], Awaitable[None]]`

---

## 7. Inbound event semantics

- `event_id` is provider-scoped and installation-scoped (not globally unique).
- Product dedupe keys should combine `provider_id + installation_id + event_id`.
- Product-level idempotency remains application-owned.
- Vendor acknowledgement is internal to the provider (no shared `acknowledge` API).
- No exactly-once, durable queue, or strict ordering promises.

---

## 8. Outbound message semantics

v1 supports:

```text
text only
or
text + one single-choice component
```

Delivery receipt confirms vendor acceptance/creation only.

---

## 9. Single-choice v1 boundary

Declared features for all seven providers: `text`, `single_choice`.

Not in v1: multi-select, forms, modals, attachments, reactions, rich layout trees.

---

## 10. Acknowledgement ownership

Vendor transport ack/retry stays inside the concrete provider runtime.
Not exposed on the category contract.

---

## 11. Idempotency responsibility

Applications perform product-level idempotency where required.
Providers do not encode product deduplication.

---

## 12. Runtime support status

```text
slack conversation_channel:
  default_enabled = false
  runtime_binding_supported = true
  transport = Socket Mode inbound + Web API outbound

teams / discord / telegram / mattermost / rocket_chat / google_chat:
  default_enabled = false
  runtime_binding_supported = false
```

Slack notification_channel registration remains a separate identity.

---

## 13. Explicit exclusions

Teams Bot Framework / Discord Gateway / Telegram poll-webhook /
Mattermost WS / Rocket.Chat realtime / Google Chat events, LKW Slack workflow,
Ask invocation, product auth/dedupe, and universal rich UI remain out of scope
for this shared contract document.

`interaction_surface` remains removed.

---

## 14. Provider-specific organizational-history extension

**Classification:** `ARCHITECTURALLY FROZEN` — not part of the mandatory shared v1 contract.

The shared `ConversationChannelIntegrationContract` remains responsible for near-real-time conversational behavior only:

```text
start
stop
send
health
inbound conversation events
outbound replies
conversation addressing
actions
attachment fetch where supported
```

Do not add generic knowledge-history methods to the shared category contract.

A concrete conversation-channel integration may expose additional typed, bounded provider read primitives when that provider is also a source of organizational knowledge. These methods remain provider-specific operations of the same concrete integration. They do not become mandatory methods of every conversation channel provider and do not create another integration category.

**Slack example (conceptual — `PLANNED`, not implemented):**

```text
list accessible conversation inventory
read bounded conversation-history page
read bounded thread/reply page
read an exact message or exact revision
read safe attachment inventory
read explicit edit/deletion state
read bounded search result where Slack and policy support it
```

These names are conceptual only. Do not freeze Python method signatures here. Do not claim implementation.

---

## 15. Current status and Slack Knowledge direction

```text
CURRENTLY IMPLEMENTED:
  slack conversation_channel runtime (Socket Mode inbound + Web API outbound)
  LKW Slack frontend and Ask workflow foundations

ARCHITECTURALLY FROZEN (SLACK-KNOWLEDGE-THREE-MODE-ARCH-1):
  SlackConversationChannelIntegration remains the only public Slack integration
  shared typed Slack read primitives on the same integration
  three consumption modes: indexed RAG, durable materialization, live access
  Slack-as-frontend and Slack-as-knowledge-source are independent roles

PLANNED — NOT IMPLEMENTED:
  SLACK-KNOWLEDGE-FOUNDATION-1 (platform — NEXT)
  LKW-SLACK-CONNECTED-SOURCE-1 (LKW application)
  SLACK-LIVE-CAPABILITY-1 (platform)
  LKW-SLACK-KNOWLEDGE-PROOF-1 (LKW application)

Rejected:
  SlackKnowledgeIntegration, SlackRagIntegration, SlackDatabaseIntegration,
  SlackLiveIntegration, LkwSlackClient, or any LKW-owned Slack vendor client
```

Binding architecture: [`KNOWLEDGE_SOURCE_INTEGRATIONS.md`](KNOWLEDGE_SOURCE_INTEGRATIONS.md) §13.7 · Plan: [`plan/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../maintainers/plans/KNOWLEDGE_SOURCE_INTEGRATIONS.md) Phase 8.

Previous next task `LKW-SLACK-WORKFLOW-1` conversational workflow slices are **DONE** or superseded by the LKW implementation plan. Slack historical knowledge access does not yet exist.
