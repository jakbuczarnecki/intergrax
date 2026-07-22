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

## 14. Next task

```text
LKW-SLACK-WORKFLOW-1 — product-local Slack conversational workflow
(authorization, tenant mapping, workspace selection, Ask HTTP, rendering)
```
