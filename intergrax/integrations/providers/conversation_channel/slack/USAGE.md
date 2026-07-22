# Slack conversation channel (`slack`)

Category: `conversation_channel`

Public Integration class: `SlackConversationChannelIntegration`
Backend class: `SlackConversationChannelBackend`

## Current status

```text
BETA
Socket Mode inbound + Web API outbound runtime supported
default_enabled = false
runtime_binding_supported = true
```

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
```

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
- LKW product behavior (auth, tenant, Ask, dedupe, citations)

## Declared conversation features

```text
text
single_choice
```
