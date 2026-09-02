# Discord conversation channel (`discord`)

Category: `conversation_channel`

Public Integration class: `DiscordConversationChannelIntegration`

## Current status

```text
contract-defined, runtime-unbound
default_enabled = false
runtime_binding_supported = false
```

## Shared operations

- `start(handler)` - begin inbound event delivery
- `stop()` - stop delivery and release resources
- `send(message)` - outbound text (+ optional single-choice)
- `health()` - provider readiness via injected backend

## Declared conversation features (contract intent)

```text
text
single_choice
```

These features are declared intent, not proof of implemented runtime.

## Expected future vendor transport

```text
Gateway + REST API
```

## Explicitly unsupported today

- vendor SDK clients
- credentials / secrets
- live sockets, webhooks, polling
- network I/O on construction or registration
