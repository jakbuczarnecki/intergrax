# Webhook (webhook)

Category: `notification_channel`

## Legacy facade

- `create_webhook_integration()` remains backward-compatible.

## Contract-based integration

- `WebhookNotificationChannelIntegration` derives from the category-specific contract.
- Factory: `create_webhook_notification_channel_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
