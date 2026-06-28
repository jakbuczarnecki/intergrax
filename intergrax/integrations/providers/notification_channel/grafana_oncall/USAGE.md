# Grafana Oncall (grafana_oncall)

Category: `notification_channel`

## Legacy facade

- `create_grafana_oncall_notification_channel()` remains backward-compatible.

## Contract-based integration

- `GrafanaOncallNotificationChannelIntegration` derives from the category-specific contract.
- Factory: `create_grafana_oncall_notification_channel_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
