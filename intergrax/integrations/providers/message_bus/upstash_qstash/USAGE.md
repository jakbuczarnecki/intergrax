# Upstash Qstash (upstash_qstash)

Category: `message_bus`

## Legacy facade

- `create_upstash_qstash_message_bus()` remains backward-compatible.

## Contract-based integration

- `UpstashQstashMessageBusIntegration` derives from the category-specific contract.
- Factory: `create_upstash_qstash_message_bus_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
