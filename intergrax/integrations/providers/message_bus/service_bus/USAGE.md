# Service Bus (service_bus)

Category: `message_bus`

## Legacy facade

- `create_service_bus_message_bus()` remains backward-compatible.

## Contract-based integration

- `ServiceBusMessageBusIntegration` derives from the category-specific contract.
- Factory: `create_service_bus_message_bus_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
