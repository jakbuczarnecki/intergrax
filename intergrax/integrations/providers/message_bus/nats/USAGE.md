# `nats` integration — usage

**Category:** ``message_bus``  
**Catalog factory:** ``create_nats_message_bus()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(message_bus="nats")
backend = profile.resolve(IntegrationCategory.MESSAGE_BUS)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.message_bus.nats.bundle import create_nats_message_bus

backend = create_nats_message_bus(**config_overrides)
```


## Environment variables

`INTERGRAX_NATS_CONNECTION_STRING` (default ``nats://localhost:4222``)

## Example

```python
from intergrax.integrations.providers.message_bus.nats.bundle import create_nats_message_bus

bus = create_nats_message_bus(connection_string="nats://localhost:4222")
```

## Notes

Lightweight event bus facade.
