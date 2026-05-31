# `temporal` integration — usage

**Category:** ``message_bus``  
**Catalog factory:** ``create_temporal_message_bus()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(message_bus=IntegrationSlug.TEMPORAL)
backend = profile.resolve(IntegrationCategory.MESSAGE_BUS)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.message_bus.temporal.bundle import create_temporal_message_bus

backend = create_temporal_message_bus(**config_overrides)
```


## Environment variables

`INTERGRAX_TEMPORAL_CONNECTION_STRING`

## Example

```python
from intergrax.integrations.providers.message_bus.temporal.bundle import create_temporal_message_bus

bus = create_temporal_message_bus(connection_string="localhost:7233")
```

## Notes

Durable workflow enqueue facade (``temporalio`` optional at runtime).
