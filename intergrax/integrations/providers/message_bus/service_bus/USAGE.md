# `service_bus` integration — usage

**Category:** ``message_bus``  
**Catalog factory:** ``create_service_bus_message_bus()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(message_bus="service_bus")
backend = profile.resolve(IntegrationCategory.MESSAGE_BUS)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.message_bus.service_bus.bundle import create_service_bus_message_bus

backend = create_service_bus_message_bus(**config_overrides)
```


## Environment variables

`INTERGRAX_SERVICE_BUS_CONNECTION_STRING`, `INTERGRAX_SERVICE_BUS_QUEUE`

## Example

```python
from intergrax.integrations.providers.message_bus.service_bus.bundle import create_service_bus_message_bus

from intergrax.queueing.contracts.task_queue import TaskRequest

bus = create_service_bus_message_bus(
    connection_string="Endpoint=sb://....servicebus.windows.net/;...",
    queue_name="intergrax-tasks",
)
handle = bus.enqueue(TaskRequest(tenant_id="t1", run_id="r1", task_name="echo", payload=b"{}", idempotency_key=None))
```

## Notes

``azure-servicebus`` opened lazily. Default ``message_bus`` when ``cloud_platform=azure``.
