# `rabbitmq` integration — usage

**Category:** ``message_bus``  
**Catalog factory:** ``create_rabbitmq_message_bus()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(message_bus="rabbitmq")
backend = profile.resolve(IntegrationCategory.MESSAGE_BUS)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.message_bus.rabbitmq.bundle import create_rabbitmq_message_bus

backend = create_rabbitmq_message_bus(**config_overrides)
```


## Environment variables

`INTERGRAX_RABBITMQ_HOST`, `INTERGRAX_RABBITMQ_QUEUE`; optional USER/PASSWORD/VHOST

## Example

```python
from intergrax.integrations.providers.message_bus.rabbitmq.bundle import create_rabbitmq_message_bus

from intergrax.queueing.contracts.task_queue import TaskRequest

bus = create_rabbitmq_message_bus(host="localhost", queue="intergrax.tasks", kv_store=cache)
handle = bus.enqueue(TaskRequest(task_id="t-1", payload={"step": "run"}))
```

## Notes

Requires a ``kv_store`` (e.g. Redis) for task status. ``pika`` only in ``opens.py``.
