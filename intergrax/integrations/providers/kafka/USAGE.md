# `kafka` integration — usage

**Category:** ``message_bus``  
**Catalog factory:** ``create_kafka_message_bus()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(message_bus=IntegrationSlug.KAFKA)
backend = profile.resolve(IntegrationCategory.MESSAGE_BUS)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.kafka.bundle import create_kafka_message_bus

backend = create_kafka_message_bus(**config_overrides)
```


## Environment variables

`INTERGRAX_KAFKA_BOOTSTRAP_SERVERS`, `INTERGRAX_KAFKA_TOPIC`, `INTERGRAX_KAFKA_CONSUMER_GROUP`

## Example

```python
from intergrax.integrations.providers.kafka.bundle import create_kafka_message_bus

from intergrax.queueing.contracts.task_queue import TaskRequest

bus = create_kafka_message_bus(bootstrap_servers="localhost:9092", topic="intergrax.tasks")
handle = bus.enqueue(TaskRequest(task_id="t-1", payload={"agent": "echo"}))
status = bus.get_status(handle)
result = bus.get_result(handle)
```

## Notes

``confluent_kafka`` only in ``opens.py``.
