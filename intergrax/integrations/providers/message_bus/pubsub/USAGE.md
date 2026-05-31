# `pubsub` integration — usage

**Category:** ``message_bus``  
**Catalog factory:** ``create_pubsub_message_bus()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(message_bus=IntegrationSlug.PUBSUB)
backend = profile.resolve(IntegrationCategory.MESSAGE_BUS)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.message_bus.pubsub.bundle import create_pubsub_message_bus

backend = create_pubsub_message_bus(**config_overrides)
```


## Environment variables

`INTERGRAX_PUBSUB_PROJECT_ID`, `INTERGRAX_PUBSUB_TOPIC`; GCP ADC or service account

## Example

```python
from intergrax.integrations.providers.message_bus.pubsub.bundle import create_pubsub_message_bus

from intergrax.queueing.contracts.task_queue import TaskRequest

bus = create_pubsub_message_bus(project_id="my-project", topic="intergrax-tasks")
handle = bus.enqueue(TaskRequest(tenant_id="t1", run_id="r1", task_name="echo", payload=b"{}", idempotency_key=None))
```

## Notes

``google-cloud-pubsub`` opened lazily. Default ``message_bus`` when ``cloud_platform=gcp``.
