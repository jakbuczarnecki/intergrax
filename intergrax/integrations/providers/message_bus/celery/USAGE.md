# `celery` integration — usage

**Category:** ``message_bus``  
**Catalog factory:** ``create_celery_message_bus()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(message_bus=IntegrationSlug.CELERY)
backend = profile.resolve(IntegrationCategory.MESSAGE_BUS)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.message_bus.celery.bundle import create_celery_message_bus

backend = create_celery_message_bus(**config_overrides)
```


## Environment variables

`INTERGRAX_CELERY_BROKER_URL`, `INTERGRAX_CELERY_BACKEND_URL`

## Example

```python
from intergrax.integrations.providers.message_bus.celery.bundle import create_celery_message_bus

from intergrax.queueing.contracts.task_queue import TaskRequest

bus = create_celery_message_bus(broker_url="redis://localhost:6379/1")
handle = bus.enqueue(TaskRequest(task_id="t-1", payload={"graph": "demo"}))
```

## Notes

You may inject an existing Celery ``app``. Workers: ``create_celery_worker_app()``.
