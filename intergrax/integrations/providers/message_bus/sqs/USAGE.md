# `sqs` integration — usage

**Category:** ``message_bus``  
**Catalog factory:** ``create_sqs_message_bus()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(message_bus=IntegrationSlug.SQS)
backend = profile.resolve(IntegrationCategory.MESSAGE_BUS)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.message_bus.sqs.bundle import create_sqs_message_bus

backend = create_sqs_message_bus(**config_overrides)
```


## Environment variables

`INTERGRAX_SQS_QUEUE`; optional `INTERGRAX_SQS_REGION`; AWS credential vars

## Example

```python
from intergrax.integrations.providers.message_bus.sqs.bundle import create_sqs_message_bus

from intergrax.queueing.contracts.task_queue import TaskRequest

bus = create_sqs_message_bus(queue_name="intergrax-tasks", region="eu-central-1")
handle = bus.enqueue(TaskRequest(tenant_id="t1", run_id="r1", task_name="echo", payload=b"{}", idempotency_key=None))
status = bus.get_status(handle)
```

## Notes

``CloudTaskQueue`` over boto3 SQS. Default ``message_bus`` when ``cloud_platform=aws``.
