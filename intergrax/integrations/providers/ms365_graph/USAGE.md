# `ms365_graph` integration — usage

**Category:** ``collaboration_suite``  
**Catalog factory:** ``create_ms365_graph_collaboration_suite()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(collaboration_suite=IntegrationSlug.MS365_GRAPH)
backend = profile.resolve(IntegrationCategory.COLLABORATION_SUITE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.ms365_graph.bundle import create_ms365_graph_collaboration_suite

backend = create_ms365_graph_collaboration_suite(**config_overrides)
```


## Environment variables

`INTERGRAX_MS365_TENANT_ID`, `INTERGRAX_MS365_CLIENT_ID`, `INTERGRAX_MS365_CLIENT_SECRET`

## Example

```python
from intergrax.integrations.providers.ms365_graph.bundle import create_ms365_graph_collaboration_suite

suite = create_ms365_graph_collaboration_suite(tenant_id="...", client_id="...", client_secret="...")
user = suite.get_user("user@contoso.com")
events = suite.list_calendar_events(user_id=user.id, start="2026-05-01", end="2026-05-31")
suite.send_mail(to=["user@contoso.com"], subject="Report", body="...")
```

## Notes

OAuth client credentials in ``opens.py``.
