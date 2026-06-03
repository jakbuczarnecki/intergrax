# `google_workspace` integration — usage

**Category:** ``collaboration_suite``  
**Catalog factory:** ``create_google_workspace_collaboration_suite()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(collaboration_suite="google_workspace")
backend = profile.resolve(IntegrationCategory.COLLABORATION_SUITE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.collaboration_suite.google_workspace.bundle import create_google_workspace_collaboration_suite

backend = create_google_workspace_collaboration_suite(**config_overrides)
```


## Environment variables

OAuth bearer via `INTERGRAX_GOOGLE_WORKSPACE_TOKEN` or service account; optional `INTERGRAX_GOOGLE_WORKSPACE_URL`

## Example

```python
from intergrax.integrations.providers.collaboration_suite.google_workspace.bundle import create_google_workspace_collaboration_suite

suite = create_google_workspace_collaboration_suite(token="ya29....")
user = suite.get_user("user@example.com")
messages = suite.list_messages("user@example.com", folder="inbox", limit=10)
suite.send_mail("user@example.com", subject="Report", body="...", to=["ops@example.com"])
events = suite.list_calendar_events("primary", start="2026-05-01T00:00:00Z", end="2026-05-31T23:59:59Z")
```

## Notes

Gmail / Calendar / Directory REST. Google-tenant parity with ``ms365_graph``.
