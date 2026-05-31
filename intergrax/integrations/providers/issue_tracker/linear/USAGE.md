# `linear` integration — usage

**Category:** ``issue_tracker``  
**Catalog factory:** ``create_linear_issue_tracker()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(issue_tracker=IntegrationSlug.LINEAR)
backend = profile.resolve(IntegrationCategory.ISSUE_TRACKER)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.issue_tracker.linear.bundle import create_linear_issue_tracker

backend = create_linear_issue_tracker(**config_overrides)
```


## Environment variables

`INTERGRAX_LINEAR_API_KEY`; optional `INTERGRAX_LINEAR_URL`

## Example

```python
from intergrax.integrations.providers.issue_tracker.linear.bundle import create_linear_issue_tracker

tracker = create_linear_issue_tracker(api_key="lin_api_...")
issue = tracker.get_issue("ENG-123")
tracker.add_comment("ENG-123", "Automated triage complete.")
results = tracker.search_issues("priority:1 state:open", limit=20)
```

## Notes

httpx REST client opened lazily.
