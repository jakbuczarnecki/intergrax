# `jira` integration — usage

**Category:** ``issue_tracker``  
**Catalog factory:** ``create_jira_issue_tracker()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(issue_tracker=IntegrationSlug.JIRA)
backend = profile.resolve(IntegrationCategory.ISSUE_TRACKER)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.issue_tracker.jira.bundle import create_jira_issue_tracker

backend = create_jira_issue_tracker(**config_overrides)
```


## Environment variables

`INTERGRAX_JIRA_BASE_URL`, `INTERGRAX_JIRA_EMAIL`, `INTERGRAX_JIRA_API_TOKEN`

## Example

```python
from intergrax.integrations.providers.issue_tracker.jira.bundle import create_jira_issue_tracker

tracker = create_jira_issue_tracker(base_url="https://acme.atlassian.net", email="bot@acme.com", api_token="...")
issue = tracker.get_issue("PROJ-123")
tracker.add_comment("PROJ-123", "Agent update: analysis complete")
results = tracker.search_issues('project = PROJ AND status = "In Progress"', limit=20)
```

## Notes

httpx only in ``opens.py``.
