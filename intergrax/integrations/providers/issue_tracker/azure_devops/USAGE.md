# `azure_devops` integration — usage

**Category:** ``issue_tracker``  
**Catalog factory:** ``create_azure_devops_issue_tracker()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(issue_tracker="azure_devops")
backend = profile.resolve(IntegrationCategory.ISSUE_TRACKER)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.issue_tracker.azure_devops.bundle import create_azure_devops_issue_tracker

backend = create_azure_devops_issue_tracker(**config_overrides)
```


## Environment variables

`INTERGRAX_AZURE_DEVOPS_TOKEN`; optional `INTERGRAX_AZURE_DEVOPS_ORG`, `INTERGRAX_AZURE_DEVOPS_REPO`, `INTERGRAX_AZURE_DEVOPS_URL`

## Example

```python
from intergrax.integrations.providers.issue_tracker.azure_devops.bundle import create_azure_devops_issue_tracker

tracker = create_azure_devops_issue_tracker(token="...", org="acme", repo="Platform")
issue = tracker.get_issue("12345")
tracker.add_comment("12345", "Agent update posted.")
results = tracker.search_issues("[System.State] = 'Active'", limit=20)
```

## Notes

REST work-item facade; WIQL passed via ``search_issues``.
