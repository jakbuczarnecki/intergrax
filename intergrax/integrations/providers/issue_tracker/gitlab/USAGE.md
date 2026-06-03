# `gitlab` integration — usage

**Category:** ``issue_tracker``  
**Catalog factory:** ``create_gitlab_issue_tracker()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(issue_tracker="gitlab")
backend = profile.resolve(IntegrationCategory.ISSUE_TRACKER)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.issue_tracker.gitlab.bundle import create_gitlab_issue_tracker

backend = create_gitlab_issue_tracker(**config_overrides)
```


## Environment variables

`INTERGRAX_GITLAB_URL`, `INTERGRAX_GITLAB_TOKEN`, `INTERGRAX_GITLAB_REPO` (project id/path)

## Example

```python
from intergrax.integrations.providers.issue_tracker.gitlab.bundle import create_gitlab_issue_tracker

tracker = create_gitlab_issue_tracker(base_url="https://gitlab.com/api/v4", repo="group/project")
```

## Notes

GitLab REST issue tracker.
