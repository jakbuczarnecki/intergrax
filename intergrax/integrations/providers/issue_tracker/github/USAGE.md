# `github` integration — usage

**Category:** ``issue_tracker``  
**Catalog factory:** ``create_github_issue_tracker()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(issue_tracker="github")
backend = profile.resolve(IntegrationCategory.ISSUE_TRACKER)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.issue_tracker.github.bundle import create_github_issue_tracker

backend = create_github_issue_tracker(**config_overrides)
```


## Environment variables

`INTERGRAX_GITHUB_TOKEN`; optional `INTERGRAX_GITHUB_ORG`, `INTERGRAX_GITHUB_REPO`, `INTERGRAX_GITHUB_URL`

## Example

```python
from intergrax.integrations.providers.issue_tracker.github.bundle import create_github_issue_tracker

tracker = create_github_issue_tracker(token="ghp_...", org="acme", repo="platform")
issue = tracker.get_issue("42")
tracker.add_comment("42", "Agent: root cause identified.")
results = tracker.search_issues("is:open label:agent", limit=20)
```

## Notes

httpx REST client opened lazily. ``search_issues`` accepts GitHub search query syntax.
