# GitLab tool bundle

**Bundle id:** `gitlab`  
**Tools:** `gitlab.create_issue`

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `issue_tracker` | Yes | GitLab `IssueTracker` with `create_issue()` (full adapter) |

Tier-3 example:

```python
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_integrations()
register_default_tools()

profile = IntegrationProfile(issue_tracker="gitlab")
ctx = ToolWiringContext.from_integration_profile(profile)
registry = build_registry_from_profile(ToolProfile(enabled_bundles=["gitlab"]), ctx=ctx)
```

Env: `INTERGRAX_GITLAB_TOKEN`, `INTERGRAX_GITLAB_REPO` (project id or path).

## Agent allow-list

```python
AgentContract(allowed_tools=["gitlab.create_issue"], ...)
```
