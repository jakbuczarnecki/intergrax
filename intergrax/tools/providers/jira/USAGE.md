# Jira tool bundle

**Bundle id:** `jira`  
**Tools:** `jira.get_issue`, `jira.add_comment`, `jira.search_tasks`

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `issue_tracker` | Yes | `IssueTracker` contract (typically `jira` integration) |

Tier-3 example:

```python
from intergrax.integrations import IntegrationProfile, IntegrationSlug, register_default_integrations
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_integrations()
register_default_tools()

profile = IntegrationProfile(issue_tracker=IntegrationSlug.JIRA)
ctx = ToolWiringContext.from_integration_profile(profile)
registry = build_registry_from_profile(ToolProfile(enabled_bundles=["jira"]), ctx=ctx)
```

## Agent allow-list

```python
AgentContract(allowed_tools=["jira.get_issue", "jira.search_tasks", "jira.add_comment"], ...)
```
