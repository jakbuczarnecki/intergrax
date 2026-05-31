# Sandbox tool bundle

**Bundle id:** `sandbox`  
**Tools:** `sandbox.exec`

## Dependencies (`ToolWiringContext`)

| Field | Required | Purpose |
|-------|----------|---------|
| `sandbox_session` | Yes | Active `SandboxSession` from `intergrax/runtime/sandbox/` |

Tier-3 example:

```python
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.registry import ToolProfile, ToolWiringContext, build_registry_from_profile, register_default_tools

register_default_tools()
session = SandboxSession.create(base_dir, tenant_id="t1", task_id="task-1")
ctx = ToolWiringContext(sandbox_session=session)
registry = build_registry_from_profile(ToolProfile(enabled=["sandbox.exec"]), ctx=ctx)
```

## Allowlisted operations

Default: `echo`, `read_file`, `write_file`, `list_files` (see `DEFAULT_SANDBOX_OPERATIONS`).

## Agent allow-list

```python
AgentContract(allowed_tools=["sandbox.exec"], ...)
```
