# `sandbox.code_exec`

**Bundle:** `sandbox` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Sandboxed code execution with workspace IO for coding agents.

## How it works

sandbox.exec + workspace read/write via ToolWiringContext.

## How to use

sandbox_skill_profile(); wire sandbox_session on host.

## What you get

Isolated exec without host filesystem access.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `sandbox.exec` | Run allowlisted sandbox operation |
| `workspace.read_file` | Read script/input |
| `workspace.write_file` | Write output |

## Related skills

- `workspace.authoring`
- `ops.security_audit`
