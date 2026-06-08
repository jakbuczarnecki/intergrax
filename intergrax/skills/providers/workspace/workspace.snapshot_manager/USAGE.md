# `workspace.snapshot_manager`

**Bundle:** `workspace` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Workspace snapshot and cleanup for long authoring sessions.

## How it works

workspace.snapshot + list_files + delete_file.

## How to use

lkw_skill_profile() or sandbox_skill_profile() hosts.

## What you get

Checkpoint/rollback without host filesystem access.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `workspace.snapshot` | Create workspace snapshot |
| `workspace.list_files` | List workspace files |
| `workspace.delete_file` | Delete stale file |

## Related skills

- `workspace.authoring`
- `storage.artifact_sync`
