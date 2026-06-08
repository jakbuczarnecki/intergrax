# `workspace.authoring`

**Bundle:** `workspace` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

**Shadow workspace drafting** — read, write, search files in the isolated shadow workspace and persist notes to task memory. Use for LKW `local_synthesizer`, report generators, and coding assistants that produce artifacts without touching the host filesystem directly.

## How it works

1. Resolves `workspace.read_file`, `workspace.write_file`, `workspace.search`, `memory.write`.
2. Workspace tools bind to `ShadowWorkspace` on `ToolWiringContext` (Tier-3 bootstrap).
3. `memory.write` stores draft metadata or summaries in task KV for multi-step pipelines.
4. LLM calls tools individually; skill only expands the allow-list at registration.

## How to use

```python
from intergrax.skills.providers.workspace.manifests import WORKSPACE_AUTHORING
from intergrax.applications._shared.skill_wiring import lkw_skill_profile

# Host
SkillProfile(enabled_bundles=["workspace"])  # or lkw_skill_profile()

# Agent
AgentContract(id="local_synthesizer", skills=[WORKSPACE_AUTHORING], ...)
```

Enable shadow workspace in environment profile / `wire_shadow_workspace(env)` so workspace tools resolve.

## What you get

| Benefit | Detail |
|---------|--------|
| **Safe artifact IO** | Shadow workspace isolation per task |
| **Grep across drafts** | `workspace.search` without shell access |
| **Memory handoff** | Persist draft pointers for later steps |
| **Reusable across products** | Same pack for LKW, research reports, legal memos |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `workspace.read_file` | Read UTF-8 artifact from shadow workspace |
| `workspace.write_file` | Create or overwrite draft files |
| `workspace.search` | Text search across workspace files |
| `memory.write` | Store draft metadata in task memory |

## Related skills

- `research.citation_synthesis` — writes reports to workspace
- `rag.hybrid_qa` — source material for drafts
- `legal.clause_compare` — clause diff output to workspace
