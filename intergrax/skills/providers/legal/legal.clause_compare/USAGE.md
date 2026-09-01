# `legal.clause_compare`

**Bundle:** `legal` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

**Side-by-side clause comparison**: retrieve variants from index, draft comparison memo in shadow workspace, supplement with web evidence. Use after `legal.contract_review` baseline is active - skill **requires** parent pack tools and policy.

## How it works

1. `requires_skills=("legal.contract_review",)` - resolver expands dependency first; merged tools include parent + `workspace.write_file`.
2. Inherits `legal.contract_review.policy` policy fragment.
3. Agent writes diff output via `workspace.write_file` for human review.
4. Prompt ref: `legal.clause_compare.system`.

## How to use

```python
from intergrax.skills.providers.legal.manifests import LEGAL_CLAUSE_COMPARE

AgentContract(id="legal", skills=[LEGAL_CLAUSE_COMPARE], ...)
# Resolver expands to: legal.contract_review + legal.clause_compare
```

Enable `workspace` bundle on host (`legal_skill_profile` includes workspace).

## What you get

| Benefit | Detail |
|---------|--------|
| **Transitive composition** | Parent legal tools merged automatically |
| **Artifact output** | Comparison memo in shadow workspace |
| **Policy inheritance** | Same legal policy fragment as contract review |
| **Dispute / redline workflows** | Pairs with DSW analyst agents |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.retrieve` | From parent + this pack |
| `websearch.query` | From parent + this pack |
| `workspace.write_file` | Write comparison draft |

## Related skills

- `legal.contract_review` - required dependency
- `workspace.authoring` - broader workspace editing
