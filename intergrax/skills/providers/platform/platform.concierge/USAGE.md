# `platform.concierge`

**Bundle:** `platform` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

**Intergrax Assistant hub pack** — retrieval, web evidence, session memory read, and skill introspection. Designed for `intergrax_assistant` (`platform.assist`): everyday chat with grounded answers and the ability to inspect which skills/tools would apply before delegating to specialists.

## How it works

1. Unions `rag.retrieve`, `websearch.query`, `memory.read`, `skill.resolve`.
2. `skill.resolve` is a catalog diagnostic tool — returns merged tool/prompt/policy ids for given `skill_ids` without executing them.
3. Hub agent uses skills for allow-list; Nexus delegation routes to Legal/Research when capability classifier decides.
4. Prompt ref: `platform.concierge.system`.

## How to use

```python
from intergrax.skills.providers.platform.manifests import PLATFORM_CONCIERGE
from intergrax.applications._shared.skill_wiring import platform_skill_profile

# intergrax_assistant_application host
env.skill_profile = platform_skill_profile()

AgentContract(
    id="intergrax_assistant",
    skills=[PLATFORM_CONCIERGE],
    capabilities=["platform.assist"],
)
```

Wire `search_provider` + RAG stack on integration profile for retrieval and web tools.

## What you get

| Benefit | Detail |
|---------|--------|
| **Chat-shaped harness** | Curated tools for general Q&A, not full 200-tool surface |
| **Skill introspection** | `skill.resolve` helps explain capability packs to users |
| **Session-aware** | `memory.read` for multi-turn context |
| **Delegation-ready** | Narrow hub surface; specialists carry domain skills |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.retrieve` | Grounded answers from vector index |
| `websearch.query` | Fresh external evidence |
| `memory.read` | Prior turn / task context |
| `skill.resolve` | Inspect skill composition (diagnostic) |

## Related skills

- `rag.hybrid_qa` — deeper index Q&A without web/skill.resolve
- `research.web_evidence` — heavier web fetch pack
- `legal.contract_review` — delegate target for legal turns
