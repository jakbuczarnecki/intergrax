# `openai.vector_admin`

**Bundle:** `openai` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

OpenAI vector store admin: upload, clear, and file_search query.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Use `openai_skill_profile()`; Enable bundle `openai` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `openai.vector_store.upload`, `openai.vector_store.clear`, `openai.file_search.query`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `openai.vector_store.upload` | Catalog tool |
| `openai.vector_store.clear` | Catalog tool |
| `openai.file_search.query` | Catalog tool |

## Related skills

-
 
`
o
p
e
n
a
i
.
*
`
 
p
e
e
r
s
 
i
n
 
s
a
m
e
 
b
u
n
d
l
e
