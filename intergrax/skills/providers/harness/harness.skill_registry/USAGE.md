# `harness.skill_registry`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

**Skill resolver smoke**: minimal pack for registry merge unit tests (Phase S harness completion). Proves `SkillResolver` + `AgentRegistry` path with a single tool allow-list.

## How it works

Single `rag.retrieve` tool - name reflects registry testing intent, not registry introspection (use `skill.resolve` tool for that).

## How to use

```python
from intergrax.skills.providers.harness.manifests import HARNESS_SKILL_REGISTRY

AgentContract(id="skill_reg_lab", skills=[HARNESS_SKILL_REGISTRY], ...)
```

## What you get

Stable manifest for skill registry factory and resolver gate tests.

## Tools unlocked

`rag.retrieve`
