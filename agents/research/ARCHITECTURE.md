# Research agents — architecture

## Purpose

Two-agent **research pipeline** for `research_application`: primary research + summary agents sharing graph delegation intent.

## Agents

| Module | Capability | Role |
|--------|------------|------|
| `research_agent.py` | `research.pipeline` | Main research UAEP flow |
| `summary_agent.py` | `research.summarize` | Summary step / delegation target |

## Runtime

- `HarnessReferenceAgent` pattern where lab hosts inject `LabHarnessContext`
- No Tier-3 `applications` imports in agent packages

## Skills

- Skill ids registered on each agent `contract.py` per `docs/SKILLS.md`

## Tests

- UAEP smoke under `agents/research/tests/`
- Application wiring: `research_application_tests/`

## Host

- Composed only in `applications/research_application/` (manifest + `environment` profile)
