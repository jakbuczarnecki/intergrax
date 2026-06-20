---
id: IJ-2026-06-10-014
date: 2026-06-10
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-0b
  - ACP-1
  - ACP-2
  - ACP-3
  - ACP-4
  - ACP-5
  - ACP-6
  - ACP-8
  - ACP-9
  - ACP-11
  - ACP-13
  - ACP-LEG-4
status: completed
commit: pending
adr: none — implements architecture §24–§26 pattern library per ADR-AGENT-001
---

# ACP Wave 5 — cognitive patterns and typed scaffold

## Operator request

Deliver Wave 5: cognitive pattern library, `cognitive_pattern` on contract, and scaffold `--pattern` without UAEP boilerplate.

## Summary

Added `CognitiveAgent` ABC and five pattern bases (reflex, react, plan_execute, decomposition, reflection) with typed session state subclasses. Extended `AgentContract` with `cognitive_pattern`, `pattern_version`, and `pattern_config` plus assembly validation. Scaffold `new-agent --pattern <name>` emits typed `on_next_step` hooks. Reference probes and CI scripts `check_scaffold_acp_pattern.py` / `check_agent_pattern_conformance.py`.

## Project impact

Authors can scaffold readable typed agents in one command; harness provides pattern bases and probes for lab wiring and conformance gates.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §24–§26 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` §6.1aw Wave 5 |

## Changed artifacts

- `intergrax/agents/authoring/patterns/` — pattern library + reference probes
- `intergrax/contracts/agent_contract_meta.py` — cognitive_pattern fields
- `intergrax/runtime/registry/agent_assembly_resolver.py` — pattern validation
- `intergrax/scaffold/new_agent.py`, `cli.py` — `--pattern` flag
- `scripts/check_scaffold_acp_pattern.py`, `check_agent_pattern_conformance.py`

## Verification

```bash
uv run pytest tests/unit/agents/authoring/patterns tests/unit/scaffold/test_acp_pattern_scaffold.py -q
uv run python scripts/check_scaffold_acp_pattern.py
```

Result: 5 passed; scaffold script OK with `uv run`.

## Risks and follow-ups

- Default scaffold (no `--pattern`) still UAEP-first until fleet migration program (Wave 8).
- `ReflectionAgent` CVL critic hook deferred to CRITIC_VERIFICATION iteration.
- ACP-12 `agent_os` pattern extension optional next gate maintenance item.
