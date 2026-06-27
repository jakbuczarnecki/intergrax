---
id: IJ-2026-06-17-002
date: 2026-06-17
tiers:
  - tier-0
  - tier-2
scope: AGENT_CONTRACTS_AND_ASSEMBLY
plan_ref:
  - ACP-TOK-CI
status: completed
commit: pending
adr: no ADR needed — deterministic test wiring and CI error reporting only
---

# ACP-TOK-CI — stabilize token budget smoke tests and CI diagnostics

## Operator request

Fix GitHub nightly regression failure on `check_agent_token_budget_contract.py` where smoke tests reported only `RequestsDependencyWarning` and ACP-CLOSE CI exited with code 1.

## Summary

Stabilized ACP-TOK smoke tests by routing host-context runs through `MeteringFakeLLMAdapter` instead of live `LLMProfile.lab()` Ollama. Fixed CI script failure reporting to surface pytest stdout. Pinned `chardet<6` and silenced `RequestsDependencyWarning` in pytest config.

## Project impact

ACP-TOK-CI gate is deterministic on CI runners without Ollama; smoke subprocess failures are diagnosable from pytest output.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §25.4–§25.5 |
| Plan | `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` — ACP-TOK-CI |
| ADR | none |

## Changed artifacts

- `testing_support/builder.py` — `MeteringFakeLLMAdapter` with response `LLMTokenUsage`
- `tests/unit/agents/conftest.py` — autouse fake LLM + identity `compile_prompt_text` for `test_acp_token_*`
- `scripts/maintenance/check_agent_token_budget_contract.py` — smoke failure detail + warning filter
- `pyproject.toml` — `chardet>=5.2,<6`, `filterwarnings` for requests pin warning
- `uv.lock` — chardet 5.2.0 pin

## Verification

```bash
uv run pytest tests/unit/agents/test_acp_token_usage_metering.py tests/unit/agents/test_acp_token_budget_enforcement.py tests/unit/agents/test_acp_token_budget_reactions.py -q
uv run python scripts/maintenance/check_agent_token_budget_contract.py
uv run python scripts/gates/check_agent_acp_close_ci.py
python scripts/maintenance/check_implementation_journal.py
```

Result: pass (13/13 token budget tests × 10 consecutive runs; ACP-TOK-CI and ACP-CLOSE CI OK).

## Risks and follow-ups

- Other host-context agent tests outside `test_acp_token_*` may still hit live Ollama when `ACPSessionHostContext` uses `lab_defaults()` without similar patching.
