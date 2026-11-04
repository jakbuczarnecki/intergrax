# Cursor project rules

| File | alwaysApply | Purpose |
|------|-------------|---------|
| `intergrax-token-budget.mdc` | **Yes** | I1 input + O1 output budget; mandatory preflight before implementation |
| `intergrax-test-process-budget.mdc` | **Yes** | One pytest/ruff per verification step; avoid parallel `uv` storms |
| `intergrax-session-artifacts.mdc` | **Yes** | Session/proof output under `.tmp/` only |
| `intergrax-git-branch-safety.mdc` | **Yes** | Stay on named branch; no detached HEAD without operator OK |
| `intergrax-env-vendor-qualification.mdc` | **Yes** | Env vars for secrets; C10/C15 freeze constraints |
| `intergrax-hep-step.mdc` | **No** | HEP / EVID scoped steps — `@` on demand |
| `intergrax-iteration.mdc` | **No** | F3 workflow, domain read scope, tier boundaries — `@` on demand |
| `intergrax-micro-implement.mdc` | **No** | Bounded micro implementation — `@` on demand |
| `intergrax-ci-hotfix.mdc` | **No** | CI/test hotfix — new chat; ultra scoped — `@` on demand |
| `intergrax-agent-roles.mdc` | **No** | Delivery DevOS role routing — `@intergrax-agent-roles` |
| `intergrax-independent-audit.mdc` | **No** | G6 independent audit handoff — `@intergrax-independent-audit` |

**Always-on:** token budget, test process budget, session artifacts, git branch safety, env/vendor qualification. **On demand:** iteration, HEP, micro-implement, CI hotfix, agent roles, independent audit.

Root **`AGENTS.md`** = Cursor auto-load stub (~350 tok). Full reference: [`docs/project/technical/guides/AGENT_INSTRUCTIONS.md`](../../docs/project/technical/guides/AGENT_INSTRUCTIONS.md). **Do not delete** via Settings trash icon.

Delivery DevOS hub: [`docs/project/technical/devos/README.md`](../../docs/project/technical/devos/README.md). Regenerate control-plane matrix: `uv run python -m scripts.governance.igx_gov sync-adapters`.

Setup: [`docs/project/technical/guides/CURSOR_TOKEN_SETUP.md`](../../docs/project/technical/guides/CURSOR_TOKEN_SETUP.md)
