# Cursor project rules

| File | alwaysApply | Purpose |
|------|-------------|---------|
| `intergrax-token-budget.mdc` | **Yes** | I1 input + O1 output budget; mandatory preflight before implementation |
| `intergrax-hep-step.mdc` | **No** | HEP / EVID scoped steps — `@` on demand |
| `intergrax-iteration.mdc` | **No** | F3 workflow, domain read scope, tier boundaries — `@` on demand |

**Only `intergrax-token-budget.mdc` is always-on.** All other rules load on explicit `@` reference or operator request.

Root **`AGENTS.md`** = Cursor auto-load stub (~350 tok). Full reference: [`docs/project/technical/guides/AGENT_INSTRUCTIONS.md`](../../docs/project/technical/guides/AGENT_INSTRUCTIONS.md). **Do not delete** via Settings trash icon.

Setup: [`docs/project/technical/guides/CURSOR_TOKEN_SETUP.md`](../../docs/project/technical/guides/CURSOR_TOKEN_SETUP.md)
