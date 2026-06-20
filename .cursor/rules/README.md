# Cursor project rules

| File | alwaysApply | Purpose |
|------|-------------|---------|
| `intergrax-iteration.mdc` | **Yes** | Workflow, F3 session protocol, context budget (~900 tok/turn) |
| `intergrax-idea-audit.mdc` | **No** | Mode I — idea audit on trigger |
| `intergrax-agents-reference.mdc` | **No** | Pointer to `@AGENTS.md` on demand |
| `intergrax-implementation-journal.mdc` | **No** | Journal milestones only |

**Do not** add root `AGENTS.md` as an always-on Project Rule — duplicates ~3k tokens/turn.  
Setup: [`docs/guides/CURSOR_TOKEN_SETUP.md`](../docs/guides/CURSOR_TOKEN_SETUP.md)
