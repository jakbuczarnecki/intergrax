# InterGRAX policy hierarchy (Delivery DevOS)

When instructions conflict, resolve in this order (highest first):

| Priority | Source | Enforcement |
|----------|--------|-------------|
| P0 | Root [`AGENTS.md`](../AGENTS.md) | Procedural (Cursor auto-load) + tier boundaries in code review |
| P1 | [`governance/`](./) YAML + tickets | Technical (hooks, `igx_gov`, pytest verifiers) + procedural (ticket lifecycle) |
| P2 | [`agent-env/rules/`](../agent-env/rules/) | Procedural (load on demand except token budget) |
| P3 | [`docs/project/technical/guides/AGENT_INSTRUCTIONS.md`](../docs/project/technical/guides/AGENT_INSTRUCTIONS.md) | Procedural (explicit `@` or operator keyword) |
| P4 | Domain architecture ↔ maintainer plan pairs | Procedural (F3 one domain per chat) |

## Delivery SSOT vs runtime governance

- **Delivery DevOS SSOT:** repository root `governance/` (tickets, agent control plane, policy registry).
- **Product runtime governance:** `intergrax/runtime/governance/` — do not conflate with Delivery DevOS paths.

## Qualification / freeze

FRZ-* and qualification statuses live under `docs/project/maintainers/qualification/`. The policy registry **points to** those artifacts; DCP does **not** change FRZ statuses.

## Enforcement classes

- **TECHNICAL:** exit-code gates (`verify_commit_ticket_refs.py`, push gate, CI jobs, governance pytest).
- **PROCEDURAL:** Cursor rules, skills, ticket lifecycle fields — honored by agents and operators; verified by consistency scripts where noted.
