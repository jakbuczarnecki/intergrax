# InterGRAX Delivery DevOS documentation hub

Delivery Quality / DevOS (DCP) layers:

| Doc | Purpose |
|-----|---------|
| [DEVELOPER-QUALITY-PROCESS.md](./DEVELOPER-QUALITY-PROCESS.md) | Local quality ticket, hooks, profiles |
| [AGENT-ARCHITECTURE.md](./AGENT-ARCHITECTURE.md) | Roles, skills, control plane |
| [PARALLEL-EXECUTION.md](./PARALLEL-EXECUTION.md) | Worktrees + subagents (no PostgreSQL) |
| [CI-TIER-MAP.md](./CI-TIER-MAP.md) | DevOS tiers ↔ InterGRAX CI |
| [INDEPENDENT-REVIEW-HANDOFF.md](./INDEPENDENT-REVIEW-HANDOFF.md) | G6 handoff |
| [EVIDENCE-AND-CERTIFICATION.md](./EVIDENCE-AND-CERTIFICATION.md) | SHA ladder, manifest flags |
| [OPERATIONS-AND-RECOVERY.md](./OPERATIONS-AND-RECOVERY.md) | Locks, bypass, recovery |

SSOT paths:

- Policy: `governance/GOVERNANCE-MANIFEST.yaml`, `governance/policy-registry.yaml`
- Tickets: `governance/tickets/`
- Agent control plane: `governance/agent-control-plane/`
- CLI: `uv run python -m scripts.governance.igx_gov`
- Skills: **11** `igx-devos-*` entries in `skill_routing_registry.yaml` (parity with `.cursor/skills/`; reference DevOS skill count is higher — GitHub Issues and other packs intentionally omitted)
- `igx_gov sync-adapters` — Cursor-only; runs `generate_control_plane_matrix.py` (see manifest GOV-02 note)

Certification flags in the manifest remain **false** until external G6 audit.
