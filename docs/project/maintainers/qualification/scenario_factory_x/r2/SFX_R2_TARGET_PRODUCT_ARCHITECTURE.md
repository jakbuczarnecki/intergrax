# SFX R2 — Target Product Architecture (PROPOSED)

Preserves R1 Option B diagram with **Mission / Autonomy** plane added at Tier-2 SFX only.

```text
Customer / Studio / CLI / MCP
        │
        ▼
Tier-3 — auth, policy, parity ops (incl. mission.* PROPOSED)
        ▼
Tier-2 agents/scenario_factory — PRODUCT ONLY
  • intake, discovery, spec, goals
  • blueprint, resolution
  • mission driver + autonomy envelope (PROPOSED)
  • candidate iteration views + business evaluator
  • evidence compare + commercial package views
        │ engineering handoff          │ sandbox execution
        ▼                              ▼
   SEF-X EEA                     Intergrax Runtime
   plan/code/repair              execute_scenario_task
        └──────── evidence ────────┘
                    ▼
          champion + dossier (not admission)
```

## Semantic ownership (unchanged + R2)

| Concern | Owner |
| --- | --- |
| Business goals, mission mode, champion pick | SFX |
| Engineering submit authority | Human grant → **SEF validates** (PROPOSED) |
| Scenario runs | Runtime only |
| Grants / approvals | Existing decision mechanisms |
| Production admission | External programme |