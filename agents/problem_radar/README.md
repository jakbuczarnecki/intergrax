# Problem Radar Agent (Phase K.1)

Architecture: [ARCHITECTURE.md](../../docs/project/technical/agents/problem_radar/ARCHITECTURE.md) · Plan: [IMPLEMENTATION_PLAN.md](../../docs/project/technical/agents/problem_radar/IMPLEMENTATION_PLAN.md). Tier-2 business agent prototype — discovers and clusters user pain signals (canon §36).

## Status

- **Deferred (2026-06-02):** Harness-first policy — no further K.1 work until plan §4.1 backlog is Done.
- **Frozen wave 1:** typed `ProblemRadarOutput`, stub domain logic, UAEP + `HarnessReferenceAgent`, Nexus gate test.

## Run (pytest)

```bash
uv run pytest agents/problem_radar/tests -m gate -q
```

## Capabilities

- `problem_radar.source_monitoring`
- `problem_radar.clustering`
- `problem_radar.scan` (lab / routing convenience)
