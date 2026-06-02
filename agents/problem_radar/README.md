# Problem Radar Agent (Phase K.1)

Tier-2 business agent prototype — discovers and clusters user pain signals (canon §36).

## Status

- **K.1 wave 1:** typed `ProblemRadarOutput`, stub domain logic, UAEP + `HarnessReferenceAgent`, Nexus gate test.
- **Next:** live `websearch` ingestion, multi-step UAEP pipeline, lab opt-in (`LAB_INCLUDE_PROBLEM_RADAR`).

## Run (pytest)

```bash
uv run pytest agents/problem_radar/tests -m gate -q
```

## Capabilities

- `problem_radar.source_monitoring`
- `problem_radar.clustering`
- `problem_radar.scan` (lab / routing convenience)
