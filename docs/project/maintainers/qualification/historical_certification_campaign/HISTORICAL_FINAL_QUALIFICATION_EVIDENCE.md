# HIST Phase 6 — QUAL-X Dimension A replay evidence

## Current record — POST_ACTIVATION (authoritative)

| Field | Value |
|---|---|
| Label | **POST_ACTIVATION** (P5-VERIFY green after N-03 re-audit) |
| EXECUTION_SHA | `bf03550aa95ce239a63542916331c6a648617e33` |
| IIAC effective_from_sha | `8045915a353bb04ea5a290f9ef8b6def33dee96d` |
| Command | `uv run pytest -p no:xdist` — 35 P5 Dimension A modules + BND-03 + W2-C |
| Result | **595 passed**, 0 failed |
| Exit code | **0** |
| Duration | 368.98s |
| Log | `HISTORICAL_P6_DIMENSION_A_REPLAY_2026-11-09.log` |
| Baseline reference | AUD-QUALX @ `d9ea0d6e8`: 630 passed (manifest node count differs) |

Does not certify ENT Population E, ARCH-FREEZE, or campaign terminal closure.

## Superseded — PRE_ACTIVATION_DIAGNOSTIC (final-docs branch)

| Field | Value |
|---|---|
| Label | `PRE_ACTIVATION_DIAGNOSTIC` |
| Replay SHA | `427646f41de0df992cd54c0e157de8f36b560f39` |
| Outcome | 595 passed, exit 0, 304.13s |
| Log SHA-256 | `d42cd8b9810f2141dd0e4ff5741ad5ef8dfd21cec7d798bef8fa8f7ee2c73a5f` |

Retained for lineage; not used for P7 promotion after POST_ACTIVATION record above.
