# Incident Report - Orion

## Summary

Incident Orion occurred on 2026-08-17 during a scheduled maintenance window overlap.

## Timeline

- 2026-08-17 02:14 UTC - anomaly detected in workspace search latency
- 2026-08-17 02:31 UTC - incident declared
- 2026-08-17 04:05 UTC - mitigated

## Root cause

Stale vector index segment on the primary collection after partial reindex.
