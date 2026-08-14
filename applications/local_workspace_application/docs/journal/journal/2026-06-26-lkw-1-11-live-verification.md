# 2026-06-26 — LKW.1.11 live verification blocker

## Summary

LKW.1.11 fixed the registry parity implementation path in unit scope, but manual live HTTP verification did not pass.

Reported implementation commit:

```text
47b8667e48fb834829bcb321b37367789e62e896
```

Focused tests passed:

```text
tests/unit/applications/test_application_tool_registry_runtime_parity.py -> 1 passed
tests/unit/tools/providers/rag/test_rag_scope.py -> 10 passed
```

Docker/live checks passed for host readiness:

```text
Docker stack built and started
/health -> ok
/agents -> local_indexer, local_search, local_synthesizer
fixture visible inside container at /data/user_docs/lkw-live-smoke.txt
```

Live index still failed to ingest:

```text
accepted=1
rejected=0
ingested=0
chunks=0
total_tool_calls=0
```

No new Qdrant live smoke collection appeared for the `lkw-registryfix-*` collection id.

## New platform blocker

The live container log showed:

```text
RuntimeEventSchemaError: phase mismatch for decision_emitted: expected step_execution, got planning
```

This is now the active blocker before LKW.1 can close.

## Classification

```text
Platform-reusable
```

The issue is in runtime event schema/emission semantics, not in LKW product logic. Future Tier-3 applications using the same runtime event bus, task trace, and validating event store can hit the same failure mode.

## Queue update

```text
LKW.1.11 — implementation/unit scope passed; live not passed
LKW.1.12 — fix decision_emitted phase mismatch
LKW.1.13 — final live index/search/synthesize smoke
LKW-H1 — local trace/evidence inspection after live execution blockers
```

## LKW.1.12 scope guardrails

Do not broaden into hosted observability or full H1. The next task should be a small runtime event contract fix:

```text
Fix the phase contract for decision_emitted so emitted event phase and schema expectation match.
```

Acceptance:

```text
- focused regression covers decision_emitted phase validity
- live index no longer logs the phase mismatch
- live index reaches RAG tool path or exposes a new non-event-schema blocker
- no Qdrant point-id changes
- no tenant scope changes
- no broad runtime refactor
```
