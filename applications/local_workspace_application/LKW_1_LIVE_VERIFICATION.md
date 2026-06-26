# LKW.1 live verification status — 2026-06-26

## Status

```text
LKW.1.11 — implementation/unit scope passed, live HTTP smoke not passed
LKW.1.12 — next: fix RuntimeEventSchemaError for decision_emitted phase mismatch
LKW.1.13 — final live index/search/synthesize smoke after LKW.1.12
LKW-H1 — local trace/evidence inspection after live execution blockers
```

## LKW.1.11 result

Implementation commit reported by operator:

```text
47b8667e48fb834829bcb321b37367789e62e896
```

The LKW.1.11 fix addressed the previous registry parity issue where `ApplicationToolWiring.registry` was built in Tier-3 but the live runtime gateway/invoker path used a different registry.

Changed files reported by the implementation:

```text
intergrax/applications/_shared/catalog_runtime_bridge.py
intergrax/runtime/nexus/config.py
intergrax/runtime/nexus/config_sections.py
intergrax/runtime/nexus/engine/runtime_context.py
tests/unit/applications/test_application_tool_registry_runtime_parity.py
```

Focused tests passed:

```text
uv run pytest tests/unit/applications/test_application_tool_registry_runtime_parity.py -q
-> 1 passed

uv run pytest tests/unit/tools/providers/rag/test_rag_scope.py -q
-> 10 passed
```

## Manual live verification

Docker stack built and started. `/health` returned `{"status":"ok"}`. `/v1/local_workspace/agents` listed:

```text
local_indexer       -> local.workspace.index
local_search        -> local.workspace.search
local_synthesizer   -> local.workspace.synthesize
```

The fixture file was visible inside the container at:

```text
/data/user_docs/lkw-live-smoke.txt
```

Live index request returned completed state, but did not ingest:

```text
accepted=1
rejected=0
ingested=0
chunks=0
total_tool_calls=0
```

Qdrant did not show a new live smoke collection for the `lkw-registryfix-*` collection id.

## New blocker

Container logs showed a new runtime event validation issue:

```text
RuntimeEventSchemaError: phase mismatch for decision_emitted: expected step_execution, got planning
```

This means LKW.1.11 should not be marked as live-passed. The registry parity implementation has a passing focused test and is present in the Docker image, but the live run still does not reach a successful RAG ingest.

## Platform classification

```text
Platform-reusable
```

Reason: runtime event schema phase correctness affects every product application using the same runtime event bus, task trace, and validating event store.

## Next task

```text
LKW.1.12 — Fix RuntimeEventSchemaError for decision_emitted phase mismatch
```

Acceptance:

```text
- Focused regression proves decision_emitted phase matches schema.
- Live index smoke no longer logs the decision_emitted phase mismatch.
- Live index smoke reaches the RAG tool path or exposes a new non-event-schema blocker.
- No Qdrant point-id changes.
- No tenant scope changes.
- No broad LKW-H1/observability implementation.
```

## Final closeout after LKW.1.12

```text
LKW.1.13 — Re-run final live HTTP smoke:
index -> ingested>0, chunks>0
search -> evidence references fixture
synthesize -> shadow artifact only
```
