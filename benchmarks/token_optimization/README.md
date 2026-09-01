# Token Optimization Benchmark Dataset

Synthetic regression and safety fixtures for token optimization benchmarks.

## Overview

1. This dataset supports token optimization regression and safety benchmarks.
2. Each test case lives in its own folder under `fixtures/regression_synthetic_v1/cases/`.
3. To add a new case, copy the closest existing case folder, update `fixture.json`, and edit local text files.
4. Do not add cases via a global manifest - discovery is recursive over `fixture.json` files.
5. Text content must live in `.txt` files; JSON holds structure, configuration, and expectations only.
6. For `tool_schema`, long tool descriptions are in `texts/*.txt`.
7. For `context_pack`, fragments are in `fragments/*.txt`.
8. For `memory_summary`, input text is in `input.txt`.
9. All data must be synthetic or redacted - no private, production, secret, or client payload data.

## Layout

```text
benchmarks/token_optimization/
  README.md
  fixtures/
    regression_synthetic_v1/
      dataset.json
      cases/
        tool_schema/
        context_pack/
        memory_summary/
```

Each case folder contains `fixture.json` plus local input files (`.txt`, `schema.json`, etc.).

## How to add a new fixture

1. Choose category: `tool_schema` / `context_pack` / `memory_summary`.
2. Copy the closest case folder.
3. Update `fixture_id`, `eval_case`, and `expected_behavior`.
4. Edit local text files.
5. Keep expected checks inside `fixture.json`.
6. Run the benchmark loader once implemented.

Example:

```bash
cp -R fixtures/regression_synthetic_v1/cases/memory_summary/compact_summary \
      fixtures/regression_synthetic_v1/cases/memory_summary/compact_long_summary
```

## Fixture contract

Every `fixture.json` must include:

```json
{
  "schema_version": 1,
  "fixture_id": "...",
  "source_type": "...",
  "eval_case": "...",
  "category": "...",
  "expected_behavior": "...",
  "description": "...",
  "input": {},
  "optimizer": {},
  "protected_values": [],
  "expected": {}
}
```

Allowed `source_type`: `tool_schema`, `context_pack`, `memory_summary`.

Allowed `eval_case`: `compactable`, `protected`, `fallback`.

Allowed `expected.validation`: `pass_like`, `failed`.

Allowed `expected.fallback`: `forbidden`, `required`.

Allowed `expected.receipt`: `required`.
