# Token Optimization Benchmarks

This directory contains benchmark datasets for token optimization regression and safety checks.

The primary rule for this directory is human usability: every test case must live in its own folder. To add a new benchmark case, copy an existing case folder, edit `fixture.json`, edit the local `input.*` file, and run the benchmark. Do not add new cases by editing a shared global manifest.

## Current datasets

```text
fixtures/
  regression_synthetic_v1/
    dataset.json
    cases/
      tool_schema/
        compact_catalog/
          fixture.json
          input.json
        protected_description/
          fixture.json
          input.json
      context_pack/
        compact_fragments/
          fixture.json
          input.json
        protected_evidence/
          fixture.json
          input.json
      memory_summary/
        compact_summary/
          fixture.json
          input.txt
        protected_dates/
          fixture.json
          input.txt
        fallback_validation/
          fixture.json
          input.txt
```

## How to add a new fixture

1. Pick the source category:

```text
tool_schema
context_pack
memory_summary
```

2. Copy the closest existing case folder.

Example:

```text
cp -R fixtures/regression_synthetic_v1/cases/memory_summary/compact_summary \
      fixtures/regression_synthetic_v1/cases/memory_summary/compact_long_summary
```

3. Edit `fixture.json` inside the new case folder.

Update at least:

```json
{
  "fixture_id": "memory_summary.compact_long_summary",
  "eval_case": "compactable",
  "expected_behavior": "applies_structural_compaction_with_receipt"
}
```

4. Edit the local input file in the same folder.

Use:

```text
input.json for structured JSON payloads
input.txt  for plain text payloads
```

5. Keep the expectation inside the same `fixture.json`.

Do not create or edit a global expectation file. A fixture must be understandable by opening only its own folder.

6. Run the benchmark loader command once it is implemented.

Expected future command shape:

```powershell
uv run python scripts/check_token_regression_benchmarks.py --report --fixture-dataset benchmarks/token_optimization/fixtures/regression_synthetic_v1
uv run python scripts/check_token_regression_benchmarks.py --gate --fixture-dataset benchmarks/token_optimization/fixtures/regression_synthetic_v1
```

## Fixture file contract

Each case folder contains:

```text
fixture.json
input.json | input.txt
```

`fixture.json` must describe everything needed to run and evaluate the case:

```json
{
  "schema_version": 1,
  "fixture_id": "memory_summary.compact_summary",
  "source_type": "memory_summary",
  "eval_case": "compactable",
  "category": "memory",
  "expected_behavior": "applies_structural_compaction_with_receipt",
  "input": {
    "file": "input.txt",
    "format": "plain_text"
  },
  "optimizer": {
    "kind": "memory_summary",
    "config": {}
  },
  "protected_values": [],
  "expected": {
    "min_saved_ratio": 0.0,
    "receipt": "required",
    "validation": "pass_like",
    "fallback": "forbidden"
  }
}
```

## Fixture classes

```text
compactable — input should be safely compacted and save tokens when the optimizer can do so.
protected   — protected values such as URLs, evidence references, and dates must not be damaged; zero savings can be correct.
fallback    — unsafe lossy compression should fail validation and fall back to the original content.
```

## Data safety

Benchmark fixtures must be synthetic or redacted. Do not add private user data, production documents, credentials, API keys, raw customer content, or proprietary evidence payloads.

The goal is realistic shape, not real private data.
