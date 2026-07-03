# Token Optimization Regression Fixtures v1

This directory contains file-backed regression fixtures for token optimization benchmarks.

The dataset is intentionally synthetic. It mirrors the current hard-coded benchmark cases without using private or production data.

## Dataset

- Dataset id: `regression_synthetic_v1`
- Version: `1.0.0`
- Token counter: `default_word_count`
- Scope: helper-only token optimization regression and safety checks

## Fixture classes

- `compactable` — input should be safely compacted and save tokens when possible.
- `protected` — protected values such as URLs, evidence references, and dates must not be damaged; zero savings can be the correct result.
- `fallback` — unsafe lossy compression should fail validation and fall back to the original content.

## Files

- `manifest.json` describes fixture metadata, optimizer dispatch, input paths, input formats, configs, and protected values.
- `expectations.json` describes pass/fail checks for each fixture.
- `inputs/` contains the raw fixture inputs consumed by the benchmark loader.

This dataset should remain redaction-safe and deterministic. Do not add real user data, production documents, secrets, tokens, or private evidence payloads.
