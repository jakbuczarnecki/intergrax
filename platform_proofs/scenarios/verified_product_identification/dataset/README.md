# WDC dataset builder

Offline, streaming builder for `verified_product_identification`.

## Dependency

Install the scenario-local tooling group once:

```bash
uv sync --group platform-proofs-vpi-dataset
```

`pyarrow` is declared only in that dependency group, not in the core Intergrax runtime.

## Build

```bash
uv run --group platform-proofs-vpi-dataset python platform_proofs/scenarios/verified_product_identification/dataset/build_wdc_dataset.py \
  --input <path-to-nonnormalized_offersV2> \
  --output <path-to-output.parquet>
```

Optional manifest path:

```bash
  --manifest <path-to-manifest.json>
```

## Selection rule

Keep every source record where `keyValuePairs != null` OR `specTableContent != null`.

## Output

- ZSTD-compressed Parquet with one column: `record_json` (lossless UTF-8 JSON per selected record).
- Small JSON manifest with build stats and output checksum.
