# WDC dataset builder

Offline, streaming builder for `verified_product_identification`.

## Layout

```text
dataset/
  raw/         # local WDC source NDJSON (gitignored contents)
  processed/   # generated Parquet + manifest (gitignored contents)
```

## Dependency

Install the scenario-local tooling group once:

```bash
uv sync --group platform-proofs-vpi-dataset
```

`pyarrow` is declared only in that dependency group, not in the core Intergrax runtime.

## Build

Place source file at `raw/nonnormalized_offersV2`, then:

```bash
uv run --group platform-proofs-vpi-dataset python platform_proofs/scenarios/verified_product_identification/dataset/build_wdc_dataset.py
```

Defaults:

- input: `dataset/raw/nonnormalized_offersV2`
- output: `dataset/processed/selected_offers.parquet`
- manifest: `dataset/processed/selected_offers_manifest.json`

Override paths explicitly when needed:

```bash
uv run --group platform-proofs-vpi-dataset python platform_proofs/scenarios/verified_product_identification/dataset/build_wdc_dataset.py \
  --input <path-to-nonnormalized_offersV2> \
  --output <path-to-output.parquet> \
  --manifest <path-to-manifest.json>
```

## Selection rule

Keep every source record where `keyValuePairs != null` OR `specTableContent != null`.

## Output

- ZSTD-compressed Parquet with one column: `record_json` (lossless UTF-8 JSON per selected record).
- Small JSON manifest with build stats and output checksum.

## Sample

Create a smaller Parquet subset (default: 1 000 records) from `processed/selected_offers.parquet`:

```bash
uv run --group platform-proofs-vpi-dataset python platform_proofs/scenarios/verified_product_identification/dataset/sample_wdc_dataset.py
```

Defaults:

- input: `dataset/processed/selected_offers.parquet`
- output: `dataset/processed/selected_offers_sample_1000.parquet`
- manifest: `dataset/processed/selected_offers_sample_1000_manifest.json`
- sample size: `1000`
- random seed: `42`

Override when needed:

```bash
uv run --group platform-proofs-vpi-dataset python platform_proofs/scenarios/verified_product_identification/dataset/sample_wdc_dataset.py \
  --input <path-to-selected_offers.parquet> \
  --output <path-to-sample.parquet> \
  --size 1000 \
  --seed 42
```

Sampling uses single-pass reservoir sampling over `record_json` rows (bounded memory).
