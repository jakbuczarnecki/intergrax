# VPI Embedding Performance Qualification (5C4A)

Evidence contract for real WDC → derivation → BGE-M3 embedding → Parquet artifact → PostgreSQL + Qdrant bootstrap.

## Reference model

| Field | Value |
|-------|-------|
| Provider | `hf` |
| Model | `BAAI/bge-m3` |
| Dimension | `1024` |
| Derivation | `v2` (unchanged in 5C4A) |
| Embedding configuration version | `v1` |

## Hardware

Measured at qualification time via `qualification/integration/hardware_probe.py` (optional torch imports only).

Report fields: Python, platform, torch build, CUDA availability, GPU name/count/VRAM, SentenceTransformers version, configured vs resolved provider device.

**Do not assume CUDA from reported hardware** — qualification probes the actual runtime torch build.

## Configuration

### Semantic identity (artifact compatibility)

- `VPI_EMBEDDING_PROVIDER` (default `hf`)
- `VPI_EMBEDDING_MODEL` (default `BAAI/bge-m3`)
- `VPI_EMBEDDING_DIMENSION` (default `1024`)

### Execution tuning (not artifact identity)

- `VPI_EMBEDDING_DEVICE` — explicit `cuda` fails closed when unavailable
- `VPI_EMBEDDING_PROVIDER_BATCH_SIZE` — provider-neutral inner batch size preference

Execution config is provider-neutral. Provider-specific constructor mapping lives in Intergrax embedding factory registration. `device` and `batch_size` are execution tuning only and are not part of artifact semantic identity.

`max_length` is not configurable in 5C4A1 because truncation changes semantic embedding output and must be versioned separately before operator tuning is exposed.

Device and batch size may produce non-bit-identical floating-point output across hardware while preserving compatible semantic behavior. Artifact checksum identity belongs to the built artifact itself.

### Diagnostics

Provider execution diagnostics use a typed provider-neutral capability (`execution_snapshot`). Qualification requests diagnostics through the VPI adapter without importing concrete provider classes.

### Materialization batching (outer orchestrator)

- `VPI_EMBEDDING_MATERIALIZATION_BATCH_SIZE` (default `64`) — texts per `embed_batch` call
- Inner provider may sub-batch again via `encode(batch_size=...)`

## Microbenchmark

Bounded real-WDC sample (default 192 records) using current `semantic_text` derivation.

Candidate inner provider batch sizes: `16`, `32`, `64` (optional `128` when VRAM permits).

Selection rule: highest stable throughput among passing candidates, with VRAM headroom preference when throughput is similar.

CUDA OOM candidates are recorded as `FAILED_OOM` without aborting the full qualification run.

## 1K materialization

```bash
uv run python platform_proofs/scenarios/verified_product_identification/run_embedding_qualification.py
```

Or direct materialization:

```bash
uv run python platform_proofs/scenarios/verified_product_identification/materialize_embeddings.py --max-records 1000
```

## Restart

Re-run against the same READY artifact — required `embedding_calls=0`.

## Storage E2E

Requires real `INTERGRAX_POSTGRESQL_*` and `INTERGRAX_QDRANT_*` configuration. Storage bootstrap uses artifact reader only — zero live embedding execution.

If storage providers are unavailable, qualification returns `BLOCKED_STORAGE_ENVIRONMENT` while preserving materialization evidence.

## Storage restart

Second bootstrap run at the same target must hit READY fast path with no duplicate ingest.

## Full-dataset estimate

From steady-state 1K embedding throughput (GPU when CUDA qualified, CPU otherwise):

- `3,770,377` records
- linear scaling for derive and artifact write per-record averages from 1K run
- central estimate only (no invented confidence bands)

## Bottleneck analysis

Uses measured derive / embedding / artifact-write shares from 1K materialization.

If embedding > ~90% of total, defer I/O parallelization to **5C4C**.

## Limitations

- No semantic representation changes in 5C4A
- No FP16 tuning unless supported via public provider APIs (deferred to 5C4C)
- No full 3.77M build in this task
- Model comparison deferred to **5C4B**

## Next decision

After 5C4A evidence:

1. **5C4B** — embedding model arena (BGE-M3 vs challengers)
2. **5C4C** — scaling / parallel pipeline optimization
3. Final GO/NO-GO for full build remains **out of scope** for 5C4A

## Runbook

```bash
New-Item -ItemType Directory -Force -Path .tmp/session/vpi-5c4a
uv run python platform_proofs/scenarios/verified_product_identification/run_embedding_qualification.py 2>&1 | Tee-Object -FilePath .tmp/session/vpi-5c4a/run.log
```

JSON report: `.tmp/session/vpi-5c4a/qualification-report.json`
