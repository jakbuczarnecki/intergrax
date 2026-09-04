# VPI Embedding Materialization — Performance Baseline (5C1)

Evidence captured during VPI-IMPLEMENTATION-5C1 on development branch.

## Pre-change synchronous path (derive + embed only)

Bounded measurement on real WDC parquet (`selected_offers.parquet`), **64 records**.

Rationale for 64 (not 256): BGE-M3 CPU embedding for 64 records already required **~17 minutes**; larger batches would not change the bottleneck conclusion.

Pipeline measured without PostgreSQL/Qdrant:

| Stage | Seconds | % of total |
|-------|---------|------------|
| Dataset read | 0.115 | 0.0% |
| Derive | 0.029 | 0.0% |
| Gate 0 probe | 6.299 | (model load + probe) |
| Embedding batch | 1027.336 | **100.0%** |
| **Total (read+derive+embed)** | **1027.481** | |

- **Records:** 64
- **Semantic text avg / p95 / max:** 1199.1 / 2453 / 11444 chars
- **Embedding throughput:** 0.06 records/sec (BGE-M3 CPU)
- **Embedding share of measured wall time:** **100.0%** (read+derive negligible vs embed)

Raw capture: `.tmp/session/vpi-5c1-baseline/pre-change-baseline.log`

## Post-change materialization path

Bounded artifact materialization:

```bash
uv run python platform_proofs/scenarios/verified_product_identification/materialize_embeddings.py \
  --max-records 64 \
  --artifact-dir .tmp/session/vpi-5c1-baseline/artifact-64
```

Raw capture: `.tmp/session/vpi-5c1-baseline/post-change-materialization.log`

Restart proof: re-run same command after READY — `embed_calls=0` on second run.

## Conclusion

CPU embedding dominates wall time (>99.9%). Separating embedding into a restartable Parquet artifact is required before full-scale storage bootstrap. Synchronous bootstrap embed path remains for qualification until 5C2.
