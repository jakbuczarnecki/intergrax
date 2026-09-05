# VPI Embedding Model Arena (5C4B)

Scenario-owned qualification arena comparing embedding model **quality + throughput + VRAM + artifact size + license + reproducibility** without changing the canonical VPI default (`BAAI/bge-m3`).

## Purpose

Answer:

> Can we materially reduce the ~99h BGE-M3 full-build estimate without materially damaging VPI dense retrieval quality?

This is an **arena**, not a migration. Canonical VPI embedding configuration remains unchanged until a separate operator decision.

## Versions

| Artifact | Version |
|---|---|
| Arena | `5c4b-v1` |
| Sample | `arena-sample-v1` |
| Query benchmark | `arena-query-benchmark-v1` |

## Baseline

| Field | Value |
|---|---|
| Provider | `hf` |
| Model | `BAAI/bge-m3` |
| Dimension | `1024` |
| Known batch (5C4A2) | `16` |
| Known throughput | ~`10.2–10.6` rec/s |

## Candidates

Mandatory:

- `bge-m3` — baseline
- `qwen3-0.6b` — primary challenger (`Qwen/Qwen3-Embedding-0.6B`)
- `nomic-v2-moe` — efficiency challenger (`nomic-ai/nomic-embed-text-v2-moe`)

Optional control:

- `e5-large-instruct` (`intfloat/multilingual-e5-large-instruct`) via `--include-e5-control`

## Architecture

Arena core is provider-neutral:

```text
Arena definition → Candidate spec → EmbeddingExecutionPort → HF provider → model
```

Model-specific behavior is expressed only through typed candidate configuration and versioned input policies at the composition boundary (`arena/composition/`).

Arena core imports no `torch`, `sentence_transformers`, `transformers`, HF providers, or Qdrant SDK.

## Canonical vs model input

- `semantic_text` remains canonical and unchanged.
- Query/document prefixes and instructions are applied only at embedding execution boundary.
- Each candidate records `INPUT_POLICY_VERSION`.

## Sample methodology

`ARENA_SAMPLE_V1` (`arena/sampling/arena_sample.py`):

- deterministic scan of WDC rows (default first `50_000`)
- strata quotas across identifiers, brand, description length, structured fields, cluster shape, title strength
- fixed ordering via seeded SHA-256 rank
- shared sample for all candidates

Stages:

| Stage | Records | Purpose |
|---|---:|---|
| A | 100 | load/OOM/dimension/initial throughput |
| B | 500 | stable throughput + preliminary retrieval quality |
| C | 1000 | finalists only — final throughput, quality, projections |

## Query benchmark

Deterministic query cases (`arena/evaluation/query_builder.py`) from WDC evidence:

- `STRONG_IDENTITY`
- `TITLE_BRAND`
- `TITLE_ONLY`
- `STRUCTURED_ATTRIBUTES`
- `PARTIAL_NOISY`
- `LONG_DESCRIPTION_SIGNAL`

`cluster_id` is benchmark-only evidence and never appears in query text.

Metrics: Recall@1/5/10, MRR@10, nDCG@10 on in-memory cosine search over the arena corpus.

## Throughput methodology

Reuses 5C4A microbenchmark infrastructure (`qualification/integration/microbenchmark.py`).

Batch candidates: `8, 16, 32, 64` (baseline reuses known batch `16`).

## Truncation analysis

Tokenizer-based profile per candidate (`arena/integration/truncation_probe.py`).

512-token models report truncation separately; long-input quality subset is mandatory when truncation occurs.

## License / runtime notes

License metadata is recorded per candidate from model card references. Non-commercial licenses are blocked before large runs.

`trust_remote_code` operational cost is recorded (Nomic candidate).

## Run

```powershell
uv run python platform_proofs/scenarios/verified_product_identification/run_embedding_arena.py
```

Options:

- `--include-e5-control`
- `--skip-gpu-stages` (sample/query evidence only)
- `--session-dir .tmp/session/vpi-5c4b`

Outputs:

- `.tmp/session/vpi-5c4b/arena-report.json`
- `.tmp/session/vpi-5c4b/ARENA_SUMMARY.md`

## Winner recommendation

Typed decision enum (`KEEP_BGE_M3`, `PROMOTE_*`, `NO_CLEAR_WINNER`, `MORE_EVIDENCE_REQUIRED`).

Dominance gates — not a weighted score:

1. correctness
2. license
3. quality non-regression vs baseline
4. operational complexity
5. throughput
6. VRAM / artifact size

## 5C4C handoff

Maximum `1–2` finalists proceed to 10k/100k scaling qualification.

## Limitations

- Full-build and artifact projections are `PRELIMINARY 1K LINEAR ESTIMATE`.
- Dense-only evaluation; hybrid fusion impact is out of scope unless existing infrastructure is reused later.
- Resolved HF revision pinning is reported as a gap when unavailable from provider surface.

## Tests

```powershell
uv run pytest tests/unit/platform_proofs/scenarios/verified_product_identification/test_embedding_arena.py -q
```
