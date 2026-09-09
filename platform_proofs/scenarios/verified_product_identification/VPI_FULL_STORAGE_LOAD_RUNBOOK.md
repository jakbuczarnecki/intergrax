# VPI Full Storage Load Runbook

Operator surface for loading a **READY** canonical VPI Data Pack into PostgreSQL + Qdrant using the 5C5D storage bootstrap core.

> **Not the legacy bootstrap path.** [`bootstrap.py`](bootstrap.py) (`BootstrapRunMode`, manifest orchestrator) is a separate, older embedding-artifact bootstrap. Full Data Pack load uses the operator module below.

## Workflow gate (external)

```text
5C4F artifact build → READY
        ↓
5C4G full production validation → PASS
        ↓
operator FULL LOAD (this runbook)
```

The operator does **not** implement 5C4G. It refuses non-READY artifacts and optional expected identity pins.

## Preconditions

1. **5C4F** canonical Data Pack status == `READY`
2. **5C4G** full production validation == `PASS` (external gate)
3. Runtime provider qualification == `PASS` for PostgreSQL + Qdrant
4. Provider credentials configured via environment (never CLI):
   - `INTERGRAX_POSTGRESQL_*` (or `INTERGRAX_POSTGRESQL_DSN`)
   - `INTERGRAX_QDRANT_*` (or `INTERGRAX_QDRANT_URL`)
   - Optional schema: `INTERGRAX_POSTGRESQL_SCHEMA` (default `vpi`)
5. External checkpoint and evidence directories (writable, outside the artifact)
6. Frozen operator parameters for the run:
   - `--batch-size` (checkpoint compatibility — **must not change on RESUME**)
   - `--expected-record-count 3770377` (recommended)
   - `--expected-content-identity <5C4G-confirmed value>` (set after 5C4G)

## Canonical operator module

```bash
python -m platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator
```

## PLAN (safe preflight)

Performs manifest/readiness checks, target resolution, provider configuration presence, and `BootstrapPlan` computation.

**Does not:** iterate all rows, write PostgreSQL/Qdrant, or initialize a durable load checkpoint.

```bash
python -m platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator \
  --artifact-root "<CANONICAL DATA PACK ROOT>" \
  --checkpoint-root "<EXTERNAL CHECKPOINT PATH>" \
  --evidence-root "<EXTERNAL EVIDENCE PATH>" \
  --relational-target "vpi-products" \
  --vector-target "vpi-product-embeddings" \
  --batch-size "<FROZEN VALUE>" \
  --verification strict \
  --plan
```

Exit `0` = plan succeeded.

## FRESH (new logical load)

Fails closed if a checkpoint already exists for the same run identity.

```bash
python -m platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator \
  --artifact-root "<CANONICAL DATA PACK ROOT>" \
  --checkpoint-root "<EXTERNAL CHECKPOINT PATH>" \
  --evidence-root "<EXTERNAL EVIDENCE PATH>" \
  --relational-target "vpi-products" \
  --vector-target "vpi-product-embeddings" \
  --expected-record-count 3770377 \
  --expected-content-identity "<SET AFTER 5C4G>" \
  --batch-size "<FROZEN VALUE>" \
  --verification strict \
  --fresh
```

## Monitor progress

Evidence (outside artifact):

| File | Purpose |
|------|---------|
| `run.json` | Immutable run metadata (no secrets) |
| `progress.jsonl` | Bounded per-batch progress |
| `final-report.json` | Typed final summary |

Progress is emitted per committed batch (phase, batch number, records processed, ETA).

## Interrupt safely

- Use `Ctrl+C` or `SIGTERM`.
- Last **committed** logical batch remains checkpointed.
- Incomplete batch is replayable on resume.
- Process exits with code `3` (`INTERRUPTED`).
- **Do not** delete checkpoint or operator lock manually unless ownership is verified.

## RESUME

Uses existing 5C5D checkpoint semantics unchanged. Operator does not compute its own offset.

```bash
python -m platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator \
  --artifact-root "<SAME AS FRESH>" \
  --checkpoint-root "<SAME AS FRESH>" \
  --evidence-root "<NEW OR SAME EVIDENCE PATH>" \
  --relational-target "vpi-products" \
  --vector-target "vpi-product-embeddings" \
  --expected-record-count 3770377 \
  --expected-content-identity "<SAME AS FRESH>" \
  --batch-size "<SAME AS FRESH>" \
  --verification strict \
  --resume
```

Mismatch in batch size, targets, verification mode, or Data Pack identity ⇒ **fail closed**.

## What NEVER to do

- Do not put credentials in CLI, evidence, checkpoints, or logs.
- Do not delete/truncate PostgreSQL tables or Qdrant collections automatically.
- Do not delete checkpoints automatically on FRESH.
- Do not change `--batch-size` between FRESH and RESUME.
- Do not run two concurrent writers against the same checkpoint root (operator lock + checkpoint CAS).
- Do not run full load before 5C4F READY and 5C4G PASS.

## Final PASS criteria (future full load)

| Check | Requirement |
|-------|-------------|
| Data Pack | `record_count == 3,770,377` |
| Bootstrap result | `SUCCESS` |
| Checkpoint | Complete committed prefix covers full artifact |
| Relational | All expected canonical identities present |
| Vector | All expected logical point identities present |
| Strict verification | PASS, failures = 0 |
| Resume after complete run | 0 provider writes |

## Failure recovery

| Symptom | Action |
|---------|--------|
| Precondition error (exit 2) | Fix config/artifact/provider env; re-run PLAN |
| Load failed (exit 1) | Inspect `final-report.json`; fix root cause; RESUME if checkpoint partial |
| Interrupted (exit 3) | Re-run with `--resume` (same parameters) |
| Operator lock held | Ensure no live process; inspect `checkpoint_root/operator.lock`; do not delete without ownership proof |
| FRESH + existing checkpoint | Use `--resume` or choose a new checkpoint root after explicit operator decision |
| Wrong expected count/identity | Fix pins to match 5C4G-confirmed manifest values |

## Production command template (document only — do not run until gates pass)

```bash
python -m platform_proofs.scenarios.verified_product_identification.storage_bootstrap.operator \
  --artifact-root "D:\Projekty\intergrax-runtime-artifacts\vpi\canonical-v1" \
  --checkpoint-root "<EXTERNAL CHECKPOINT PATH>" \
  --evidence-root "<EXTERNAL EVIDENCE PATH>" \
  --relational-target "vpi-products" \
  --vector-target "vpi-product-embeddings" \
  --expected-record-count 3770377 \
  --expected-content-identity "<SET AFTER 5C4G>" \
  --batch-size "<FROZEN VALUE>" \
  --verification strict \
  --fresh
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Load failed |
| 2 | Operator / precondition error |
| 3 | Interrupted |

## Single-writer protection

`checkpoint_root/operator.lock` provides single-host protection in addition to checkpoint CAS semantics. A second live writer fails immediately. Stale locks are not auto-deleted.
