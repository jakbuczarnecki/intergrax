# VPI Full Data Pack Validation Runbook

Operator surface for **5C4G** — full production validation of the canonical VPI Data Pack v1 artifact (3,770,377 records) using the existing canonical validator.

> **Read-only gate.** 5C4G validates portable Data Pack integrity. It does **not** load storage, call embedding models, or repair artifacts.

## Workflow gate

```text
5C4F artifact build → READY
        ↓
5C4G full production validation → PASS  (this runbook)
        ↓
freeze content_identity
        ↓
5C5E full storage load (see VPI_FULL_STORAGE_LOAD_RUNBOOK.md)
```

**5C4G may start ONLY after the operator independently confirms 5C4F has completed.**

Required 5C4F completion evidence (do not infer from this runbook — verify operationally):

- builder process finished normally
- canonical build reports final `READY`
- all planned shards completed (3,771 relational + 3,771 embedding)
- manifest finalized
- no writer process remains active

Expected canonical values (code-owned by `canonical_v1_validation_expectations()` — do not pass manually):

| Field | Value |
|-------|-------|
| `record_count` | 3,770,377 |
| `shard_size` | 1,000 |
| `shard_count` | 3,771 |
| final shard record count | 377 |
| `embedding_dimension` | 1,024 |

---

## 1. Purpose

Run the **already implemented** full Data Pack validator against the finalized canonical artifact at:

`D:\Projekty\intergrax-runtime-artifacts\vpi\canonical-v1`

Produce external operator evidence proving semantic integrity before any full storage load (5C5E).

**5C4G PASS ≠ public redistribution approval.** Technical validation does not remove legal/redistribution gates.

---

## 2. Hard prerequisites

1. **5C4F complete** — artifact status `READY`, all shards written, manifest finalized, no active builder/writer.
2. **No concurrent 5C4G** — exactly one validator process against `canonical-v1`.
3. **No concurrent 5C4F** — do not validate while build is still writing.
4. **External scratch** — dedicated per-run directory outside the artifact (see §5).
5. **External report root** — dedicated per-run directory outside the artifact (see §6).
6. **Validator code** — record Git SHA of the repository used for the run.
7. **Resource isolation** — no GPU, embedding model, PostgreSQL, Qdrant, pgvector, storage load, or concurrent benchmarks (see §9–10).

---

## 3. Artifact identity

| Property | Production value |
|----------|------------------|
| Artifact root | `D:\Projekty\intergrax-runtime-artifacts\vpi\canonical-v1` |
| Data Pack version | v1 (canonical) |
| Record count | 3,770,377 |
| Shard layout | 3,771 shards × 1,000 (final shard: 377) |
| Embedding dimension | 1,024 |

Canonical expectations are owned by `canonical_v1_validation_expectations()` in `dataset/data_pack/validation/plan.py`. The operator does **not** pass record count, shard count, shard size, or embedding dimension on the CLI.

---

## 4. Execution environment

- **Host:** operator workstation with read access to runtime artifacts.
- **Python:** repository `uv` environment (Python 3.12).
- **Process model:** single validator process — no external shard parallelization, no secondary checksum tooling, no antivirus/hash scans against the artifact during validation.
- **Forbidden services:** PostgreSQL, Qdrant, pgvector, embedding service, CUDA, HuggingFace, sentence-transformers, torch.
- **Embedding model calls:** 0 (vectors already materialized in the Data Pack).
- **Concurrency:** do not set high BLAS/OpenMP thread counts; avoid multiprocessing or arbitrary thread pools beyond what the validator already uses internally.

---

## 5. Scratch location

Global duplicate detection uses bounded-memory scratch partitions. Scratch **must** be outside `canonical-v1`.

**Production convention:**

```text
D:\Projekty\intergrax-runtime-artifacts\vpi\validation-scratch\5c4g-<run-id>
```

Requirements:

- dedicated per-run directory (empty or new)
- sufficient free disk (see §12 — evidence-based minimum, not guessed)
- not shared with 5C4F
- not Git-tracked
- not inside the Data Pack
- disposable after successful validation

**Default production:** do **not** use `--keep-scratch`.

Use `--keep-scratch` only for deliberate forensic investigation after a failure.

---

## 6. Evidence / report location

Reports **must** be outside the canonical Data Pack.

**Production convention:**

```text
D:\Projekty\intergrax-runtime-artifacts\vpi\operator-evidence\5c4g\<run-id>
```

The canonical CLI creates (when `--report-root` is set):

| File | Role |
|------|------|
| `full-data-pack-validation-report.json` | Machine-readable source of truth for downstream gates |
| `FULL_DATA_PACK_VALIDATION_REPORT.md` | Human review evidence |
| `validation.log` | Execution summary |

**Recommended operator metadata** (capture manually alongside the report):

| File | Content |
|------|---------|
| `command.txt` | Exact invocation used |
| `git-sha.txt` | `git rev-parse HEAD` at run start |
| `started-at.txt` | UTC timestamp before launch |
| `finished-at.txt` | UTC timestamp after exit |
| `exit-code.txt` | Process exit code |
| `content_identity.txt` | Frozen value extracted after PASS (see §16) |

Git SHA is **not** embedded in `FullDataPackValidationReport` by design — store it in operator evidence.

---

## 7. Pre-run checklist

Complete **after** 5C4F READY confirmation, **without** reading the entire artifact:

- [ ] Sufficient free disk for external scratch root
- [ ] Sufficient free disk for external report root
- [ ] No active 5C4F writer process
- [ ] No other 5C4G validator process running
- [ ] Git SHA recorded (`git rev-parse HEAD`)
- [ ] Target artifact root frozen (`canonical-v1` — no concurrent writers)
- [ ] External scratch root empty or newly created
- [ ] External report root empty or newly created
- [ ] Single validator process planned (no parallel validation)
- [ ] No GPU / model / PostgreSQL / Qdrant / storage load / benchmark running

**Optional lightweight read-only preflight** (after 5C4F READY only — prefer validator precondition logic):

- artifact root directory exists
- `manifest.json` exists and reports `READY`
- `build-state.json` exists

Do **not** enumerate shards, calculate checksums, or run recursive filesystem scans as a preflight.

---

## 8. Canonical execution command

**Module entrypoint** (verified against repository packaging):

```powershell
python -m platform_proofs.scenarios.verified_product_identification.dataset.run_data_pack_validation `
  --artifact-root "D:\Projekty\intergrax-runtime-artifacts\vpi\canonical-v1" `
  --scratch-root "<SCRATCH_ROOT>" `
  --report-root "<REPORT_ROOT>"
```

**Example with production path conventions:**

```powershell
$RunId = Get-Date -Format "yyyyMMdd-HHmmss"
$ScratchRoot = "D:\Projekty\intergrax-runtime-artifacts\vpi\validation-scratch\5c4g-$RunId"
$ReportRoot  = "D:\Projekty\intergrax-runtime-artifacts\vpi\operator-evidence\5c4g\$RunId"

git rev-parse HEAD | Out-File -Encoding utf8 "$ReportRoot\git-sha.txt"
Get-Date -Format o | Out-File -Encoding utf8 "$ReportRoot\started-at.txt"
@"
python -m platform_proofs.scenarios.verified_product_identification.dataset.run_data_pack_validation --artifact-root D:\Projekty\intergrax-runtime-artifacts\vpi\canonical-v1 --scratch-root $ScratchRoot --report-root $ReportRoot
"@ | Out-File -Encoding utf8 "$ReportRoot\command.txt"

python -m platform_proofs.scenarios.verified_product_identification.dataset.run_data_pack_validation `
  --artifact-root "D:\Projekty\intergrax-runtime-artifacts\vpi\canonical-v1" `
  --scratch-root $ScratchRoot `
  --report-root $ReportRoot

$ExitCode = $LASTEXITCODE
$ExitCode | Out-File -Encoding utf8 "$ReportRoot\exit-code.txt"
Get-Date -Format o | Out-File -Encoding utf8 "$ReportRoot\finished-at.txt"
```

**Do not** add `--keep-scratch` on the normal production PASS path.

---

## 9. Resource expectations

- **One validator process** — shard-streaming, bounded-memory design.
- **No GPU, no embedding model, no database providers.**
- **No concurrent benchmarks** or storage load.
- **No explicit high process/thread fan-out** — do not wrap with multiprocessing or external parallel shard workers.
- **FULL VALIDATION DURATION: UNKNOWN UNTIL FIRST PRODUCTION RUN** — no ETA without measured evidence from a completed 3.77M run.

Scratch free space must exceed an **evidence-based minimum** determined from implementation/qualification (duplicate partition writer uses bounded partitions under `--scratch-root`). Do not guess fixed GB requirements without measurement.

---

## 10. Monitoring

During execution:

- confirm validator process is alive
- monitor host CPU/RAM — avoid severe memory pressure
- monitor free disk on scratch volume
- watch `validation.log` / stdout for phase progress (`validation progress phase=...` log lines)
- treat report files as **non-authoritative until the process exits**

Do **not**:

- poll canonical artifact files continuously
- run secondary checksum or hash tooling against the artifact
- launch a second validator or validation phase in parallel

---

## 11. Exit codes

| Code | Meaning |
|------|---------|
| `0` | **PASS** — all validation phases passed |
| `1` | **VALIDATION FAIL** — semantic/data integrity failure |
| `2` | **PRECONDITION FAIL** — cannot start (missing artifact, unreadable structure, invalid invocation/scratch setup) |

**Distinction:** exit `2` is **not** a Data Pack semantic FAIL. Exit `1` means validation ran and found integrity problems.

---

## 12. PASS criteria

5C4G PASS requires **all** of the following simultaneously:

| Check | Requirement |
|-------|-------------|
| CLI exit code | `0` |
| Report verdict | `PASS` |
| `summary.finalized_artifact_valid` | `true` |
| Expected records | 3,770,377 |
| Observed relational | 3,770,377 |
| Observed embeddings | 3,770,377 |
| Relational shards | 3,771 |
| Embedding shards | 3,771 |
| Ready shards | 3,771 |
| `first_invalid_shard_ordinal` | `None` |
| Embedding dimension | 1,024 |
| Non-finite vectors | 0 |
| Zero vectors | 0 |
| Semantic hash mismatches | 0 |
| Duplicate `global_row_index` | 0 |
| Duplicate `source_ref` | 0 |
| Duplicate `logical_point_id` | 0 |
| Coverage phase | PASS |
| Checksums phase | PASS |
| Build-state phase | PASS |
| Finalization phase | PASS |
| Every validation check | PASS |

No "mostly PASS". Warnings do not substitute for FAIL verdict.

---

## 13. FAIL handling

For **every** semantic validation FAIL → **FULL STORAGE LOAD IS BLOCKED.**

| Failure category | Operator action |
|------------------|-----------------|
| `STRUCTURE_FAIL` | Stop. Do not load. Inspect artifact layout. |
| `MANIFEST_FAIL` | Stop. Inspect builder/finalization evidence. |
| `SHARD_INDEX_FAIL` | Stop. No manual repair. |
| `SHARD_PAIR_FAIL` | Stop. |
| `RELATIONAL_FAIL` | Stop. |
| `EMBEDDING_FAIL` | Stop. |
| `CROSS_IDENTITY_FAIL` | Stop. |
| `COVERAGE_FAIL` | Stop. |
| `DUPLICATE_FAIL` | Stop. |
| `INTEGRITY_FAIL` | Stop. |
| `BUILD_STATE_FAIL` | Stop. |
| `FINALIZATION_FAIL` | Stop. |

**Fail-closed rule:** never repair `canonical-v1` in place after validation failure.

Preserve: canonical artifact, 5C4G report root, `validation.log`, optional scratch (forensic rerun with `--keep-scratch`). Open a **separate correction/rebuild task**.

Do **not**: edit manifest, rewrite checksums, remove duplicates, regenerate vectors, replace Parquet shards, or mark `READY` manually.

---

## 14. Interruption / restart procedure

5C4G is **not** a resumable build. There are no checkpoints or resume semantics.

```text
Ctrl+C / process termination
        ↓
canonical artifact remains untouched (read-only validation)
        ↓
discard incomplete report as non-authoritative
        ↓
clean or create new external scratch root
        ↓
create new report run directory
        ↓
rerun validation from beginning
```

Do not reuse partial PASS evidence.

If failure is due to scratch/disk problems (not semantic corruption): fix the operational issue and start a **new** 5C4G run. The canonical artifact remains untouched.

---

## 15. Artifact immutability rules

5C4G is **read-only** with respect to the canonical artifact.

**Prohibited during and after validation:**

- scratch under `artifact_root`
- report files under `artifact_root`
- temporary duplicate partitions under `artifact_root`
- manual manifest edits
- build-state edits
- checksum regeneration
- shard repair
- renaming or deleting files
- intentional timestamp touches

5C4G validates. It does not repair.

---

## 16. Post-PASS handoff to 5C5E

After 5C4G PASS:

1. Confirm exit code `0` and report verdict `PASS`.
2. Extract `content_identity` from the validated manifest (`manifest.json` → `content_identity` field). This is the exact pin for 5C5E.
3. Record `content_identity` in operator evidence (`content_identity.txt`).
4. Preserve report JSON, Markdown, log, and execution metadata permanently.
5. Delete scratch after evidence is safely preserved (unless forensic retention required).

```text
5C4G PASS
    ↓
freeze content_identity (from manifest)
    ↓
5C5E PLAN (preflight)
    ↓
5C5E FRESH (full storage load)
```

See [`VPI_FULL_STORAGE_LOAD_RUNBOOK.md`](VPI_FULL_STORAGE_LOAD_RUNBOOK.md) for 5C5E commands.

### 5C4G → 5C5E handoff checklist

Only when **all** items are satisfied → **FULL STORAGE LOAD GATE = OPEN**

- [ ] 5C4G verdict: `PASS`
- [ ] Artifact: `canonical-v1`
- [ ] `record_count`: 3,770,377
- [ ] `content_identity`: `<frozen exact value from manifest>`
- [ ] Report JSON: `<path to full-data-pack-validation-report.json>`
- [ ] Report Markdown: `<path to FULL_DATA_PACK_VALIDATION_REPORT.md>`
- [ ] Validator Git SHA: `<sha from git-sha.txt>`
- [ ] 5C4G exit code: `0`

5C5E FRESH requires `--expected-content-identity` matching the frozen value exactly.

---

## 17. Validation phases (execution order)

The canonical validator owns this pipeline. Do not run phases separately or in parallel.

| # | Phase | Purpose |
|---|-------|---------|
| 1 | `ARTIFACT_STRUCTURE` | Required files and layout |
| 2 | `MANIFEST_IDENTITY` | Manifest fields vs canonical expectations |
| 3 | `SHARD_INDEX` | Shard index contract and counts |
| 4 | `RELATIONAL_SHARD` | Per-shard relational record validation |
| 5 | `EMBEDDING_SHARD` | Per-shard embedding record validation |
| 6 | `CROSS_ARTIFACT_IDENTITY` | Relational ↔ embedding identity per shard |
| 7 | `GLOBAL_COVERAGE` | Global row index continuity and totals |
| 8 | `GLOBAL_DUPLICATES` | Duplicate detection via external scratch |
| 9 | `CHECKSUMS` | File checksum integrity |
| 10 | `BUILD_STATE` | Build state consistency with manifest/index |
| 11 | `FINALIZATION` | READY semantics and artifact validity |

---

## 18. Cleanup

### After PASS

Preserve permanently: report JSON, report Markdown, `validation.log`, operator execution metadata, `content_identity.txt`.

Delete scratch unless legal/forensic retention applies. Canonical artifact: unchanged.

### After FAIL

Preserve: report root, `validation.log`, optionally scratch (forensic rerun with `--keep-scratch`).

Do not delete canonical artifact. Do **not** run 5C5E.

---

## 19. Final operator checklist

Before starting production 5C4G:

- [ ] 5C4F independently confirmed READY
- [ ] No concurrent 5C4F or 5C4G processes
- [ ] External scratch and report roots prepared
- [ ] Git SHA captured
- [ ] Canonical command ready (no `--keep-scratch`)
- [ ] No GPU / model / DB / storage load running

After production 5C4G:

- [ ] Exit code recorded
- [ ] Report JSON + Markdown + log preserved
- [ ] If PASS: `content_identity` frozen for 5C5E
- [ ] If PASS: scratch cleaned (unless forensic hold)
- [ ] If FAIL: storage load blocked; correction task opened
- [ ] Canonical artifact untouched

---

## Canonical validator reference

| Item | Location |
|------|----------|
| CLI | `dataset/run_data_pack_validation.py` |
| Service | `dataset/data_pack/validation/service.py` |
| Expectations | `dataset/data_pack/validation/plan.py` → `canonical_v1_validation_expectations()` |
| Phases / contracts | `dataset/data_pack/validation/contracts.py` |
| Report serialization | `dataset/data_pack/validation/report.py` |
