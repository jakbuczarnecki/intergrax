# Intergrax Token Optimization — vLLM Prefix-Cache Live Proof

> **See also:** [Token Optimization Engine guide](../README.md) · [Architecture](../../architecture/TOKEN_OPTIMIZATION.md) · [Plan](../../plan/TOKEN_OPTIMIZATION.md)

## Purpose

This guide shows how to start the canonical local vLLM proof environment and verify real prefix-cache reuse through Intergrax.

It is the technical reviewer path for **TOKEN-10C** (vLLM prefix-cache live proof). You do not need to read Intergrax runtime internals to follow it.

As a reviewer, do **not**:

- infer cache reuse from response latency alone;
- parse full Docker container logs during the normal path;
- guess what Prometheus metric names mean without the checks below;
- assume that repeating the same prompt text automatically proves an exact-prefix cache hit.

The runner and generated report provide explicit PASS/FAIL gates and reason codes.

## What this proof demonstrates

When the environment is available and the proof completes, the checked-in runner exercises:

1. connectivity to the pinned vLLM server (`0.23.0`);
2. automatic prefix caching enabled in the canonical Compose command;
3. presence of required prefix-cache Prometheus metrics;
4. real inference through the OpenAI-compatible vLLM endpoint (`/v1/chat/completions`);
5. a deterministic stable prefix assembled by Intergrax proof helpers;
6. a **cold-cache control** (first measured case after startup warmup);
7. a **warm-cache reuse** case with the same stable prefix and a different dynamic tail;
8. a **changed-prefix negative control** with a different stable prefix;
9. prefix-hash and tool-schema-hash recording for send-payload integrity;
10. separation of provider prefix-cache reuse from content-reduction mechanisms;
11. generation of a redaction-safe JSON and Markdown report;
12. explicit hard gates and reason codes for cold, warm, and changed-prefix cases.

**Implementation status:** the runner, evaluation gates, and report serializer are **implemented** and unit-tested.

**Live verification status:** the Windows reviewer path below was **live-verified** with vLLM `0.23.0`, `Qwen/Qwen2.5-3B-Instruct`, and an NVIDIA RTX 4080 Laptop GPU (12 GB VRAM). That verification confirms environment and mechanism behavior; it does **not** certify every GPU, OS, or model profile.

**Not yet live-certified:** full checked-in universal proof packaging (TOKEN-10F/TOKEN-10G) and public promotion (TOKEN-10H).

## What this proof does not demonstrate

This proof does **not** demonstrate:

- universal percentage savings across workloads;
- cost reduction for every LLM provider;
- Claude Prompt Caching behavior;
- OpenAI Prompt Caching behavior;
- production readiness;
- behavior on every model;
- behavior on every GPU;
- semantic equivalence for arbitrary lossy compression;
- guaranteed cache retention over time;
- real-customer workload savings;
- that cached tokens are removed tokens.

Content reduction and provider prefix-cache reuse are separate mechanisms and must be reported separately.

## Current maturity and certification status

| Label | Meaning here |
| --- | --- |
| **Implemented** | Runner (`vllm_prefix_cache_live.py`), evaluation (`vllm_prefix_cache_proof.py`), report writer, unit tests |
| **Live-verified** | Manual Windows path with Docker Desktop / WSL2, RTX 4080 Laptop 12 GB, vLLM `0.23.0`, `Qwen/Qwen2.5-3B-Instruct` |
| **Live-certified** | Not claimed for TOKEN-10C alone |
| **Planned** | Universal TOML harness (TOKEN-10F), checked-in proof corpus and hard gates (TOKEN-10G), README/public claims (TOKEN-10H) |

**Live-verified environment (documented reference, not a universal guarantee):**

- Windows host
- Docker Desktop / WSL2 runtime
- NVIDIA GeForce RTX 4080 Laptop GPU, 12 GB VRAM
- vLLM `0.23.0` (`vllm/vllm-openai:v0.23.0`)
- Model `Qwen/Qwen2.5-3B-Instruct`
- `/health` HTTP `200`
- `/version` returns `0.23.0`
- `/metrics` exposes required prefix-cache metrics
- automatic prefix caching active (`--enable-prefix-caching`)

Do not extend this list to native Linux, macOS, other GPUs, the 7B model, or TOKEN-10G without separate evidence.

The full checked-in universal proof remains tied to **TOKEN-10F** and **TOKEN-10G**.

## Prerequisites

- Docker Desktop or Docker Engine with Docker Compose
- NVIDIA GPU visible to Docker (`nvidia-container-toolkit` on Linux; GPU support enabled in Docker Desktop on Windows)
- Compatible NVIDIA driver
- Python 3.12
- `uv` (repository dependency manager)
- Repository checkout at the repository root
- PowerShell for the currently live-verified Windows path

The canonical live-verified configuration used a **12 GB RTX 4080 Laptop GPU**. VRAM requirements depend on model size; this guide does not state a single universal minimum VRAM for all models.

## Canonical proof model

**Canonical reviewer model:** `Qwen/Qwen2.5-3B-Instruct`

- The proof validates the **prefix-caching mechanism**, not 7B model quality.
- The 3B model is sufficient to exercise cold/warm/changed-prefix gates.
- In the live-verified 12 GB environment, `Qwen/Qwen2.5-7B-Instruct` failed: model weights consumed about **14.29 GiB**, KV cache memory was negative, and vLLM reported `No available memory for the cache blocks`.
- With the 3B model on the same hardware, weights used about **5.79 GiB**, available KV cache memory was about **3.29 GiB**, and GPU KV cache size was about **95,872 tokens** (values vary by device and load).
- `Qwen/Qwen2.5-7B-Instruct` may remain an optional profile for hosts with more VRAM; it is **not** the canonical reviewer default for 12 GB GPUs.

Compose defaults to the canonical 3B model when `VLLM_MODEL` is unset. Reviewers may still set `VLLM_MODEL` explicitly for clarity.

## Reviewer workflow overview

```text
preflight
→ select canonical 3B model
→ start canonical vLLM (manual, observable)
→ verify health and version
→ verify prefix-cache metrics
→ run smoke inference
→ run Intergrax cold/warm/changed-prefix proof
→ inspect generated report
→ stop the environment
```

Use **two terminals** on Windows: Terminal A for vLLM logs, Terminal B for checks and the proof runner.

## Step-by-step Windows reviewer path

All commands assume the **repository root** as the working directory.

### Step 1 — Verify repository tooling

**Goal:** Confirm `uv` can run Python in this checkout.

**Command:**

```powershell
uv run python --version
```

**Expected result:**

```text
Python 3.12.x
```

**Meaning:** The repository environment is usable. Commands in this guide use `uv run`; you do not need a separate virtualenv activation if `uv` is installed.

**Failure signal:** `uv` or Python not found → **ENVIRONMENT_BLOCKED**.

### Step 2 — Verify Docker

**Goal:** Confirm Docker client, server, and Compose are available.

**Commands:**

```powershell
docker version
docker compose version
```

**Expected result:**

- `docker version` shows Client and Server sections.
- `docker compose version` prints a Compose version string.

**Meaning:** Docker lifecycle commands for vLLM can run.

**Failure signal:** Cannot connect to Docker daemon → **ENVIRONMENT_BLOCKED**.

### Step 3 — Verify NVIDIA GPU

**Goal:** Confirm a GPU is visible on the host.

**Command:**

```powershell
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
```

**Expected result (example from live-verified host):**

```text
NVIDIA GeForce RTX 4080 Laptop GPU, 12282 MiB, ...
```

`memory.used` and `memory.free` change with workload; only presence of the GPU row matters here.

**Meaning:** Host GPU is available for Docker GPU reservation.

**Failure signal:** `nvidia-smi` fails or returns no GPU → **ENVIRONMENT_BLOCKED**.

### Step 4 — Select canonical model

**Goal:** Pin the 3B proof model for Compose and the Intergrax runner.

**Commands:**

```powershell
$env:VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
$env:INTERGRAX_DEFAULT_VLLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
$env:INTERGRAX_DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8100/v1"
$env:VLLM_MODEL
```

**Expected result:**

```text
Qwen/Qwen2.5-3B-Instruct
```

**Meaning:** Compose will start the 3B weights; the proof runner will target the same model and base URL.

**Note:** Assigning `$env:VLLM_MODEL` produces no output; that is normal PowerShell behavior.

**Failure signal:** Empty or wrong model name → fix before starting vLLM.

### Step 5 — Start vLLM (Terminal A)

**Goal:** Start pinned vLLM with prefix caching and observe startup logs.

**Command (Terminal A — leave running):**

```powershell
docker compose -f infra/docker/vllm/docker-compose.yml up --force-recreate vllm
```

Do **not** add `-d` on the reviewer path: you need visible startup logs and the first model download progress.

**Expected important log fragments (order may vary):**

```text
model Qwen/Qwen2.5-3B-Instruct
enable_prefix_caching=True
Available KV cache memory: <positive value>
GPU KV cache size: <positive token count>
Starting vLLM server on http://0.0.0.0:8000
Application startup complete
GET /health HTTP/1.1 200 OK
```

**Live-verified examples (same hardware class, values not universal):**

```text
Available KV cache memory: 3.29 GiB
GPU KV cache size: 95,872 tokens
```

**Meaning:** Model fits in VRAM, KV cache is positive, server is listening inside the container on port 8000 (host port **8100**).

**Failure signals (environment, not proof mechanism):**

```text
Available KV cache memory: <negative value>
No available memory for the cache blocks
Engine core initialization failed
```

Classification: **ENVIRONMENT_BLOCKED** — typically wrong model for available VRAM (7B on 12 GB) or GPU not passed to the container.

**Container restart loop:** Compose sets `restart: unless-stopped`. After a failed start, the same startup sequence may repeat in Terminal A. That is one container being restarted, not multiple proof runs. Stop it with the cleanup command in Step 18 before retrying.

**First start:** Hugging Face model download can take several minutes; wait until health succeeds (Step 6).

### Step 6 — Verify health (Terminal B)

**Goal:** Confirm the vLLM HTTP server is reachable on the host-mapped port.

**Command:**

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8100/health | Select-Object StatusCode
```

**Expected result:**

```text
StatusCode
----------
       200
```

**Meaning:** vLLM health endpoint is up on the host.

**Failure signal:** Connection error or non-200 after startup should have finished → **ENVIRONMENT_BLOCKED**.

### Step 7 — Verify version

**Goal:** Confirm the pinned vLLM release.

**Command:**

```powershell
Invoke-RestMethod http://127.0.0.1:8100/version
```

**Expected result:**

```text
version
-------
0.23.0
```

**Meaning:** Server matches the image pinned in `infra/docker/vllm/docker-compose.yml` (`vllm/vllm-openai:v0.23.0`). The proof gates expect version `0.23.0`.

**Failure signal:** Version mismatch → **ENVIRONMENT_BLOCKED** (or **PROOF_FAILED** if the runner completes against a non-canonical server).

### Step 8 — Verify required metrics

**Goal:** Confirm prefix-cache metrics exist without dumping the full `/metrics` body.

vLLM `0.23.0` exposes these **base metric names** (parsed by Intergrax diagnostics):

```text
vllm:kv_cache_usage_perc
vllm:prefix_cache_queries
vllm:prefix_cache_hits
vllm:prompt_tokens_cached
```

**Command:**

```powershell
$metrics = (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8100/metrics).Content -split "`n"
$metrics | Where-Object {
    $_ -match '^vllm:(kv_cache_usage_perc|prefix_cache_queries|prefix_cache_hits|prompt_tokens_cached)(\{|\s)'
}
```

**Expected result:** At least one line per required metric name (labels may appear on some series).

**Before the first inference request**, counter values are often `0` or `0.0`. Zero counters before the first request do not mean that prefix caching is disabled.

**Meaning:** Prometheus exposition includes the metrics the proof runner reads for deltas.

**Failure signal:** Filter returns no lines for a required name → **ENVIRONMENT_BLOCKED**.

### Step 9 — Smoke inference

**Goal:** Confirm OpenAI-compatible chat completion works before the full proof.

**Command:**

```powershell
$body = @{
  model = "Qwen/Qwen2.5-3B-Instruct"
  messages = @(@{ role = "user"; content = "Reply with one word: ok." })
  max_tokens = 8
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "http://127.0.0.1:8100/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
```

**Expected result:**

- HTTP request completes without error.
- Response object includes `choices` with generated text.
- `usage` block present; with `--enable-prompt-tokens-details`, `usage.prompt_tokens_details` may appear after this request.

**Meaning:** Endpoint and model inference work. This is **not** proof of prefix-cache reuse.

**Failure signal:** HTTP error or empty choices → **ENVIRONMENT_BLOCKED**.

## Intergrax live proof entrypoint

**Module:** `intergrax.runtime.token_optimization.proofs.vllm_prefix_cache_live`

**Canonical manual path (vLLM already running from Step 5):**

```powershell
uv run python -m intergrax.runtime.token_optimization.proofs.vllm_prefix_cache_live `
  --model Qwen/Qwen2.5-3B-Instruct `
  --base-url http://127.0.0.1:8100/v1 `
  --runs 3 `
  --minimum-prefix-chars 4096
```

**Behavior:**

- `manage_vllm` defaults to **false** — the runner **does not** start or stop Docker; it connects to the server you started manually.
- Shared/manual servers are **never** stopped by the runner.
- Managed lifecycle (`--manage-vllm`): the runner may start vLLM via Compose; after proof it runs `docker compose … stop vllm` only when it started or recreated the container and `--keep-vllm-running` is not set.
- The runner performs one **warmup** inference (non-measured) then **three canonical runs**.
- Each run executes three cases in order: **COLD**, **WARM**, **CHANGED_PREFIX**.
- Default report directory: `build/proofs/token_optimization/vllm_prefix_cache/<timestamp_utc>/`.

**Optional gated E2E test (same runner, single run):**

```powershell
$env:INTERGRAX_TOKEN_OPTIMIZATION_VLLM_E2E = "1"
uv run pytest tests/e2e/token_optimization/test_vllm_prefix_cache_live_e2e.py -m e2e -q
```

**Expected terminal summary (example when all gates pass on a manually started server):**

```text
final status: PASS
pass count: 3/3
failure reason codes: none
json report: build/proofs/token_optimization/vllm_prefix_cache/20260130T120000Z/vllm-prefix-cache-proof.json
markdown report: build/proofs/token_optimization/vllm_prefix_cache/20260130T120000Z/vllm-prefix-cache-proof.md
```

A manually started vLLM server can produce a canonical **PASS** when the runner independently verifies the expected model (`Qwen/Qwen2.5-3B-Instruct` by default), pinned vLLM version (`0.23.0`), required metrics, and all cold/warm/changed-prefix gates. **Managed/shared** describes lifecycle ownership, not proof correctness.

**Optional managed lifecycle path** (runner starts vLLM via Compose; less observable in Terminal A):

```powershell
uv run python -m intergrax.runtime.token_optimization.proofs.vllm_prefix_cache_live `
  --manage-vllm --force-recreate-vllm --keep-vllm-running `
  --model Qwen/Qwen2.5-3B-Instruct `
  --runs 3
```

That path starts vLLM detached (`docker compose up -d`); prefer the manual path for first-time environment review.

**Adapter usage fields (vLLM):** `cached_input_tokens` is mapped from OpenAI-style `usage.prompt_tokens_details.cached_tokens` when `--enable-prompt-tokens-details` is enabled. `uncached_input_tokens` is derived as `input_tokens - cached_input_tokens` when valid.

## Cold-cache control

**Run A (first case in each measured run, after runner warmup):**

- Stable prefix **P1** (synthetic cache-stable block, ≥4096 chars by default)
- Dynamic tail **D1** (unique per case)
- Canonical tool schema included

**Check in report (case `COLD`):**

| Field | Role |
| --- | --- |
| `prefix_hash` | Stable prefix fingerprint recorded |
| `tool_envelope_hash` | Canonical tool schema hash |
| `input_tokens` | Provider prompt/input count |
| `cached_input_tokens` | Cached prefix tokens from adapter usage |
| `metric_deltas.prefix_cache_hits` | Hit counter delta for this request |
| `metric_deltas.prefix_cache_queries` | Query counter delta |
| `passed` | Case-level gate |

**Expected:**

- Request succeeds.
- Cache reuse may be **zero** on cold case; cold run does **not** require a cache hit.
- `prompt_tokens_details_reported` should be **true**.

## Warm-cache reuse

**Run B (second case in the same measured run):**

- Same stable prefix **P1** as Run A
- Different dynamic tail **D2**

**Hard conditions:**

```text
Run A prefix_hash == Run B prefix_hash
prefix-cache query counter delta increases on warm case
prefix-cache hit counter delta increases on warm case
cached_input_tokens > 0 (warm > cold cached_input_tokens)
request succeeds
```

Lower latency alone is not accepted as proof of cache reuse. Provider metrics or adapter `cached_input_tokens` must confirm cached prefix tokens.

**Failure examples in report reason codes:** `WARM_CACHED_TOKENS_NOT_POSITIVE`, `WARM_NOT_GREATER_THAN_COLD`, `WARM_HIT_DELTA_NOT_GREATER_THAN_COLD`.

## Changed-prefix negative control

**Run C (third case in the same measured run):**

- Changed stable prefix **P2** (different `prefix_variant` label in proof assembly)
- Dynamic tail **D3**

**Expected:**

```text
Run C prefix_hash != Run A prefix_hash
prefix change is recorded
Run C cached_input_tokens < warm cached_input_tokens (changed-prefix reuse gate)
request succeeds
```

This prevents treating accidental repetition or global cache state as an exact-prefix hit on P1.

**Failure example:** `CHANGED_PREFIX_REUSE_NOT_LOWER_THAN_WARM`.

## Tool-schema integrity

Aligned with **TOKEN-10B-R1/R2** send-path boundaries (not a separate optimization algorithm):

```text
canonical tool order
→ exact schema hash (tool_envelope_hash)
→ send-time integrity validation (materialize_cache_stable_send_payload)
→ adapter invocation only after validation
```

Integrity cases enforced before adapter calls:

| Case | Expected |
| --- | --- |
| Identical canonical schema | Accepted; stable `tool_envelope_hash` |
| Reordered outer tool list | Rejected (`CacheStablePromptIntegrityError`) |
| Mutated schema vs recorded hash | Rejected |
| Mutated messages vs `messages_hash` | Rejected |
| Invalid payload | Adapter not invoked |

The live proof uses one canonical synthetic tool (`token_optimization_proof_echo`) per measured case; integrity is validated on every send.

## Generated proof artifacts

**Directory pattern:**

```text
build/proofs/token_optimization/vllm_prefix_cache/<YYYYMMDDTHHMMSSZ>/
```

**Files:**

| File | Content |
| --- | --- |
| `vllm-prefix-cache-proof.json` | Full safe structured result |
| `vllm-prefix-cache-proof.md` | Human-readable summary |

**Report includes (safe fields only):**

- task ID (`TOKEN-10C-LIVE-PROOF-1`), timestamps, repository commit SHA
- vLLM image/version, model, GPU summary
- per-run and per-case: `prefix_hash`, `tool_envelope_hash`, token counts, metric deltas, pass/fail, reason codes
- aggregate summary and process exit code
- known limitations section in Markdown

**Report does not include:**

- raw prompts, documents, RAG fragments, tool arguments, secrets, tokens, private customer content, or arbitrary unredacted metadata
- absolute user-specific file paths (report paths are relative to the repository layout)

## Final statuses

Do not infer status from Docker logs alone. Use terminal summary, exit code, and report.

| Status | Meaning |
| --- | --- |
| **PASS** | Aggregate `canonical_pass` is true (three runs, environment verified, all gates pass, exit code `0`) |
| **PROOF_FAILED** | Environment verified enough to execute cases, but proof gates or aggregate rules failed (exit code `3`) |
| **ENVIRONMENT_BLOCKED** | Docker, GPU, model memory, download, health, version, or required metrics blocked execution (exit code `2` when no completed runs) |
| **INTERNAL_ERROR** | Unexpected runner failure (exit code `4`) |

**Exit code map:**

| Code | Typical classification |
| --- | --- |
| `0` | PASS (`canonical_pass`) |
| `2` | ENVIRONMENT_BLOCKED |
| `3` | PROOF_FAILED |
| `4` | INTERNAL_ERROR |

For manual vLLM review, prioritize:

- `aggregate.all_runs_passed`
- per-case `passed` and cached-token evidence in the Markdown report
- explicit `reason_codes`

## Troubleshooting

### Insufficient GPU memory

**Symptoms:**

```text
Available KV cache memory: negative value
No available memory for the cache blocks
```

**Action:** Use the canonical `Qwen/Qwen2.5-3B-Instruct` proof model (Step 4). Do not raise `gpu_memory_utilization` as the first fix when model weights alone exceed device VRAM (7B failure on 12 GB).

### Container restart loop

**Symptom:** The same startup sequence appears repeatedly in Terminal A.

**Explanation:** `restart: unless-stopped` restarts the failed vLLM container.

**Stop:**

```powershell
docker compose -f infra/docker/vllm/docker-compose.yml stop vllm
```

Fix the model (3B) or GPU visibility, then repeat Step 5.

### Health connection failure

**Check:**

```powershell
docker compose -f infra/docker/vllm/docker-compose.yml ps
docker compose -f infra/docker/vllm/docker-compose.yml logs --tail 200 vllm
```

Use logs only for troubleshooting, not as the normal proof path.

### Hugging Face unauthenticated warning

A warning about missing `HUGGING_FACE_HUB_TOKEN` may mean slower downloads or rate limits. It is not automatically a proof failure for public models such as Qwen2.5-3B-Instruct.

### WSL `pin_memory` warning

```text
Using 'pin_memory=False' as WSL is detected
```

Performance warning only; not an automatic proof failure.

## Cleanup

**Stop vLLM (Terminal A may show container stopped):**

```powershell
docker compose -f infra/docker/vllm/docker-compose.yml stop vllm
```

Do not delete the `vllm_cache` volume after each proof; retained Hugging Face weights speed up reruns.

## Reviewer checklist

- [ ] Docker client and server are available.
- [ ] NVIDIA GPU is visible.
- [ ] Canonical 3B model is selected (`VLLM_MODEL` and runner model).
- [ ] vLLM `0.23.0` starts successfully.
- [ ] Prefix caching is enabled in logs.
- [ ] KV cache memory is positive.
- [ ] Health returns HTTP 200.
- [ ] Required metrics are present.
- [ ] Smoke inference succeeds.
- [ ] Cold control succeeds.
- [ ] Warm run confirms provider cache reuse (`cached_input_tokens` and metric deltas).
- [ ] Changed-prefix negative control succeeds.
- [ ] Tool-schema integrity holds (no integrity errors during run).
- [ ] Report contains no raw/private content.
- [ ] Final status is explicit in report and terminal summary.

## Known limitations

- Windows / WSL2 / Docker Desktop is the currently live-verified environment; other OS profiles are not automatically live-certified.
- Default proof model: `Qwen/Qwen2.5-3B-Instruct` (runner and Compose).
- Managed/shared lifecycle describes server ownership, not proof correctness.
- Cache reuse metrics are not a content-reduction benchmark.
- vLLM prefix-cache reuse does not imply billing savings on other providers.
- Full universal proof, checked-in public proof, and README promotion belong to **TOKEN-10F**, **TOKEN-10G**, and **TOKEN-10H** — not completed by this document.
- TOKEN-10G and TOKEN-10H must not be treated as closed based on TOKEN-10C alone.
