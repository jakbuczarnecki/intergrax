# Try Local Knowledge Workspace

Want to understand the product experience before running it? See the [LKW Product Tour](LKW_PRODUCT_TOUR.md).

## PRODUCT QUICKSTART — supported entry point

**Prerequisite:** Clone or open this repository locally. Commands below assume
the repository root as the working directory.

Run exactly one command from the repository root:

| Operating system | Command |
|---|---|
| Windows | `applications\local_workspace_application\scripts\run-lkw-product-quickstart-windows.bat` |
| Linux | `./applications/local_workspace_application/scripts/run-lkw-product-quickstart-linux.sh` |
| macOS | `./applications/local_workspace_application/scripts/run-lkw-product-quickstart-macos.sh` |

This is the supported zero-to-value path. Docker Desktop/Engine with Compose and
`uv` must already be installed by the user; the launcher does not install
software or use elevated installers. Git is needed to obtain the repository, not
for each subsequent rerun. Keep at least **20 GiB free** for the bounded
first-bootstrap check; this is a safety floor, not a prediction of download
size.

## What this does

This quickstart is a supported local product-evaluation path. One command starts the canonical local stack (unless you already have it running), uploads a bundled non-sensitive sample document through managed-file Knowledge Intake, waits for indexing, asks a grounded question over indexed knowledge, shows the answer with a source citation, and verifies the persisted Ask run.

The canonical stack includes: `local_workspace`, MongoDB, Qdrant, Ollama, and
the OTEL collector. Optional proof overlays are not started by this quickstart.

Workflow:

```text
start canonical local stack
→ create workspace
→ upload managed sample file
→ wait for indexing
→ ask a question
→ receive grounded answer
→ inspect source citation
→ verify persisted Ask run
```

This Quick Start exercises the indexed **Ask V1** product path — real application
boundaries, indexed knowledge only. It is **not** the separate Hybrid Ask
proof/certification path, not a platform certification run, and not a production
deployment. Accepted public evidence for the bounded indexed Hybrid Ask branch
lives in [PROOFS](../../../../docs/project/proofs/PROOFS.md) and
[LKW Platform Proof](../proof/LKW_PLATFORM_PROOF.md); mixed indexed +
authorized-live Hybrid Ask in one answer remains **not proven**.

## After the run

The quickstart is **script-driven**: one command uploads, asks, cites, and
verifies the persisted Ask run. There is no polished end-user UI on this path.
On success the stack stays running for inspection — health check
(`http://127.0.0.1:8020/health`), Docker logs, and persisted run read. For
deeper bounded platform verification, see [Core Platform Proof](../proof/LKW_PLATFORM_PROOF.md).

Product Quick Start and Core Platform Proof use separate execution lifecycles.
Stop this quickstart stack before starting the isolated Core Platform Proof; see
[Stop the stack](#stop-the-stack) and the proof document's
[After Product Quick Start](../proof/LKW_PLATFORM_PROOF.md#after-product-quick-start)
section.

**Proofs:** `LKW-PRODUCT-QUICKSTART-WINDOWS`, `LKW-PRODUCT-QUICKSTART-LINUX`, `LKW-PRODUCT-QUICKSTART-MACOS`

## Prerequisites

- Git
- Docker Desktop or Docker Engine with Compose
- `uv` installed and available on `PATH`
- Sufficient disk space for Docker images and configured models when the
  selected generation or embedding provider is Ollama (model pull on first run)

First-run duration depends on image downloads, model download, network speed, and machine performance. A 15-minute target is not yet externally validated.

## What you should see

`AURORA-17` is the expected success marker for this quickstart proof.

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="../assets/lkw-grounded-result-dark.svg"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="../assets/lkw-grounded-result-light.svg"
  >
  <img
    alt="LKW quickstart flow showing the approved sample file lkw_product_quickstart.txt, the question “What is the project codename?”, the grounded answer “AURORA-17”, its source reference, and persisted Ask-run verification."
    src="../assets/lkw-grounded-result-light.svg"
  >
</picture>

The visual summarizes the documented result shape. The text block below remains the exact reviewable output contract.

A successful run ends with a concise summary like:

```text
LKW quickstart: PASS

Question:
What is the project codename?

Answer:
<grounded answer containing AURORA-17>

Source:
lkw_product_quickstart.txt

Workspace:
<workspace_id>

Ask run:
<run_id>

Persisted Ask run verified:
yes
```

Stable machine-readable lines:

```text
lkw_quickstart_result=PASS
answer_marker=AURORA-17
citation_file=lkw_product_quickstart.txt
persisted_run_verified=true
stack_left_running=true
```

## What actually happened

The runner reused existing production-shaped boundaries:

- existing Docker bootstrap (`applications/local_workspace_application/scripts/build-local-docker.bat` on Windows or `applications/local_workspace_application/scripts/build-local-docker.sh` on Linux/macOS)
- managed workspace HTTP API
- managed-file Knowledge Intake (no arbitrary local folder read)
- durable ingestion operation polling
- indexed Ask V1
- persisted Ask-run read

You did not need to write API JSON, copy operation IDs manually, or configure `INTERGRAX_ALLOWED_READ_ROOTS`.

## First-run downloads

- Docker images for the LKW stack may be downloaded on first run.
- When the selected **generation** provider is Ollama, the configured generation
  model may be downloaded.
- When the selected **embedding** provider is Ollama, the configured embedding
  model may be downloaded.

**Canonical model-selection contract** (see
[Platform Configuration](../../../../docs/project/technical/guides/PLATFORM_CONFIGURATION.md)):

| Role | Provider variable | Model variable |
| --- | --- | --- |
| Generation | `INTERGRAX_LLM_PROVIDER` | `INTERGRAX_LLM_MODEL` |
| Embedding | `INTERGRAX_EMBEDDING_PROVIDER` | `INTERGRAX_EMBEDDING_MODEL` |

`PROVIDER` selects the adapter; `MODEL` selects the configured model.
Generation model selection uses only `INTERGRAX_LLM_PROVIDER` and
`INTERGRAX_LLM_MODEL`. Provider-specific `INTERGRAX_DEFAULT_*_MODEL` variables
are not supported.

Canonical local LKW example:

```text
INTERGRAX_LLM_PROVIDER=ollama
INTERGRAX_LLM_MODEL=llama3.1:latest
INTERGRAX_EMBEDDING_PROVIDER=ollama
INTERGRAX_EMBEDDING_MODEL=nomic-embed-text
```

- An existing `applications/local_workspace_application/.env` is not modified; a missing `applications/local_workspace_application/.env` is created once from `applications/local_workspace_application/.env.example`.
- Operational failures return `failed_stage`, `failure_reason`, and
  `recommended_action` instead of raw Docker, HTTP or subprocess logs.
- Duration varies by environment; timing is not claimed as validated until external sessions confirm it.

## Safety

- The sample file is bundled and non-sensitive (`applications/local_workspace_application/sample_docs/lkw_product_quickstart.txt`).
- Managed upload is used; no arbitrary local folder is read.
- Only loopback HTTP (`127.0.0.1`, `localhost`, `::1`) is allowed.
- Your original local files are not modified.
- If `applications/local_workspace_application/.env` is missing, it is created from `applications/local_workspace_application/.env.example` once; an existing `.env` is never overwritten.
- The stack remains running after success for inspection.
- Local Docker volumes may retain evaluation data.
- Rerunning preserves `.env`, downloaded models, and named volumes. A healthy
  existing stack is reused; a partial start can be retried without deleting
  volumes or application data.
- `--skip-stack-start` is an advanced rerun option for an already-running
  canonical stack; the normal OS launcher starts or reuses the stack.

### Stop the stack

From `applications/local_workspace_application`:

```sh
docker compose -p intergrax_lkw -f docker/docker-compose.yml down
```

On Windows, from the repository root, run `cd /d applications\local_workspace_application` first, then run the same `docker compose` command.

## Troubleshooting

- For a normal failure, follow `failed_stage`, `failure_reason`, and
  `recommended_action`. Docker, Compose, and `uv` remain user-managed
  prerequisites.
- **Port already in use (for example `8020`):** another documented LKW stack or
  process may already own the port. Common Compose project names are
  `intergrax_lkw` (Product Quick Start), `lkw-core-platform-proof` (Core
  Platform Proof), and `lkw-trusted-ask-workspace-proof` (Trusted Ask proof).
  Do not delete volumes or reset data by default — stop the conflicting
  documented stack non-destructively (see [Stop the stack](#stop-the-stack)
  or the proof document's stop guidance), then rerun the path you want.
- From `applications/local_workspace_application`, inspect service status with `docker compose -p intergrax_lkw -f docker/docker-compose.yml ps`.
- From `applications/local_workspace_application`, inspect logs with `docker compose -p intergrax_lkw -f docker/docker-compose.yml logs --tail 200 local_workspace`.
- Health check: `http://127.0.0.1:8020/health` should return `status: ok`.
- Advanced troubleshooting commands above may show Docker details; they are
  not part of the normal product failure output.

## What this does not prove

- the separate Hybrid Ask proof/certification path (see [PROOFS](../../../../docs/project/proofs/PROOFS.md))
- mixed indexed + authorized-live Hybrid Ask in one answer
- Live-provider access
- Production readiness or security/compliance certification
- Commercial validation
- Full LKW platform certification ([LKW Platform Proof](../proof/LKW_PLATFORM_PROOF.md) remains the deeper technical path)
- Linux or macOS live certification unless you actually run the quickstart on those systems

## Primary next action

**Inspect the bounded technical evidence:** [LKW Platform Proof](../proof/LKW_PLATFORM_PROOF.md)

## Other routes

- Product orientation: [LKW Product Tour](LKW_PRODUCT_TOUR.md)
- Proof status dashboard: [docs/project/proofs/PROOFS.md](../../../../docs/project/proofs/PROOFS.md)
- Builder route: [Builder Quick Start](../../../../docs/project/builders/BUILDER_QUICKSTART.md)
- Deeper build planning: [docs/project/builders/BUILD_WITH_INTERGRAX.md](../../../../docs/project/builders/BUILD_WITH_INTERGRAX.md)
