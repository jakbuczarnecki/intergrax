# Try Local Knowledge Workspace

Want to understand the product experience before running it? See the [LKW Product Tour](../../../LKW_PRODUCT_TOUR.md).

## What this does

This quickstart is a supported local product-evaluation path. One command starts the canonical local stack (unless you already have it running), uploads a bundled non-sensitive sample document through managed-file Knowledge Intake, waits for indexing, asks a grounded question over indexed knowledge, shows the answer with a source citation, and verifies the persisted Ask run.

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

This is indexed-only LKW behavior. It is not Hybrid Ask, not a platform certification run, and not a production deployment.

## Prerequisites

- Git
- Docker Desktop or Docker Engine with Compose
- `uv`
- Sufficient disk space for Docker images and the configured local model (Ollama pull on first run)

First-run duration depends on image downloads, model download, network speed, and machine performance. A 15-minute target is not yet externally validated.

## Windows

From the repository root:

```bat
applications\local_workspace_application\scripts\run-lkw-product-quickstart-windows.bat
```

## Linux

From the repository root:

```sh
./applications/local_workspace_application/scripts/run-lkw-product-quickstart-linux.sh
```

## macOS

From the repository root:

```sh
./applications/local_workspace_application/scripts/run-lkw-product-quickstart-macos.sh
```

## What you should see

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="../../../docs/assets/public/lkw-grounded-result-dark.svg"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="../../../docs/assets/public/lkw-grounded-result-light.svg"
  >
  <img
    alt="LKW quickstart flow showing the approved sample file lkw_product_quickstart.txt, the question “What is the project codename?”, the grounded answer “AURORA-17”, its source reference, and persisted Ask-run verification."
    src="../../../docs/assets/public/lkw-grounded-result-light.svg"
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

- existing Docker bootstrap (`build-local-docker`)
- managed workspace HTTP API
- managed-file Knowledge Intake (no arbitrary local folder read)
- durable ingestion operation polling
- indexed Ask V1
- persisted Ask-run read

You did not need to write API JSON, copy operation IDs manually, or configure `INTERGRAX_ALLOWED_READ_ROOTS`.

## First-run downloads

- Docker images for the LKW stack may be downloaded on first run.
- The configured generation model (`llama3.1:latest` by default) may be downloaded by the stack.
- The quickstart resolves the embedding model configured in the running LKW container and pulls that exact model.
- An existing `.env` is not modified; a missing `.env` is created once from `.env.example`.
- Operational failures return a safe stage and reason instead of raw Docker, HTTP or subprocess logs.
- Duration varies by environment; timing is not claimed as validated until external sessions confirm it.

## Safety

- The sample file is bundled and non-sensitive (`sample_docs/lkw_product_quickstart.txt`).
- Managed upload is used; no arbitrary local folder is read.
- Only loopback HTTP (`127.0.0.1`, `localhost`, `::1`) is allowed.
- Your original local files are not modified.
- If `.env` is missing, it is created from `.env.example` once; an existing `.env` is never overwritten.
- The stack remains running after success for inspection.
- Local Docker volumes may retain evaluation data.

### Stop the stack

From `applications/local_workspace_application/`:

```sh
docker compose -p intergrax_lkw -f docker/docker-compose.yml down
```

On Windows, use the same command from a shell after `cd` to that directory.

## Troubleshooting

- Ensure Docker is running and `uv` is on `PATH`.
- Inspect service status: `docker compose -p intergrax_lkw -f docker/docker-compose.yml ps`
- Inspect logs: `docker compose -p intergrax_lkw -f docker/docker-compose.yml logs --tail 200 local_workspace`
- Health check: `http://127.0.0.1:8020/health` should return `status: ok`.
- Re-run with an already-running stack: add `--skip-stack-start` to the Python runner (wrappers normally start the stack).

## What this does not prove

- Hybrid Ask
- Live-provider access
- Production readiness or security/compliance certification
- Commercial validation
- Full LKW platform certification ([LKW Platform Proof](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md) remains the deeper technical path)
- Linux or macOS live certification unless you actually run the quickstart on those systems

## Next steps

- Product orientation: [LKW Product Tour](../../../LKW_PRODUCT_TOUR.md)
- Bounded technical proof: [LKW Platform Proof](../../../docs/public-adoption/LKW_PLATFORM_PROOF.md)
- Proof status dashboard: [PROOFS.md](../../../PROOFS.md)
- Build and evaluate: [BUILD_WITH_INTERGRAX.md](../../../BUILD_WITH_INTERGRAX.md)
