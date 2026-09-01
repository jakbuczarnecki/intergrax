# LKW PRODUCT-2 Installation Gap Audit

## 1. Status

**Status:** READY_FOR_REVIEW
**Task:** LKW-PRODUCT-2A - ZERO-TO-VALUE INSTALLATION GAP AUDIT AND IMPLEMENTATION PLAN
**Mode:** discovery / gap analysis only

The accepted contract ancestor is present. The audit was performed on branch
`development` at starting HEAD `d88db5d410e5ebe8bf116db624e342bfb8d82415`;
`origin/development` was `4e78c011d445d6e17d1c5c91ecff416d21ce8c9c` after the
required fetch. Existing unrelated dirty work was preserved.

No production code, Docker/bootstrap script, vendor integration, or test was
modified by this audit.

## 2. Audited product outcome

The accepted outcome is the technically comfortable user's path:

```text
clone/download
→ one obvious supported command
→ prerequisite and preflight feedback
→ automatic preparation of local runtime dependencies
→ canonical LKW stack
→ usable running application boundary
```

The current product quickstart reaches a stronger bounded evaluation result:
it starts the stack, creates a workspace, uploads the bundled sample, waits for
indexing, performs grounded Ask with a citation, and verifies the persisted Ask
run. This is sufficient evidence of a usable application boundary for
PRODUCT-2, but it does not yet provide production-grade installation
preflight or action-oriented failure recovery.

The audit does not treat onboarding with the user's own sources, vendor
connections, or the daily-use UI as PRODUCT-2 evidence. Those belong to later
product tasks.

## 3. Current supported installation paths

| Path | Classification | Current role |
|---|---|---|
| Windows `run-lkw-product-quickstart-windows.bat` | PRODUCT PATH | Delegates to the shared product runner. |
| Linux `run-lkw-product-quickstart-linux.sh` | PRODUCT PATH | Delegates to the shared product runner. |
| macOS `run-lkw-product-quickstart-macos.sh` | PRODUCT PATH | Delegates to the shared product runner. |
| `build-local-docker.bat` / `build-local-docker.sh` | ADVANCED/DEVELOPER PATH | Direct Docker bootstrap; not the normal product UX. |
| README manual `uvicorn` plus curl/API examples | ADVANCED/DEVELOPER PATH | Local development and API proof flow. |
| Platform Proof route referenced by the README and quickstart | INTERNAL/PROOF PATH | Deeper technical verification, explicitly separate from product quickstart. |

The product entry point is documented, but not fully unambiguous in the
repository: the application README presents the advanced Docker and manual
uvicorn routes close to the supported quickstart. A technically comfortable
user can identify the product route, but the documentation still exposes
competing commands.

The three product launchers are transport wrappers. The shared runner owns
product orchestration, which is the correct reuse point for common behavior.

## 4. Current zero-to-value journey

The current product path is:

1. The OS wrapper checks that `uv` exists, locates the shared runner, and
   invokes it with an OS/wrapper identity.
2. The shared runner validates the supported OS/wrapper pair, loopback base
   URL, and bundled sample.
3. It creates `.env` from `.env.example` only when missing, then appends the
   default embedding model. An existing `.env` is preserved.
4. It invokes the OS-specific Docker bootstrap with the fixed Compose project
   `intergrax_lkw`.
5. The bootstrap materializes the runtime context, starts the canonical
   Compose stack, and pulls `llama3.1:latest`.
6. The runner waits for `http://127.0.0.1:8020/health` to report `status=ok`.
7. It resolves the embedding model from the running LKW container and pulls
   that exact model through Ollama.
8. It creates an evaluation workspace, uploads the bundled managed-file sample,
   polls the ingestion operation, asks the fixed grounded question, and reads
   back the persisted Ask result.
9. It emits human-readable output and stable machine-readable success lines:
   `lkw_quickstart_result=PASS`, citation, persisted-run verification, and
   `stack_left_running=true`.

The success journey is coherent and uses product-facing API boundaries. The
preflight journey is incomplete: it mostly discovers prerequisites indirectly
through the bootstrap command, and it does not turn failures into explicit
user actions.

## 5. What already satisfies PRODUCT-2

- One product launcher exists for each Windows, Linux, and macOS.
- All three launchers share one orchestration runner rather than duplicating
  product behavior.
- `.env` is prepared automatically when absent and is not overwritten when
  present.
- The canonical Compose project is consistently `intergrax_lkw`.
- MongoDB, Qdrant, Ollama, the LKW host, and durable named volumes are part of
  the canonical stack.
- Compose healthchecks and `depends_on` health conditions already provide
  reusable dependency readiness mechanisms.
- The host health endpoint is polled before product operations begin.
- Generation and embedding model preparation is automatic on the product path;
  embedding preparation is resolved from the running application configuration.
- Bootstrap and HTTP subprocess output is captured by the shared runner.
- Failure output is reduced to a safe `failed_stage` and
  `failure_reason`; raw Docker, HTTP, and subprocess output is not emitted by
  the runner.
- The final proof is stronger than a health check: it verifies indexed
  knowledge, grounded answer, citation, and persisted Ask readback.
- No normal product step asks the user to edit MongoDB/Qdrant, use manual
  curl/JSON, or understand service topology.

These mechanisms are a solid base for a single PRODUCT-2 implementation block.
They do not by themselves prove that all prerequisite failures are safely
detected and explained before a user encounters them.

## 6. Gap matrix

Status meanings: **GOOD** satisfies the current PRODUCT-2 need,
**PARTIAL** works on the happy path but lacks a production-grade guard or UX,
**MISSING** has no detection or useful handling on the supported path.

| User-visible requirement | Current behavior | Status | Evidence path | Reuse decision | Required change |
|---|---|---:|---|---|---|
| One unambiguous product entry point | Quickstart is labelled supported, but README also exposes direct Docker and manual uvicorn/API routes. | PARTIAL | [`QUICKSTART.md`](QUICKSTART.md); [`application README`](../../../../applications/local_workspace_application/README.md) | EXTEND EXISTING | Make the product path visually primary and label all other routes advanced or proof-only. |
| Git availability, if needed for acquisition | Git is listed as a prerequisite for cloning, but the launchers do not check it. After a repository is already downloaded, Git is not needed to run the path. | PARTIAL | [`QUICKSTART.md`](QUICKSTART.md) | REUSE EXISTING | State the boundary clearly; add a check only if download/clone is part of the supported installer rather than a precondition. |
| Docker executable missing | The wrapper checks `uv`, not Docker. The bootstrap fails later and is mapped to generic `stack_start_failed`. | PARTIAL | [`windows launcher`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart-windows.bat); [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py) | EXTEND EXISTING | Add an early Docker CLI check with a safe reason and user action. |
| Docker daemon unavailable | No explicit daemon probe; `docker compose` failure is discovered during stack start. | PARTIAL | [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py); [`docker-compose.yml`](../../../../applications/local_workspace_application/docker/docker-compose.yml) | EXTEND EXISTING | Preflight daemon availability and report “start Docker” rather than only `stack_start_failed`. |
| Compose unavailable | No explicit Compose capability check; the command fails during bootstrap. | PARTIAL | [`build-local-docker.bat`](../../../../applications/local_workspace_application/scripts/build-local-docker.bat); [`build-local-docker.sh`](../../../../applications/local_workspace_application/scripts/build-local-docker.sh) | EXTEND EXISTING | Check `docker compose version` before bootstrap and map failure to a clear action. |
| `uv` missing | All OS wrappers report `uv was not found on PATH` and stop. | GOOD | [`Windows launcher`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart-windows.bat); [`Linux launcher`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart-linux.sh); [`macOS launcher`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart-macos.sh) | REUSE EXISTING | Preserve the shared wrapper behavior. |
| Unsupported OS | The runner recognizes Windows/Linux/macOS and rejects an OS/wrapper mismatch, but there is no supported path or tailored guidance for another OS. | PARTIAL | [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py) | EXTEND EXISTING | Keep the bounded support list and add an actionable unsupported-platform message. |
| Insufficient disk space | Disk space is documented as a prerequisite, but no check or specific failure mapping exists. | MISSING | [`QUICKSTART.md`](QUICKSTART.md) | NEW CHECK WITHIN EXISTING RUNNER | Add a bounded preflight check or an explicit safe “free disk space” failure contract. |
| Required port conflict | Ports such as 8020 and 4318 are fixed in Compose; no preflight probe explains a conflict. | MISSING | [`docker-compose.yml`](../../../../applications/local_workspace_application/docker/docker-compose.yml) | NEW CHECK WITHIN EXISTING RUNNER | Probe required host ports before starting and identify the action without printing topology or secrets. |
| Generation model preparation | Bootstrap always pulls hardcoded `llama3.1:latest`; failure is safely reduced by the runner, but a configured generation model is not consistently honored. | PARTIAL | [`build-local-docker.bat`](../../../../applications/local_workspace_application/scripts/build-local-docker.bat); [`build-local-docker.sh`](../../../../applications/local_workspace_application/scripts/build-local-docker.sh); [`.env.example`](../../../../applications/local_workspace_application/.env.example) | EXTEND EXISTING | Use the configured/default generation model consistently in both OS bootstraps and retain safe failure mapping. |
| Embedding model preparation | The runner resolves the model from the running LKW container and pulls it; the model is reused by Ollama on repeat runs. | GOOD | [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py) | REUSE EXISTING | Preserve this shared implementation and test configured/default resolution. |
| Ollama availability and pull failure | Compose healthchecks gate the host; pull failures become `embedding_model_pull_failed` or generic stack-start failure, without raw logs. | GOOD | [`docker-compose.yml`](../../../../applications/local_workspace_application/docker/docker-compose.yml); [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py) | EXTEND EXISTING | Add the missing action text while preserving safe reasons and idempotent pulls. |
| MongoDB startup/readiness | Mongo has a healthcheck and LKW depends on it, but the user receives no service-specific readiness/action result. | PARTIAL | [`docker-compose.yml`](../../../../applications/local_workspace_application/docker/docker-compose.yml); [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py) | REUSE EXISTING | Reuse Compose health state and map a failed dependency to a safe retry/action message. |
| Qdrant startup/readiness | Qdrant has a healthcheck and LKW depends on it, but the user receives no service-specific readiness/action result. | PARTIAL | [`docker-compose.yml`](../../../../applications/local_workspace_application/docker/docker-compose.yml); [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py) | REUSE EXISTING | Reuse Compose health state and map a failed dependency to a safe retry/action message. |
| LKW host readiness | The runner polls `/health` until `status=ok` and reports a bounded `health_timeout` otherwise. | GOOD | [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py); [`QUICKSTART.md`](QUICKSTART.md) | REUSE EXISTING | Keep `/health` as the application readiness gate. |
| Automatic `.env` preparation and preservation | Missing `.env` is copied from `.env.example`; existing `.env` is not overwritten. | GOOD | [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py); [`.env.example`](../../../../applications/local_workspace_application/.env.example) | REUSE EXISTING | Preserve the behavior and verify it with a checksum-based rerun test. |
| Invalid or missing configuration feedback | Missing example is mapped safely, but values are not validated before Compose and invalid configuration becomes a generic start/health failure. | PARTIAL | [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py); [`.env.example`](../../../../applications/local_workspace_application/.env.example) | EXTEND EXISTING | Validate only configuration mandatory for initial start and report a safe corrective action. Do not expose values. |
| User-safe failure experience | The runner emits stage and reason and captures Docker/HTTP subprocess output, but it emits no action; wrapper-level `uv run` setup failures can still expose tool output. | PARTIAL | [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py); OS launchers | EXTEND EXISTING | Add a stage/reason/action contract and contain the supported wrapper's setup failures where feasible. |
| Canonical stack and durable runtime | Both bootstraps use project `intergrax_lkw`; Compose includes LKW host, MongoDB, Qdrant, Ollama, OTEL, and named durable volumes. | GOOD | [`docker-compose.yml`](../../../../applications/local_workspace_application/docker/docker-compose.yml); bootstrap scripts | REUSE EXISTING | Do not redesign topology or introduce a second Compose project. |
| Safe and resumable rerun | `.env` and models survive; Linux/macOS use Compose up, while Windows deliberately downs the stack first. `--skip-stack-start` supports an already-running stack. No checkpointed bootstrap resume exists. | PARTIAL | [`QUICKSTART.md`](QUICKSTART.md); bootstrap scripts; [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py) | EXTEND EXISTING | Define and test safe reuse/retry semantics, including already-running services and partial bootstrap. |
| Equivalent Windows/Linux/macOS outcome and error semantics | All wrappers use the shared runner and stable result keys; bootstrap mechanics differ and Linux/macOS are not live-certified by the quickstart documentation. | GOOD | OS launchers; [`QUICKSTART.md`](QUICKSTART.md) | REUSE EXISTING | Keep common orchestration and validate each wrapper's equivalent safe result contract. |
| Installation completion proof | PASS output includes host-backed Ask, citation, persisted Ask readback, and `stack_left_running=true`. | GOOD | [`QUICKSTART.md`](QUICKSTART.md); [`shared runner`](../../../../applications/local_workspace_application/scripts/run-lkw-product-quickstart.py) | REUSE EXISTING | Use this existing proof; do not add a competing readiness mechanism. |

**Count:** GOOD 8 · PARTIAL 12 · MISSING 2.

The highest-priority user-visible gaps are missing Docker/Compose/daemon
preflight, missing port and disk-space feedback, lack of configuration
validation, hardcoded generation-model preparation, and failure output that
has a safe reason but no corrective action.

Secret leakage risk in the shared runner is currently low: subprocess output is
captured, failure reasons are restricted to safe tokens, and user-facing
success text is checked against forbidden internal/configuration fragments.
The advanced scripts and the outer `uv` process are not subject to the same
complete output contract, which is another reason they must remain outside the
normal product path.

## 7. Reuse opportunities

### Reusable mechanisms found

- The shared cross-platform runner is the natural product orchestration
  boundary.
- `ensure_env_file()` already provides safe one-time configuration
  materialization and preservation.
- Compose service healthchecks and dependency conditions already define
  MongoDB, Qdrant, and Ollama readiness.
- `/health` already defines the LKW host readiness gate.
- The fixed Compose project `intergrax_lkw` and named volumes already define
  the durable local runtime.
- The runner's safe stage/reason output and stable PASS lines already form the
  beginnings of an installation completion contract.
- The runner's embedding-model resolution is shared across all OS wrappers.

### Existing mechanisms requiring extension

- Extend the shared runner's preflight instead of adding OS-specific
  prerequisite logic.
- Extend bootstrap model selection so both scripts honor the same configured
  generation model.
- Extend failure mapping with an action field and dependency-aware reasons,
  while retaining safe output.
- Clarify the existing documentation hierarchy instead of creating another
  launcher.

### New capability required

**No new reusable product capability is required for the smallest PRODUCT-2
block.** The gaps are preflight, configuration validation, model-selection
consistency, and error UX around existing contracts.

If the product later requires a detailed per-service status page or a
machine-readable installer diagnostic API, that would be a separate capability
and must not be introduced speculatively in PRODUCT-2.

## 8. Required minimal implementation block

Implement one coherent block: **Product Quickstart Installation/Preflight
and Failure UX Extension**.

It should:

1. Run early checks for Docker CLI, daemon, Compose, required ports, supported
   OS, mandatory initial configuration, and the bounded disk-space policy.
2. Preserve the current one-time `.env` creation and existing `.env` contents.
3. Keep the canonical Compose project and durable services unchanged.
4. Make generation-model preparation consume the configured/default model in
   both OS bootstrap scripts; retain the shared runner's configured embedding
   model resolution.
5. Reuse Compose health state plus the existing `/health` gate, mapping
   dependency and host failures to safe stage/reason/action output.
6. Define rerun behavior explicitly: an existing environment survives,
   downloaded models are reused, an already-running stack can be reused, and a
   partial bootstrap can be retried without manual database or vector-store
   manipulation.
7. Finish with the existing PASS evidence rather than adding a second
   readiness mechanism.

The block must not add vendor setup, onboarding, a new UI, a new service, or a
new database/vector-store contract.

## 9. Explicit out-of-scope items

- PRODUCT-3 first-run onboarding and user-owned source setup.
- Slack, Google Workspace, Microsoft 365, or any vendor integration.
- MongoDB or Qdrant schema/data manipulation.
- Compose topology redesign, service replacement, or new readiness service.
- Production deployment, HA, security/compliance certification, or external
  model hosting.
- Full platform proof and observability certification.
- Manual API/JSON as a normal user flow.
- Automatic installation of Docker, Docker Desktop, Compose, or `uv` with
  elevated privileges.
- Changes to the accepted product contract.
- Changes to unrelated concurrent work.

## 10. Proposed PRODUCT-2 acceptance proof

On clean supported Windows, Linux, and macOS environments, a technically
comfortable evaluator should be able to:

1. Start from the repository root and run only the documented OS product
   launcher.
2. Receive a safe, actionable result for each supported negative case:
   missing `uv`, missing Docker, unavailable daemon, unavailable Compose,
   occupied required port, invalid mandatory configuration, insufficient disk
   space, model pull failure, dependency readiness failure, and host
   readiness timeout.
3. Verify that no negative-case output contains secrets, raw Python traceback,
   raw Docker/Compose logs, internal paths, Mongo/Qdrant operation details, or
   manual API instructions.
4. Verify that a missing `.env` is created, an existing `.env` is byte-for-byte
   preserved, and no normal edit is required for the default initial start.
5. Verify that the stack uses Compose project `intergrax_lkw`, that required
   services become ready, and that configured generation and embedding models
   are available.
6. Observe the existing successful proof:
   `lkw_quickstart_result=PASS`, grounded answer marker, citation file,
   `persisted_run_verified=true`, and `stack_left_running=true`.
7. Rerun with existing `.env`, downloaded models, and running services; verify
   safe reuse or retry without manual Mongo/Qdrant operations.

This proof demonstrates the PRODUCT-2 installation outcome. It does not claim
vendor onboarding, production readiness, or the later daily-use product UX.

## 11. Exact proposed implementation file scope

The future implementation block should be limited to:

1. `applications/local_workspace_application/scripts/run-lkw-product-quickstart.py`
   - centralize Docker/Compose/daemon/port/configuration/disk preflight;
   - add safe action-bearing failure output;
   - map Compose dependency state without printing raw logs;
   - preserve environment/model/idempotency behavior.
2. `applications/local_workspace_application/scripts/build-local-docker.bat`
   - use the configured/default generation model instead of an unconditional
     hardcoded pull;
   - preserve the existing `intergrax_lkw` project and named volumes.
3. `applications/local_workspace_application/scripts/build-local-docker.sh`
   - provide the same configured-model and machine-contract behavior as the
     Windows bootstrap without requiring identical shell implementation.
4. `applications/local_workspace_application/docs/product/QUICKSTART.md`
   - document the one supported installation command per OS, preflight result
     meanings, rerun contract, and safe actions.
5. `applications/local_workspace_application/README.md`
   - make the product quickstart primary and explicitly label manual uvicorn,
     direct Docker, and proof routes as advanced/developer or proof-only.
6. `applications/local_workspace_application/tests/scripts/test_run_lkw_product_quickstart.py`
   - **new test file, justified for isolated runner-contract tests**;
     cover pure preflight mapping, safe output, environment preservation,
     model resolution, port/configuration checks, and rerun decisions with
     subprocess/HTTP calls mocked.

No new production component is justified by the current evidence. If service
diagnostics cannot be expressed using existing Compose health state and
`/health`, stop for an architectural decision before adding a new readiness
endpoint or diagnostic service.

## 12. Risks / architectural decisions requiring user approval

- **Supported-host policy:** decide whether Docker, Compose, and `uv` remain
  explicit host prerequisites. Automatic installation is not a safe default
  for a local/self-hosted product and is outside this minimal block.
- **Generation-model override policy:** decide whether an existing configured
  generation model is authoritative, with `llama3.1:latest` only as the
  default. The current bootstrap behavior is inconsistent with that policy.
- **Diagnostic detail policy:** decide how much opt-in diagnostic detail is
  allowed after a safe failure without printing secrets or raw implementation
  traces. The minimum contract needs stage, reason, and action only.
- **Dependency readiness granularity:** current Compose healthchecks and
  `/health` are reusable. A new per-service readiness API is an
  **ARCHITECTURAL DECISION REQUIRED** item, not part of the minimum block.
- **Cross-OS acceptance evidence:** the scripts expose equivalent contracts,
  but live Linux/macOS runs are still needed for a complete acceptance proof;
  static parity must not be represented as live certification.
