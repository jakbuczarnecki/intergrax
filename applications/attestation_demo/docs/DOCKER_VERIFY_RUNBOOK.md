# Docker build, run & verify — step-by-step

**Audience:** Operators and partners validating the Attestation Demo PoC before handoff.

This runbook walks through **building the Docker image**, **starting the host**, and **checking PoC v2 + EBE-9 assumptions** (two boundary events, host signing by default) from [`ARCHITECTURE.md`](ARCHITECTURE.md) §17.

**Default manifest:** `host_signing_enabled=true` (signed events, `trust_model.host_attested`). For unsigned v2 regression, set `host_signing_enabled=false` in `manifest.py` or use pytest unsigned tests.

Related docs: [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md) · [`partner_handoff/README.md`](../partner_handoff/README.md)

---

## What you are verifying

| # | Project assumption (ARCHITECTURE §17) | How this runbook checks it |
|---|--------------------------------------|----------------------------|
| 1 | `POST /poc/run` works without Intergrax fork | Step 5 — HTTP trigger |
| 2 | Response has `execution_boundary_event.v1` for `records.put` **and** `harness_step` | Step 6 — two events, `event_sequence` 1 and 2 |
| 3 | Fields map to BoundaryAttest adapter | Step 6 — mapping table |
| 4 | Debug journal available for same `run_id` | Step 8 — trace endpoint |
| 5 | No dishonest `server_attested` claim from Intergrax | Step 6 — `trust_model` uses `host_attested` when signed |
| 6 | EBE-9 host attestation verifiable per event | Step 6 — `signed: true`, `host_attestation` envelope |
| 7 | HOS trace reconstructs the run | Step 8 — non-empty `trace_events` |
| 8 | No partner packages in Intergrax | Design invariant (no runtime check) |

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Git clone of **intergrax** monorepo | All commands run from **repository root** |
| Docker Desktop (or Docker Engine) **running** | `docker version` must succeed |
| `curl` or PowerShell | For HTTP smoke tests |
| Optional: [uv](https://docs.astral.sh/uv/) | For automated pytest in Step 10 |

**Build context is always the monorepo root** — not `applications/attestation_demo` alone.

---

## Step 1 — Prepare environment file

From repository root:

```bash
cp applications/attestation_demo/.env.example applications/attestation_demo/.env
```

Edit `applications/attestation_demo/.env` if needed. Defaults are fine for local Docker verification.

| Variable | Default | Role |
|----------|---------|------|
| `ATTESTATION_DEMO_BACKEND_PORT` | `8097` | Host port |
| `INTERGRAX_ENV` | `dev` | Use `prod` for production-like container |
| `INTERGRAX_HARNESS_API_KEY` | *(unset)* | Set to enable API key on `/poc/run` (Step 9) |

---

## Step 2 — Build the Docker image

**Recommended (all platforms):** classic `docker build` from repo root — no BuildKit `--ignorefile` required. This is the path partners should use when `build-docker.sh` fails locally.

```bash
docker build \
  -f applications/attestation_demo/docker/Dockerfile \
  -t attestation-demo \
  .
```

**Expected:** build completes without errors; image tagged `attestation-demo` (or `attestation-demo:latest`).

**First build** may take several minutes (`uv sync` inside the image).

### Wrapper scripts

#### Linux / macOS / Git Bash

```bash
applications/attestation_demo/docker/build-docker.sh
```

Uses BuildKit with `--ignorefile` when available; **automatically falls back** to classic `docker build` if that fails (common on older Docker Desktop / buildx builds).

#### Windows (cmd or PowerShell)

```bat
applications\attestation_demo\docker\build-docker.bat
```

Uses classic `docker build` only (same as manual build above).

### BuildKit fallback (manual)

If you see errors mentioning `--ignorefile` or an unsupported buildx flag, run the **recommended** classic build at the top of this section.

Optional BuildKit path (when your Docker supports `--ignorefile`):

```bash
docker buildx build \
  -f applications/attestation_demo/docker/Dockerfile \
  --ignorefile applications/attestation_demo/docker/.dockerignore \
  -t attestation-demo \
  --load \
  .
```

If BuildKit is unavailable, copy the app ignore rules to repo root before classic build:

```bash
cp applications/attestation_demo/docker/.dockerignore .dockerignore
docker build -f applications/attestation_demo/docker/Dockerfile -t attestation-demo .
```

### Alternative: compose build

```bash
docker compose -f applications/attestation_demo/docker/docker-compose.yml build
```

---

## Step 3 — Run the container

### Option A — `docker run` (foreground)

```bash
docker run --rm \
  --name attestation-demo \
  --env-file applications/attestation_demo/.env \
  -p 8097:8097 \
  attestation-demo
```

Leave this terminal open. The API is available at `http://127.0.0.1:8097`.

### Option B — `docker compose` (foreground)

```bash
docker compose -f applications/attestation_demo/docker/docker-compose.yml up
```

### Option C — detached background

```bash
docker run -d \
  --name attestation-demo \
  --env-file applications/attestation_demo/.env \
  -p 8097:8097 \
  attestation-demo
```

Stop later: `docker stop attestation-demo && docker rm attestation-demo`

**Wait ~30–120 s** after start for the healthcheck / uvicorn to be ready.

---

## Step 4 — Health check (agent roster)

Base URL for all steps below: `http://127.0.0.1:8097`

### curl (bash)

```bash
curl -s http://127.0.0.1:8097/v1/attestation_demo/agents | jq .
```

### PowerShell

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8097/v1/attestation_demo/agents"
```

**Pass criteria:**

- HTTP **200**
- JSON contains `"agents"` array
- At least one agent with `"agent_id": "boundary_demo_agent"`

---

## Step 5 — Trigger the PoC run (primary endpoint)

### curl

```bash
curl -s -X POST "http://127.0.0.1:8097/v1/attestation_demo/poc/run" \
  -H "Content-Type: application/json" \
  -d @applications/attestation_demo/partner_handoff/poc_run_request.v1.json \
  | tee /tmp/poc_response.json
```

### PowerShell

```powershell
$body = Get-Content "applications/attestation_demo/partner_handoff/poc_run_request.v1.json" -Raw
$response = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8097/v1/attestation_demo/poc/run" `
  -ContentType "application/json" `
  -Body $body
$response | ConvertTo-Json -Depth 10
```

**Pass criteria:**

- HTTP **200**
- `"state": "completed"`
- `"agent_id": "boundary_demo_agent"`
- `"boundary_events"` is a non-empty array with **at least 2** elements
- Compare shape to [`partner_handoff/poc_run_response.v2.json`](../partner_handoff/poc_run_response.v2.json)

Save `run_id` from the response for Steps 7–8.

---

## Step 6 — Verify project assumptions (response checklist)

Inspect the JSON from Step 5. Every item below must pass.

### 6.1 Run outcome

| Field | Expected (EBE-9 default) |
|-------|--------------------------|
| `state` | `"completed"` |
| `agent_id` | `"boundary_demo_agent"` |
| `run_id` | non-empty string |
| `trust_model.platform_signed` | `"true"` |
| `trust_model.recommended_receipt_role` | `"host_attested"` |

> **Unsigned v2** (`host_signing_enabled=false`): `platform_signed` = `"false"`, `recommended_receipt_role` = `"client_observed"`.

### 6.2 Boundary events (PoC v2 — two claims per run)

Expect **two** events in `boundary_events[]`, ordered by `event_sequence`.

**Event 1 — `tool_execution` (`event_sequence: 1`)**

| Field | Expected (EBE-9 default) |
|-------|--------------------------|
| `schema_id` | `"execution_boundary_event.v1"` |
| `signed` | `true` |
| `host_attestation.schema_id` | `"host_attestation_envelope.v1"` |
| `host_attestation.context` | `"boundaryattest.host-attestation.v1"` |
| `host_attestation.public_key_id` | `"attestation-demo-host-1"` |
| `host_attestation.signature` | non-empty base64 |
| `boundary_type` | `"tool_execution"` |
| `event_id` | non-empty UUID (unique per event) |
| `tool_id` | `"records.put"` |
| `agent_id` | `"boundary_demo_agent"` |
| `action_status` | `"executed"` |
| `side_effects` | `true` |
| `step_id` | `"store_demo_record"` |
| `lineage.type` | `"execution_record"` |
| `lineage.ref` | contains `run_id` |
| `input.partition_key` | `"attestation_demo"` |
| `output.stored` | `true` |
| `input_hash` / `output_hash` | start with `sha256:` |

**Event 2 — `harness_step` (`event_sequence: 2`)**

| Field | Expected |
|-------|----------|
| `boundary_type` | `"harness_step"` |
| `action_status` | `"completed"` |
| `tool_id` | `null` |
| `policy_verdicts` | non-empty array (pre/post allow) |
| `step_outcome.status` | `"completed"` |
| `lineage.ref` | contains `run_id` and `:harness_step` |

**Failure path:** if storage fails, event 1 has `action_status: failed`; event 2 may still be `harness_step` / `completed` (separate claims). See [`partner_handoff/poc_run_response.failed.v2.json`](../partner_handoff/poc_run_response.failed.v2.json).

### 6.3 BoundaryAttest mapping readiness (partner-side)

After verifying `host_attestation` (see [`partner_handoff/EBE-9_HOST_SIGNING.md`](../partner_handoff/EBE-9_HOST_SIGNING.md)), these Intergrax fields must be present for the partner wrapper:

| Intergrax field | Partner use |
|-----------------|-------------|
| `event_id` | stable evidence / receipt key |
| `event_sequence` | ordering within run |
| `boundary_type` | distinguishes tool vs harness claim |
| `agent_id` | `agentId` |
| `tool_id` | `tool` (tool events only) |
| `action_status` | `actionStatus` |
| `input` / `output` | `input` / `output` |
| `lineage.ref` | `lineage.ref` |
| `host_attestation` | host signature verify (pinned pubkey) |
| *(partner sets)* | `receiptRole: "client_observed"` on separate wrapper |

Create **one receipt per** `boundary_events[]` element — not one composite receipt per run.

**Do not** label receipts as `server_attested` based on this PoC alone.

---

## Step 7 — Debug: buffered boundary events

Replace `{run_id}` with the value from Step 5.

```bash
curl -s "http://127.0.0.1:8097/v1/attestation_demo/poc/runs/{run_id}/boundary-events"
```

**Pass criteria:**

- HTTP **200**
- `"count"` equals length of `boundary_events` from Step 5
- Same `tool_id` and `signed: true` with non-null `host_attestation` in buffered events (EBE-9 default)

---

## Step 8 — Debug: HOS journal comparison

```bash
curl -s "http://127.0.0.1:8097/debug/tasks/{run_id}/trace"
```

**Pass criteria:**

- HTTP **200**
- `"run_id"` matches Step 5
- `"trace_events"` is a non-empty array
- Trace content references the demo run (`boundary_demo_agent`, `attestation.demo` capability, completed graph node, critic verdict, or task state)

**Scope note:** the HOS journal trace correlates at **run/task level**. It does not expose EBE `event_id`, `step_id`, or `tool_id`. For exact per-event correlation (receipt key, tool claim, harness claim), use `boundary_events[]` from Step 5 or Step 7 — not the trace alone.

This satisfies ARCHITECTURE §17 item 4 — partner can compare run-level journal facts with receipt grouping (`run_id`, `step_id`, `lineage.ref`).

---

## Step 9 — Optional: API key authentication

Skip if `INTERGRAX_HARNESS_API_KEY` is not set in `.env`.

1. Add to `applications/attestation_demo/.env`:

   ```env
   INTERGRAX_HARNESS_API_KEY=your-partner-secret
   ```

2. Restart the container (Step 3).

3. Request **without** key → expect **401**:

   ```bash
   curl -s -o /dev/null -w "%{http_code}" -X POST \
     "http://127.0.0.1:8097/v1/attestation_demo/poc/run" \
     -H "Content-Type: application/json" \
     -d '{"message":"x","capability":"attestation.demo"}'
   ```

4. Request **with** key → expect **200**:

   ```bash
   curl -s -X POST "http://127.0.0.1:8097/v1/attestation_demo/poc/run" \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: your-partner-secret" \
     -d @applications/attestation_demo/partner_handoff/poc_run_request.v1.json
   ```

---

## Step 10 — Automated verification (pytest)

From repository root (host need **not** be running — uses in-process `TestClient`):

```bash
uv run pytest applications/attestation_demo/tests -q
uv run pytest tests/unit/runtime/attestation/ -q
```

**Pass criteria:** all tests green (smoke, full event contract, optional auth).

---

## Step 11 — Cleanup

```bash
docker stop attestation-demo 2>/dev/null || true
docker rm attestation-demo 2>/dev/null || true
```

Or for compose:

```bash
docker compose -f applications/attestation_demo/docker/docker-compose.yml down
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Cannot connect to Docker daemon` | Docker Desktop not running | Start Docker Desktop; retry `docker version` |
| Connection refused on `:8097` | Container still starting | Wait 30–120 s; check `docker logs attestation-demo` |
| `boundary_events` empty | EBE profile not wired | Rebuild image from current `main`; run Step 10 pytest |
| HTTP 401 on `/poc/run` | API key required | Set `X-Api-Key` header or unset `INTERGRAX_HARNESS_API_KEY` for dev |
| Build fails on Windows path | Wrong build context | Run `docker build` from **repo root**, not `applications/attestation_demo` |
| `unknown flag: --ignorefile` or buildx error | Local BuildKit / buildx version | Use classic `docker build` (Step 2 — recommended); or `build-docker.bat` on Windows |
| `jq` not found | Optional formatter | Use `python -m json.tool` or PowerShell `ConvertTo-Json` |

---

## Sign-off checklist

Before sharing with a partner, confirm:

- [ ] Step 4 — agents list returns `boundary_demo_agent`
- [ ] Step 5 — PoC run returns `state: completed`
- [ ] Step 6 — both boundary events pass (tool + harness, `event_sequence` 1 and 2, `signed: true`)
- [ ] Step 6 — `host_attestation` verifies against golden vector pubkey (optional crypto check)
- [ ] Step 8 — debug trace available for `run_id`
- [ ] Step 10 — pytest green (recommended)
- [ ] `INTERGRAX_HARNESS_API_KEY` set if exposing publicly (Step 9)

Handoff package: [`partner_handoff/README.md`](../partner_handoff/README.md) · EBE-9 spec: [`partner_handoff/EBE-9_HOST_SIGNING.md`](../partner_handoff/EBE-9_HOST_SIGNING.md)
