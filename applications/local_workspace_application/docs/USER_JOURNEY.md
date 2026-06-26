# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

# Local Knowledge Workspace (LKW) — User Journey

**Status:** target onboarding narrative for the final LKW experience  
**Scope:** from GitHub discovery to first useful local workspace run  
**Related:** [`README.md`](../README.md) · [`ARCHITECTURE.md`](ARCHITECTURE.md) · [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) · [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md)

---

## 1. What the user should understand first

A new user lands on the Intergrax repository and should understand this in the first minute:

**Intergrax** is the platform / harness.

**LKW** is the first product-grade proof application built on that platform.

LKW gives the user a local assistant over their own files:

- it runs on the user's machine;
- it reads only allowed folders or explicitly provided files;
- it indexes documents into a local knowledge workspace;
- it answers questions with evidence from those files;
- it writes generated drafts only into a shadow workspace;
- it exposes what happened through trace/evidence inspection;
- it proves that Intergrax can create, configure, run, package, deploy, and observe real agent applications.

The first expected mental model:

```text
User files
  -> local LKW backend
  -> Intergrax Nexus
  -> local_indexer / local_search / local_synthesizer agents
  -> RAG + shadow workspace
  -> answer, draft, and trace/evidence
```

---

## 2. Target user paths

LKW should support three entry paths.

| User type | Goal | Entry path |
|-----------|------|------------|
| Product evaluator | See that LKW solves a real local knowledge problem | README -> LKW quickstart -> index fixture -> search -> synthesize |
| Platform evaluator | See that Intergrax can produce repeatable agent applications | README -> LKW platform proof loop -> scaffold/build/deploy docs |
| Developer contributor | Extend or improve LKW / Intergrax | architecture -> implementation plan -> one wave -> tests -> platform propagation |

The first user path is the most important. If a new user cannot reach a useful local run quickly, LKW is not ready.

---

## 3. Final GitHub discovery flow

When a new user opens GitHub, the intended flow is:

```text
1. Open repository README.
2. Understand: Intergrax is the platform; LKW is the first proof application.
3. Open applications/local_workspace_application/README.md.
4. Read what LKW does and what it does not do.
5. Choose a run mode: local dev, Docker, or packaged local daemon.
6. Copy .env.example to .env.
7. Start the LKW host.
8. Check /health and /agents.
9. Index a sample document or allowed local folder.
10. Ask a question against indexed content.
11. Generate a draft into the shadow workspace.
12. Inspect trace/evidence for the run.
13. Optional: review how the same pattern is propagated back into platform/scaffold/deploy templates.
```

---

## 4. Step-by-step final local run

### Step 1 — Clone the repository

```bash
git clone <repo-url>
cd intergrax
```

### Step 2 — Install dependencies

Final target:

```bash
uv sync --extra lkw
```

Current development fallback:

```bash
uv sync
```

The final product should not require every heavy experimental dependency for the first LKW run. If the current dependency model still requires a full install, that is a platform packaging gap to close through `pyproject.toml` and optional dependency groups.

### Step 3 — Open the LKW application folder

```bash
cd applications/local_workspace_application
```

The user should see:

```text
README.md
.env.example
docs/
docker/
host/
serving/
```

### Step 4 — Create local environment config

```bash
cp .env.example .env
```

Minimum expected variables:

```text
INTERGRAX_ENV=dev
LOCAL_WORKSPACE_BACKEND_ENV=dev
LOCAL_WORKSPACE_BACKEND_HOST=127.0.0.1
LOCAL_WORKSPACE_BACKEND_PORT=8020
LOCAL_WORKSPACE_ROUTE_PREFIX=/v1/local_workspace
LOCAL_WORKSPACE_INCLUDE_MCP=true
LOCAL_WORKSPACE_ENABLE_RAG=true
LOCAL_WORKSPACE_ENABLE_RAG_INGEST=true
```

Final product rule:

- dev mode may be permissive when explicitly configured;
- prod mode must fail closed when required security/policy/auth settings are missing;
- `.env.example` must match `host/settings.py` and generated application scaffold behavior.

### Step 5 — Choose data access

The user chooses one of two safe input modes.

#### Explicit files for a single run

The user sends `metadata.source_paths` in the request.

This is the first LKW.1 path.

#### Allowed folders

The user configures allowed read roots:

```text
INTERGRAX_ALLOWED_READ_ROOTS=/path/to/docs,/path/to/projects
```

LKW may read from those folders. It must not write into them.

### Step 6 — Start the local backend

From repository root:

```bash
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020
```

Alternative module run:

```bash
uv run python -m local_workspace_application.host.main
```

The host is the product runtime boundary:

```text
LKW backend
  -> FastAPI HTTP routes
  -> optional MCP endpoint
  -> NexusLoop
  -> local agents
  -> tools, RAG, memory, policy, trace
```

Thin clients must not contain RAG, LLM, or agent-loop logic.

### Step 7 — Verify the host

```bash
curl -s http://127.0.0.1:8020/health
curl -s http://127.0.0.1:8020/v1/local_workspace/agents
```

Expected result:

- `/health` confirms that the host is running;
- `/agents` shows the local agent roster;
- the user can see at least `local_indexer`, `local_search`, and `local_synthesizer` or their declared capabilities.

### Step 8 — Index a document

Target request:

```bash
curl -s -X POST http://127.0.0.1:8020/v1/local_workspace/run \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Index this project document",
    "capability": "local.workspace.index",
    "metadata": {
      "source_paths": ["/path/to/document.pdf"],
      "collection_id": "my_workspace"
    }
  }'
```

Expected behavior:

```text
Request
  -> LocalIndexerAgent
  -> path validation
  -> rag.ingest_document
  -> parsed chunks
  -> local vector index
  -> trace/evidence event
  -> completed result
```

The user should not need to understand internal RAG wiring to use this.

### Step 9 — Search indexed content

```bash
curl -s -X POST http://127.0.0.1:8020/v1/local_workspace/run \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What does the document say about project risks?",
    "capability": "local.workspace.search",
    "metadata": {
      "collection_id": "my_workspace"
    }
  }'
```

Expected behavior:

```text
Request
  -> LocalSearchAgent
  -> rag.retrieve
  -> evidence package
  -> grounded answer
  -> source references
  -> trace/evidence event
```

The answer should explain what was found and where it came from.

### Step 10 — Generate a draft into the shadow workspace

```bash
curl -s -X POST http://127.0.0.1:8020/v1/local_workspace/run \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Prepare a short report based on the retrieved project risks.",
    "capability": "local.workspace.synthesize",
    "metadata": {
      "collection_id": "my_workspace",
      "shadow_workspace": true,
      "output_name": "project-risk-report.md"
    }
  }'
```

Expected behavior:

```text
Request
  -> LocalSynthesizerAgent
  -> retrieved context / memory / prior evidence
  -> draft generation
  -> workspace.write_file
  -> artifact under shadow workspace
  -> trace/evidence event
```

LKW must not modify original user files.

### Step 11 — Inspect what happened

The user should be able to inspect a run without reading internal code.

Minimum inspection fields:

- submitted task and capability;
- task id and run id;
- selected agent;
- step sequence;
- invoked tools and outcomes;
- policy decisions;
- RAG ingest/retrieve evidence;
- shadow workspace artifact path;
- terminal outcome;
- diagnostics from non-fatal lifecycle/finalization failures.

Final target:

```bash
uv run intergrax trace show <run-id>
uv run intergrax trace export <run-id>
```

or an equivalent LKW debug endpoint / UI surface.

### Step 12 — Continue through MCP or another thin client

Once HTTP works, the same backend can be controlled through MCP:

```text
http://127.0.0.1:8020/mcp
```

MCP is a control surface only. It must call the same LKW backend and the same Nexus task path.

Later product surfaces may include:

- tray app;
- Slack slash command;
- local CLI;
- IDE/Cursor integration.

They must remain thin clients.

---

## 5. Final Docker path

The final Docker path should allow a user to verify LKW without understanding Python packaging internals.

### Build

```bash
applications/local_workspace_application/docker/build-docker.sh
```

or:

```bash
docker buildx build -f applications/local_workspace_application/docker/Dockerfile \
  --ignorefile applications/local_workspace_application/docker/.dockerignore \
  -t local_workspace-application .
```

### Run

```bash
docker run --rm \
  --env-file applications/local_workspace_application/.env \
  -e INTERGRAX_ENV=prod \
  -e LOCAL_WORKSPACE_BACKEND_HOST=0.0.0.0 \
  -e LOCAL_WORKSPACE_BACKEND_PORT=8020 \
  -p 8020:8020 \
  local_workspace-application
```

### Verify

```bash
curl -s http://127.0.0.1:8020/health
curl -s http://127.0.0.1:8020/v1/local_workspace/agents
```

If Docker requires special dependency workarounds, copied files, or modified uv settings, those are platform lessons and must be reflected in scaffold Docker templates.

---

## 6. Final packaged local daemon path

For an end-user oriented release, the target flow becomes:

```text
1. Download/install LKW package.
2. Start local daemon.
3. Open tray or local web UI.
4. Pick folders to allow for indexing.
5. Click "Index".
6. Ask questions.
7. Generate reports/drafts.
8. Review source evidence and generated artifacts.
9. Stop or uninstall without losing original files.
```

The daemon owns:

- local API;
- Nexus runtime;
- agent registry;
- RAG index;
- shadow workspace;
- trace/evidence store;
- policy and auth boundaries.

The tray/UI owns only:

- folder selection;
- command entry;
- status display;
- opening generated artifacts;
- links to trace/evidence.

---

## 7. What must be obvious to a new user

The final GitHub/project experience should make these points obvious:

1. **What problem LKW solves:** local knowledge over user-controlled files.
2. **What is safe:** read allowlist, shadow-only writes, local-first execution.
3. **How to run it:** local dev, Docker, later packaged daemon.
4. **How to prove it works:** index, search, synthesize, inspect trace.
5. **How it proves Intergrax:** same platform/scaffold/deploy loop should support the next application.
6. **What is not ready yet:** anything not validated by live LKW proof should not be marketed as production-proven.

---

## 8. First-run success criteria

A new technical user should be successful when they can complete this sequence:

```text
clone repository
  -> install dependencies
  -> configure .env
  -> start LKW host
  -> GET /health
  -> GET /agents
  -> POST /run local.workspace.index
  -> POST /run local.workspace.search
  -> POST /run local.workspace.synthesize
  -> find generated artifact in shadow workspace
  -> inspect trace/evidence
```

This is the minimum LKW product proof.

---

## 9. Platform proof success criteria

LKW proves the platform only when the above user path also produces reusable platform outcomes:

- app settings model is reusable by generated product applications;
- env template is aligned with settings validation;
- Docker template can build and run the app host;
- CI can verify the app proof path or at least the stable smoke subset;
- scaffolded applications inherit improved patterns;
- scaffolded agents inherit improved contract/test/doc patterns;
- build/deploy docs describe a real verified path;
- trace/evidence inspection is not LKW-specific when the pattern is generic.

If these are not true, then LKW works, but the platform proof is incomplete.

---

## 10. Final user story

A final user should be able to say:

> I found Intergrax on GitHub. I understood that LKW is the first local knowledge product built on the platform. I cloned the repo, started the LKW backend, allowed a folder or passed a document path, indexed my files, asked questions, generated a draft into a safe shadow workspace, and inspected the trace showing which agent and tools were used. I can also see that the same patterns are part of the platform and scaffold, so the next Intergrax application should not have to reinvent this setup.
