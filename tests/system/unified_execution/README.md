# UE-11G-C1 - Real Agentic Production E2E

Platform-native Docker certification for `scenario.real_agentic` using the production
Local Workspace HTTP application and `local.workspace.search`.

## One command

From repository root:

```bash
./tests/system/unified_execution/run_c1_proof.sh
```

Windows PowerShell:

```powershell
./tests/system/unified_execution/run_c1_proof.ps1
```

Direct compose (after runtime-context materialization):

```bash
docker compose -f tests/system/unified_execution/docker-compose.yml up --build --exit-code-from proof-runner
```

Teardown:

```bash
docker compose -f tests/system/unified_execution/docker-compose.yml down -v
```

## Production path

```text
HTTP client (proof-runner)
  → POST /v1/local_workspace/run (X-API-Key auth)
  → LocalWorkspaceTaskExecutor
  → HostTaskExecution.execute
  → Execution.execute
  → StrategyExecutionRouter (AGENTIC)
  → local_search production agent
  → rag.retrieve + Ollama embeddings
  → persisted OTLP runtime evidence
```

## Artifacts

Proof JSON report:

`.tmp/session/ue-11g-c1/docker-run/proof-report.json`
