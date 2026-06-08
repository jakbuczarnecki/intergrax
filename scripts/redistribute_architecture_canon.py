# © Artur Czarnecki. All rights reserved.
"""Redistribute PLATFORM §53 hardening into domain architecture docs (1:1 model)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "docs" / "architecture"

PLATFORM_SLIM_53 = """# 53. Harness Architecture Hardening Index

Post-U hardening topics are **owned by domain pairs** (architecture + plan), not this file.

| Topic | Architecture | Plan |
|-------|--------------|------|
| Capability graph | [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §19 | [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) |
| Agent lifecycle governance | [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §20 | same |
| Prompt registry | [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §17 | same |
| Registry snapshots / assembly | [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §18 | same |
| Context quality | [`MEMORY.md`](MEMORY.md) §19 | [`plan/MEMORY.md`](../plan/MEMORY.md) |
| Knowledge graph / hybrid retrieval | [`MEMORY.md`](MEMORY.md) §20 · [`INTEGRATIONS.md`](INTEGRATIONS.md) | [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md) |
| Evaluation operations | [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) §42 | [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| Architecture metrics / debt | [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) §43 | same |
| Security / tenant isolation | [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.45–§42.46 | [`plan/UNIFIED_EXECUTION_RUNTIME.md`](../plan/UNIFIED_EXECUTION_RUNTIME.md) |
| Cost governance | [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.47 | same |
| Identity / trust / tenancy | [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) §42.44 | same |
| Multi-agent coordination patterns | [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §27 | [`plan/NEXUS_EXECUTION_FLOW.md`](../plan/NEXUS_EXECUTION_FLOW.md) |
| Modality plane | [`MODALITY.md`](MODALITY.md) | [`plan/MODALITY.md`](../plan/MODALITY.md) |

**Harness-first lock (normative):**

```text
Harness -> Runtime -> Agents -> Applications -> Products
```

Business-agent work (K.1/K.2, product apps) remains deferred until explicit product reprioritization — see [`plan/PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md) §6.3.

**Phase V implementation baseline** (code modules, not duplicated here): `intergrax/runtime/architecture/` — capability graph, lifecycle, eval, security, cost, context/prompt quality, graph RAG helpers. Traceability: [`plan/PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md) Phase V / FAUDIT-32.

---
"""

APPEND: dict[str, str] = {
    "AGENT_CONTRACTS_AND_ASSEMBLY.md": """

---

# 17. Prompt Registry Architecture

Prompt artifacts are **governed platform assets**, not ad-hoc strings in agents.

## 17.1 Requirements

- ownership and versioning on every prompt id (`PromptMeta`),
- composable layers: system / task / policy / context,
- deterministic policy injection overlays,
- regression suites on golden prompt catalogs,
- Tier-3 `PromptProfile` selects YAML catalog path per host.

## 17.2 Code map

| Module | Role |
|--------|------|
| `intergrax/prompts/registry/` | YamlPromptRegistry, governance validation |
| `intergrax/runtime/architecture/prompt_registry_governance.py` | Ownership / risk tier gates |
| `intergrax/runtime/architecture/prompt_composition.py` | Layer composition |
| `intergrax/runtime/architecture/prompt_policy_overlay.py` | Policy overlays |
| `intergrax/runtime/architecture/prompt_regression_suite.py` | Golden regression |
| `intergrax/applications/_shared/prompt_wiring.py` | Environment → Nexus prompt registry |

**Authoring:** [`guides/AGENT_CREATION_GUIDE.md` Appendix M](../guides/AGENT_CREATION_GUIDE.md) · **Plan:** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Phase PE.

---

# 18. Registry Architecture

Registries are versioned, snapshot-capable catalogs — not mutable globals.

## 18.1 Registry types

| Registry | Tier | Consumed by |
|----------|------|-------------|
| Agent registry | 1 | Nexus agent selection |
| Tool registry | 0 | `ToolRuntime` |
| Skill registry | 0 | Skill resolver |
| Integration registry | 0 | Provider hosts |
| Prompt registry | 0/1 | Nexus steps, eval |
| Evaluation registry | 1 | EvalRunner, release gates |

## 18.2 Assembly pattern

Tier-3 `wire_application_environment()` materializes registries from `ApplicationEnvironmentProfile` tool/skill/integration/prompt profiles → `RuntimeConfig` via `runtime_config_bridge.py` and domain `*_assembly_resolver.py` modules.

Snapshots and conformance CI validate registry shape before release (`scripts/check_agents_lifecycle_metadata.py`, harness registry guards).

**Plan:** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Phase REG.

---

# 19. Capability Graph Architecture

Registries and capability layers MUST be represented as a typed dependency graph:

```text
Integration -> Tool -> Skill -> Policy -> Agent -> Application -> Product
```

## 19.1 Minimum requirements

- typed node and edge taxonomy,
- dependency lineage and provenance,
- blast-radius impact analysis for version/policy/runtime changes,
- compatibility validation on graph edges before release.

## 19.2 Code map

| Module | Role |
|--------|------|
| `runtime/architecture/capability_graph.py` | Core graph model |
| `capability_graph_lineage.py` | Lineage / provenance |
| `capability_graph_compatibility.py` | Edge compatibility |
| `capability_graph_applications.py` | Application slice |
| `scripts/phase_v_capability_graph_guard.py` | CI guard |

Nexus routes to **capabilities** (§16), not hardcoded class names. Graph edges MUST reflect manifest roster per application — not global cross-product shortcuts.

**Plan:** [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Phase CG.

---

# 20. Agent Lifecycle Governance

Beyond contract shape (§12) and registry metadata (§15):

| Stage | Requirement |
|-------|-------------|
| Certification | quality + policy + security gates before production |
| Promotion | dev → staging → production with evidence |
| Deprecation | migration windows, runtime filters for retired agents |
| Retirement | rollback/archive semantics |
| Ownership | explicit owner + escalation path |

**Code:** `runtime/architecture/agent_lifecycle_governance.py`, `agent_certification.py`, `agent_promotion.py`, `production_ownership.py`.

Runtime MUST reject or reroute retired/deprecated agents in production mode (V-REM-ALG.*). **Plan:** Phase AS + V-REM in [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md).

---
""",
    "TIER3_APPLICATION_ENVIRONMENT.md": """

---

# 22. Application Environment Profile (canonical)

Tier-3 hosts are configured through **`ApplicationEnvironmentProfile`** — a typed umbrella aggregating every harness control plane slice.

## 22.1 Profile composition

| Sub-profile | Purpose |
|-------------|---------|
| `IdentityProfile` | API key, tenant_required, service identities |
| `PolicyRulesProfile` + `ExecutionMode` | Declarative rules + STRICT/BALANCED/EXPLORATORY |
| `ApplicationSecurityProfile` | Per-app V-SEC toggles |
| `ToolProfile` / `SkillProfile` | Allowed catalogs |
| `IntegrationProfile` | Provider stack |
| `LLMProfile` / `ModalityProfile` | Model and modality posture |
| `ContextProfile` / `MemoryProfile` / `ContextDecisionProfile` | Assembly and stores |
| `PromptProfile` | YAML prompt catalog path |
| `ReliabilityProfile` | Idempotency, circuit breaker, checkpoint |
| `ObservabilityProfile` | Trace, OTEL, metrics plugins |
| `OrchestrationProfile` | Planner/classifier kinds, delegation depth |
| `ApplicationGraphSpec` | Declarative multi-agent topology |

**Contract:** `intergrax/applications/contracts/environment_profile.py`

## 22.2 Unified wiring entrypoints

```text
ApplicationManifest
    -> build ApplicationBuildContext
    -> wire_application_environment(ctx, profile)
    -> materialize_runtime_config(request, harness_ctx, env)
    -> build_nexus_loop_from_environment(...)
    -> UnifiedTaskRunner (§41)
```

| Module | Role |
|--------|------|
| `applications/_shared/environment_wiring.py` | Single wiring entry |
| `runtime_config_bridge.py` | Environment → `RuntimeConfig` |
| `nexus_factory.py` | NexusLoop from profile |
| `identity_wiring.py` | Host auth from `IdentityProfile` |
| `shadow_wiring.py` / `sandbox_wiring.py` | Isolated execution |
| `*_runtime_bridge.py` | Domain bridges (RAG, memory, policy, …) |

## 22.3 Interaction surfaces (intake)

Normalized intake MUST converge on the same Nexus lifecycle:

| Surface | Typical entry |
|---------|---------------|
| HTTP API | `applications/*/host/` FastAPI routers |
| CLI | `intergrax` CLI / lab commands |
| Slack / Teams | `POST /v1/interactions/intake` + adapters |
| Webhook / worker | `applications/_shared/task_intake.py`, queue consumers |
| Scheduler | `intergrax/queueing/` + long-running task API |

See [`ORCHESTRATION.md`](ORCHESTRATION.md) §48 for `TaskEnvelope` normalization.

## 22.4 Host migration rule

Every Tier-3 application MUST:

1. declare `environment` on `ApplicationManifest`,
2. wire through `wire_application_environment` (no ad-hoc `getattr` profile access),
3. keep business logic in Tier-2 agents — hosts only compose harness.

**Plan:** [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](../plan/TIER3_APPLICATION_ENVIRONMENT.md) Phase H-APP (43 tasks, Done).

## 22.5 Related documents

| Document | Relationship |
|----------|--------------|
| [`applications/USAGE.md`](../../applications/USAGE.md) | Authoring Tier-3 hosts |
| [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) | UAEP + policy runtime |
| [`ORCHESTRATION.md`](ORCHESTRATION.md) | Nexus orchestration fields on profile |
| [`guides/HARNESS_ENVIRONMENT.md`](../guides/HARNESS_ENVIRONMENT.md) | Lab stack operator guide |

---
""",
    "RELIABILITY_FAILURE_AND_HITL.md": """

---

# 33. Reliability Primitives

Reliability is enforced at **graph**, **run**, and **integration** layers.

## 33.1 Idempotency and deduplication

- Side-effectful tools SHOULD accept `idempotency_key` on `ToolRequest` (§42.12).
- Tier-3 `ReliabilityProfile` enables idempotency stores via integration `key_value_cache` backends.
- Duplicate intake deduplication uses stable task/run identifiers on `TaskEnvelope`.

## 33.2 Circuit breaker and timeouts

| Layer | Mechanism |
|-------|-----------|
| Integration calls | Circuit breaker on provider hosts; wired from `ReliabilityProfile` |
| LLM adapters | Retry/backoff profiles on `LLMProfile` |
| Graph steps | `RetryEngine` + `RetryPolicy` (§31.1) |
| UAEP run | `RuntimeConfig.max_run_retries` |

## 33.3 Checkpoint, resume, compensation

- `RuntimeCheckpoint` captures plan snapshot, graph snapshot, UAEP cursor (§42.9).
- HITL pause creates `PauseRecord`; resume restores checkpoint.
- Long-running tasks expose partial results API + scheduler hooks (§26 in [`ORCHESTRATION.md`](ORCHESTRATION.md)).

## 33.4 Error taxonomy (Harness)

| Class | Examples | Typical response |
|-------|----------|------------------|
| `UserError` | Invalid input, denied permission | Fail fast, no retry |
| `PolicyError` | Guardrail violation | DENY / REQUIRE_HUMAN |
| `DependencyError` | Provider down | Retry + circuit breaker |
| `RuntimeError` | Timeout, state corruption | Retry run or escalate |
| `QualityError` | Schema / rubric failure | Retry alternate agent or critic loop |

## 33.5 Code map

| Module | Role |
|--------|------|
| `runtime/nexus/retry/retry_engine.py` | Graph-level retry |
| `runtime/resilience/` | Circuit breaker helpers |
| `applications/_shared/reliability_wiring.py` | Profile → runtime |
| `runtime/sandbox/`, `runtime/shadow/` | Isolated risky execution |
| `runtime/human/` | HITL approval flow (§32, UAEP §42.10) |

**Plan:** [`plan/RELIABILITY_FAILURE_AND_HITL.md`](../plan/RELIABILITY_FAILURE_AND_HITL.md) Phase REL.

---
""",
    "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md": """

---

# 42. Evaluation and Benchmarking Operations

Evaluation is a **first-class runtime subsystem**, not a post-hoc script.

## 42.1 Modes

| Mode | Purpose |
|------|---------|
| Offline | Golden datasets, regression before merge |
| Online | Production sampling, score trends |
| Shadow | Compare candidate path without user impact |
| Human | HITL rubric scoring |

## 42.2 Components

| Module | Role |
|--------|------|
| `runtime/architecture/evaluation_modes.py` | Mode contracts |
| `evaluation_automation.py` | Runner automation |
| `evaluation_registry_trends.py` | Score history / trends |
| `online_evaluation_registry.py` | Live eval registry |
| `evaluation_assets.py` | Golden asset catalog |
| `runtime/eval/` | NexusEvalRunner integration |

Evaluators: rule-based, schema, LLM-judge (see [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md)).

**Plan:** [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) Phase EVAL · [`plan/CRITIC_VERIFICATION.md`](../plan/CRITIC_VERIFICATION.md) CRIT-V.

---

# 43. Architecture Metrics, Debt, and CI Gates

Architecture health MUST be measured, not inferred.

## 43.1 Metric families

- modularity and coupling indicators,
- dependency graph health and incompatibility rate,
- observability and governance coverage on critical paths,
- policy / context / prompt / test coverage,
- architecture debt index with trend tracking.

**Code:** `runtime/architecture/architecture_metrics.py`, `architecture_metrics_pipeline.py`, `debt_governance.py`, `architecture_coverage.py`, `maturity_gate_evidence.py`.

## 43.2 Developer experience surface

| Surface | Role |
|---------|------|
| `intergrax/scaffold/` | `new-agent`, `new-application`, `new-skill` |
| `intergrax/cli/doctor.py` | Harness health checks |
| `scripts/test.bat` / `pytest -m gate` | Mandatory merge gates |
| `guides/AGENT_CREATION_GUIDE.md` | Author workflow |

**TTFRun** (idea → first Nexus run) is the primary DX metric. **Plan:** Phase DX, AA, W-OPS in [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md).

## 43.3 Operational L3 evidence

Release cycles, SLO snapshots, and ops sign-off are tracked via `scripts/phase_w_ops_evidence.py` and release cycle artifacts under `build/architecture_hardening/`.

---
""",
    "ORCHESTRATION.md": """

---

# 48. Task Intake and TaskEnvelope

All entrypoints MUST normalize into a common intake contract before NexusLoop.

## 48.1 TaskEnvelope (minimum)

```text
TaskEnvelope:
    task_id / run_id
    tenant_id
    user_id | service_id
    source_channel          # api | cli | slack | teams | webhook | scheduler
    raw_input
    constraints             # SLA, risk class, budget caps
    correlation_ids         # trace_id parent
```

## 48.2 Intake pipeline

```text
Surface adapter -> contract validation -> TaskEnvelope -> TaskClassifier -> Planner
```

| Module | Role |
|--------|------|
| `applications/_shared/task_intake.py` | Shared intake helpers |
| `fastapi_core/` | HTTP auth + request context |
| `runtime/nexus/orchestration/` | Classifier, planner, graph |
| `runtime/interactions/` | Slack/Teams interaction adapters |

**Audit layer:** INTEGRAX_HARNESS_AUDIT_MAP §3 (Interface and Task Intake).

---

# 49. Scheduler and Queueing

Long-running and asynchronous work uses the Tier-0 queueing plane — not ad-hoc threads in agents.

## 49.1 Components

| Module | Role |
|--------|------|
| `intergrax/queueing/` | Task index, registry, worker contracts |
| `intergrax/distributed/` | Rate limiting, distributed locks |
| Integration `message_bus` providers | Celery, RabbitMQ, Redis, Kafka |

## 49.2 Orchestration integration

- `OrchestrationProfile.long_running` enables checkpointed schedules.
- Graph batch concurrency caps prevent provider overload.
- Backpressure and semaphore limits are policy-aware (see [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) §9).

**Plan:** [`plan/ORCHESTRATION.md`](../plan/ORCHESTRATION.md) Phase ORCH.

---
""",
}

# Appended via separate write for large files
UNIFIED_APPEND = """

---

## 42.44 Identity, Trust, and Tenancy

Every execution MUST carry identity, scope, and data boundaries (AUDIT_MAP §4).

### 42.44.1 Identity kinds

| Kind | Examples | Propagation |
|------|----------|-------------|
| User | Human operator, API user | `tenant_id`, roles → tool policy |
| Service | Tier-3 host, worker | `service_identities` on `IdentityProfile` |
| Agent | `agent_id` on contract | Scoped tool allow-list |

### 42.44.2 Tenancy rules

- `tenant_id` REQUIRED on trace events and policy evaluation for multi-tenant hosts.
- Subagents MUST NOT inherit unrestricted parent permissions — delegation contracts cap scope.
- Secrets ONLY via integration secrets backends — never in agent code or manifests.

### 42.44.3 Code map

| Module | Role |
|--------|------|
| `fastapi_core/auth/` | API key extraction, request context |
| `applications/_shared/identity_wiring.py` | Profile → host auth |
| `runtime/architecture/tenant_security.py` | Tenant isolation verification |
| `integrations/providers/.../identity_*` | Auth0, Keycloak, WorkOS hosts |
| `tools/providers/identity/` | `identity.*` tools |

Tier-3 declares posture in `IdentityProfile`; Tier-1 enforces on execution path. **Plan:** [`plan/UNIFIED_EXECUTION_RUNTIME.md`](../plan/UNIFIED_EXECUTION_RUNTIME.md) V-REM-SEC, SEC.

---

## 42.45 Security and Data Governance

Agent-native threats MUST have explicit defenses (AUDIT_MAP §23):

| Threat | Defense module |
|--------|----------------|
| Prompt injection | `prompt_security.py` |
| Tool injection | `tool_security.py` + middleware |
| Retrieval poisoning | `retrieval_security.py`, `retrieval_security_wiring.py` |
| Tenant isolation | `tenant_security.py` |
| Audit trail | Policy + trace on governance-critical actions |

`ApplicationSecurityProfile` (Tier-3) toggles defenses per host. Wiring MUST reach `ToolRuntime` and RAG retrieval path — not documentation-only.

**Authoring:** [`guides/AGENT_CREATION_GUIDE.md` Appendix S](../guides/AGENT_CREATION_GUIDE.md).

---

## 42.46 Cost and Resource Governance

Cost control MUST be enforceable at runtime (AUDIT_MAP §24):

- budget envelopes by tenant / application / agent / model / tool,
- token and tool quotas,
- forecast and anomaly signals,
- optimization recommendations under policy constraints.

| Module | Role |
|--------|------|
| `cost_budget.py` | Budget envelopes |
| `cost_quota.py` | Quotas |
| `cost_forecast.py` | Forecasting |
| `cost_optimization.py` | Optimization loops |

`RuntimePolicyBundle.budget` merges into Nexus and UAEP. Observability emits cost signals (see [`OBSERVABILITY.md`](OBSERVABILITY.md)).

**Plan:** [`plan/UNIFIED_EXECUTION_RUNTIME.md`](../plan/UNIFIED_EXECUTION_RUNTIME.md) Phase COST.

---
"""

MEMORY_APPEND = """

---

## 19. Context Quality Hardening

Context engineering MUST include explicit quality controls (not only token budgeting):

| Control | Mechanism |
|---------|-----------|
| Relevance / freshness / confidence | Scoring in context assembly |
| Duplicate suppression | Context noise controls |
| Regression benchmarks | `context_regression_benchmark.py` |
| Lineage | Traceable chain from output evidence → source |

**Code:** `runtime/architecture/context_engineering.py`, `retrieval_effectiveness.py`, `runtime/nexus/context/`.

**Authoring:** [`guides/AGENT_CREATION_GUIDE.md` Appendix L](../guides/AGENT_CREATION_GUIDE.md).

---

## 20. Knowledge Graph and Hybrid Retrieval

Graph-native knowledge evolves from optional enhancement to first-class capability:

- graph RAG support (`intergrax/rag/graph/`),
- entity–relation semantic modeling,
- hybrid retrieval: vector + keyword + graph traversal,
- graph-backed explainability in reasoning traces.

| Module | Role |
|--------|------|
| `runtime/architecture/graph_rag.py` | Graph RAG contracts |
| `hybrid_retrieval.py` | Hybrid strategy |
| `graph_provenance.py` | Lineage for graph edges |

**Distinction:** Graph RAG indexes **document knowledge** — not user episodic memory (§4). Integration backends (Neo4j, etc.) are catalog providers in [`INTEGRATIONS.md`](INTEGRATIONS.md).

---
"""

NEXUS_APPEND = """

---

## 27. Multi-Agent Coordination Model Catalog

Pattern selection MUST be explicit and policy-aware (IDEAL §6.4, AUDIT_MAP §10).

| Pattern | When to use |
|---------|-------------|
| Hierarchical | Top-down plan with delegated executors |
| Orchestrator–worker | Central planner, specialized workers |
| Supervisor–worker | Quality/policy supervision over workers |
| Peer-to-peer | Parallel decomposition with merge policy |
| Evaluator-loop / critique–revise | Quality gate before finalize |

**Code:** `runtime/architecture/multi_agent_coordination.py`, `multi_agent_acceptance.py`.

Narrative flows: §12–§14 in this document. Critic depth: [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md).

---
"""


def slim_platform() -> None:
    path = ARCH / "PLATFORM_FOUNDATION.md"
    text = path.read_text(encoding="utf-8")
    pattern = r"# 53\. Harness Architecture Hardening Addendum \(Post-U\).*"
    if not re.search(pattern, text, flags=re.DOTALL):
        print("PLATFORM §53 pattern not found — skip slim")
        return
    text = re.sub(pattern, PLATFORM_SLIM_53.strip() + "\n", text, flags=re.DOTALL)
    path.write_text(text, encoding="utf-8")
    print("slimmed PLATFORM_FOUNDATION §53")


def append_if_missing(path: Path, marker: str, content: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"skip {path.name} ({marker} exists)")
        return
    path.write_text(text.rstrip() + content, encoding="utf-8")
    print(f"updated {path.name}")


def main() -> None:
    slim_platform()
    for name, content in APPEND.items():
        append_if_missing(ARCH / name, content.strip().split("\n")[2], content)
    append_if_missing(ARCH / "UNIFIED_EXECUTION_RUNTIME.md", "§42.44 Identity", UNIFIED_APPEND)
    append_if_missing(ARCH / "MEMORY.md", "## 19. Context Quality", MEMORY_APPEND)
    append_if_missing(ARCH / "NEXUS_EXECUTION_FLOW.md", "## 27. Multi-Agent", NEXUS_APPEND)
    print("redistribute_architecture_canon: done")


if __name__ == "__main__":
    main()
