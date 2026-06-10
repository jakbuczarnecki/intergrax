# © Artur Czarnecki. All rights reserved.
"""Generate docs/guides/audit/<DOMAIN>.md prompt files. Idempotent."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "guides" / "audit"

DOMAINS: list[dict[str, str | list[str]]] = [
    {
        "id": "PLATFORM_FOUNDATION",
        "title": "Platform Foundation",
        "layers": "1–2, 32",
        "code": "docs/, AGENTS.md, .cursor/rules/, tier boundaries across repo",
        "mission": "Verify Intergrax remains a Harness AI / Agent OS — durable runtime, replaceable agents — with correct four-tier model, documentation governance, and strategic alignment to the ideal architecture.",
        "dimensions": [
            "Harness vs agent prioritization — product logic not in Nexus; agents not hard-wiring platform internals.",
            "Four-tier dependency rules enforced in code imports (`intergrax/` ↔ `agents/` ↔ `applications/`).",
            "Documentation model: 21 domain pairs 1:1, hub-only root, no monolithic plan files.",
            "Strategic principles in canon match implementation reality (policy-first, trace-everything, composable-by-default).",
            "Gate maintenance workflow and PLATFORM_FOUNDATION ladder — plan rows match evidence.",
            "Architecture governance loop — audits update paired docs, ADRs, plan registers.",
        ],
        "scale": "N/A — meta-layer; sample multiple tiers for boundary violations.",
        "overrides": "Tier placement rules for new components; scaffold defaults; extension author boundaries.",
        "appendix": "N/A",
    },
    {
        "id": "UNIFIED_EXECUTION_RUNTIME",
        "title": "Unified Execution Runtime (UAEP)",
        "layers": "4–5, 8, 23–24",
        "code": "intergrax/runtime/nexus/, intergrax/harness/, policy engine, UAEP steps, identity/cost hooks",
        "mission": "Audit the Agent OS execution substrate: policy-first UAEP, identity/trust propagation, security and cost governance on every runtime path.",
        "dimensions": [
            "Single canonical path: UnifiedTaskRunner → NexusLoop → AgentEngine → UAEP steps.",
            "PolicyEngine coverage: pre-run, pre-plan, pre-LLM, pre-tool, post-tool, pre-output, memory writes.",
            "RuntimePolicyBundle completeness — no bypass routes for catalog tools, RAG, memory, delegation.",
            "Identity, tenant, and permission context on every Run/Step/ToolInvocation.",
            "Security profile: data classification, redaction, guardrail middleware integration.",
            "Cost governance: token/cost metering, budgets, throttles, per-tenant accounting.",
            "Checkpoint, pause/resume, interrupt semantics — recoverable state.",
            "Forbidden patterns absent: agent-specific Nexus branches, duplicate policy engines.",
        ],
        "scale": "Concurrent runs, budget exhaustion, policy denial storms, large delegation trees.",
        "overrides": "ApplicationEnvironmentProfile policy bundle, security profile, execution mode (strict/balanced/exploratory).",
        "appendix": "Appendix H (governance control plane)",
    },
    {
        "id": "ORCHESTRATION",
        "title": "Orchestration",
        "layers": "3, 9",
        "code": "intergrax/runtime/nexus/orchestration/, intake, scheduler, ExecutionGraph, GraphExecutor",
        "mission": "Audit intake normalization, scheduling, graph execution, parallelism, merge policies, and resilience — as formal runtime responsibilities, not agent code.",
        "dimensions": [
            "TaskEnvelope / intake convergence — API, CLI, worker, queue paths equivalent.",
            "ExecutionGraph typed nodes/edges, observable execution, deterministic merge.",
            "Scheduler: priority, concurrency caps, backpressure, retry budgets per step.",
            "Fan-out/fan-in, batch parallelism, graph strategies (single-agent, orchestrator-worker, evaluator loop).",
            "Planning strategy catalog wired to runtime — not ad-hoc agent loops.",
            "Configuration completeness: Tier-3 profile fields map to runtime (CFG-* rows).",
            "Execution-surface parity across postures (sync/async/streaming).",
        ],
        "scale": "Deep graphs, wide fan-out, scheduler under load, stuck-node recovery.",
        "overrides": "OrchestrationProfile, graph spec from Tier-3, hook plugins, planning strategy selection.",
        "appendix": "Appendix I (orchestration control plane)",
    },
    {
        "id": "NEXUS_EXECUTION_FLOW",
        "title": "Nexus Execution Flow",
        "layers": "8–10",
        "code": "intergrax/runtime/nexus/runtime_steps/, NexusLoop, delegation, subagent runners",
        "mission": "Audit the end-to-end Nexus loop narrative: step ordering, subagent delegation, coordination, and flow-level observability against the execution flow canon.",
        "dimensions": [
            "Runtime step pipeline matches architecture narrative (intake → plan → context → act → verify → complete).",
            "Each step: typed inputs/outputs, policy hooks, trace events, failure classification.",
            "Subagent delegation: SubtaskContract, scoped tools/memory, parent policy retention, merge policy.",
            "Multi-agent coordination patterns (§27) — isolation, trace lineage, budget split.",
            "No duplicate execution engines; legacy paths removed or gated.",
            "Flow gaps in plan (FLOW-*) resolved in code or explicitly deferred with risk.",
        ],
        "scale": "Long-running loops, nested delegation depth, parallel subagents.",
        "overrides": "Runtime step hooks, custom runners at Tier-3, delegation profiles.",
        "appendix": "Appendix I §I.6 (delegation)",
    },
    {
        "id": "AGENT_CONTRACTS_AND_ASSEMBLY",
        "title": "Agent Contracts and Assembly",
        "layers": "17–20, 31",
        "code": "intergrax/agents/, agents/, registry, prompt registry, capability graph",
        "mission": "Audit agent contracts, registry resolution, prompt assembly, capability graph, and agent lifecycle governance.",
        "dimensions": [
            "AgentContract: capabilities, allowed_tools, skill_ids resolution, risk metadata.",
            "Registry: host resolution, version pinning, discovery — CI wiring scripts green.",
            "Prompt registry: YamlPromptRegistry, PromptProfile, layered compilation (system/task/policy/context).",
            "Capability graph: agent ↔ tool ↔ skill edges, policy-aware traversal.",
            "Agent assembly path — no vendor SDKs in Tier-2; composition from Tier-0 only.",
            "Lifecycle: scaffold → register → evaluate → deprecate; governance documented.",
        ],
        "scale": "Large agent rosters, many capabilities, prompt template fan-out.",
        "overrides": "Per-agent manifest, host registry overrides, custom PromptRegistryProtocol impl at Tier-3.",
        "appendix": "Appendix M (prompt), Appendix N/O/P (assembly, registry, capability graph)",
    },
    {
        "id": "INTEGRATIONS",
        "title": "Integration Library",
        "layers": "13",
        "code": "intergrax/integrations/, IntegrationProfile, integration_runtime_bridge",
        "mission": "Audit the Integration Library as the sole vendor boundary — adapters, health probes, profiles, and bridge wiring to tools/RAG/LLM.",
        "dimensions": [
            "No direct vendor SDK imports in agents or Nexus business logic.",
            "Integration slugs, contracts, and capability metadata complete per adapter family.",
            "IntegrationProfile drives backend selection at Tier-3 — wired through bridges.",
            "Health probes and circuit behaviour for external dependencies.",
            "Secrets and credentials via integration layer — not hardcoded.",
            "Guardrail integrations (llm_guardrail) as policy extension, not parallel tier.",
            "Test coverage and CI checks for vendor import boundaries.",
        ],
        "scale": "Many backends, failover between providers, rate limits, bulk operations.",
        "overrides": "IntegrationProfile per host, custom IntegrationAdapter plugins (EXTENSION_AUTHOR_GUIDE).",
        "appendix": "Appendix K (integration control plane)",
    },
    {
        "id": "RAG",
        "title": "RAG and Retrieval Engine",
        "layers": "14",
        "code": "intergrax/rag/, RetrievalService, IngestPipeline, rag.* catalog tools, RagStep",
        "mission": "Deep audit of the Tier-0 retrieval engine: ingest, chunking, indexing, retrieval modes, resilience, and production posture vs state-of-the-art RAG systems.",
        "dimensions": [
            "Single canonical path: RagProfile → RetrievalService → catalog tools / Nexus RagStep.",
            "No agent direct vectorstore.query shortcuts.",
            "Retrieval modes: vector, keyword, hybrid, fusion, graph, rerank, agentic, hierarchical — wired vs documented-only.",
            "Ingest: parser catalog, chunking strategies, contextual enrich, dual-index / TOC for large docs.",
            "Strategy selection: explicit Tier-3 policy vs autonomous (AHI deferred) — dead config flagged.",
            "Short/medium vs multi-GB corpus behaviour — sync vs job orchestration.",
            "Resilience: retry, fallback chains, circuit breakers on embedding/retriever paths.",
            "Security: retrieval poisoning defence on **all** surfaces (Nexus + catalog tools).",
            "Citations, tenant MetadataFilter, multi-tenant isolation with prod backends.",
            "Observability: RetrievalTrace, parser trace, metrics, OTel on hot paths.",
            "Golden retrieval tests, recall/MRR eval harness, load/soak gaps.",
        ],
        "scale": "Single-page doc, 100-page PDF, book-scale corpus, high QPS retrieve, poisoned chunks.",
        "overrides": "RagProfile, IntegrationProfile vector_store/document_parser/rerank_provider, rag_runtime_bridge.",
        "appendix": "Appendix K §K.5",
    },
    {
        "id": "TOOLS",
        "title": "Tool Library and ToolRuntime",
        "layers": "11",
        "code": "intergrax/tools/, ToolRuntime, RuntimeToolInvoker, tool_planning_service, tool_selection",
        "mission": "Audit the Tool Library and Nexus ToolRuntime: catalog completeness, execution pipeline, selection/planning, policy, and production tool-engine posture.",
        "dimensions": [
            "Every tool: tool_id, schemas, risk level, idempotency hints, observability tags.",
            "Single execution path: ToolRuntime → policy → invoker → integration/RAG/sandbox.",
            "Tool selection and planning: strategies, constraints, budgets — not boolean flags.",
            "Plugin model: ToolPlugin, entry points, MCP export parity with OpenAI schema.",
            "Concurrency, timeout, retry, idempotency keys on invocation.",
            "Tool audit signals (ops:tool_audit) and trace taxonomy.",
            "No duplicate tool mechanisms; legacy tool_gateway paths removed.",
            "Catalog scale: bundle registration, bootstrap_catalogs, Tier-3 ToolProfile wiring.",
        ],
        "scale": "Large tool catalogs, parallel invocations, slow tools, tool-plan combinatorics.",
        "overrides": "ToolProfile, allowed_tools per agent, external tool plugins, custom invoker hooks.",
        "appendix": "Appendix J",
    },
    {
        "id": "CODE_CRAFT",
        "title": "Ephemeral Code Craft",
        "layers": "11b",
        "code": "intergrax/codecraft/, sandbox tools, codegen loop",
        "mission": "Audit the dynamic code-generation loop: sandbox isolation, execution governance, observability, and safe failure for generated code paths.",
        "dimensions": [
            "Codegen loop contract: plan → generate → execute → verify — bounded iterations.",
            "Sandbox tiers: local workspace vs container vs cloud — risk alignment.",
            "Policy and permission gates before code execution.",
            "Output validation and artifact handling.",
            "Trace of generated code, execution results, and failures.",
            "No arbitrary code execution bypassing ToolRuntime/policy.",
        ],
        "scale": "Large generated artifacts, long-running sandbox jobs, concurrent codegen sessions.",
        "overrides": "Sandbox profile, execution backend via IntegrationProfile, Tier-3 risk posture.",
        "appendix": "Appendix J (tool surfaces)",
    },
    {
        "id": "SKILLS",
        "title": "Skill Library",
        "layers": "12",
        "code": "intergrax/skills/, SkillResolverProtocol, skill→tool resolution",
        "mission": "Audit skills as composable capability bundles above tools — resolution, policy, registration, and agent consumption paths.",
        "dimensions": [
            "SkillContract: skill_id, required tools, schemas, risk inheritance.",
            "SkillResolver resolves skill_ids → allowed_tools with policy checks.",
            "Single registration/bootstrap path; no agent self-registration.",
            "Skill plugins and scaffold alignment.",
            "Observability on skill resolution and invocation chain.",
            "Clear separation: skill (composition) vs tool (atomic operation).",
        ],
        "scale": "Deep skill graphs, many skills per agent, resolution caching.",
        "overrides": "SkillProfile, custom SkillResolverProtocol at Tier-3.",
        "appendix": "Appendix J",
    },
    {
        "id": "LLM_ADAPTERS",
        "title": "LLM Adapters",
        "layers": "6",
        "code": "intergrax/llm_adapters/, LLMAdapter, LLMProfile, streaming, structured output",
        "mission": "Audit LLM abstraction: provider replaceability, response envelopes, routing, metering, retries, guardrails, and structured output validation.",
        "dimensions": [
            "All LLM calls through LLMAdapter — no direct OpenAI/Anthropic/Gemini SDK in agents/runtime business code.",
            "LLMAdapterResponse / LLMStructuredResult typed envelopes on all completion paths.",
            "LLMProfile: model selection by cost/latency/quality/risk/capability.",
            "Streaming events (LLMStreamEvent) parity with non-streaming.",
            "Token/cost usage metering per call, aggregated per run/tenant.",
            "Retries, fallbacks, timeout, rate-limit handling.",
            "Structured output schema validation — not manual JSON parse.",
            "Guardrail middleware integration (AFTER_LLM_OUTPUT).",
        ],
        "scale": "High token volume, long contexts, tool-call heavy turns, provider failover.",
        "overrides": "LLMProfile per host/agent step, provider plugins, model routing policy.",
        "appendix": "N/A",
    },
    {
        "id": "MEMORY",
        "title": "Memory and Context Engineering",
        "layers": "15–16",
        "code": "intergrax/memory/, ContextManager, context_runtime_bridge, MemoryView, consolidation",
        "mission": "Audit memory stores, scopes, lifecycle, context assembly, budgets, and Knowledge vs LTM boundary — explicit, governed, observable.",
        "dimensions": [
            "Memory type separation: STM, task, session, user LTM, tenant, procedural, shared context.",
            "Every read/write scoped; namespace isolation between runs/subagents.",
            "MemoryWritePolicy and BEFORE_MEMORY_WRITE hooks enforced.",
            "Context assembly: ContextProfile, ContextBudgetPolicy, provenance on fragments.",
            "Retrieval-first for large history — not full dumps into prompt.",
            "Knowledge (RAG) vs agent memory boundary — graph RAG ≠ user memory.",
            "Retention, TTL, forget/delete mechanisms.",
            "No direct DB access from agents.",
        ],
        "scale": "Long sessions, large LTM corpora, tight token budgets, multi-agent shared context.",
        "overrides": "ContextProfile, MemoryProfile, context_runtime_bridge, custom store backends at Tier-3.",
        "appendix": "Appendix L",
    },
    {
        "id": "MODALITY",
        "title": "Modality (Vision, Audio, ML)",
        "layers": "29",
        "code": "intergrax/modality/, vision/audio adapters, modality tools",
        "mission": "Audit multimodal planes: vision, audio, dedicated ML — as Tier-0 capabilities consumed through policy and tools, not agent SDK bypass.",
        "dimensions": [
            "Modality operations exposed as tools/skills — consistent with ToolRuntime.",
            "Adapter abstraction for vision/audio providers.",
            "Payload size limits, streaming where required, timeout handling.",
            "Policy classification for sensitive media.",
            "Trace and redaction for binary/media metadata.",
            "Integration with context assembly — not ad-hoc inline blobs in agents.",
        ],
        "scale": "Large images, long audio, batch media processing.",
        "overrides": "Modality profiles, integration backends, Tier-3 enablement flags.",
        "appendix": "N/A",
    },
    {
        "id": "OBSERVABILITY",
        "title": "Observability Spine",
        "layers": "21, 30",
        "code": "intergrax/observability/, journal, trace bus, OTel exporters, event catalog",
        "mission": "Audit the Harness Observability Spine: one bus, causal trees, complete catalog emission, redaction, and operator reconstructability of every run.",
        "dimensions": [
            "Single observability spine — no per-agent private trace DBs.",
            "Event catalog completeness for intake → plan → context → tools → LLM → policy → terminal.",
            "trace_id/run_id/tenant_id propagation across async boundaries.",
            "Redaction policy enforced in production profiles.",
            "OTel export optional but canonical journal always populated.",
            "ops:* signals, metrics, SLI hooks for SLO monitoring.",
            "CI: check_harness_observability_wiring.py and related gates.",
        ],
        "scale": "High event volume, long runs, nested subagents, export backpressure.",
        "overrides": "ObservabilityProfile, sink configuration, optional external APM — not parallel event models.",
        "appendix": "Appendix H (observability mandatory vs optional)",
    },
    {
        "id": "RELIABILITY_FAILURE_AND_HITL",
        "title": "Reliability, Failure Model, and HITL",
        "layers": "22",
        "code": "retry policies, failure taxonomy, HITL runners, ReliabilityProfile",
        "mission": "Audit failure classification, retry/resume, circuit breaking, human-in-the-loop gates, and safe-failure semantics across the runtime.",
        "dimensions": [
            "Typed failure taxonomy — reasoning vs dependency vs policy vs user errors.",
            "Retry budgets at step/graph/tool level — not unbounded agent loops.",
            "HITL connected to policy decisions and risk levels.",
            "Checkpoint recovery after transient failures.",
            "ReliabilityProfile wired at Tier-3.",
            "Incident-worthy failures emit observability signals.",
            "CI reliability wiring scripts green.",
        ],
        "scale": "Flaky integrations, cascading failures, HITL queue backlog.",
        "overrides": "ReliabilityProfile, HITL routing per application, custom escalation hooks.",
        "appendix": "Appendix H (risk/HITL)",
    },
    {
        "id": "TIER3_APPLICATION_ENVIRONMENT",
        "title": "Tier-3 Application Environment",
        "layers": "3, 28",
        "code": "applications/, ApplicationEnvironmentProfile, catalog_runtime_bridge, host wiring",
        "mission": "Audit deployable application hosts: profile composition, catalog bootstrap, runtime bridges, and product wiring without business logic in Nexus.",
        "dimensions": [
            "ApplicationEnvironmentProfile as single composition root.",
            "All *Profile sections wired through runtime bridges — no orphan profile fields.",
            "bootstrap_catalogs and wiring modules per host — CI audited.",
            "Agents and tools selected by profile — not hardcoded in Nexus.",
            "Serving/deployment patterns documented per reference host.",
            "Tier-3 contains orchestration of runtime+agents — not agent pipeline logic duplicated.",
        ],
        "scale": "Multi-host fleet, profile variants, cold start, config drift across environments.",
        "overrides": "This IS the override layer — verify hosts can customize without platform forks.",
        "appendix": "Appendix H (full profile map)",
    },
    {
        "id": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
        "title": "Experimentation and Developer Experience",
        "layers": "25–27, 30",
        "code": "intergrax/scaffold/, scripts/check_*.py, eval harness, doctor CLI",
        "mission": "Audit DX: scaffold, eval, CI architecture gates, lab environment, and operational excellence hooks for developers and operators.",
        "dimensions": [
            "Scaffold commands produce tier-correct artifacts (agent, app, tool bundle, skill).",
            "CI gates enforce boundaries (no getattr, vendor imports, doc pairs, observability).",
            "Eval/benchmark harness integrated with critic/RAG/tool quality loops.",
            "intergrax doctor and lab stack (HARNESS_ENVIRONMENT.md) accurate.",
            "W-OPS / release cycle documentation matches scripts.",
            "Developer time-to-first-run metric supported (<1h agent creation goal).",
        ],
        "scale": "Large monorepo CI time, many gates, parallel eval workloads.",
        "overrides": "Lab presets, local gate subsets, extension author workflows.",
        "appendix": "EXTENSION_AUTHOR_GUIDE",
    },
    {
        "id": "ADAPTIVE_HARNESS_INTELLIGENCE",
        "title": "Adaptive Harness Intelligence (L4)",
        "layers": "L4 AHI",
        "code": "adaptive routing, feedback loops, AHI modules in runtime",
        "mission": "Audit L4 adaptive loops: bounded self-tuning, evaluation-driven routing, and safe automation without uncontrolled behaviour changes.",
        "dimensions": [
            "Feedback signals defined and consumed (eval scores, latency, cost, quality).",
            "Adaptive routing bounded by policy — not unconstrained model/tool switching.",
            "Human approval gates for adaptive policy changes.",
            "Separation from L3 production paths — experimental flags clear.",
            "Observability of adaptive decisions (why route changed).",
            "Deferred vs implemented features explicitly marked in plan.",
        ],
        "scale": "Continuous adaptation under load, feedback delay, exploration/exploitation balance.",
        "overrides": "AHI profiles, opt-in per Tier-3 host, kill switches.",
        "appendix": "N/A",
    },
    {
        "id": "CRITIC_VERIFICATION",
        "title": "Critic and Verification",
        "layers": "25 (depth)",
        "code": "critic modules, verification layers, eval runners",
        "mission": "Audit output verification: LLM-as-judge, rule checks, trajectory eval, human gates — integrated with runtime, not bolt-on scripts.",
        "dimensions": [
            "Critic invoked at defined runtime points (pre-output, post-step, final).",
            "Verdict types and escalation to HITL/policy.",
            "Trajectory and rubric evaluation harness.",
            "Separation: closeout documentation vs execution depth.",
            "Trace of critic decisions and scores.",
            "False positive/negative handling and retry semantics.",
        ],
        "scale": "High-volume eval, multi-layer critics, latency impact on user path.",
        "overrides": "Critic profiles per agent/application, custom rule plugins.",
        "appendix": "N/A",
    },
    {
        "id": "REASONING_AND_COGNITION",
        "title": "Reasoning and Cognition",
        "layers": "7",
        "code": "planners, classifiers, DecisionRecord, cognition runtime steps",
        "mission": "Audit explicit reasoning: planning contracts, DecisionRecord, strategy selection, separation from side-effectful execution.",
        "dimensions": [
            "Planning as structured contract — not free-text-only plans.",
            "DecisionRecord or equivalent for major choices.",
            "Planning strategies: none, deterministic, LLM, graph — explicit and wired.",
            "Reasoning separated from tool execution in UAEP steps.",
            "Classifier/planner outputs validated against typed schemas.",
            "Reasoning failures classified separately from tool/runtime failures.",
            "Prompt compilation layers feed cognition inputs.",
        ],
        "scale": "Complex multi-step plans, replanning loops, classifier fan-out.",
        "overrides": "Planning strategy per profile, custom PlannerProtocol implementations.",
        "appendix": "Appendix I §I.4",
    },
    {
        "id": "ELASTIC_CAPACITY_AND_SCALING",
        "title": "Elastic Capacity and Platform Scaling",
        "layers": "30",
        "code": "capacity simulation, backpressure, scaling hooks, ECP modules",
        "mission": "Audit platform scaling: elastic capacity, backpressure, resource pools, contention handling, and SLO-oriented capacity governance.",
        "dimensions": [
            "Backpressure signals from orchestration to intake.",
            "Concurrency and queue depth limits enforced.",
            "Capacity model or simulation aligned with architecture (ECP).",
            "Horizontal scaling assumptions documented — state externalization.",
            "Observability SLIs for saturation and queue latency.",
            "Cost/capacity coupling — scale without budget blowout.",
        ],
        "scale": "Burst traffic, multi-tenant contention, regional failover.",
        "overrides": "Capacity profiles, queue tuning per deployment, elastic worker pools at Tier-3 infra.",
        "appendix": "OBSERVABILITY (SLIs), ORCHESTRATION (backpressure)",
    },
]


def render(domain: dict[str, str | list[str]]) -> str:
    did = str(domain["id"])
    title = str(domain["title"])
    layers = str(domain["layers"])
    code = str(domain["code"])
    mission = str(domain["mission"])
    dims = domain["dimensions"]
    assert isinstance(dims, list)
    scale = str(domain["scale"])
    overrides = str(domain["overrides"])
    appendix = str(domain["appendix"])

    dim_lines = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(dims))
    appendix_line = (
        f"6. `docs/guides/AGENT_CREATION_GUIDE.md` **{appendix}** — control-plane wiring\n"
        if appendix != "N/A"
        else ""
    )

    return f"""# {title} — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/{did}.md`](../architecture/{did}.md) · [`plan/{did}.md`](../plan/{did}.md)  
**Audit map layers:** {layers} · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with repository access.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus`).
4. Output must follow [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: {did}
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice, e.g. "ingest pipeline only" or "ToolRuntime policy path"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — {title} (`{did}`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **{title}** domain — architecture canon, implementation plan, source code, tests, and CI gates. Compare against production-grade systems in this problem space. Do **not** produce a shallow documentation survey.

**Mission:** {mission}

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state for this concern
2. `docs/architecture/{did}.md` — current architecture canon
3. `docs/plan/{did}.md` — implementation status and gap registers
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers {layers}
5. `docs/guides/audit/README.md` — shared production Harness checklist (mandatory)
{appendix_line}
---

## 2. Code and test paths (inspect concretely)

Search and read — do not rely on memory:

```text
{code}
tests/unit/ and tests/integration/ matching the above
scripts/check_harness_*.py and scripts/check_* relevant to this domain
```

---

## 3. Domain-specific audit dimensions

Answer each with **Yes / Partial / No / Unknown** and **evidence** (file + symbol or test name):

{dim_lines}

---

## 4. Workload and scale probes

Evaluate behaviour for:

{scale}

For each probe: describe actual code path, limits, and failure mode — not hypothetical design.

---

## 5. Tier-3 and agent override surfaces

Verify customization without forking Tier-0/Tier-1:

{overrides}

Confirm overrides are **wired**, not documentation-only.

---

## 6. Cross-cutting checklist (mandatory)

Apply every item in `docs/guides/audit/README.md` §Shared production Harness checklist:

- Architecture & modularity
- Configuration & strategy selection
- Override & customization surfaces
- Observability, tracing & logging
- Security & governance
- Reliability & error handling
- Performance & scale
- Testing & verification
- Documentation alignment

---

## 7. Production comparison

Compare the implementation to **production-grade systems** in this domain (commercial and open-source). State clearly:

- What Intergrax already matches at L3 production Harness OS level
- What is L2 or below with specific gaps
- What is intentionally deferred (design boundary) vs **niedoróbka** / missing wiring

---

## 8. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5:

```text
L0 — Fragmented
L1 — Operational MVP
L2 — Scalable Harness
L3 — Production Harness OS
L4 — Adaptive Agent OS
```

Report **score before**, **target for current milestone**, evidence, and **remaining risks**.

---

## 9. Verification commands

Run applicable checks; cite results:

```bash
uv run pytest -m gate -q
uv run pytest tests/unit/<relevant>/ -q
python scripts/check_harness_no_getattr.py
# plus domain-specific scripts discovered during inspection
```

---

## 10. Output and mode rules

- Follow output format in `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 (Audit Result template).
- End with §8 Completion Summary.
- `audit-only`: **no file edits**
- `audit-and-fix`: update `docs/plan/{did}.md` gap rows and `docs/architecture/{did}.md` audit register if present; **no code changes** unless user requests separately
- Never declare the whole platform complete
- Record out-of-scope findings with suggested next domain

Begin the audit now.

---END PROMPT---
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        path = OUT / f"{domain['id']}.md"
        path.write_text(render(domain), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
