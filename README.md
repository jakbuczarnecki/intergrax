# Intergrax

<!-- [![Regression gate](https://github.com/jakbuczarnecki/intergrax/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/jakbuczarnecki/intergrax/actions/workflows/unit-tests.yml) (https://github.com/jakbuczarnecki/intergrax/actions/workflows/unit-tests.yml)-->
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Harness AI](https://img.shields.io/badge/Harness%20AI-Agent%20OS-6c5ce7.svg)](#harness-ai--the-core-idea)
[![Docs](https://img.shields.io/badge/docs-canonical-green.svg)](#documentation-index)
[![LLM context](https://img.shields.io/badge/llms.txt-available-orange.svg)](llms.txt)
[![LKW proof](https://img.shields.io/badge/LKW%20platform%20proof-run%20locally-2ea44f.svg)](docs/public-adoption/LKW_PLATFORM_PROOF.md)

**Agent OS and Harness AI runtime** for building, orchestrating, experimenting with, and validating specialized AI agents — with a **clear separation between who decides, who executes, and who orchestrates.**

---

## See Intergrax running

The fastest way to evaluate Intergrax is the **Local Knowledge Workspace (LKW) platform proof** — a real Tier-3 application that runs a governed agent workflow through Nexus, `rag.retrieve`, runtime events, policy-safe observability export, Elasticsearch, and Kibana.

The proof validates a real `run_id`, `tool_requested` / `tool_completed` events, duplicate-free export, and safety-checked observability documents.

[Run the LKW platform proof →](docs/public-adoption/LKW_PLATFORM_PROOF.md)

## Overview

- **What:** Intergrax is a **Harness AI platform** — the durable runtime that runs many agents, not a single chatbot or domain bot.
- **What it provides:** Nexus Agent OS, Tier-0 catalogs (**197** integrations · **200** tools · **150** skills in **42** bundles), LLM, RAG, memory, **Ephemeral Code Craft** (**Done** ECC-0…ECC-6), policy, trace, multi-agent graphs, and Tier-3 application hosts.
- **Who it is for:** Teams building **governed multi-agent systems** — platform engineers, agent architects, Harness AI researchers, and product teams shipping agent-backed applications.
- **Why it is different:** **The Harness is the product; agents are replaceable.** Agents own **domain decisions** inside a typed session loop; the harness owns **policy, trace, and execution**; Nexus owns **multi-agent orchestration**; applications own **environment, identity, and production gates** — without collapsing these into one mega-class.
- **Problem it solves:** Stop rebuilding infrastructure for every new agent. Target: **idea → first traced Nexus run in under one hour**, then **the same agent class** moving from lab evaluation toward governed deployment paths when explicitly permitted.

```text
Intergrax — Harness AI

┌──────────────────────────────────────────┐
│ Tier-3 Application environment           │
│ identity · profiles · org policy         │
│ AgentBinding · production scoreboard     │
└─────────────────────┬────────────────────┘
                      │ Task
┌─────────────────────▼────────────────────┐
│ Tier-1 NexusLoop (Agent OS)              │
│ graphs · capability routing · HITL       │
└─────────────────────┬────────────────────┘
                      │ one agent node → Agent.run()
┌─────────────────────▼────────────────────┐
│ Tier-2 Agent session                     │
│ HarnessKernel                            │
│ on_next_step · policy · trace · state    │
│ budgets                                  │
└─────────────────────┬────────────────────┘
                      │
┌─────────────────────▼────────────────────┐
│ Tier-0 Platform                          │
│ Integration → Tool → Skill → LLM · RAG   │
│ Memory                                   │
└──────────────────────────────────────────┘
```

Strategic direction: [Development Strategy](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) · [System invariants](docs/guides/SYSTEM_INVARIANTS.md) · Ideal target: [IDEAL_HARNESS_AI_ARCHITECTURE.md](docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) · **Agent model canon:** [AGENT_CONTRACTS_AND_ASSEMBLY.md](docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §13–§40

---

## Current platform maturity

**Harness baseline:** **32/32** audit layers at **L3** ([scorecard](scripts/gates/harness_maturity_report.py) · [IDEAL_HARNESS_L3](docs/plan/IDEAL_HARNESS_L3.md) · [audit map §5](docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md#5-maturity-scoring-model) · [Maturity Taxonomy](docs/guides/MATURITY_TAXONOMY.md)).

Maturity uses **L0–L4** levels (not arbitrary percentages) — navigation summaries in this README. Authoritative readiness claims use the four-axis **A/I/P/E** vocabulary in [MATURITY_TAXONOMY.md](docs/guides/MATURITY_TAXONOMY.md). Per-domain evidence lives in **domain-layer pairs**: `docs/architecture/<DOMAIN>.md` ↔ `docs/plan/<DOMAIN>.md`. **Cross-layer capabilities** use **multi-layer feature pairs**: `docs/features/architecture/<FEATURE>.md` ↔ `docs/features/plan/<FEATURE>.md` — see [Multi-layer feature docs](docs/features/README.md).

| Area | Maturity | Evidence / open gap |
|------|----------|---------------------|
| **Agent contracts (ACP)** | **L3** | [Phase ACP](docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) **Done** · fleet `on_next_step` migration ongoing |
| **Tools** | **L3** | [TOOL-ENG](docs/plan/TOOLS.md) closeout **Done** · deferred: hierarchical LLM category pass |
| **Tier-3 hosts** | **L3** (depth partial) | [APP-PROD](docs/plan/TIER3_APPLICATION_ENVIRONMENT.md) gates **Done** · §22 profile bundles M1–M3 **Done** · enterprise distribution **P4** |
| **Memory** | **L3** | [MEM / MEM-VEC / MEM-DEPTH](docs/plan/MEMORY.md) **Done** · L3 harness · P2 depth: procedural store, org LTM parity, per-step budget caps before CE collect |
| **RAG** | **L3** (profile-driven) | [M-RAG-DEPTH / GRAPH](docs/plan/RAG.md) **Done** · Tier-3 `RagProfile` required · autonomous parser/chunker/retriever selection **Frozen/P4** → AHI |
| **Adaptive harness (AHI)** | **L4-ready** (mechanisms) | [W-ADAPT](docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md) **Done** · L4 runtime mechanisms implemented · production L4 evidence requires product-host run volume |

**Also at L3:** LLM adapters · Observability · Nexus flow · UAEP · Skills · Integrations — full index in [architecture hub](docs/intergrax_runtime_architecture.md).

---

## License and collaboration model

Intergrax is **public and source-available** for evaluation and technical partner discovery. It is **not** distributed under an open-source license. Production, commercial, and redistribution use require **explicit permission** from the copyright holder.

External feedback, proof-path testing, integration proposals, and design-partner discussions are welcome under the collaboration model described in [COLLABORATION.md](COLLABORATION.md). Full terms: [LICENSE](LICENSE).

Active public feedback paths are listed in the [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md). Use it to choose the right curated issue for proof-path feedback, documentation clarity, integration feedback, or design-partner interest.

---

## Public Discussion Map

The open GitHub issues are **not** a generic implementation backlog. They are a maintainer-curated public discussion map for evaluating Intergrax as a **Harness AI / Agent OS** platform.

Use the [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md) to choose the right entry point.

| Track | Use when you want to discuss |
|-------|------------------------------|
| **Proof path feedback** | First-run setup, README quick start, evidence and trace inspection |
| **Architecture discussion** | Harness AI boundaries, Nexus as Agent OS, policy-first execution, agent contracts |
| **Integration feedback** | Attestation, trace/evidence export, MCP-style controlled task surfaces |
| **Product validation** | Legal review, research workflows, local knowledge workspace, lab application, assistant hub |
| **Deep technical review** | Capability graph, evaluation gates, cost governance, reliability, security, observability, developer experience |

Public discussion issues do **not** create support obligations, roadmap commitments, production-use permission, commercial-use permission, redistribution rights, derivative-work rights, SLA commitments, or security vulnerability handling. For security reports, use [SECURITY.md](SECURITY.md). For commercial licensing, production use, partnerships, redistribution, derivative works, or substantial implementation permission, contact the maintainer directly.

---

## Who this repository is for

Intergrax is currently most useful for technical readers evaluating how to build governed agent applications with explicit runtime boundaries, policy-controlled tool execution, trace/evidence surfaces, and external verification hooks.

It is especially relevant for:

- AI platform engineers designing agent infrastructure beyond a single demo agent
- Teams building governed agent applications that need policy, HITL, trace, evidence, or evaluation surfaces
- Builders working on [attestation](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md), receipts, boundary events, auditability, observability, or agent governance
- Developers evaluating Nexus orchestration, tool/skill boundaries, RAG, memory, and Tier-3 application hosts
- Potential technical design partners willing to run proof paths and report friction

Intergrax is not presented as a finished SaaS, a general-purpose open-source framework, or a production certification claim. See [COLLABORATION.md](COLLABORATION.md), [ROADMAP.md](ROADMAP.md), and [LICENSE](LICENSE).

---

## Start here

| If you are… | Start with |
|-------------|------------|
| Evaluating Intergrax for the first time | [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md) · [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) · [FAQ.md](FAQ.md) · [Proof of platform](#proof-of-platform) |
| Checking use-case fit | [USE_CASES.md](USE_CASES.md) · [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) |
| Checking collaboration or license boundaries | [COLLABORATION.md](COLLABORATION.md) · [LICENSE](LICENSE) |
| Reviewing the Harness AI / Agent OS model | [INTERGRAX_HARNESS_NARRATIVE.md](docs/guides/INTERGRAX_HARNESS_NARRATIVE.md) · [AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) |
| Exploring attestation, boundary events, or external verification | [BOUNDARYATTEST_ATTESTATION_POC.md](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) · [attestation_demo README](applications/attestation_demo/README.md) |
| Exploring product-validation directions | [LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) · [USE_CASES.md](USE_CASES.md) |
| Interested in feedback, design-partner work, or integration proposals | [PARTNERS.md](PARTNERS.md) · [COLLABORATION.md](COLLABORATION.md) · [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md) |

Looking for common questions? See [FAQ.md](FAQ.md).

This navigation is intentionally public-facing; canonical technical architecture remains in the architecture and plan documents linked below.

---

## The agent model — why architects choose Intergrax

Most agent frameworks mix **planning, policy, tool I/O, and multi-agent routing** into a single author-facing class. That works for demos; it breaks for **governed products** — opaque control flow, untyped state, and agents that secretly become mini operating systems.

Intergrax treats the **agent as a domain decision unit** running inside a **rich, typed environment**. Four roles stay explicit:

| Layer | Responsibility | Answers |
|-------|----------------|---------|
| **Application (Tier-3)** | Environment & product wiring | *Who is the tenant? Which tools/memory/RAG profile? Org policy? Production gates?* |
| **NexusLoop (Tier-1)** | Multi-agent orchestration | *Which agents run on this Task? Graph, HITL, checkpoints at task level?* |
| **Agent (Tier-2)** | Domain cognition per session | *What is the next move? Plan valid? Terminal? Pause for human?* — via **`on_next_step` → `StepOutcome`** |
| **HarnessKernel** | Deterministic harness cycle | *Policy allowed? State merged safely? Trace recorded? Budgets enforced?* |

**Author mental model (one session, many steps):**

```text
result = await agent.run(AgentRunRequest(...))     # once per graph node (or direct in lab)

inside run():
    loop:
        state = load_session_state(step_ctx)       # READ  — typed AcpSessionState
        outcome = await on_next_step(step_ctx)     # DECIDE — StepOutcome factories
        HarnessKernel.execute_step(outcome, ...)   # EXECUTE — policy, trace, merge
```

**What you get as a platform owner:**

- **Readable agent code** — reviewers see *continue / complete / fail / HITL* from the final `return StepOutcome.*`, not scattered flags in dicts.
- **Same agent, many deployments** — lab vs legal vs research host = different `ApplicationEnvironmentProfile`, **zero agent forks**.
- **Virtual workforce ready** — organizational policy envelope on the host, not `if customer == acme` in agent source.
- **Operational readiness is measurable** — [Agent Production Readiness Scoreboard](docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md#4015-agent-production-readiness-scoreboard) (contract, runtime, policy, observability, checkpointing, idempotency, security, evaluation, lifecycle, routing).
- **Agents are swappable** — capability-based routing (`research.web_search`), not hardcoded class names in Nexus.

**Canonical decisions:** [ADR-AGENT-001](docs/adr/entries/2026-06-11/ADR-AGENT-001.md) (Nexus stays Agent OS) · [ADR-AGENT-002](docs/adr/entries/2026-06-11/ADR-AGENT-002.md) (`Agent.run()` facade) · [ADR-AGENT-003](docs/adr/entries/2026-06-11/ADR-AGENT-003.md) (step loop + dual observability).

**Implementation status:** architecture **decision-complete** (§13–§40); delivery via plan [Phase ACP](docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) (typed contracts → step loop → fleet migration → prod gates). Today: UAEP bridge + Nexus path; target: typed `on_next_step` on all roster agents.

**Deep dive:** [Agent Creation Guide — Appendix AC](docs/guides/AGENT_CREATION_GUIDE.md#appendix-ac--agent-run-cognitive-patterns-and-environment-acp) · [Architecture hub — agent in environment](docs/intergrax_runtime_architecture.md#agent-in-the-harness-environment)

---

## Why another AI framework?

Most AI projects ship **one agent class that secretly is the whole stack** — planner, executor, policy, and orchestrator in one file.

**Intergrax ships the Harness** — a governed Agent OS where **agents decide domain steps**, the **kernel executes under policy**, and **Nexus orchestrates graphs** — so you can field dozens of specialized agents and Tier-3 products on one platform without architectural collapse.

If you evaluate GitHub projects by *“does the architecture scale beyond the demo?”* — this separation is the answer.

---

## Audience

This repository is for you if you are:

| Role | Why Intergrax |
|------|----------------|
| **AI systems architect** | Four-tier Harness AI model; **agent / kernel / Nexus split**; policy-first execution; L0–L4 maturity |
| **Agent platform engineer** | Typed `Agent.run()` + `on_next_step`, `HarnessKernel`, `ToolRuntime`, dual observability planes |
| **Multi-agent runtime developer** | Delegation, subagents, parallel graphs, HITL — without nested OS forks |
| **Harness AI researcher** | Lab workflow, trace inspection, evaluation hooks, adaptive harness (L4) |
| **Product team shipping agents** | Tier-3 application shells — isolated deployable hosts composing Tier-2 agents |

**Not the primary audience:** teams looking for a finished SaaS chatbot, a prompt library, or a no-code workflow builder.

---

## Quick start

**Goal:** clone → install → verify → run → inspect.

### Prerequisites

Python 3.12 · [`uv`](https://github.com/astral-sh/uv) · Git

### 1. Install

```bash
git clone https://github.com/jakbuczarnecki/intergrax.git
cd intergrax
uv sync --extra dev
```

### 2. Verify

```bash
uv run intergrax doctor
uv run pytest -m gate -q

# Optional — core certification evidence
uv run intergrax certify core --level L2

# Optional — report-derived evidence timeline
uv run intergrax trace show
uv run intergrax trace export
```

`trace show` renders the report-derived timeline to stdout.
`trace export` writes `build/evidence/trace/timeline.json` and `timeline.md`.
This is deterministic mock evidence derived from the certification report, not live runtime tracing.

See [Harness Environment — core certification](docs/guides/HARNESS_ENVIRONMENT.md#core-certification-evidence-path-hep).

### 3. Run the lab host

```bash
uv run uvicorn lab_application.host.main:app --host 127.0.0.1 --port 8090
```

### 4. Execute and inspect

```bash
# Submit a run (Echo agent via capability routing)
curl -s -X POST http://127.0.0.1:8090/v1/lab/run \
  -H "Content-Type: application/json" \
  -d '{"message":"hello","capability":"echo.basic"}'

# Inspect trace (replace {task_id} from response)
curl -s "http://127.0.0.1:8090/debug/tasks/{task_id}/trace?include_runtime=true"
```

**Next steps:** scaffold your own agent, register it, rerun through the lab, inspect `/debug/tasks/{id}/metrics` and `/events`.

| Command | Purpose |
|---------|---------|
| `python -m intergrax.scaffold new-agent {name} --capability domain.action` | Create Tier-2 agent skeleton |
| `python -m intergrax.scaffold new-application {name} --profile lab` | Create Tier-3 host |
| `python -m intergrax.scaffold new-stack {name}` | Agent + application bundle |
| `uv run intergrax run {module}:app` | Launch any ASGI application host |
| `uv run intergrax certify core --level L2` | Core certification report (deterministic mock contract evidence) |
| `uv run intergrax trace show` | Render report-derived evidence timeline to stdout |
| `uv run intergrax trace export` | Write timeline JSON/Markdown under `build/evidence/trace/` |
| `python -m intergrax.debug` | Debug CLI |

**Full workflow:** [Agent Creation Guide](docs/guides/AGENT_CREATION_GUIDE.md) · **Contributing setup:** [CONTRIBUTING.md](CONTRIBUTING.md)

After running the quick start, share structured feedback via [#186 README quick start feedback](https://github.com/jakbuczarnecki/intergrax/issues/186) or choose another path from the [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md).

---

## Proof of platform

**Live application proof:** [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md) — run a real Local Knowledge Workspace platform path with Docker Compose, Elasticsearch, Kibana, and duplicate/safety validation.

External narrative: [Intergrax Harness Narrative](docs/guides/INTERGRAX_HARNESS_NARRATIVE.md)

**External integration proof:** [BoundaryAttest Attestation PoC](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) — external validation of host-signed execution boundary events (technical integration validation, not production certification).

**Product-validation direction:** [Local Knowledge Workspace alpha](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) — product-validation narrative for local governed knowledge workflows (alpha track, not a finished product or SaaS).

**What is it?** The fastest local way to verify Intergrax as an evidence-backed harness platform — not a production readiness or compliance claim.

**Why run it?** You get a repeatable, offline proof that the harness can produce and aggregate the core evidence surfaces an early adopter needs to trust the platform locally. The A2 end-to-end evidence smoke audit verified this exact command sequence on a clean local run.

Run the canonical proof path:

```bash
uv run intergrax certify core --level L2
uv run intergrax trace export
uv run intergrax evidence live-core
uv run intergrax evidence eval
uv run intergrax evidence cost
uv run intergrax evidence posture
uv run intergrax evidence posture export
```

All artifacts land under `build/evidence/`:

| Surface | Artifacts |
| ------- | --------- |
| Core certification | `build/evidence/core_certification/report.json`, `build/evidence/core_certification/report.md` |
| Trace evidence | `build/evidence/trace/timeline.json`, `build/evidence/trace/timeline.md` |
| Live Tier-0 probes | `build/evidence/live_core_probes/live_core_report.json`, `build/evidence/live_core_probes/live_core_report.md` |
| Eval evidence | `build/evidence/eval/report.json`, `build/evidence/eval/report.md` |
| Cost evidence | `build/evidence/cost/report.json`, `build/evidence/cost/report.md` |
| Evidence posture | `build/evidence/posture/posture.json`, `build/evidence/posture/posture.md` |

**What this proves** — local ability to produce and aggregate:

* core certification evidence
* trace evidence
* selected local live Tier-0 probe evidence
* eval regression evidence
* cost evidence
* evidence posture scoreboard

**What this does not prove:**

* production runtime certification
* security/compliance attestation
* real provider execution
* real LLM evaluation
* billing
* provider pricing
* cloud cost estimation
* product-specific acceptance

**Verify artifacts and docs** (after running the proof path):

```bash
python scripts/maintenance/check_evidence_artifacts.py
```

Confirms expected evidence artifacts exist and README still documents the canonical proof path.

**Next steps:** After the proof path, inspect `build/evidence/posture/posture.md` first, then drill into the individual evidence reports. For roadmap and status, see [HARNESS_EVIDENCE_PACK.md](docs/plan/HARNESS_EVIDENCE_PACK.md). For architecture framing, see [EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md](docs/architecture/satellites/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md).

---

## What you can do today

| Action | How | Learn more |
|--------|-----|------------|
| **Scaffold a new agent** | `python -m intergrax.scaffold new-agent …` | [Agent Creation Guide](docs/guides/AGENT_CREATION_GUIDE.md) |
| **Build a Tier-3 application** | `new-application` / `new-stack` | [applications/USAGE.md](applications/USAGE.md) |
| **Connect integrations** | `IntegrationProfile` + Tier-3 wiring | [architecture/INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) |
| **Attach tools and skills** | `ToolProfile`, `SkillProfile`, `skill_ids` on contract | [architecture/TOOLS.md](docs/architecture/TOOLS.md) · [architecture/SKILLS.md](docs/architecture/SKILLS.md) |
| **Run through Nexus** | Lab or product host → `NexusLoop` → `Agent.run()` / `AgentEngine` | [NEXUS_EXECUTION_FLOW.md](docs/architecture/NEXUS_EXECUTION_FLOW.md) · [Agent model](#the-agent-model--why-architects-choose-intergrax) |
| **Inspect traces** | `/debug/tasks/{id}/trace`, `intergrax.debug` | [HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md) |
| **Evaluate execution** | Evaluation profile, online registry, CVL hooks | [CRITIC_VERIFICATION.md](docs/architecture/CRITIC_VERIFICATION.md) |
| **Ephemeral code craft** | Dynamic codegen loop in sandbox (**Done** ECC-0…ECC-6) | [CODE_CRAFT.md](docs/architecture/CODE_CRAFT.md) |
| **Extend via plugins** | `ToolPlugin`, `IntegrationPlugin`, `SkillPlugin` EPs | [EXTENSION_AUTHOR_GUIDE.md](docs/guides/EXTENSION_AUTHOR_GUIDE.md) |

Reference hosts: [`applications/README.md`](applications/README.md) · Reference agents: [`agents/README.md`](agents/README.md)

---

## Harness AI — the core idea

> **The future value is not in building one agent. The value is in building the runtime that allows many agents to be built, tested, and orchestrated quickly.**

Intergrax implements the Harness AI chain:

```text
Harness  →  Runtime (Nexus)  →  Agents  →  Applications  →  Products
```

| Term | Intergrax implementation |
|------|---------------------------|
| **Harness** | Tier-1 Nexus + Tier-0 catalogs + Tier-3 wiring (policy, tools, integrations, trace) |
| **Scaffold** | `python -m intergrax.scaffold` — `new-agent`, `new-application`, `new-stack`, `new-skill` |
| **Runnable agent instance** | Harness + agent + `LLMProfile` + resolved `skill_ids` / `allowed_tools` + `RuntimePolicyBundle` |
| **Tool** | Atomic `ToolContract` — LLM/MCP invocable operation |
| **Skill** | Composable `SkillManifest` — tools + prompts + policy fragment (not an LLM function) |
| **Subagent** | Graph delegation via `ExecutionGraph` — not a nested OS |
| **Policy** | `PolicyEngine`, budgets, HITL, `RuntimePolicyBundle` |

**Agent composition flow:**

```text
Application profile (Tier-3)  →  merge_environment  →  Agent.run(AgentRunRequest)
    → NexusLoop (multi-agent) or direct run (lab)
        → Agent.on_next_step  →  StepOutcome  (domain decides)
        → HarnessKernel       →  policy · trace · state merge (harness executes)
        → SkillManifest(s) · ToolRuntime.invoke  →  Integration adapters
        → LLM / RAG / memory gateways (per-step, policy-bound)
```

**Vocabulary canon:** [architecture/PLATFORM_FOUNDATION.md §5.3](docs/architecture/PLATFORM_FOUNDATION.md) · **Target model:** [IDEAL_HARNESS_AI_ARCHITECTURE.md](docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)

---

## Laboratory vs production harness

Two modes on **one codebase**:

| Mode | Purpose | Primary metric |
|------|---------|----------------|
| **Laboratory** | Fast hypothesis validation | Idea → first traced run in under **1 hour** |
| **Production harness** | Governed Agent OS at organizational scale | Stable integration paths + ops SLOs |

New capabilities start in the lab (`lab_application`, pytest, debug API). Capabilities that ship to users graduate through maturity gates. Business agents (Phase K) require **explicit product prioritization** — default harness queue is [gate maintenance](docs/plan/PLATFORM_FOUNDATION.md#61-harness-platform-maintenance-default--band-1).

Details: [INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) · [HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md)

---

## Four-tier platform model

```text
Tier-3  Applications     →  deployable products (legal API, lab host, research service)
Tier-2  Agents           →  specialized capability modules (LegalAgent, ResearchAgent)
Tier-1  Nexus Runtime    →  Agent OS (NexusLoop, AgentEngine, UAEP, governance)
Tier-0  Platform         →  universal building blocks (integrations, tools, skills, LLM, RAG)
```

| Tier | Role | Path |
|------|------|------|
| **Tier-0 — Platform** | Integrations, tools, skills, LLM, RAG, memory | `intergrax/` (outside Nexus orchestration) |
| **Tier-1 — Nexus** | Task lifecycle, graphs, governance, event bus | `intergrax/runtime/` |
| **Tier-2 — Agents** | Domain logic: contracts, pipelines, prompts | `agents/` |
| **Tier-3 — Applications** | Isolated deployable environments | `applications/` |

**Dependency rules:**

```text
intergrax/       MUST NOT import from agents/ or applications/
agents/          MUST NOT import from applications/
applications/    MAY import from agents/ and intergrax/
```

Agents consume Tier-0 through Nexus policy and **`ToolRuntime`** — never vendor SDKs directly. Tier-1/2/3 work is **composition and wiring**, not parallel platform mechanisms.

Canon: [architecture/PLATFORM_FOUNDATION.md §5.2](docs/architecture/PLATFORM_FOUNDATION.md) · Hub: [intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md)

---

## Capability stack (Integration → Tool → Skill → Agent)

| Layer | What it is | Invoked by LLM? | Example |
|-------|------------|-----------------|---------|
| **Integration** | Swappable backend contract | No | PostgreSQL, Bing, Jira REST |
| **Tool** | Single atomic operation | **Yes** | `rag.retrieve`, `jira.search_tasks` |
| **Skill** | Reusable pack: `tool_ids` + prompts + policy | **No** | `legal.contract_review`, `harness.tool_smoke` |
| **Agent** | Domain module: contract + `on_next_step` / cognitive pattern, `skill_ids[]` | — | `LegalAgent` in `agents/legal/` |

```text
Integration  →  Tool  →  Skill  →  Agent  →  Nexus (Harness)  →  Application wiring
```

Skills are **not** tools — the runtime resolves skills into allow-lists and instructions before the run.

Catalogs: [INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) · [TOOLS.md](docs/architecture/TOOLS.md) · [SKILLS.md](docs/architecture/SKILLS.md)

---

## Nexus runtime and agent execution

**Nexus** (Tier-1) is the Agent Operating System. It **orchestrates Tasks and graphs** — it is **not** the agent’s reasoning engine.

| Component | Role |
|-----------|------|
| **NexusLoop** | Task intake, multi-agent graph, capability routing, HITL, application orchestration log |
| **AgentRegistry** | Registration, capability tokens, skill/tool resolution, lifecycle gates |
| **AgentEngine** | Bridge graph node → `Agent.run()` / session loop |
| **HarnessKernel** | Per-step harness cycle — policy, trace, state merge, budgets (not domain planning) |
| **ExecutionGraph** | Multi-agent workflows, delegation, parallel cap |
| **ToolRuntime** | Unified tool gateway — policy, trace, idempotency (§42.12) |
| **PolicyEngine** | Governance on tool/LLM/RAG/memory paths |
| **ContextManager** | Context assembly, budget trimming, memory views |

### Target author API (ACP — canonical)

```text
Agent.run(AgentRunRequest)  →  loop: on_next_step → StepOutcome  →  HarnessKernel.execute_step
```

- **One `run()`** per graph node — **many `on_next_step`** inside the session.
- **Typed state** (`AcpSessionState`) and **typed outcomes** — readability at code-review time.
- **Dual observability:** `AgentRunTrace` (agent plane) + `ApplicationRunSummary` (orchestration plane).

### UAEP (bridge today)

```text
get_steps  →  run_step  →  decide_after_step   # maps to on_next_step + kernel during migration
```

All agents conform to the **Unified Execution Runtime Specification** (§42). **Registration:** `AgentRegistry.register()` — never fork `NexusLoop` for one agent.

Canon: [AGENT_CONTRACTS_AND_ASSEMBLY.md](docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) · Plan: [Phase ACP](docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · [UNIFIED_EXECUTION_RUNTIME.md](docs/architecture/UNIFIED_EXECUTION_RUNTIME.md) · [NEXUS_EXECUTION_FLOW.md](docs/architecture/NEXUS_EXECUTION_FLOW.md) · [ORCHESTRATION.md §50–§54](docs/architecture/ORCHESTRATION.md)

---

## Tier-0 catalog summary

Shipped first-party catalogs (verified via `bootstrap_catalogs(integration_preset='full')` + `snapshot_catalogs()` — **2026-06-23**).

```text
Integration  →  vendor backend (Postgres, Bing, Jira, …)
Tool         →  atomic LLM/MCP operation (rag.retrieve, websearch.query, …)
Skill        →  composable pack (tool_ids + prompts + policy fragment)
```

| Layer | Catalog size | Module | Architecture | Plan | Usage / authoring |
|-------|--------------|--------|--------------|------|-------------------|
| **Integrations** | **197** slugs · **31** contract categories (116 STABLE · 81 BETA) | [`intergrax/integrations/`](intergrax/integrations/) | [INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) | [plan/INTEGRATIONS.md](docs/plan/INTEGRATIONS.md) | Per-provider USAGE.md under `intergrax/integrations/providers/` |
| **Tools** | **200** `tool_id`s · **49** bundles | [`intergrax/tools/`](intergrax/tools/) | [TOOLS.md](docs/architecture/TOOLS.md) | [plan/TOOLS.md](docs/plan/TOOLS.md) | [intergrax/tools/USAGE.md](intergrax/tools/USAGE.md) |
| **Skills** | **150** `skill_id`s · **42** bundles | [`intergrax/skills/`](intergrax/skills/) | [SKILLS.md](docs/architecture/SKILLS.md) | [plan/SKILLS.md](docs/plan/SKILLS.md) | Per-skill `USAGE.md` under `intergrax/skills/providers/{bundle}/{skill_id}/` |

**Control plane (profiles, wiring, resolver):** [AGENT_CREATION_GUIDE.md Appendix J](docs/guides/AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane) · **Extension plugins:** [EXTENSION_AUTHOR_GUIDE.md](docs/guides/EXTENSION_AUTHOR_GUIDE.md)

**Skill bundles (42):** `agent`, `billing`, `browser`, `cache`, `catalog`, `cloud_platform`, `code`, `codecraft`, `collaboration`, `context`, `cost`, `crm`, `data`, `dev`, `eval`, `filesystem`, `gitlab`, `graph`, `harness`, `health`, `hitl`, `http`, `identity`, `interaction`, `jira`, `knowledge`, `legal`, `memory`, `message_bus`, `metrics`, `ml`, `modality`, `notify`, `openai`, `ops`, `platform`, `rag`, `research`, `sandbox`, `storage`, `vector_store`, `workspace` — **150** skills — full index in [SKILLS.md](docs/architecture/SKILLS.md).

---

## Platform capabilities

Tier-0 building blocks — one canonical path per concern. Agents use these through Nexus; they do not reimplement them.

| Concern | Scale / module | Documentation |
|---------|----------------|---------------|
| **Integrations** | **197** providers · `intergrax/integrations/` | [architecture/INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) · [plan](docs/plan/INTEGRATIONS.md) |
| **Tools** | **200** catalog tools · **49** bundles · `intergrax/tools/` | [architecture/TOOLS.md](docs/architecture/TOOLS.md) · [plan](docs/plan/TOOLS.md) · [USAGE](intergrax/tools/USAGE.md) |
| **Skills** | **150** skills · **42** bundles · `intergrax/skills/` | [architecture/SKILLS.md](docs/architecture/SKILLS.md) · [plan](docs/plan/SKILLS.md) |
| **LLM adapters** | 19 providers · typed `LLMAdapterResponse` | [architecture/LLM_ADAPTERS.md](docs/architecture/LLM_ADAPTERS.md) |
| **RAG** | Retrieval, ingest, hybrid/graph/agentic (profile-configured) · golden + load/soak gates | [architecture/RAG.md](docs/architecture/RAG.md) · [plan](docs/plan/RAG.md) |
| **Ephemeral Code Craft** | Dynamic codegen, test/fix loop, sandbox promotion (**Done** ECC-0…ECC-6) | [architecture/CODE_CRAFT.md](docs/architecture/CODE_CRAFT.md) · [plan](docs/plan/CODE_CRAFT.md) |
| **Memory** | STM/LTM, context compiler, Knowledge vs LTM boundary | [architecture/MEMORY.md](docs/architecture/MEMORY.md) · [plan](docs/plan/MEMORY.md) |
| **Modality / ML** | Vision, speech, classical ML via catalog tools | [architecture/MODALITY.md](docs/architecture/MODALITY.md) |
| **Governance & HITL** | Policy bundle, budgets, shadow workspace, sandbox | [UAEP §42.11](docs/architecture/UNIFIED_EXECUTION_RUNTIME.md) · [Appendix H](docs/guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane) |
| **LLM guardrails** | Vendor scanners via Integration `llm_guardrail` (M.12) | [INTEGRATIONS §47](docs/architecture/INTEGRATIONS.md) · [UAEP §42.11.6](docs/architecture/UNIFIED_EXECUTION_RUNTIME.md) · [ADR-GR-001](docs/adr/entries/2026-06-09/ADR-GR-001.md) |
| **Observability** | Event bus, trace DB, unified journal, OTLP | [architecture/OBSERVABILITY.md](docs/architecture/OBSERVABILITY.md) |
| **Plugins** | pip-installable integration/tool/skill catalogs | [EXTENSION_AUTHOR_GUIDE.md](docs/guides/EXTENSION_AUTHOR_GUIDE.md) |

**Control-plane authoring maps:** [AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) Appendices A–U · **32-layer audit:** [INTEGRAX_HARNESS_AUDIT_MAP.md](docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md)

---

## Applications

**Applications** turn agent capabilities into **isolated, deployable products** — own env, host, Docker, integration profile. Domain logic stays in `agents/`; applications **wire only**.

```text
agents/legal/  ──mount──►  applications/legal_application/  ──►  NexusLoop + FastAPI
agents/*       ──mount──►  applications/lab_application/      ──►  universal lab + /debug/*
```

| Application | Port | Role |
|-------------|------|------|
| [`lab_application/`](applications/lab_application/) | 8090 | Universal lab + debug trace API |
| [`poc_template_application/`](applications/poc_template_application/) | 8095 | Canonical Tier-3 scaffold reference |
| [`legal_application/`](applications/legal_application/) | 8000 | Contract review product API |
| [`research_application/`](applications/research_application/) | 8010 | Research → summarize pipeline |
| [`local_workspace_application/`](applications/local_workspace_application/) | 8020 | Local Knowledge Workspace (LKW) |
| [`dispute_sim_application/`](applications/dispute_sim_application/) | 8025 | Dispute Simulation Workspace (DSW) |
| [`intergrax_assistant_application/`](applications/intergrax_assistant_application/) | 8096 | Harness chat lab (IAA) |

Full index: [`applications/README.md`](applications/README.md) · Composition engine: [`intergrax/applications/USAGE.md`](intergrax/applications/USAGE.md) · Tier-3 guide: [Appendix F](docs/guides/AGENT_CREATION_GUIDE.md#appendix-f--tier-3-application-environment)

---

## Experimentation workflow

```text
new idea  →  scaffold agent  →  contract + on_next_step (READ/UPDATE/DECIDE)
  →  register  →  wire host profile / AgentBinding
  →  agent.run() in pytest  →  Nexus graph in prod  →  inspect AgentRunTrace
  →  production readiness scoreboard  →  promote roster
```

Regression gate: `uv run pytest -m gate -q`

---

## Repository layout

```text
intergrax/              # Tier-0 platform + Tier-1 Nexus
  integrations/         # Integration Library
  tools/                # Tool Library + MCP export
  skills/               # Skill Library
  llm_adapters/         # LLM providers
  rag/ · memory/        # Retrieval and memory
  codecraft/            # Ephemeral Code Craft engine (ECC-0…ECC-6 Done)
  runtime/nexus/        # NexusLoop, AgentEngine, UAEP, orchestration
  runtime/adaptive/     # L4 Adaptive Control Plane
  applications/         # Tier-3 composition engine
  scaffold/             # new-agent, new-application, new-stack
agents/                 # Tier-2 specialized agents
applications/           # Tier-3 deployable hosts
docs/                   # Architecture canon (22 domain pairs) + guides
infra/                  # Local Docker compose for backends
tests/ · scripts/       # Gate tests and harness CI checks
```

---

## Documentation index

> **Doc roles and Cursor workflow:** [DOCUMENTATION_MAP.md](docs/DOCUMENTATION_MAP.md) — single navigation hub (human · operator · AI agent).

### Quick doc routing

| You need… | Read |
|-----------|------|
| Map of all doc roles | [DOCUMENTATION_MAP.md](docs/DOCUMENTATION_MAP.md) |
| Architecture hub + domain pairs | [intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md) |
| Cursor audit / implement session | [bootstrap/README.md](docs/bootstrap/README.md) → [audit/README.md](docs/audit/README.md) |
| AI agent instructions (full) | [AGENT_INSTRUCTIONS.md](docs/guides/AGENT_INSTRUCTIONS.md) |
| Milestone history | [implementation-journal/README.md](docs/implementation-journal/README.md) |

### Internal documentation routing

| I want to… | Read |
|------------|------|
| Understand strategic direction | [INTERGRAX_DEVELOPMENT_STRATEGY.md](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) |
| **Rules you must not break** | [SYSTEM_INVARIANTS.md](docs/guides/SYSTEM_INVARIANTS.md) — `SYS-INV-*` index (P2-ARCH-01) |
| **Close out a full harness layer** | [LAYER_COMPLETION_MODE.md](docs/guides/LAYER_COMPLETION_MODE.md) |
| Understand the platform | [intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md) → pick a domain pair |
| Exploring multi-layer platform features | [Multi-layer feature docs](docs/features/README.md) |
| **Understand the agent model** | [The agent model](#the-agent-model--why-architects-choose-intergrax) · [AGENT_CONTRACTS §13–§40](docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) · [ADR-AGENT-001..003](docs/adr/entries/2026-06-11/ADR-AGENT-001.md) |
| See implementation status | [plan/PLATFORM_FOUNDATION.md](docs/plan/PLATFORM_FOUNDATION.md) · [plan ACP](docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) |
| Create a new agent | [AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) · [Appendix AC](docs/guides/AGENT_CREATION_GUIDE.md#appendix-ac--agent-run-cognitive-patterns-and-environment-acp) |
| Full Nexus execution flow | [NEXUS_EXECUTION_FLOW.md](docs/architecture/NEXUS_EXECUTION_FLOW.md) |
| See catalog sizes (integrations / tools / skills) | [Tier-0 catalog summary](#tier-0-catalog-summary) |
| Wire integrations / tools / skills | [INTEGRATIONS.md](docs/architecture/INTEGRATIONS.md) · [TOOLS.md](docs/architecture/TOOLS.md) · [SKILLS.md](docs/architecture/SKILLS.md) · [Appendix J](docs/guides/AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane) |
| RAG engine / retrieval | [RAG.md](docs/architecture/RAG.md) · [plan/RAG.md](docs/plan/RAG.md) · [Appendix K §K.5](docs/guides/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) |
| Ephemeral Code Craft | [CODE_CRAFT.md](docs/architecture/CODE_CRAFT.md) · [plan/CODE_CRAFT.md](docs/plan/CODE_CRAFT.md) |
| All agents / applications | [agents/README.md](agents/README.md) · [applications/README.md](applications/README.md) |
| Harness audit (32 layers) | [INTEGRAX_HARNESS_AUDIT_MAP.md](docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md) |
| Business backlog only | [plan/PLATFORM_FOUNDATION.md §6.3a](docs/plan/PLATFORM_FOUNDATION.md#63a-business-backlog-register-consolidated) |

**AI context:** [llms.txt](llms.txt) · [llms-full.txt](llms-full.txt) · [AGENTS.md](AGENTS.md) (stub) · [AGENT_INSTRUCTIONS.md](docs/guides/AGENT_INSTRUCTIONS.md) · [CURSOR_TOKEN_SETUP.md](docs/guides/CURSOR_TOKEN_SETUP.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

**One source of truth per topic.** Platform docs live in [`docs/`](docs/); product and agent docs live under `applications/{product}/` and `agents/{name}/`.

### Key operating guides

- [Maturity Taxonomy](docs/guides/MATURITY_TAXONOMY.md) — four-axis A/I/P/E readiness vocabulary.
- [Agent Author Minimal Path](docs/guides/AGENT_AUTHOR_MINIMAL_PATH.md) — safe Tier-2 agent authoring path.
- [Tier-3 Product Hypothesis Contract](docs/guides/TIER3_PRODUCT_HYPOTHESIS_CONTRACT.md) — required before new application hosts.
- [Cursor Token Setup](docs/guides/CURSOR_TOKEN_SETUP.md) — keeps Cursor context bounded.

### Canonical map

| Area | Links |
|------|-------|
| **Strategy & hub** | [Strategy](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) · [System invariants](docs/guides/SYSTEM_INVARIANTS.md) · [Layer completion](docs/guides/LAYER_COMPLETION_MODE.md) · [Architecture hub](docs/intergrax_runtime_architecture.md) · [Ideal model](docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) |
| **Domain canon (22 pairs)** | `docs/architecture/{DOMAIN}.md` ↔ `docs/plan/{DOMAIN}.md` — indexed in [hub](docs/intergrax_runtime_architecture.md) |
| **Multi-layer features** | `docs/features/architecture/{FEATURE}.md` ↔ `docs/features/plan/{FEATURE}.md` — [features/README.md](docs/features/README.md) |
| **Execution** | [UAEP / §42](docs/architecture/UNIFIED_EXECUTION_RUNTIME.md) · [Nexus flow](docs/architecture/NEXUS_EXECUTION_FLOW.md) · [Orchestration](docs/architecture/ORCHESTRATION.md) |
| **Authoring** | [Agent guide](docs/guides/AGENT_CREATION_GUIDE.md) · [Extension guide](docs/guides/EXTENSION_AUTHOR_GUIDE.md) · [applications/USAGE.md](applications/USAGE.md) |
| **Operations** | [HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md) · [infra/README.md](infra/README.md) |
| **ADRs** | [docs/adr/README.md](docs/adr/README.md) |

**Documentation boundary:** platform `docs/` describe the Harness / Agent OS. Each business environment and agent maintains its own `ARCHITECTURE.md` and local plan — see [Strategy §Documentation boundary](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md#documentation-boundary).

**Update rules:** canonical file per topic — strategy → hub → domain pair → guides. Details in [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).

---

## Project snapshot

**Last updated:** 2026-06-17 · **Stage:** active private R&D

| Dimension | Status |
|-----------|--------|
| **Platform maturity** | **32/32 L3** harness baseline — see [Current platform maturity](#current-platform-maturity) · [Maturity Taxonomy](docs/guides/MATURITY_TAXONOMY.md); default [gate maintenance](docs/plan/PLATFORM_FOUNDATION.md#61-harness-platform-maintenance-default--band-1) active |
| **Active development** | Default queue: [§6.1 gate maintenance](docs/plan/PLATFORM_FOUNDATION.md#61-harness-platform-maintenance-default--band-1) · depth bands: [MEM-DEPTH](docs/plan/MEMORY.md), [CRIT-V](docs/plan/CRITIC_VERIFICATION.md), [OBS-BUS](docs/plan/OBSERVABILITY.md) |
| **Business agents** | Phase K — **end of plan** until explicit product prioritization ([§6.3](docs/plan/PLATFORM_FOUNDATION.md#63-end-of-plan--deferred-product-work-only)) |
| **Regression gate** | `uv run pytest -m gate -q` — CI green ([workflow badge](#intergrax)) |

**Also in the platform:**

| Capability | Doc |
|------------|-----|
| **Adaptive Harness Intelligence (L4)** | [architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md](docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) |
| **Critic & Verification (PEV)** | [architecture/CRITIC_VERIFICATION.md](docs/architecture/CRITIC_VERIFICATION.md) |
| **Reasoning & cognition** | [architecture/REASONING_AND_COGNITION.md](docs/architecture/REASONING_AND_COGNITION.md) |
| **Elastic capacity** | [architecture/ELASTIC_CAPACITY_AND_SCALING.md](docs/architecture/ELASTIC_CAPACITY_AND_SCALING.md) |
| **Ephemeral Code Craft** | [architecture/CODE_CRAFT.md](docs/architecture/CODE_CRAFT.md) · [ADR-CODECRAFT-001](docs/adr/entries/2026-06-10/ADR-CODECRAFT-001.md) |

Full phase tracker: [plan/PLATFORM_FOUNDATION.md](docs/plan/PLATFORM_FOUNDATION.md) · [intergrax_runtime_architecture.md](docs/intergrax_runtime_architecture.md)

---

## Local infrastructure

Optional Docker backends for integration development:

```bash
cd infra && ./manage.sh up redis qdrant postgresql
```

[infra/README.md](infra/README.md) · [infra/PORTS.md](infra/PORTS.md) · Lab stack: [HARNESS_ENVIRONMENT.md](docs/guides/HARNESS_ENVIRONMENT.md)

---

## License

All rights reserved © Artur Czarnecki. See [LICENSE](LICENSE).

This repository is in active proprietary R&D and source-available evaluation stage. Commercial licensing, production use, and partnership opportunities require explicit maintainer permission.

---

## Contributing & community

### Public evaluation and collaboration

| Resource | Purpose |
|----------|---------|
| [FAQ.md](FAQ.md) | Common external-reader questions |
| [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) | Evaluation path for technical reviewers, design partners, and integration builders |
| [USE_CASES.md](USE_CASES.md) | Use-case map for governed agent applications, controlled RAG, trace/evidence, orchestration, and tool governance |
| [PARTNERS.md](PARTNERS.md) | Partner and design-partner brief |
| [ROADMAP.md](ROADMAP.md) | Public adoption roadmap, collaboration tracks, and near-term public-facing priorities |
| [COLLABORATION.md](COLLABORATION.md) | Source-available collaboration model, permitted use, contact |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, work cycle, PR process |
| [SECURITY.md](SECURITY.md) | Security policy |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community standards |
| [CITATION.cff](CITATION.cff) | Citation metadata |

### Maintainer and operator resources

The following resources are maintainer/operator-facing and do not create public commitments, support obligations, partnership terms, or license grants.

| Resource | Purpose |
|----------|---------|
| [docs/public-adoption/PUBLIC_LAUNCH_CHECKLIST.md](docs/public-adoption/PUBLIC_LAUNCH_CHECKLIST.md) | Maintainer checklist for public outreach readiness |
| [docs/public-adoption/OUTREACH_KIT.md](docs/public-adoption/OUTREACH_KIT.md) | Maintainer-facing outreach drafts for technical reviewers, integration builders and design partners |
| [AGENTS.md](AGENTS.md) | Cursor auto-load stub — tiers, boundaries, pointers |
| [docs/guides/AGENT_INSTRUCTIONS.md](docs/guides/AGENT_INSTRUCTIONS.md) | Full instructions for AI coding agents |

**Maintainer:** Artur Czarnecki · **Repository:** [Intergrax](https://github.com/jakbuczarnecki/intergrax) · **Contact:** jakbu.czarnecki.83@gmail.com
