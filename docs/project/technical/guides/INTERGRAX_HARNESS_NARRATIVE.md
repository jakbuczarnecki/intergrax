# Intergrax Harness Narrative

## One-sentence summary

Intergrax is an evidence-backed Harness AI platform for building governed multi-agent systems where agents decide, the harness executes under policy, and Nexus orchestrates.

## The problem

Many agent frameworks work well for demos but collapse responsibilities when you move toward production. Planning, policy enforcement, tool I/O, memory and RAG wiring, trace capture, evaluation hooks, multi-agent routing, and production evidence often land in a single author-facing class or an ad hoc script layer. That pattern hides control flow, makes state untyped, and turns every new agent into a miniature operating system. Teams then rebuild the same infrastructure—budgets, observability, policy gates, evidence packaging—for each product or pilot, with no durable substrate to swap agents, compare runs, or onboard external reviewers.

## The Intergrax position

Intergrax separates concerns across four explicit tiers so each layer answers a different question:

| Layer | Role | Answers |
|-------|------|---------|
| **Application (Tier-3)** | Environment and product wiring | Who is the tenant? Which tools, memory, and RAG profile apply? What org policy and production gates apply? |
| **Nexus (Tier-1)** | Multi-agent orchestration | Which agents run on this task? How does the graph, HITL, and checkpointing behave at task level? |
| **Agent (Tier-2)** | Domain cognition per session | What is the next move? Is the plan valid? Should the session complete, fail, or pause for a human? |
| **Harness (Tier-0 / execution substrate)** | Deterministic execution cycle | Is policy allowed? Is state merged safely? Is trace recorded? Are budgets enforced? Where is evidence collected? |

Agents own **domain decisions** inside a typed session loop. The harness owns **policy, tools, trace, memory/RAG integration, budgets, and evidence**. Nexus owns **graph-level coordination and routing**. Applications own **identity, environment profiles, and production gates**. None of these roles are folded into a single mega-class.

## Why this is a harness, not just an agent framework

**The harness is the product; agents are replaceable.**

An agent framework optimizes for authoring one clever agent quickly. A harness platform optimizes for **reusable execution infrastructure** that survives agent churn, product changes, and external review. Value comes from portable policy, consistent trace and observability, governed tool and skill catalogs, memory and RAG profiles wired through the kernel, multi-agent orchestration without hardcoded class names, and an evidence path that shows what the platform can produce locally—not from any single agent implementation.

When you swap a research agent for a legal or operations agent, the harness cycle, evidence surfaces, and orchestration model stay the same. When a technical partner evaluates the platform, they inspect durable substrate and proof artifacts, not a demo-specific prompt stack.

## What the evidence proof path shows

Intergrax ships a canonical **local** proof path so early adopters, platform engineers, and external reviewers can verify evidence production without running production workloads or calling real providers. Run:

```bash
uv run intergrax certify core --level L2
uv run intergrax trace export
uv run intergrax evidence live-core
uv run intergrax evidence eval
uv run intergrax evidence cost
uv run intergrax evidence posture
uv run intergrax evidence posture export
```

This sequence verifies local ability to produce and aggregate:

- core certification evidence
- trace evidence
- selected local live Tier-0 probe evidence
- eval regression evidence
- cost evidence
- evidence posture scoreboard

Artifacts land under `build/evidence`. Start with `build/evidence/posture/posture.md`, then drill into individual reports.

After the proof path has been run, a lightweight checker confirms expected artifacts and README proof-path references:

```bash
python scripts/maintenance/check_evidence_artifacts.py
```

The checker validates local files and documentation consistency; it does not execute the proof path or import runtime modules.

## What it does not claim

The proof path is **local verification of evidence packaging and aggregation**, not a substitute for production certification. It is explicitly **not**:

- production runtime certification
- security or compliance attestation
- real provider execution
- real LLM evaluation
- billing
- provider pricing
- cloud cost estimation
- product-specific acceptance

Treat posture and certification artifacts as harness-platform evidence for onboarding and review—not as enterprise attestation or go-live approval for a specific product.

## Who this is for

- **AI systems architects** designing governed multi-agent platforms with clear separation of orchestration, cognition, and execution
- **Agent platform engineers** who need reusable policy, trace, tool, and evidence infrastructure across agents
- **Multi-agent runtime developers** building Nexus graphs and typed agent session loops
- **Product teams shipping governed agents** who want the same agent class from lab to Tier-3 host without architectural forks
- **Technical partners evaluating the platform** who need a skimmable narrative and a repeatable local proof path before deeper integration

## Current proof path

- **Run and interpret locally:** [README.md](../../../../README.md) § Proof of platform
- **Roadmap and HEP status:** [HARNESS_EVIDENCE_PACK.md](../../maintainers/plans/HARNESS_EVIDENCE_PACK.md)
- **Architecture framing:** [EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md](../../architecture/satellites/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_production_gates.md)
