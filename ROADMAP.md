# Intergrax Public Roadmap

This roadmap is **public-facing**. It describes the active product-validation program, adoption paths, feedback priorities, proof paths, demos, and maintainer-approved collaboration tracks. It does **not** replace the technical implementation plan or architecture canon.

Canonical technical architecture and implementation status remain in:

- [`docs/intergrax_runtime_architecture.md`](docs/intergrax_runtime_architecture.md)
- [`docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)
- Domain-layer pairs under [`docs/architecture/`](docs/architecture/) and [`docs/plan/`](docs/plan/)
- Multi-layer feature pairs under [`docs/features/architecture/`](docs/features/architecture/) and [`docs/features/plan/`](docs/features/plan/) — [`docs/features/README.md`](docs/features/README.md)
- The canonical Local Knowledge Workspace execution plan: [`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md)

---

## Status

- Intergrax is **source-available and proprietary**.
- A **harness platform baseline** exists and is available for evaluation.
- The primary active development program is the **Local Knowledge Workspace (LKW) MVP**.
- Intergrax is being developed **application-first**: a real LKW product workflow drives discovery, implementation and validation of reusable platform capabilities.
- The primary public goals are **technical review**, **design-partner discovery**, **proof-path feedback**, **selected integrations**, and eventual **real-user validation of LKW**.
- Intergrax is **not** currently presented as a finished SaaS or a general open-source framework.

---

## Primary product-validation program: Local Knowledge Workspace

Local Knowledge Workspace is the current primary product-development and platform-validation program for Intergrax.

LKW is a **deployment-neutral Hybrid Knowledge Workspace**: private by default, tenant-scoped, with indexed RAG knowledge, controlled live provider access (planned), unified evidence provenance, and storage location selected by configuration and provider wiring. **“Local”** means user-controlled deployment and first-class self-hosted topology — not “all data always on one device.” Canonical contract: [LKW Architecture — Deployment, storage and tenancy model](applications/local_workspace_application/docs/ARCHITECTURE.md#deployment-storage-and-tenancy-model). Hybrid knowledge access: [KNOWLEDGE_ACCESS_ARCHITECTURE.md](applications/local_workspace_application/docs/KNOWLEDGE_ACCESS_ARCHITECTURE.md).

Intergrax vendor integrations are designed as **reusable foundations** for three consumption modes:

```text
indexed RAG
durable application/database materialization
controlled live access
```

The same provider integration is reused across modes; the lifecycle is **not** the same. Database materialization does not automatically mean RAG. Live access remains **planned** — not yet implemented as a provider-neutral executor.

The immediate product goal is:

```text
controlled channel-neutral knowledge intake (indexed)
→ shared provider foundation (durable + live branches)
→ Workspace Knowledge Configuration (Connections, Indexed Sources, Live Access Bindings)
→ Hybrid Ask (indexed + authorized live evidence)
→ natural-language frontends (Slack and others)
→ live platform proof
```

The first implemented source slice commonly uses **local-folder** documents. Slack is the way the user talks to LKW today. Separately, Slack is also targeted as an explicitly connected and searchable knowledge source — that capability is **not** available yet.

LKW has three connected roles:

| Role | Priority | Meaning |
|------|----------|---------|
| Real product | Primary | Solve a real private knowledge-workspace workflow for a knowledge worker |
| Platform proof | Secondary | Demonstrate Intergrax capabilities through a complete working application |
| Platform problem detector | Secondary | Expose concrete reusable platform gaps through real product pressure |

Implementation order is driven by user value. Platform mechanisms are added when the active LKW workflow requires them, not as an independent platform-first backlog.

The canonical source for the current LKW stage, completed slices, active task and deferred scope is the [LKW Implementation Plan](applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md).

---

## Public Discussion Map

The open curated GitHub issues are a public discussion map for evaluating Intergrax as a **Harness AI / Agent OS** platform. They are not a generic implementation backlog, public support queue, roadmap commitment, security vulnerability channel, commercial-use permission path, redistribution permission path, or open-source task board.

The canonical entry point is the [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md). The source data for the active issue map is [curated_public_issues.yml](docs/public-adoption/curated_public_issues.yml).

---

## Now

| Item | Notes |
|------|-------|
| **Build the LKW MVP as the primary active Intergrax program** | Hybrid Knowledge Workspace: indexed intake → Workspace Knowledge Configuration → Hybrid Ask → natural-language Slack proof (see [Implementation Plan](applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md)) |
| **WEB_URL end-to-end intake** | `LKW-WORKSPACE-CONTENTS-1B-5-2` — **ACCEPTED** (including C1 and C2 corrections) |
| **Hybrid knowledge access architecture** | `LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1` — **ACCEPTED** — [KNOWLEDGE_ACCESS_ARCHITECTURE.md](applications/local_workspace_application/docs/KNOWLEDGE_ACCESS_ARCHITECTURE.md) |
| **Vendor knowledge three-mode reuse architecture** | `VENDOR-KNOWLEDGE-THREE-MODE-REUSE-ARCH-1` — **ACCEPTED** — one provider integration reused for indexed RAG, durable materialization and planned live access; [KNOWLEDGE_SOURCE_INTEGRATIONS.md](docs/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md) |
| **Slack Knowledge three-mode architecture** | `SLACK-KNOWLEDGE-THREE-MODE-ARCH-1` — **DONE** — freeze one `SlackConversationChannelIntegration` reused across indexed RAG, durable materialization and live access; Slack frontend and knowledge-source roles are independent |
| **Model runtime portability** | `LKW-MODEL-RUNTIME-1` — **ACCEPTED** — full canonical LKW proof for Ollama `qwen2.5:14b` and vLLM `Qwen/Qwen2.5-3B-Instruct`; [evidence](applications/local_workspace_application/docs/evidence/LKW_MODEL_RUNTIME_PORTABILITY.md) |
| **Workspace Knowledge Configuration** | `LKW-KNOWLEDGE-ACCESS-1` — **NEXT** — durable tenant Connection Catalog, restart-safe Connection configuration, SecretsStore-owned credentials, runtime registry rehydration; Remote Resources, Indexed Sources, Live Access Bindings and bounded Query Policies |
| Trusted Ask Workspace available | Surface-neutral HTTP Ask Workspace over **indexed** knowledge; grounded answers, citations and persisted runs are implemented and live-verified |
| Slack conversational MVP in active development | User can operate LKW through Slack **DM** and ask about knowledge already in the workspace (temporary in-memory personal selection); provider-neutral conversation context architecture with observed-audience validation frozen for review (`LKW-CONVERSATION-CONTEXT-ARCH-1`); durable bindings, shared-channel runtime, shared source eligibility and Slack channel/conversation history as searchable workspace knowledge are **not** yet available |
| Knowledge Intake architecture being frozen | Channel-neutral intake and asynchronous ingestion contract documented; managed-file upload, Source Candidate intake and end-to-end `WEB_URL` indexed intake are **ACCEPTED** — see [Knowledge Intake discovery](applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md) |
| Preserve application-first platform development | Concrete LKW blockers may produce reusable Intergrax improvements; unrelated platform expansion does not override the LKW MVP path |
| Source-available collaboration model clarified | See [COLLABORATION.md](COLLABORATION.md) and [LICENSE](LICENSE) |
| Core harness proof path available from README | Local evaluation path documented in [README.md](README.md) |
| Lab host / local execution path available | Tier-3 local host and lab workflows |
| Attestation Demo available as the primary external-integration proof | [applications/attestation_demo/README.md](applications/attestation_demo/README.md) |
| BoundaryAttest / attestation case study available | Includes the validation flow diagram; see [docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) |
| Agent and application scaffolding available | Scaffold tooling under `scaffold/`; see [CONTRIBUTING.md](CONTRIBUTING.md) |
| Public evaluation entry points available | [README.md](README.md) · [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) · [USE_CASES.md](USE_CASES.md) · [PARTNERS.md](PARTNERS.md) |
| Curated public issue map active | [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md) lists active discussion waves |
| Canonical issue automation available | [curated_public_issues.yml](docs/public-adoption/curated_public_issues.yml) is the source of truth; [manage_curated_issues.bat](scripts/public_adoption/manage_curated_issues.bat) provides dry/apply/check workflow |

---

## Next

| Item | Notes |
|------|-------|
| **LKW-KNOWLEDGE-ACCESS-1** | **NEXT** — durable tenant Connection Catalog, restart-safe Connection configuration, SecretsStore-owned credentials, runtime registry rehydration; Remote Resources, Indexed Sources, Live Access Bindings and Query Policy |
| **LKW-CONVERSATION-CONTEXT-ARCH-1** | **READY_FOR_REVIEW** — provider-neutral conversational context with observed-audience validation, binding identity, workspace resolution, thread memory isolation, shared `READ_ONLY_ASK` boundary and deterministic guards; proved first through Slack |
| **Slack Knowledge vertical (application-first priority)** | Platform `SLACK-KNOWLEDGE-FOUNDATION-1` **DONE** → architecture `LKW-CONVERSATION-CONTEXT-ARCH-1` **READY_FOR_REVIEW** → LKW `LKW-SLACK-CONNECTED-SOURCE-1` **DONE** → LKW `LKW-CONVERSATION-CONTEXT-1` **NEXT** → LKW `LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1` → platform `SLACK-LIVE-CAPABILITY-1`; final proof joins `LKW-HYBRID-ASK-1` at `LKW-SLACK-KNOWLEDGE-PROOF-1`; complete vertical precedes Microsoft Graph Calendar |
| **Vendor knowledge durable + live branches** | Microsoft Graph Calendar adapter after Slack vertical; durable materialization sink contract; live capability contract and executor — convergence at Hybrid Ask |
| **LKW-HYBRID-ASK-1** | Indexed RAG + authorized live evidence with unified provenance |
| **LKW-CONVERSATIONAL-FRONTEND-1** | Natural-language planner execution and Slack cutover (`CONV-1B`, `CONV-1C`) |
| **LKW-VENDOR-ACCESS-COLLABORATION-1** | Microsoft 365, Jira, Confluence indexed and live access |
| **LKW-VENDOR-ACCESS-DATA-1** | Databricks, Power BI, Atlan read-only live access |
| **LKW-KNOWLEDGE-LIFECYCLE-1** | Shared synchronization, freshness, permissions, safe removal |
| **LKW-LIVE-PLATFORM-PROOF-1** | Complete demonstrable Slack platform proof |
| Complete the end-to-end LKW Slack workflow | Natural-language Hybrid Ask with citations — final proof target |
| Provide a repeatable design-partner setup | A real user must be able to start and try the controlled LKW environment without ad hoc developer reconstruction |
| Run first real-user LKW validation | Measure usefulness, citation correctness, time saved, trust and blockers to repeated use |
| Feed validated LKW gaps back into Intergrax | Classify concrete product blockers and implement reusable platform fixes only where justified |
| Improve first-run and proof-path clarity | Reduce friction for external evaluators and design partners |
| Maintain the curated public issue map | Active issues remain a discussion and evaluation surface, not the implementation source of truth |
| Improve the public demo path for trace, evidence and boundary events | Provide clearer end-to-end evaluation of harness observability |
| Optionally add demo media for the attestation case study | The validation-flow diagram is published; short demo media remains optional |

---

## Later

| Item | Notes |
|------|-------|
| Extend LKW based on validated user demand | Candidates include workspace history and outputs, fuller document reconciliation and improved operational diagnostics |
| Develop an LKW local companion | Installation, configuration, status and diagnostics toward an installable LKW 1.0 |
| Add a second conversational adapter when justified | Microsoft Teams or another surface should follow demonstrated user or commercial demand, not abstraction proof alone |
| Harden LKW toward a 1.0 release | Security, operations, updates, recovery and broader repeatability after MVP validation |
| Expand design-partner integrations | Beyond the current LKW and attestation tracks |
| Package selected evidence and governance capabilities for production-oriented partners | For teams with explicit permission and partnership scope |
| Decide long-term commercial / open-core / source-available packaging | Depends on product validation, design-partner feedback and maintainer direction |
| Consider hosted documentation or a landing page | After the core product, proof and collaboration narratives stabilize |

---

## Collaboration tracks

| Track | Focus | Who it is for | Expected first action |
|-------|-------|---------------|----------------------|
| **LKW MVP design-partner validation** | Real private knowledge-workspace workflows (first slice: local-folder), Slack interaction, grounded answers, source verification and product fit | Knowledge workers and teams willing to test a controlled self-hosted LKW installation | Review the [LKW Implementation Plan](applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md), [alpha narrative](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) and [LKW architecture](applications/local_workspace_application/docs/ARCHITECTURE.md); discuss a concrete validation workflow |
| Proof path feedback | Run local evaluation paths; report friction, gaps and unclear steps | Engineers evaluating the harness baseline | Follow [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) and the [proof path](README.md#proof-of-platform) in [README.md](README.md); open an issue with concrete findings |
| Attestation / boundary events integration | Host attestation flows, boundary events and external trust integration | Teams building attestation, security or compliance integrations | Review [Attestation Demo](applications/attestation_demo/README.md); propose scope via issue or maintainer contact |
| Documentation clarity | Corrections, gaps, readability and navigation improvements | Anyone reading public docs | Open an issue or PR with a specific documentation fix; see [CONTRIBUTING.md](CONTRIBUTING.md) |
| Agent / application proposals | New agent classes or Tier-3 application ideas aligned with harness boundaries | Agent architects and product teams | Review [USE_CASES.md](USE_CASES.md) and [PARTNERS.md](PARTNERS.md); propose scope before substantial work; align with [COLLABORATION.md](COLLABORATION.md) |
| Governance, observability and evaluation feedback | Trace, evidence, policy and evaluation workflows | Platform engineers and governance builders | Run proof paths; inspect evidence outputs; report gaps with reproducible steps |

Prior discussion is recommended before substantial work on any track. Maintainer approval applies to production, commercial and redistribution use.

---

## What is intentionally not promised

- **No claim of a finished SaaS** — Intergrax is not offered as a hosted product today.
- **No claim that the LKW MVP is already complete** — the end-to-end user workflow and real-user validation remain active roadmap work.
- **No open-source license grant** — the repository is source-available under proprietary terms; see [LICENSE](LICENSE).
- **No production certification claim** — maturity and evidence docs describe harness readiness; they are not a production guarantee.
- **No guarantee that all demos are production-ready** — demos and proof paths are for evaluation and integration discovery.
- **No guarantee that all proposed contributions will be accepted** — scope, boundaries and maintainer direction apply.
- **No commitment to support all use cases** — collaboration is focused on aligned product-validation, design-partner and evaluation tracks.

---

## Related documents

- [README.md](README.md) — project overview, proof path and documentation index
- [LKW Implementation Plan](applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) — canonical LKW product brief, MVP execution order, active task and deferred scope
- [LKW Hybrid Knowledge Access Architecture](applications/local_workspace_application/docs/KNOWLEDGE_ACCESS_ARCHITECTURE.md) — indexed, live and hybrid modes; Connections; Live Access Bindings; Hybrid Ask roadmap
- [LKW Architecture](applications/local_workspace_application/docs/ARCHITECTURE.md) — LKW ownership, boundaries and runtime shape
- [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md) — external verification of completed LKW capabilities
- [Local Knowledge Workspace Alpha](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) — public product-validation narrative
- [Product-First MVP](docs/plan/PRODUCT_FIRST_MVP.md) — governing application-first product-development rule
- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) — focused evaluation path for reviewers and design partners
- [USE_CASES.md](USE_CASES.md) — use-case map for validation and partner-fit discussions
- [PARTNERS.md](PARTNERS.md) — partner and design-partner brief
- [COLLABORATION.md](COLLABORATION.md) — collaboration model, permitted use and contact
- [Token Optimization feature plan](docs/features/plan/TOKEN_OPTIMIZATION.md) — **TOKEN-10A–10D** accepted/closed; **CTX-UCL-2** ready for review; **TOKEN-10E** blocked until CTX-UCL-CLOSEOUT-1
- [LangChain Independence](docs/features/architecture/LANGCHAIN_INDEPENDENCE.md) — cross-layer migration roadmap; active stage **LCI-0A** (inventory); target **LangChain-free core + optional compatibility**; does not change LKW status; next after review: **LCI-0B** boundary guard — [feature plan](docs/features/plan/LANGCHAIN_INDEPENDENCE.md)
- [Public Adoption Documents](docs/public-adoption/README.md) — public-adoption control documents, issue index, triage playbook and automation source
- [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md) — active curated public issue map and recommended evaluation paths
- [Public Discussion Issue Expansion](docs/public-adoption/PUBLIC_DISCUSSION_ISSUE_EXPANSION.md) — active architecture, product-validation and deep technical discussion issue waves
- [Curated Public Issue Drafts](docs/public-adoption/CURATED_PUBLIC_ISSUES.md) — rationale for curated public discussions
- [curated_public_issues.yml](docs/public-adoption/curated_public_issues.yml) — canonical source data for public issue automation
- [LICENSE](LICENSE) — proprietary terms
- [CONTRIBUTING.md](CONTRIBUTING.md) — development workflow and requirements
- [Attestation Demo](applications/attestation_demo/README.md) — attestation integration proof
- [BoundaryAttest Case Study](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) — external validation case study
- [Intergrax Harness Narrative](docs/guides/INTERGRAX_HARNESS_NARRATIVE.md) — harness narrative for external readers
