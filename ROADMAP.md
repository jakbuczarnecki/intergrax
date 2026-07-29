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

LKW is a **deployment-neutral** knowledge workspace: private by default, tenant-scoped, with storage location selected by configuration and provider wiring. **“Local”** means user-controlled deployment and first-class self-hosted topology — not “all data always on one device.” Canonical contract: [LKW Architecture — Deployment, storage and tenancy model](applications/local_workspace_application/docs/ARCHITECTURE.md#deployment-storage-and-tenancy-model).

The immediate product goal is:

```text
controlled channel-neutral knowledge intake
→ durable asynchronous processing
→ grounded Ask across replaceable clients
→ real-user validation
```

The first implemented source slice commonly uses **local-folder** documents. Slack remains one optional frontend over the same LKW capabilities — not the ingestion engine.

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
| **Build the LKW MVP as the primary active Intergrax program** | Complete the smallest real product experience: channel-neutral knowledge intake → durable async processing → grounded Ask across replaceable clients (first source slice commonly local-folder; Slack is one optional frontend) |
| Trusted Ask Workspace available | Surface-neutral HTTP Ask Workspace, grounded answers, citations and persisted runs are implemented and live-verified; see the [LKW Implementation Plan](applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) |
| Slack conversational MVP in active development | Connect the existing Ask Workspace capability to an approved Slack user and workspace through the governed interaction path; source inspection (`sources`) is operator-verified |
| Knowledge Intake architecture being frozen | Channel-neutral intake and asynchronous ingestion contract documented for review; upload / URL / Slack attachment intake are **not** claimed as implemented yet — see [Knowledge Intake discovery](applications/local_workspace_application/docs/KNOWLEDGE_INTAKE_DISCOVERY.md) |
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
| Complete the end-to-end LKW Slack workflow | One approved user selects a workspace, asks a real question and receives a grounded answer with verifiable sources |
| Deliver durable Knowledge Intake foundation | Channel-neutral intake submission, durable ingestion operation, idempotent acceptance and queue/worker boundary before channel-specific adapters |
| Add managed file upload and later Slack attachment mapping | Core upload capability first; Slack attachments map only after the LKW capability exists |
| Add connected-source candidate and explicit web URL intake | Safe candidate selection for local-capable connectors; explicit URL intake under policy (other clients such as Teams/mobile/Telegram remain channel-neutrality examples, not committed deliverables) |
| Surface operation status and completion notification | Channel-neutral lifecycle events with conversation correlation back to replaceable clients |
| Add the minimum outbound-data warning and policy required by the MVP | Make the local-to-provider boundary understandable and operationally safe enough for controlled validation |
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
- [LKW Architecture](applications/local_workspace_application/docs/ARCHITECTURE.md) — LKW ownership, boundaries and runtime shape
- [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md) — external verification of completed LKW capabilities
- [Local Knowledge Workspace Alpha](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) — public product-validation narrative
- [Product-First MVP](docs/plan/PRODUCT_FIRST_MVP.md) — governing application-first product-development rule
- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) — focused evaluation path for reviewers and design partners
- [USE_CASES.md](USE_CASES.md) — use-case map for validation and partner-fit discussions
- [PARTNERS.md](PARTNERS.md) — partner and design-partner brief
- [Token Optimization feature plan](docs/features/plan/TOKEN_OPTIMIZATION.md) — **TOKEN-10** cache-aware universal runtime and proof (active); LKW-PF6 product proof follows universal platform proof
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
