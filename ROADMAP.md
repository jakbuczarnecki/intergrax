# Intergrax Public Roadmap

This roadmap is **public-facing**. It describes adoption paths, feedback priorities, proof paths, demos, and maintainer-approved collaboration tracks. It does **not** replace the technical implementation plan or architecture canon.

Canonical technical architecture and implementation status remain in:

- [`docs/intergrax_runtime_architecture.md`](docs/intergrax_runtime_architecture.md)
- [`docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](docs/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)
- Relevant paired documents under [`docs/architecture/`](docs/architecture/) and [`docs/plan/`](docs/plan/)

---

## Status

- Intergrax is **source-available and proprietary**.
- A **harness platform baseline** exists and is available for evaluation.
- The primary public goal is **technical review**, **design-partner discovery**, **proof-path feedback**, and **selected integrations**.
- Intergrax is **not** currently presented as a finished SaaS or a general open-source framework.

---

## Now

| Item | Notes |
|------|-------|
| Source-available collaboration model clarified | See [COLLABORATION.md](COLLABORATION.md) and [LICENSE](LICENSE) |
| Core harness proof path available from README | Local evaluation path documented in [README.md](README.md) |
| Lab host / local execution path available | Tier-3 local host and lab workflows |
| Attestation Demo available as the primary external-integration proof | [applications/attestation_demo/README.md](applications/attestation_demo/README.md) |
| BoundaryAttest / attestation case study available (includes validation flow diagram) | [docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) |
| Boundary events / host attestation flow documented | Attestation demo and related architecture docs |
| Agent and application scaffolding available | Scaffold tooling under `scaffold/`; see [CONTRIBUTING.md](CONTRIBUTING.md) |
| Local Knowledge Workspace exists as the first product-validation direction / alpha track | [Alpha narrative](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) · [Architecture](applications/local_workspace_application/ARCHITECTURE.md) |
| Public evaluation entry points available | [README.md](README.md) · [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) · [USE_CASES.md](USE_CASES.md) · [PARTNERS.md](PARTNERS.md) |
| Curated public issue routing available | [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md) |
| Public issue templates / feedback routing available | Structured entry points for feedback, bug reports, integration proposals, and design-partner interest |
| Expanded public discussion waves prepared | Architecture, product-validation, and deep technical discussion waves are defined in [Public Discussion Issue Expansion](docs/public-adoption/PUBLIC_DISCUSSION_ISSUE_EXPANSION.md) and [curated_public_issues.yml](docs/public-adoption/curated_public_issues.yml) |

---

## Next

| Item | Notes |
|------|-------|
| Improve first-run / proof-path clarity | Reduce friction for external evaluators |
| Maintain curated public issue routing | Active and prepared issues are defined in [curated_public_issues.yml](docs/public-adoption/curated_public_issues.yml); active issues are listed in [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md) |
| Open expanded discussion waves selectively | Use [curated_public_issues.yml](docs/public-adoption/curated_public_issues.yml) for Wave 3 architecture, Wave 4 product-validation, and Wave 5 deep technical discussion issues |
| Refine issue templates based on first external feedback | Feedback, bug reports, integration proposals, and design-partner interest have structured entry points |
| Optionally add demo media for the attestation case study | Validation-flow diagram is published; optional future work is short demo media for the [BoundaryAttest case study](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) |
| Collect Local Knowledge Workspace alpha feedback from design partners | [Alpha narrative](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) published; structured partner feedback welcome |
| Improve public demo path for trace, evidence and boundary events | Clearer end-to-end evaluation of harness observability |

---

## Later

| Item | Notes |
|------|-------|
| Decide long-term commercial / open-core / source-available packaging | Depends on design-partner feedback and maintainer direction |
| Expand design-partner integrations | Beyond current attestation and alpha tracks |
| Mature Local Knowledge Workspace or other product validation tracks based on feedback | Product direction follows validated partner input |
| Package selected evidence / governance capabilities for production-oriented partners | For teams with explicit permission and partnership scope |
| Consider hosted documentation / landing page after the core public narrative stabilizes | After proof paths and collaboration tracks are clear |

---

## Collaboration tracks

| Track | Focus | Who it is for | Expected first action |
|-------|-------|---------------|----------------------|
| Proof path feedback | Run local evaluation paths; report friction, gaps, and unclear steps | Engineers evaluating the harness baseline | Follow [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) and the [proof path](README.md#proof-of-platform) in [README.md](README.md); open an issue with concrete findings |
| Attestation / boundary events integration | Host attestation flows, boundary events, external trust integration | Teams building attestation, security, or compliance integrations | Review [Attestation Demo](applications/attestation_demo/README.md); propose scope via issue or maintainer contact |
| Local Knowledge Workspace alpha feedback | Early capabilities, UX, and fit for local knowledge workflows | Design partners interested in product-validation direction | Read [USE_CASES.md](USE_CASES.md), [alpha narrative](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md), and [LKW architecture](applications/local_workspace_application/ARCHITECTURE.md); share structured alpha feedback |
| Documentation clarity | Corrections, gaps, readability, and navigation improvements | Anyone reading public docs | Open an issue or PR with a specific doc fix; see [CONTRIBUTING.md](CONTRIBUTING.md) |
| Agent / application proposals | New agent classes or Tier-3 application ideas aligned with harness boundaries | Agent architects and product teams | Review [USE_CASES.md](USE_CASES.md) and [PARTNERS.md](PARTNERS.md); propose scope before substantial work; align with [COLLABORATION.md](COLLABORATION.md) |
| Governance, observability and evaluation feedback | Trace, evidence, policy, and evaluation workflows | Platform engineers and governance builders | Run proof paths; inspect evidence outputs; report gaps with reproducible steps |

Prior discussion is recommended before substantial work on any track. Maintainer approval applies to production, commercial, and redistribution use.

---

## What is intentionally not promised

- **No claim of a finished SaaS** — Intergrax is not offered as a hosted product today.
- **No open-source license grant** — the repository is source-available under proprietary terms; see [LICENSE](LICENSE).
- **No production certification claim** — maturity and evidence docs describe harness readiness; they are not a production guarantee.
- **No guarantee that all demos are production-ready** — demos and proof paths are for evaluation and integration discovery.
- **No guarantee that all proposed contributions will be accepted** — scope, boundaries, and maintainer direction apply.
- **No commitment to support all use cases** — collaboration is focused on aligned design-partner and evaluation tracks.

---

## Related documents

- [README.md](README.md) — project overview, proof path, documentation index
- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) — focused evaluation path for reviewers and design partners
- [USE_CASES.md](USE_CASES.md) — use-case map for validation and partner-fit discussions
- [PARTNERS.md](PARTNERS.md) — partner and design-partner brief
- [COLLABORATION.md](COLLABORATION.md) — collaboration model, permitted use, contact
- [Public Adoption Documents](docs/public-adoption/README.md) — public-adoption control documents, issue index, triage playbook, and automation source
- [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md) — active curated public issues and recommended evaluation paths
- [Public Discussion Issue Expansion](docs/public-adoption/PUBLIC_DISCUSSION_ISSUE_EXPANSION.md) — expanded architecture, product-validation, and deep technical discussion issue waves
- [Curated Public Issue Drafts](docs/public-adoption/CURATED_PUBLIC_ISSUES.md) — maintainer-curated public issue drafts for proof-path feedback, documentation clarity, selected integration feedback, and design-partner discovery
- [curated_public_issues.yml](docs/public-adoption/curated_public_issues.yml) — single canonical source data for active and expanded public issue automation
- [LICENSE](LICENSE) — proprietary terms
- [CONTRIBUTING.md](CONTRIBUTING.md) — development workflow and requirements
- [applications/attestation_demo/README.md](applications/attestation_demo/README.md) — attestation integration proof
- [docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) — BoundaryAttest external validation case study
- [docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) — Local Knowledge Workspace alpha / product-validation narrative
- [applications/local_workspace_application/ARCHITECTURE.md](applications/local_workspace_application/ARCHITECTURE.md) — Local Knowledge Workspace architecture
- [docs/guides/INTERGRAX_HARNESS_NARRATIVE.md](docs/guides/INTERGRAX_HARNESS_NARRATIVE.md) — harness narrative for external readers
