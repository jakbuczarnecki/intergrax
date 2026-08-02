# Intergrax FAQ

This FAQ answers common questions for external readers evaluating Intergrax as a source-available Harness AI / Agent OS project. It does not replace [LICENSE](LICENSE), [COLLABORATION.md](COLLABORATION.md), [ROADMAP.md](ROADMAP.md), or the technical architecture canon.

## Is Intergrax open source?

No. Intergrax is public and **source-available proprietary** under the [Intergrax Evaluation and Collaboration License 1.0](LICENSE). Public availability of the repository does not grant open-source rights. Production use, commercial use, hosted services, redistribution as an independent product, and incorporation into products or services require **explicit written permission**.

## Can I clone the repository?

Yes. You may clone and download the Official Repository for evaluation under the [LICENSE](LICENSE).

## Can I run Intergrax locally?

Yes. You may install dependencies and run Quick Start, tests, benchmarks, examples, demos, proof paths, and documented evaluation workloads in a local or isolated non-production environment.

## Can I modify the code?

Yes, for permitted evaluation. You may make private modifications for analysis, testing, debugging, integration assessment, or preparation of a Code Contribution or Documentation Contribution. Production deployment or commercial use of modified code still requires explicit written permission.

## Can I create a fork?

Yes. You may create a public or private GitHub fork as an **Authorized Fork** for evaluation, discussion, and preparation of a Code Contribution or Documentation Contribution. Other evaluators may clone and test your Authorized Fork for permitted evaluation and pull-request collaboration. An Authorized Fork may not be marketed as an independent distribution, hosted service, or commercial offering without separate permission.

## Can a company evaluate Intergrax internally?

Yes. Internal, non-production Evaluation by a commercial organization is permitted. Multiple employees, contractors, and advisers may participate as **Evaluation Participants** in a controlled Evaluation Environment.

## Can multiple employees participate in an evaluation?

Yes, when they act solely as Evaluation Participants in a controlled non-production Evaluation Environment. Using Intergrax as an everyday operational tool for employees is Production Use and requires explicit written permission.

## Can I clone and test another contributor's fork?

Yes. You may clone, run, test, and collaborate on an **Authorized Fork** for Evaluation, code review, and pull-request preparation. Authorized Fork rights do not include Production Use, Commercial Use, or independent distribution.

## Does an issue comment become a code contribution?

No. Ordinary issue comments and suggestions are **Feedback**, not automatic Code Contributions or Documentation Contributions. A pull request or other material clearly offered for inclusion is a Code Contribution or Documentation Contribution under the terms in [LICENSE](LICENSE).

## Who owns external contributions?

Contributors retain copyright in their Code Contributions and Documentation Contributions. The maintainer receives a license to use submitted contributions upon submission as described in [LICENSE](LICENSE). The maintainer does not automatically become the copyright owner.

## Can users evaluate external contributions included in Intergrax?

Yes, for Evaluation and other uses expressly permitted in [LICENSE](LICENSE). Contributor-owned Code Contributions and Documentation Contributions that the Licensor is authorized to sublicense under Section 8 are part of **Licensed Materials** and may be used under the same limited grant as Licensor Materials. Third-party components remain governed by their own licenses and are not sublicensed under Intergrax terms beyond those licenses.

## Do contributions include patent rights?

No automatic patent license is granted. The maintainer may require a separate Contributor License Agreement for substantial or potentially patent-relevant contributions. See [LICENSE](LICENSE) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Can I create a competing product?

Local evaluation, private prototyping, comparison benchmarks, and independent original development are not prohibited by the license competitor clause. Offering, marketing, distributing, or providing to third parties a standalone or hosted platform based substantially on Intergrax and presented as a substitute for the Intergrax Harness AI / Agent OS platform requires explicit written permission. Production Use, Commercial Use, hosting, and incorporation into products still require separate permission regardless.

## Can I submit a pull request?

Yes. You may submit patches and pull requests under the Code Contribution and Documentation Contribution terms in [LICENSE](LICENSE) and [CONTRIBUTING.md](CONTRIBUTING.md). Submission does not guarantee acceptance and does not grant production or commercial rights.

## Can I use Intergrax in production?

Not without explicit written permission. The public repository supports non-production evaluation, review, proof-path feedback, and selected design-partner discussions. See [COLLABORATION.md](COLLABORATION.md).

## Can I build a commercial product on Intergrax?

Not without explicit written permission. Internal, non-production evaluation by a commercial organization is permitted. Production use, commercial use, hosted services, redistribution, and incorporation into products or services require separate permission. See [LICENSE](LICENSE) and [COLLABORATION.md](COLLABORATION.md).

## What is Intergrax in one sentence?

Intergrax is a source-available Harness AI / Agent OS for governed agent applications where agents decide, the harness executes under policy, Nexus orchestrates, and trace/evidence surfaces make runs inspectable.

## Is Intergrax another agent framework?

Not primarily. It can host agents, but its core value is harness/runtime infrastructure: policy-controlled tools, orchestration, memory/RAG, trace/evidence, evaluation, and application hosts. It is closer to a governed agent runtime / harness than a simple agent authoring library.

## What is Nexus?

Nexus is the orchestration/runtime layer that coordinates agent execution, tool use, traces, policy boundaries, and application-hosted workflows. See [INTERGRAX_HARNESS_NARRATIVE.md](docs/guides/INTERGRAX_HARNESS_NARRATIVE.md) and [AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md).

## What do the tiers mean?

- **Tier-0:** shared platform capabilities such as tools, skills, integrations, LLM/RAG/memory.
- **Tier-1:** Nexus / runtime / harness execution.
- **Tier-2:** agents.
- **Tier-3:** applications / hosts / environments.

See [AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md).

## Where should I start?

Start with the README [Start here](README.md#start-here) section, then choose the path that matches your goal:

| Goal | Start with |
|------|------------|
| First-time technical review | [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) |
| Use-case fit | [USE_CASES.md](USE_CASES.md) |
| Design-partner or partner-fit discussion | [PARTNERS.md](PARTNERS.md) |
| Permission, commercial, or production boundaries | [COLLABORATION.md](COLLABORATION.md) |
| Public adoption priorities and feedback tracks | [ROADMAP.md](ROADMAP.md) |

For deeper technical paths after the overview, see the [proof path](README.md#proof-of-platform), [architecture hub](docs/intergrax_runtime_architecture.md), [multi-layer feature docs](docs/features/README.md), [BoundaryAttest case study](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md), [Local Knowledge Workspace alpha](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md), or [Agent Creation Guide](docs/guides/AGENT_CREATION_GUIDE.md).

## What does the BoundaryAttest case study prove?

It validates an integration pattern: Intergrax host-signed boundary events can be verified by an external project and preserved in a separate `client_observed` wrapper. It does not prove production certification, compliance, security certification, or legal attestation. See [BOUNDARYATTEST_ATTESTATION_POC.md](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md).

## What is Local Knowledge Workspace?

Local Knowledge Workspace is an alpha/product-validation direction exploring local governed assistant workflows over user-controlled files. It validates the harness on document discovery, RAG, memory, policy boundaries, trace/evidence, and Tier-3 hosting. It is not a finished product or SaaS. See [LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md).

## Can I contribute?

Yes. Feedback, proof-path reports, documentation fixes, evaluation-only integrations, patches, pull requests, integration proposals, and design-partner discussions are welcome under the [Intergrax Evaluation and Collaboration License 1.0](LICENSE). Substantial work requires prior discussion. See [CONTRIBUTING.md](CONTRIBUTING.md) and [COLLABORATION.md](COLLABORATION.md).

## Can I build an application or agent on Intergrax?

For non-production evaluation and design discussion, yes, within the permitted evaluation model. You may create test agents, application hosts, plugins, and integrations for evaluation. Production and commercial use require explicit written permission. See [AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md), [COLLABORATION.md](COLLABORATION.md), and [LICENSE](LICENSE).

## How should I report security issues?

Do not open public issues for vulnerabilities. Follow [SECURITY.md](SECURITY.md).

## Related documents

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Overview and start paths |
| [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) | Focused evaluation path for reviewers and design partners |
| [USE_CASES.md](USE_CASES.md) | Use-case map for validation and partner-fit discussions |
| [PARTNERS.md](PARTNERS.md) | Partner and design-partner brief |
| [COLLABORATION.md](COLLABORATION.md) | Collaboration and permission model |
| [LICENSE](LICENSE) | Intergrax Evaluation and Collaboration License 1.0 |
| [ROADMAP.md](ROADMAP.md) | Public adoption roadmap |
| [SECURITY.md](SECURITY.md) | Security reporting |
| [INTERGRAX_HARNESS_NARRATIVE.md](docs/guides/INTERGRAX_HARNESS_NARRATIVE.md) | Harness narrative |
| [AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) | Agent/application authoring model |
| [BOUNDARYATTEST_ATTESTATION_POC.md](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) | Attestation case study |
| [LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) | LKW alpha narrative |
