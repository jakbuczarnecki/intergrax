# Intergrax FAQ

This FAQ answers common questions for external readers evaluating Intergrax as a source-available Harness AI / Agent OS project. It does not replace [LICENSE](LICENSE), [COLLABORATION.md](COLLABORATION.md), [ROADMAP.md](ROADMAP.md), or the technical architecture canon.

## Is Intergrax open source?

No. Intergrax is public and source-available for evaluation and technical partner discovery. Use, modification, redistribution, derivative works, production use, and commercial use require permission under [LICENSE](LICENSE) and [COLLABORATION.md](COLLABORATION.md).

## Can I use Intergrax in production?

Not without explicit permission. The public repository is for evaluation, review, proof-path feedback, and selected design-partner discussions. See [COLLABORATION.md](COLLABORATION.md).

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

Start with the README [Start here](README.md#start-here) section, the [proof path](README.md#proof-of-platform), [ROADMAP.md](ROADMAP.md), and [COLLABORATION.md](COLLABORATION.md). Then choose a relevant path:

- [BoundaryAttest case study](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md)
- [Local Knowledge Workspace alpha](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md)
- [Agent Creation Guide](docs/guides/AGENT_CREATION_GUIDE.md)

## What does the BoundaryAttest case study prove?

It validates an integration pattern: Intergrax host-signed boundary events can be verified by an external project and preserved in a separate `client_observed` wrapper. It does not prove production certification, compliance, security certification, or legal attestation. See [BOUNDARYATTEST_ATTESTATION_POC.md](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md).

## What is Local Knowledge Workspace?

Local Knowledge Workspace is an alpha/product-validation direction exploring local governed assistant workflows over user-controlled files. It validates the harness on document discovery, RAG, memory, policy boundaries, trace/evidence, and Tier-3 hosting. It is not a finished product or SaaS. See [LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md).

## Can I contribute?

Feedback, proof-path reports, documentation fixes, integration proposals, and design-partner discussions are welcome under the collaboration model. Substantial work requires prior discussion. Contributions do not grant extra rights. See [CONTRIBUTING.md](CONTRIBUTING.md) and [COLLABORATION.md](COLLABORATION.md).

## Can I build an application or agent on Intergrax?

For evaluation and design discussion, yes, within the permitted source-available evaluation model. Production and commercial use require permission. See [AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md), [COLLABORATION.md](COLLABORATION.md), and [LICENSE](LICENSE).

## How should I report security issues?

Do not open public issues for vulnerabilities. Follow [SECURITY.md](SECURITY.md).

## Related documents

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Overview and start paths |
| [COLLABORATION.md](COLLABORATION.md) | Collaboration and permission model |
| [LICENSE](LICENSE) | Proprietary license terms |
| [ROADMAP.md](ROADMAP.md) | Public adoption roadmap |
| [SECURITY.md](SECURITY.md) | Security reporting |
| [INTERGRAX_HARNESS_NARRATIVE.md](docs/guides/INTERGRAX_HARNESS_NARRATIVE.md) | Harness narrative |
| [AGENT_CREATION_GUIDE.md](docs/guides/AGENT_CREATION_GUIDE.md) | Agent/application authoring model |
| [BOUNDARYATTEST_ATTESTATION_POC.md](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) | Attestation case study |
| [LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) | LKW alpha narrative |
