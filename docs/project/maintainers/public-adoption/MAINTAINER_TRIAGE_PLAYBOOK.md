<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Maintainer Triage Playbook

This playbook defines how maintainer-curated public issues for Intergrax should be handled.

It applies to the public-adoption issue track listed in [Public Issue Index](PUBLIC_ISSUE_INDEX.md), especially issues #186–#194.

The goal is to keep public feedback useful, scoped, and aligned with the Intergrax source-available/proprietary collaboration model.

## Core rule

Treat public issues as structured evaluation signals, not as automatic roadmap commitments.

A public issue may help identify friction, missing explanation, integration interest, or design-partner fit. It does not create an obligation to implement, support, certify, license, partner, or accept proposed work.

## Default maintainer stance

When responding, maintainers should:

- thank the reporter for concrete feedback,
- ask for missing reproduction or evaluation details,
- keep the discussion attached to the issue purpose,
- avoid promising timelines,
- avoid promising implementation,
- avoid implying production readiness,
- avoid accepting commercial, licensing, or security-sensitive discussion in public,
- close off-topic issues politely,
- move sensitive conversations to direct maintainer contact.

## Triage categories

| Category | Meaning | Default action |
|----------|---------|----------------|
| Useful evaluation feedback | Concrete feedback from running or reviewing a public proof path | Acknowledge, capture, keep open if more signal is needed |
| Documentation friction | Specific unclear wording, missing link, confusing term, or navigation problem | Acknowledge, optionally create a doc follow-up, close when captured |
| Integration-shape feedback | Concrete external-system or integration-surface feedback | Discuss shape, do not promise implementation |
| Qualified design-partner interest | Real use case aligned with governed agents, LKW, evidence, policy, or integration | Ask for safe public summary; move details to direct contact if needed |
| Broad feature request | Large or generic request outside the curated issue purpose | Redirect or close as not planned |
| Support request | Request for production help, SLA, deployment support, debugging unrelated setup | Redirect away from curated issue track |
| Commercial/licensing request | Request for commercial use, redistribution, derivative work, or product use | Move to direct maintainer contact |
| Security-sensitive report | Vulnerability, exploit, secret, customer data, or private system detail | Do not discuss publicly; redirect to private disclosure channel/contact |

## Keep open when

Keep an issue open when:

- the feedback is on topic,
- more evaluator detail is needed,
- multiple evaluators may add comparable feedback,
- the issue is intentionally collecting design-partner interest,
- the maintainer has not yet captured the useful signal.

## Close when

Close an issue when:

- the feedback has been captured,
- the documentation fix or follow-up is complete,
- the issue is off topic,
- the issue asks for support, SLA, commercial use, or licensing in public,
- the issue is a broad feature request outside curated public-adoption scope,
- the discussion no longer produces actionable evaluation signal.

Use `completed` when the useful signal was captured or a related doc/update was made.

Use `not planned` when the request is outside the curated issue purpose or not aligned with current maintainer direction.

## Do not promise

Do not promise:

- production readiness,
- support timelines,
- implementation timelines,
- SLA,
- certification,
- compliance approval,
- security approval,
- commercial use rights,
- redistribution rights,
- derivative-work permission,
- acceptance of implementation work,
- future public roadmap inclusion.

## Response templates

### Useful proof-path feedback

Thank you for running the proof path and sharing concrete feedback.

I will treat this as evaluation feedback for the public proof path. If you can share the exact command, environment, branch or commit, and the point where the path became unclear, that will make the signal easier to capture.

This issue remains scoped to evaluation-path clarity and does not imply production support or implementation commitment.

### Documentation clarity feedback

Thank you for the documentation feedback.

The useful signal here is the point where the first-time reader lost context or needed a better link, definition, or example. I will capture this as documentation-friction feedback and decide whether it belongs in README, the public issue index, the collaboration model, or the architecture docs.

### Integration feedback

Thank you for the integration-shape feedback.

This issue is useful for understanding what an external system would need from Intergrax trace, evidence, policy, or task surfaces. I will keep the discussion focused on integration shape, required fields, boundaries, and evaluation value.

This does not imply a commitment to implement a specific exporter, vendor integration, dashboard, or hosted product feature.

### Design-partner interest

Thank you for sharing design-partner interest.

Please keep public details high level and do not include confidential information, customer data, credentials, secrets, or private system details. If the use case appears aligned, partnership or commercial discussion should move to direct maintainer contact.

Submitting interest does not create a partnership, production-use permission, commercial-use permission, support obligation, or implementation commitment.

### Off-topic or broad feature request

Thank you for the suggestion.

This curated issue track is intentionally limited to structured evaluation feedback, documentation clarity, selected integration feedback, and qualified design-partner discovery. This request is broader than the current public-adoption scope, so I am closing it as not planned for this issue track.

### Commercial or licensing request

Thank you for reaching out.

Commercial use, production use, redistribution, derivative works, incorporation into products or services, or substantial implementation work based on Intergrax require explicit maintainer permission. Please move this discussion to direct maintainer contact instead of continuing in a public issue.

### Security-sensitive report

Thank you for the report.

Please do not include security-sensitive details, secrets, exploit details, credentials, customer data, or private system information in public issues. Move this to direct maintainer contact or the repository's security disclosure path.

This public issue should not be used for vulnerability disclosure.

## Issue-specific handling notes

| Issue | Handling focus |
|-------|----------------|
| #186 README quick start | Reproducibility, command clarity, environment details, first-run friction |
| #187 First-time evaluator path | Navigation, first document opened, unclear terms, missing links |
| #188 Evidence and trace inspection | Field clarity, lineage, identifiers, trace/evidence distinction |
| #189 BoundaryAttest case study | Trust model clarity, boundary events, receipt mapping, non-claims |
| #190 Governed agent applications | Use-case fit, governance need, proof path needed, safe public summary |
| #191 Harness AI mental model | Agent/harness/Nexus/application distinction, diagrams, terms |
| #192 Trace/evidence export surfaces | External consumer needs, export shape, essential fields, out-of-scope surfaces |
| #193 Local Knowledge Workspace alpha | Product fit, local/private knowledge problem, privacy/control/reviewability |
| #194 MCP controlled task surface | Task intake vs tool access, policy boundaries, HITL, trace/evidence expectations |

## Escalation rules

Move out of public issues when the discussion involves:

- commercial licensing,
- production use,
- redistribution,
- derivative work,
- partnership negotiation,
- private customer details,
- credentials or secrets,
- security vulnerabilities,
- legal claims,
- confidential architecture.

## Related documents

- [Public Issue Index](PUBLIC_ISSUE_INDEX.md)
- [Curated Public Issue Drafts](CURATED_PUBLIC_ISSUES.md)
- [curated_public_issues.yml](curated_public_issues.yml)
- [Collaboration model](../../community/COLLABORATION.md)
- [Roadmap](../../overview/ROADMAP.md)
- [Security policy](../../../../SECURITY.md)
