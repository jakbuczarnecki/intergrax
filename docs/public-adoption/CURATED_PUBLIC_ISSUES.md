<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Curated Public Issue Rationale

This document explains why Intergrax uses maintainer-curated public GitHub issues and how those issues should be framed.

It is **not** the source of truth for active issue definitions.

Canonical sources:

- [curated_public_issues.yml](curated_public_issues.yml) — source of truth for issue IDs, titles, labels, GitHub issue numbers, URLs, and bodies.
- [Public Issue Index](PUBLIC_ISSUE_INDEX.md) — human-readable active public issue map.
- [Public Discussion Issue Expansion](PUBLIC_DISCUSSION_ISSUE_EXPANSION.md) — active Wave 1-5 discussion map and maintenance commands.
- [Maintainer Triage Playbook](MAINTAINER_TRIAGE_PLAYBOOK.md) — maintainer handling rules and response templates.

## Purpose

Curated public issues help Intergrax move from a public repository that can be inspected to a public repository where external evaluators know how to engage in a useful, controlled, and strategically aligned way.

Curated public issues are used to invite structured feedback in areas such as:

- proof-path evaluation,
- documentation clarity,
- evidence and trace inspection,
- attestation and boundary-event feedback,
- selected integration discussion,
- qualified design-partner interest,
- architecture discussion,
- product / application validation,
- deep technical review.

They are not:

- a generic implementation backlog,
- a production-support channel,
- an open-source task board,
- a free-work request queue,
- a commercial-use permission path,
- a redistribution permission path,
- a derivative-work permission path,
- a roadmap commitment.

## Public collaboration boundaries

Intergrax is a source-available proprietary project. The repository is public for evaluation, technical review, proof-path feedback, selected integration discussion, and technical partner discovery.

Intergrax is not currently presented as:

- a classical open-source framework,
- a finished SaaS product,
- a public production-support channel,
- a production-certification product,
- a compliance-attestation product,
- a general feature-request backlog.

Opening or responding to a public issue does not grant any rights to:

- production use,
- commercial use,
- redistribution,
- derivative works,
- incorporation into products or services,
- substantial implementation work based on Intergrax,
- support,
- SLA,
- automatic acceptance of proposed work.

Commercial, production, redistribution, derivative-work, or substantial implementation use requires explicit permission from the maintainer under the repository license and collaboration model.

## Issue selection principles

Curated public issues should be intentional, feedback-oriented, and aligned with the source-available/proprietary collaboration model.

### Prefer feedback over feature requests

Good public issues ask external evaluators to inspect, run, review, validate, or critique an existing path or a clearly framed architectural/product direction.

Preferred framing:

```text
Run this proof path and tell us where it breaks.
Review this architecture boundary and tell us what is unclear.
Validate whether this product direction matches a real workflow problem.
```

Avoid framing such as:

```text
Build this feature.
Pick this up.
What should we build next?
```

### Treat issues as a public discussion map

The active issue set is intentionally organized into waves:

| Wave | Issues | Focus |
|------|--------|-------|
| Wave 1 | #186-#190 | First evaluator / proof-path feedback |
| Wave 2 | #191-#194 | Architecture clarity and integration surfaces |
| Wave 3 | #205-#212 | Architecture discussion |
| Wave 4 | #213-#218 | Product / application validation |
| Wave 5 | #219-#227 | Deep technical discussion |

This structure should be described as a public discussion map, not a backlog.

### Keep every issue scoped

Each curated issue should make clear:

- who the issue is for,
- what kind of feedback is useful,
- what is out of scope,
- which rights are not granted,
- what kind of confidential information must not be posted publicly.

### Avoid open-source contribution assumptions

Avoid issue language such as:

```text
Good first issue
Help wanted
Contributors wanted
Pick this up
Implement this integration
```

Prefer language such as:

```text
Feedback requested
Evaluation path
Design partner interest
Integration feedback
Maintainer-reviewed proposal
```

### Avoid production-readiness claims

Public issues must not imply production guarantees, certification, compliance approval, enterprise support, or hosted-product availability.

Avoid language such as:

```text
production-ready
certified
enterprise support
SLA
official compliance
```

Prefer language such as:

```text
evaluation path
technical validation
proof path
case study
integration feedback
not production certification
not compliance attestation
```

## Issue categories

| Category | Purpose |
|----------|---------|
| Proof path feedback | Validate whether an external evaluator can run and inspect Intergrax without internal project knowledge. |
| Documentation clarity | Reduce conceptual and navigation friction for first-time readers. |
| Evidence / attestation / boundary events feedback | Validate whether evidence surfaces, boundary events, and attestation patterns are understandable and useful. |
| Integration feedback | Understand which integration surfaces are worth validating without creating implementation commitments. |
| Design partner interest | Identify real product or platform problems where Intergrax's Harness AI model may be relevant. |
| Architecture discussion | Validate core architecture boundaries, runtime concepts, and governance model. |
| Product / application validation | Validate whether scaffolded or early product directions solve real workflow problems. |
| Deep technical discussion | Collect advanced technical review on capability resolution, reliability, security, observability, cost governance, and developer experience. |

## Automation model

The YAML file is the source of truth:

```text
docs/public-adoption/curated_public_issues.yml
```

Normal maintainer workflow:

```bat
scripts\public_adoption\manage_curated_issues.bat dry
scripts\public_adoption\manage_curated_issues.bat apply
scripts\public_adoption\manage_curated_issues.bat check
```

Behavior:

- `dry` reads the whole YAML and shows what would be created,
- `apply` reads the whole YAML, skips existing GitHub issues by exact title, and creates missing issues,
- `check` verifies YAML-to-GitHub alignment.

Optional single-wave mode remains available:

```bat
scripts\public_adoption\manage_curated_issues.bat dry wave_3
scripts\public_adoption\manage_curated_issues.bat apply wave_3
scripts\public_adoption\manage_curated_issues.bat check wave_3
```

## Maintainer stance

Maintainer responses should keep issues focused and scoped:

- thank people for concrete feedback,
- ask for reproducible details when needed,
- avoid promising implementation,
- avoid promising support timelines,
- avoid suggesting production readiness,
- redirect commercial or licensing questions to direct maintainer contact,
- redirect security disclosures away from public issues,
- close off-topic issues politely,
- treat public issues as evaluation signals, not automatic roadmap commitments.

See [Maintainer Triage Playbook](MAINTAINER_TRIAGE_PLAYBOOK.md) for response templates and escalation rules.
