<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Public Adoption Documents

This directory contains the public-adoption control documents for Intergrax.

These documents define how the public repository accepts structured evaluation feedback, documentation feedback, selected integration feedback, and qualified design-partner interest while preserving the source-available/proprietary collaboration model.

They do not define an open-source contribution model, production-support channel, commercial-use permission path, redistribution permission path, derivative-work permission path, SLA, or public product roadmap commitment.

## Recommended reading order

| Step | Document | Purpose |
|------|----------|---------|
| 1 | [Public Issue Index](PUBLIC_ISSUE_INDEX.md) | Active curated public issues and recommended evaluation paths |
| 2 | [Maintainer Triage Playbook](MAINTAINER_TRIAGE_PLAYBOOK.md) | Maintainer handling rules, close/keep-open criteria, escalation rules, and response templates |
| 3 | [Curated Public Issue Drafts](CURATED_PUBLIC_ISSUES.md) | Strategy and draft rationale for curated public issues |
| 4 | [curated_public_issues.yml](curated_public_issues.yml) | Structured source data for issue automation |

## Operational model

The public-adoption setup has four layers:

```text
README.md / ROADMAP.md / COLLABORATION.md
  -> docs/public-adoption/README.md
  -> PUBLIC_ISSUE_INDEX.md
  -> GitHub Issues #186-#194
```

Automation support:

```text
curated_public_issues.yml
  -> scripts/public_adoption/create_curated_issues.py
  -> --check-sync for YAML <-> GitHub issue alignment
  -> --apply only for explicit issue creation
```

## Current curated issue waves

| Wave | Issues | Purpose |
|------|--------|---------|
| Wave 1 | #186-#190 | First-run proof path, documentation clarity, trace/evidence inspection, attestation feedback, governed-agent design-partner interest |
| Wave 2 | #191-#194 | Harness AI mental model, trace/evidence export surfaces, Local Knowledge Workspace alpha, MCP controlled task surface |

## Boundaries

Public-adoption issues are for:

- proof-path feedback,
- documentation clarity,
- evidence and trace inspection feedback,
- selected integration feedback,
- qualified design-partner discovery.

Public-adoption issues are not for:

- production support,
- commercial use requests,
- redistribution or derivative-work permission requests,
- broad feature requests,
- public security vulnerability disclosure,
- hosted SaaS or pricing discussions,
- requests for free implementation work,
- requests that imply an open-source contribution model.

For commercial licensing, production use, partnerships, redistribution, derivative works, or substantial implementation permission, contact the maintainer directly.

## Related root documents

- [README.md](../../README.md)
- [ROADMAP.md](../../ROADMAP.md)
- [COLLABORATION.md](../../COLLABORATION.md)
- [LICENSE](../../LICENSE)
- [SECURITY.md](../../SECURITY.md)
