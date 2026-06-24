<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Public Adoption Documents

This directory contains the public-adoption control documents for Intergrax.

These documents define how the public repository accepts structured evaluation feedback, documentation feedback, selected integration feedback, qualified design-partner interest, architecture discussion, product-validation discussion, and deep technical feedback while preserving the source-available/proprietary collaboration model.

They do not define an open-source contribution model, production-support channel, commercial-use permission path, redistribution permission path, derivative-work permission path, SLA, or public product roadmap commitment.

## Recommended reading order

| Step | Document | Purpose |
|------|----------|---------|
| 0 | [Public Launch Checklist](PUBLIC_LAUNCH_CHECKLIST.md) | Maintainer checklist before public posts, reviewer requests or design-partner outreach |
| 1 | [Public Issue Index](PUBLIC_ISSUE_INDEX.md) | Active curated public issues and recommended evaluation paths |
| 2 | [Public Discussion Issue Expansion](PUBLIC_DISCUSSION_ISSUE_EXPANSION.md) | Expanded architecture, product-validation, and deep technical discussion issue waves |
| 3 | [Maintainer Triage Playbook](MAINTAINER_TRIAGE_PLAYBOOK.md) | Maintainer handling rules, close/keep-open criteria, escalation rules, and response templates |
| 4 | [Outreach Kit](OUTREACH_KIT.md) | Maintainer-facing outreach drafts and positioning guardrails |
| 5 | [Curated Public Issue Drafts](CURATED_PUBLIC_ISSUES.md) | Strategy and draft rationale for curated public issues |
| 6 | [curated_public_issues.yml](curated_public_issues.yml) | Single canonical source data for active and expanded public issue automation |

## Operational model

The public-adoption setup has four layers:

```text
README.md / ROADMAP.md / COLLABORATION.md
  -> docs/public-adoption/README.md
  -> PUBLIC_ISSUE_INDEX.md
  -> GitHub Issues #186-#194
```

Expanded discussion waves are prepared in the same canonical YAML:

```text
PUBLIC_DISCUSSION_ISSUE_EXPANSION.md
  -> curated_public_issues.yml
  -> Wave 3 architecture discussion issues
  -> Wave 4 product / application validation issues
  -> Wave 5 deep technical discussion issues
```

Automation support:

```text
curated_public_issues.yml
  -> scripts/public_adoption/create_curated_issues.py
  -> --wave wave_1 | wave_2 | wave_3 | wave_4 | wave_5
  -> --check-sync for YAML <-> GitHub issue alignment
  -> --apply only for explicit issue creation
```

Windows wrapper for expanded waves:

```text
scripts/public_adoption/manage_discussion_issues.bat
  -> dry | apply | check
  -> wave_3 | wave_4 | wave_5 | all
```

## Current curated issue waves

| Wave | Issues | Purpose | Source |
|------|--------|---------|--------|
| Wave 1 | #186-#190 | First-run proof path, documentation clarity, trace/evidence inspection, attestation feedback, governed-agent design-partner interest | [curated_public_issues.yml](curated_public_issues.yml) |
| Wave 2 | #191-#194 | Harness AI mental model, trace/evidence export surfaces, Local Knowledge Workspace alpha, MCP controlled task surface | [curated_public_issues.yml](curated_public_issues.yml) |
| Wave 3 | prepared | Architecture discussion issues | [curated_public_issues.yml](curated_public_issues.yml) |
| Wave 4 | prepared | Product / application validation issues | [curated_public_issues.yml](curated_public_issues.yml) |
| Wave 5 | prepared | Deep technical discussion issues | [curated_public_issues.yml](curated_public_issues.yml) |

## Boundaries

Public-adoption issues are for:

- proof-path feedback,
- documentation clarity,
- evidence and trace inspection feedback,
- selected integration feedback,
- qualified design-partner discovery,
- architecture discussion,
- product / application validation,
- deep technical review.

Public-adoption issues are not for:

- production support,
- commercial use requests,
- redistribution or derivative-work permission requests,
- broad feature requests detached from architecture or product validation,
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
