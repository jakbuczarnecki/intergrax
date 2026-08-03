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
| 0 | [Intergrax Public Positioning](INTERGRAX_PUBLIC_POSITIONING.md) | Canonical source for public problem statement, value proposition, Harness AI explanation, LKW role, audience, and public claim boundaries |
| 1 | [Public Launch Checklist](PUBLIC_LAUNCH_CHECKLIST.md) | Maintainer checklist before public posts, reviewer requests or design-partner outreach |
| 2 | [Public Issue Index](PUBLIC_ISSUE_INDEX.md) | Active curated public issue map and recommended evaluation paths |
| 3 | [Public Discussion Issue Expansion](PUBLIC_DISCUSSION_ISSUE_EXPANSION.md) | Active architecture, product-validation, and deep technical discussion issue waves |
| 4 | [Maintainer Triage Playbook](MAINTAINER_TRIAGE_PLAYBOOK.md) | Maintainer handling rules, close/keep-open criteria, escalation rules, and response templates |
| 5 | [Outreach Kit](OUTREACH_KIT.md) | Maintainer-facing outreach drafts and positioning guardrails |
| 6 | [Curated Public Issue Drafts](CURATED_PUBLIC_ISSUES.md) | Strategy and draft rationale for curated public issues |
| 7 | [curated_public_issues.yml](curated_public_issues.yml) | Single canonical source data for all curated public issue automation |

## Operational model

The public-adoption setup has one canonical issue source and one normal maintainer workflow:

```text
README.md / ROADMAP.md / COLLABORATION.md
  -> docs/public-adoption/README.md
  -> PUBLIC_ISSUE_INDEX.md
  -> curated_public_issues.yml
  -> scripts/public_adoption/manage_curated_issues.bat
  -> scripts/public_adoption/manage_curated_milestones.py
  -> GitHub Issues #186-#227
```

The open curated issues are a public discussion map, not a generic implementation backlog.

## Canonical issue automation

Source of truth:

```text
docs/public-adoption/curated_public_issues.yml
```

Normal Windows workflow for issue creation and sync:

```bat
scripts\public_adoption\manage_curated_issues.bat dry
scripts\public_adoption\manage_curated_issues.bat apply
scripts\public_adoption\manage_curated_issues.bat check
```

Behavior:

```text
dry   -> read the whole YAML and show what would be created
apply -> read the whole YAML, skip existing issues by exact title, create missing issues
check -> verify YAML <-> GitHub issue alignment
```

Optional single-wave mode remains available when needed:

```bat
scripts\public_adoption\manage_curated_issues.bat dry wave_3
scripts\public_adoption\manage_curated_issues.bat apply wave_3
scripts\public_adoption\manage_curated_issues.bat check wave_3
```

## Milestone automation

Curated issues are grouped into wave milestones for GitHub UX:

| Wave | Milestone |
|------|-----------|
| Wave 1 | Public Adoption — Wave 1 |
| Wave 2 | Public Adoption — Wave 2 |
| Wave 3 | Architecture Discussion — Wave 3 |
| Wave 4 | Product Validation — Wave 4 |
| Wave 5 | Deep Technical Review — Wave 5 |

Normal milestone workflow:

```bat
python scripts\public_adoption\manage_curated_milestones.py
python scripts\public_adoption\manage_curated_milestones.py --apply
python scripts\public_adoption\manage_curated_milestones.py --check-sync
```

Token-limited fallback workflow when the GitHub token can edit issues but cannot create milestones:

```bat
python scripts\public_adoption\manage_curated_milestones.py --list-milestones
python scripts\public_adoption\manage_curated_milestones.py --assign-only
python scripts\public_adoption\manage_curated_milestones.py --check-sync
```

Behavior:

```text
default           -> dry-run milestone creation and issue assignment plan
--apply           -> create missing milestones and assign issues to expected wave milestones
--assign-only     -> assign issues to existing milestones without creating milestones
--list-milestones -> list milestones visible to GitHub API without mutating GitHub
--check-sync      -> verify issue milestone assignments without mutating GitHub
```

`--assign-only` expects the wave milestones to already exist in GitHub. The script tolerates common manual title differences such as em dash, en dash, ASCII dash, and extra spaces.

Optional single-wave mode remains available:

```bat
python scripts\public_adoption\manage_curated_milestones.py --wave wave_3
python scripts\public_adoption\manage_curated_milestones.py --wave wave_3 --apply
python scripts\public_adoption\manage_curated_milestones.py --wave wave_3 --assign-only
python scripts\public_adoption\manage_curated_milestones.py --wave wave_3 --check-sync
```

## Current curated issue waves

| Wave | Issues | Status | Purpose | Source |
|------|--------|--------|---------|--------|
| Wave 1 | #186-#190 | Open | First-run proof path, documentation clarity, trace/evidence inspection, attestation feedback, governed-agent design-partner interest | [curated_public_issues.yml](curated_public_issues.yml) |
| Wave 2 | #191-#194 | Open | Harness AI mental model, trace/evidence export surfaces, Local Knowledge Workspace alpha, MCP controlled task surface | [curated_public_issues.yml](curated_public_issues.yml) |
| Wave 3 | #205-#212 | Open | Architecture discussion issues | [curated_public_issues.yml](curated_public_issues.yml) |
| Wave 4 | #213-#218 | Open | Product / application validation issues | [curated_public_issues.yml](curated_public_issues.yml) |
| Wave 5 | #219-#227 | Open | Deep technical discussion issues | [curated_public_issues.yml](curated_public_issues.yml) |

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

## Token optimization claim guardrails

For token-optimization proof wording and claim boundaries, see [`TOKEN_OPTIMIZATION_CLAIMS.md`](TOKEN_OPTIMIZATION_CLAIMS.md).

## Related root documents

- [INTERGRAX_PUBLIC_POSITIONING.md](INTERGRAX_PUBLIC_POSITIONING.md)
- [README.md](../../README.md)
- [ROADMAP.md](../../ROADMAP.md)
- [COLLABORATION.md](../../COLLABORATION.md)
- [LICENSE](../../LICENSE)
- [SECURITY.md](../../SECURITY.md)
