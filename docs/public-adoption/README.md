<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Public Adoption Documents

This directory contains the public-adoption control documents for Intergrax.

These documents define how the public repository accepts structured evaluation feedback, documentation feedback, selected integration feedback, qualified design-partner interest, architecture discussion, product-validation discussion, and deep technical feedback while preserving the source-available/proprietary collaboration model.

They do not define an open-source contribution model, production-support channel, commercial-use permission path, redistribution permission path, derivative-work permission path, SLA, or public product roadmap commitment.

## Recommended reading order

| Step | Document | Purpose |
|------|----------|---------|
| 0 | [Intergrax Public Positioning](INTERGRAX_PUBLIC_POSITIONING.md) | Maintainer contract for exact first-contact message, product hierarchy, audience value, and CTA language |
| 1 | [Public Documentation Architecture](PUBLIC_DOCUMENTATION_ARCHITECTURE.md) | Canonical maintainer contract for public documentation layers, reader-intent routing, and proof classification |
| 1a | [Public Product Experience Roadmap](PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md) | Layer 5 product-experience transformation program, audience contract, and PX phase status |
| 2 | [Public Proof and Claims Model](PUBLIC_PROOF_AND_CLAIMS_MODEL.md) | Canonical status vocabulary, evidence requirements, claim qualification, and proof promotion rules |
| 3 | [External Reader Validation Protocol](EXTERNAL_READER_VALIDATION_PROTOCOL.md) | Canonical methodology, tasks, scoring, privacy and completion gates |
| 4 | [Public Launch Checklist](PUBLIC_LAUNCH_CHECKLIST.md) | Maintainer checklist before external-reader sessions, reviewer requests or design-partner outreach |
| 5 | [Public Issue Index](PUBLIC_ISSUE_INDEX.md) | Active curated public issue map and recommended evaluation paths |
| 6 | [Public Discussion Issue Expansion](PUBLIC_DISCUSSION_ISSUE_EXPANSION.md) | Active architecture, product-validation, and deep technical discussion issue waves |
| 7 | [Maintainer Triage Playbook](MAINTAINER_TRIAGE_PLAYBOOK.md) | Maintainer handling rules, close/keep-open criteria, escalation rules, and response templates |
| 8 | [Outreach Kit](OUTREACH_KIT.md) | Maintainer-facing recruitment templates and positioning guardrails |
| 9 | [Curated Public Issue Drafts](CURATED_PUBLIC_ISSUES.md) · [curated_public_issues.yml](curated_public_issues.yml) | Strategy, draft rationale and canonical source data for curated public issue automation |

**Intergrax Public Positioning** — [`INTERGRAX_PUBLIC_POSITIONING.md`](INTERGRAX_PUBLIC_POSITIONING.md)

```text
Role: Exact first-contact message, product hierarchy, audience value and CTA language
Status: ACTIVE — applied to root README in PX-2 ACCEPTED / CLOSED
Public-reader route: no
```

**Builder Quick Start** — [`../../BUILDER_QUICKSTART.md`](../../BUILDER_QUICKSTART.md)

```text
Role: First bounded builder orientation and progressive-disclosure route
Status: PX-6 — ACCEPTED / CLOSED
Public-reader route: yes, through the root public documentation
```

**Public Product Experience Roadmap** — [`PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md`](PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md)

```text
Role: Layer 5 roadmap and measurable product-experience contract
Status: ACTIVE — PX-8 READY_FOR_REVIEW
Public-reader route: no
```

## Reader-facing public navigation

Normal readers should start with the public documentation map:

[Intergrax Public Documentation Map](../PUBLIC_DOCUMENTATION_MAP.md)

**Public proof dashboard:** [`PROOFS.md`](../../PROOFS.md) — reader-facing proof status and verification paths.

**Maintainer proof rules:** [`PUBLIC_PROOF_AND_CLAIMS_MODEL.md`](PUBLIC_PROOF_AND_CLAIMS_MODEL.md) — status vocabulary, evidence requirements, and allowed public wording.

This directory contains maintainer controls and operational public-adoption material — not the default first-contact path for external reviewers.

## Current program status

```text
Previous phase: PX-7 ACCEPTED / CLOSED
Current phase: PX-8 READY_FOR_REVIEW
Next after acceptance: PX-9
External sessions: NOT_STARTED
```

## Audience-route summary

```text
Architect or platform engineer:
../../ARCHITECTURE_OVERVIEW.md

CTO, product lead or technical buyer:
../../USE_CASES.md

Partner, integrator or design partner:
../../PARTNERS.md

Category and alternative positioning:
../../WHY_INTERGRAX.md#where-intergrax-fits

Category map:
../assets/public/intergrax-category-map-light.svg
../assets/public/intergrax-category-map-dark.svg
```

## Validation status

```text
Protocol status:
READY_TO_RUN

External reader validation:
NOT_STARTED
```

No external result is claimed until real sessions are completed, anonymized and reviewed.

The [External Reader Validation Protocol](EXTERNAL_READER_VALIDATION_PROTOCOL.md) is a maintainer methodology document — not a normal external first-contact route. Normal readers should start with the [Public Documentation Map](../PUBLIC_DOCUMENTATION_MAP.md).

## Featured public proof routes

| Route | Classification | Entry point |
|-------|----------------|-------------|
| Product Tour | Product orientation | `../../LKW_PRODUCT_TOUR.md` |
| Product Quick Start | Supported executable product evaluation | `../../applications/local_workspace_application/docs/QUICKSTART.md` |
| Builder Quick Start | First bounded builder orientation | `../../BUILDER_QUICKSTART.md` |
| Deeper builder planning | Builder route selection and deeper planning | `../../BUILD_WITH_INTERGRAX.md` |
| Broader evaluation | Bounded evaluation execution | `../../EVALUATION_GUIDE.md` |
| Technical Platform Proof | Bounded technical reviewer evidence | `LKW_PLATFORM_PROOF.md` |
| Token Optimization | Featured platform-capability proof | `../features/token_optimization/README.md` |

Secondary control for Token Optimization public wording: [`TOKEN_OPTIMIZATION_CLAIMS.md`](TOKEN_OPTIMIZATION_CLAIMS.md)

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
