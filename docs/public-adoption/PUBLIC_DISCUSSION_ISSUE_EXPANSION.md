<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Public Discussion Issue Expansion

This document defines the expanded public discussion issue plan for Intergrax.

All public-adoption issue waves are defined in the single canonical YAML source:

- [curated_public_issues.yml](curated_public_issues.yml)

The purpose is to promote discussion around Intergrax as a Harness AI / Agent OS platform without turning the repository into a broad implementation backlog, support channel, open-source task board, or commercial-use permission path.

## Issue model

The expanded model has five waves:

| Wave | Status | Purpose |
|------|--------|---------|
| Wave 1 | Active | First evaluator / proof-path feedback |
| Wave 2 | Active | Deeper architecture and integration-surface feedback |
| Wave 3 | Prepared | Architecture discussion issues |
| Wave 4 | Prepared | Product / application validation issues |
| Wave 5 | Prepared | Deep technical discussion issues |

All waves are stored in [curated_public_issues.yml](curated_public_issues.yml).

## Active issues today

| Issue | Topic |
|-------|-------|
| #186 | README quick start feedback |
| #187 | First-time evaluator documentation clarity |
| #188 | Evidence and trace inspection |
| #189 | BoundaryAttest case study feedback |
| #190 | Governed agent applications design-partner interest |
| #191 | Harness AI mental model |
| #192 | Trace and evidence export surfaces |
| #193 | Local Knowledge Workspace alpha |
| #194 | MCP as controlled Intergrax task surface |

## Wave 3 — Architecture Discussion

| Order | Title | Purpose |
|-------|-------|---------|
| 1 | Architecture discussion: Intergrax as Harness AI, not an agent framework | Position Intergrax as a Harness AI / Agent OS platform |
| 2 | Architecture discussion: four-tier boundary model | Validate Tier-0 / Tier-1 / Tier-2 / Tier-3 clarity |
| 3 | Architecture discussion: Nexus as Agent OS | Validate Nexus as runtime / Agent OS rather than an agent |
| 4 | Architecture discussion: policy-first agent execution | Promote governance, policy checks, HITL, and traceable decisions |
| 5 | Architecture discussion: Tool / Skill / Integration separation | Validate Integration -> Tool -> Skill -> Agent boundaries |
| 6 | Architecture discussion: context engineering as a first-class runtime layer | Promote context assembly as governed runtime architecture |
| 7 | Architecture discussion: governed RAG and memory boundaries | Validate RAG / memory / context separation |
| 8 | Architecture discussion: agent contracts and production readiness | Promote agents as contracted, lifecycle-managed components |

## Wave 4 — Product / Application Validation

| Order | Title | Purpose |
|-------|-------|---------|
| 1 | Design partner interest: Legal contract review application | Validate governed contract review workflows |
| 2 | Design partner interest: Research and summarization pipeline | Validate traceable research and summary workflows |
| 3 | Design partner interest: Dispute Simulation Workspace | Validate dispute-preparation and scenario-review workflows |
| 4 | Product discussion: Intergrax Assistant as harness chat hub | Validate conversational hub as entry point into the harness |
| 5 | Product discussion: ProblemRadar and VendorDiscovery agents | Validate problem discovery and vendor discovery workflows |
| 6 | Product discussion: Lab application as universal proof environment | Validate the lab host as public proof environment |

## Wave 5 — Deep Technical Discussion

| Order | Title | Purpose |
|-------|-------|---------|
| 1 | Architecture discussion: registry-driven capability resolution | Validate registry-driven discovery, versioning, and lifecycle |
| 2 | Architecture discussion: capability graph and blast-radius analysis | Validate impact analysis and compatibility graph needs |
| 3 | Architecture discussion: evaluation gates for agent behavior | Validate quality and behavior release gates |
| 4 | Architecture discussion: cost governance and token/resource budgets | Validate token, model, tool, run, and tenant cost controls |
| 5 | Architecture discussion: reliability, retries, idempotency and HITL | Validate failure handling and recovery expectations |
| 6 | Architecture discussion: Tier-3 applications as product hosts | Validate product host model and application/agent boundary |
| 7 | Architecture discussion: security and data governance for agent platforms | Validate public security/data-governance concepts without public vulnerability disclosure |
| 8 | Architecture discussion: observability spine and event journal | Validate runtime event spine beyond basic trace/evidence inspection |
| 9 | Architecture discussion: developer experience, scaffold and lab workflow | Validate idea-to-first-run and scaffold/lab developer experience |

## Creation commands

Always run dry-run first.

```bash
python scripts/public_adoption/create_curated_issues.py --wave wave_3
```

Create Wave 3:

```bash
python scripts/public_adoption/create_curated_issues.py --wave wave_3 --apply
```

Create Wave 4:

```bash
python scripts/public_adoption/create_curated_issues.py --wave wave_4 --apply
```

Create Wave 5:

```bash
python scripts/public_adoption/create_curated_issues.py --wave wave_5 --apply
```

Create all expanded waves one by one:

```bash
for wave in wave_3 wave_4 wave_5; do
  python scripts/public_adoption/create_curated_issues.py \
    --wave "$wave" \
    --apply
done
```

Windows wrapper:

```bat
scripts\public_adoption\manage_discussion_issues.bat dry wave_3
scripts\public_adoption\manage_discussion_issues.bat apply wave_3
scripts\public_adoption\manage_discussion_issues.bat apply all
```

## Sync checks

After issues are created, check alignment by title, labels, URL metadata when present, and open state:

```bash
python scripts/public_adoption/create_curated_issues.py --check-sync
```

The newly prepared Wave 3–5 entries do not initially contain GitHub issue numbers or URLs. That is intentional. The sync check will still match by exact title and labels.

If issue numbers and URLs are later added to the YAML, `--check-sync` will also verify them.

## Safety boundaries

These expanded issues are public discussion entry points only.

They must not be treated as:

- production support requests,
- SLA commitments,
- commercial-use permission requests,
- redistribution or derivative-work permission requests,
- public security vulnerability reports,
- broad implementation backlog tasks,
- help-wanted / good-first-issue work items,
- hosted SaaS or pricing discussions,
- automatic roadmap commitments.

Maintainer handling rules are defined in [Maintainer Triage Playbook](MAINTAINER_TRIAGE_PLAYBOOK.md).
