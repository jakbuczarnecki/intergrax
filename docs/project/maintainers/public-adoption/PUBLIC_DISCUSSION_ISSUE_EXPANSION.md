<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Public Discussion Issue Expansion

This document defines the expanded public discussion issue map for Intergrax.

All public-adoption issue waves are defined in the single canonical YAML source:

- [curated_public_issues.yml](curated_public_issues.yml)

The purpose is to promote discussion around Intergrax as a Harness AI / Agent OS platform without turning the repository into a broad implementation backlog, support channel, open-source task board, or commercial-use permission path.

## Issue model

The public issue model has five active waves:

| Wave | Status | Issue range | Purpose |
|------|--------|-------------|---------|
| Wave 1 | Active | #186-#190 | First evaluator / proof-path feedback |
| Wave 2 | Active | #191-#194 | Deeper architecture and integration-surface feedback |
| Wave 3 | Active | #205-#212 | Architecture discussion issues |
| Wave 4 | Active | #213-#218 | Product / application validation issues |
| Wave 5 | Active | #219-#227 | Deep technical discussion issues |

All waves are stored in [curated_public_issues.yml](curated_public_issues.yml). That YAML file is the source of truth for issue title, ID, labels, GitHub issue number, and GitHub issue URL.

## Active issue map

| Wave | Issue | Topic |
|------|-------|-------|
| Wave 1 | [#186](https://github.com/jakbuczarnecki/intergrax/issues/186) | README quick start feedback |
| Wave 1 | [#187](https://github.com/jakbuczarnecki/intergrax/issues/187) | First-time evaluator documentation clarity |
| Wave 1 | [#188](https://github.com/jakbuczarnecki/intergrax/issues/188) | Evidence and trace inspection |
| Wave 1 | [#189](https://github.com/jakbuczarnecki/intergrax/issues/189) | BoundaryAttest case study feedback |
| Wave 1 | [#190](https://github.com/jakbuczarnecki/intergrax/issues/190) | Governed agent applications design-partner interest |
| Wave 2 | [#191](https://github.com/jakbuczarnecki/intergrax/issues/191) | Harness AI mental model |
| Wave 2 | [#192](https://github.com/jakbuczarnecki/intergrax/issues/192) | Trace and evidence export surfaces |
| Wave 2 | [#193](https://github.com/jakbuczarnecki/intergrax/issues/193) | Local Knowledge Workspace alpha |
| Wave 2 | [#194](https://github.com/jakbuczarnecki/intergrax/issues/194) | MCP as controlled Intergrax task surface |
| Wave 3 | [#205](https://github.com/jakbuczarnecki/intergrax/issues/205) | Intergrax as Harness AI, not an agent framework |
| Wave 3 | [#206](https://github.com/jakbuczarnecki/intergrax/issues/206) | Four-tier boundary model |
| Wave 3 | [#207](https://github.com/jakbuczarnecki/intergrax/issues/207) | Nexus as Agent OS |
| Wave 3 | [#208](https://github.com/jakbuczarnecki/intergrax/issues/208) | Policy-first agent execution |
| Wave 3 | [#209](https://github.com/jakbuczarnecki/intergrax/issues/209) | Tool / Skill / Integration separation |
| Wave 3 | [#210](https://github.com/jakbuczarnecki/intergrax/issues/210) | Context engineering as a first-class runtime layer |
| Wave 3 | [#211](https://github.com/jakbuczarnecki/intergrax/issues/211) | Governed RAG and memory boundaries |
| Wave 3 | [#212](https://github.com/jakbuczarnecki/intergrax/issues/212) | Agent contracts and production readiness |
| Wave 4 | [#213](https://github.com/jakbuczarnecki/intergrax/issues/213) | Legal contract review application |
| Wave 4 | [#214](https://github.com/jakbuczarnecki/intergrax/issues/214) | Research and summarization pipeline |
| Wave 4 | [#215](https://github.com/jakbuczarnecki/intergrax/issues/215) | Dispute Simulation Workspace |
| Wave 4 | [#216](https://github.com/jakbuczarnecki/intergrax/issues/216) | Intergrax Assistant as harness chat hub |
| Wave 4 | [#217](https://github.com/jakbuczarnecki/intergrax/issues/217) | ProblemRadar and VendorDiscovery agents |
| Wave 4 | [#218](https://github.com/jakbuczarnecki/intergrax/issues/218) | Lab application as universal proof environment |
| Wave 5 | [#219](https://github.com/jakbuczarnecki/intergrax/issues/219) | Registry-driven capability resolution |
| Wave 5 | [#220](https://github.com/jakbuczarnecki/intergrax/issues/220) | Capability graph and blast-radius analysis |
| Wave 5 | [#221](https://github.com/jakbuczarnecki/intergrax/issues/221) | Evaluation gates for agent behavior |
| Wave 5 | [#222](https://github.com/jakbuczarnecki/intergrax/issues/222) | Cost governance and token/resource budgets |
| Wave 5 | [#223](https://github.com/jakbuczarnecki/intergrax/issues/223) | Reliability, retries, idempotency and HITL |
| Wave 5 | [#224](https://github.com/jakbuczarnecki/intergrax/issues/224) | Tier-3 applications as product hosts |
| Wave 5 | [#225](https://github.com/jakbuczarnecki/intergrax/issues/225) | Security and data governance for agent platforms |
| Wave 5 | [#226](https://github.com/jakbuczarnecki/intergrax/issues/226) | Observability spine and event journal |
| Wave 5 | [#227](https://github.com/jakbuczarnecki/intergrax/issues/227) | Developer experience, scaffold and lab workflow |

## Wave 3 - Architecture Discussion

| Issue | Title | Purpose |
|-------|-------|---------|
| [#205](https://github.com/jakbuczarnecki/intergrax/issues/205) | Architecture discussion: Intergrax as Harness AI, not an agent framework | Position Intergrax as a Harness AI / Agent OS platform |
| [#206](https://github.com/jakbuczarnecki/intergrax/issues/206) | Architecture discussion: four-tier boundary model | Validate Tier-0 / Tier-1 / Tier-2 / Tier-3 clarity |
| [#207](https://github.com/jakbuczarnecki/intergrax/issues/207) | Architecture discussion: Nexus as Agent OS | Validate Nexus as runtime / Agent OS rather than an agent |
| [#208](https://github.com/jakbuczarnecki/intergrax/issues/208) | Architecture discussion: policy-first agent execution | Promote governance, policy checks, HITL, and traceable decisions |
| [#209](https://github.com/jakbuczarnecki/intergrax/issues/209) | Architecture discussion: Tool / Skill / Integration separation | Validate Integration -> Tool -> Skill -> Agent boundaries |
| [#210](https://github.com/jakbuczarnecki/intergrax/issues/210) | Architecture discussion: context engineering as a first-class runtime layer | Promote context assembly as governed runtime architecture |
| [#211](https://github.com/jakbuczarnecki/intergrax/issues/211) | Architecture discussion: governed RAG and memory boundaries | Validate RAG / memory / context separation |
| [#212](https://github.com/jakbuczarnecki/intergrax/issues/212) | Architecture discussion: agent contracts and production readiness | Promote agents as contracted, lifecycle-managed components |

## Wave 4 - Product / Application Validation

| Issue | Title | Purpose |
|-------|-------|---------|
| [#213](https://github.com/jakbuczarnecki/intergrax/issues/213) | Design partner interest: Legal contract review application | Validate governed contract review workflows |
| [#214](https://github.com/jakbuczarnecki/intergrax/issues/214) | Design partner interest: Research and summarization pipeline | Validate traceable research and summary workflows |
| [#215](https://github.com/jakbuczarnecki/intergrax/issues/215) | Design partner interest: Dispute Simulation Workspace | Validate dispute-preparation and scenario-review workflows |
| [#216](https://github.com/jakbuczarnecki/intergrax/issues/216) | Product discussion: Intergrax Assistant as harness chat hub | Validate conversational hub as entry point into the harness |
| [#217](https://github.com/jakbuczarnecki/intergrax/issues/217) | Product discussion: ProblemRadar and VendorDiscovery agents | Validate problem discovery and vendor discovery workflows |
| [#218](https://github.com/jakbuczarnecki/intergrax/issues/218) | Product discussion: Lab application as universal proof environment | Validate the lab host as public proof environment |

## Wave 5 - Deep Technical Discussion

| Issue | Title | Purpose |
|-------|-------|---------|
| [#219](https://github.com/jakbuczarnecki/intergrax/issues/219) | Architecture discussion: registry-driven capability resolution | Validate registry-driven discovery, versioning, and lifecycle |
| [#220](https://github.com/jakbuczarnecki/intergrax/issues/220) | Architecture discussion: capability graph and blast-radius analysis | Validate impact analysis and compatibility graph needs |
| [#221](https://github.com/jakbuczarnecki/intergrax/issues/221) | Architecture discussion: evaluation gates for agent behavior | Validate quality and behavior release gates |
| [#222](https://github.com/jakbuczarnecki/intergrax/issues/222) | Architecture discussion: cost governance and token/resource budgets | Validate token, model, tool, run, and tenant cost controls |
| [#223](https://github.com/jakbuczarnecki/intergrax/issues/223) | Architecture discussion: reliability, retries, idempotency and HITL | Validate failure handling and recovery expectations |
| [#224](https://github.com/jakbuczarnecki/intergrax/issues/224) | Architecture discussion: Tier-3 applications as product hosts | Validate product host model and application/agent boundary |
| [#225](https://github.com/jakbuczarnecki/intergrax/issues/225) | Architecture discussion: security and data governance for agent platforms | Validate public security/data-governance concepts without public vulnerability disclosure |
| [#226](https://github.com/jakbuczarnecki/intergrax/issues/226) | Architecture discussion: observability spine and event journal | Validate runtime event spine beyond basic trace/evidence inspection |
| [#227](https://github.com/jakbuczarnecki/intergrax/issues/227) | Architecture discussion: developer experience, scaffold and lab workflow | Validate idea-to-first-run and scaffold/lab developer experience |

## Sync commands

The source of truth is [curated_public_issues.yml](curated_public_issues.yml).

Show the plan for all waves:

```bat
scripts\public_adoption\manage_curated_issues.bat dry
```

Create every missing issue from the YAML and skip existing issues by exact title:

```bat
scripts\public_adoption\manage_curated_issues.bat apply
```

Check YAML-to-GitHub alignment:

```bat
scripts\public_adoption\manage_curated_issues.bat check
```

Optional single-wave mode remains available when needed:

```bat
scripts\public_adoption\manage_curated_issues.bat dry wave_3
scripts\public_adoption\manage_curated_issues.bat apply wave_3
scripts\public_adoption\manage_curated_issues.bat check wave_3
```

## Milestone commands

Curated issues should be grouped by wave milestones:

| Wave | Milestone |
|------|-----------|
| Wave 1 | Public Adoption - Wave 1 |
| Wave 2 | Public Adoption - Wave 2 |
| Wave 3 | Architecture Discussion - Wave 3 |
| Wave 4 | Product Validation - Wave 4 |
| Wave 5 | Deep Technical Review - Wave 5 |

Show milestone creation and assignment plan:

```bat
python scripts\public_adoption\manage_curated_milestones.py
```

Create missing milestones and assign issues:

```bat
python scripts\public_adoption\manage_curated_milestones.py --apply
```

Check milestone assignments:

```bat
python scripts\public_adoption\manage_curated_milestones.py --check-sync
```

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
