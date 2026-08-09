# LKW Daily-Use Product Contract

**Task:** LKW-PRODUCT-1
**Status:** READY_FOR_REVIEW
**Authority:** This document freezes the LKW 1.0 daily-use product contract. It
defines the product outcome, boundaries, and acceptance gates; it does not
certify current implementation readiness.

## One-sentence product definition

Local Knowledge Workspace (LKW) is a private AI knowledge workspace that
connects selected work sources, keeps required knowledge synchronized, and lets
people ask grounded questions with visible sources and user-controlled access.

## Product promise

LKW helps a person:

```text
connect where my work knowledge lives
→ ask questions
→ receive grounded answers
→ inspect exact sources
→ understand freshness and state
→ keep using it every day
```

The user value is knowledge access and trustworthy understanding. RAG, agents,
Mongo, Qdrant, Nexus, and Intergrax internals are implementation means, not
the primary product proposition.

## Primary users

LKW 1.0 is bounded to these first-public-product personas:

1. **Software engineer / technical lead** — reconstruct decisions and
   technical context spread across work sources.
2. **Project, product, or knowledge worker** — locate decisions, project
   context, and information distributed across communication and documents.
3. **Technical evaluator / CTO** — evaluate trustworthiness and product
   boundaries; this persona must not drive everyday UX.

LKW 1.0 does not expand its primary UX to HR, sales, support, enterprise
administration, consumers, or other broad markets.

## Jobs to be done

The product must help users:

- find everything relevant about a topic, project, customer, or decision;
- explain what was decided;
- show how and why a decision changed;
- show exact supporting sources;
- determine whether information is fresh;
- scope a question to selected sources;
- refresh selected knowledge;
- report insufficient evidence clearly instead of pretending certainty.

## Top daily workflows

The core daily-use set is intentionally limited to:

1. Open or use an existing workspace.
2. Ask a grounded question.
3. Inspect citations and open the original source.
4. Restrict a question to selected knowledge sources.
5. Check source/data freshness.
6. Connect, disable, re-enable, or detach a source through safe product
   controls.
7. Request synchronization or refresh.
8. See and recover a source that needs user attention.

## First-public-product vendor target

The target vendor family for the first public product is:

- Slack;
- Google Workspace;
- Microsoft 365.

This is a product target, not a claim that every adapter is production-ready.
PRODUCT-5 must verify each vendor end to end before public support is claimed.
Jira and Confluence are not first-public-product commitments.

## Installation persona

The first installation persona is a technically comfortable user who can
install and run a local or self-hosted application, but is not expected to
know Intergrax internals, Python/FastAPI, MongoDB, Qdrant schemas, internal
operation IDs, or internal APIs.

The final product journey must not require manual database changes, internal
scripts, manual curl/JSON as normal UX, knowledge of Intergrax structures, or
assistance from the project author.

## Minimum daily-use UX

The product must expose capability-level paths for:

- workspace selection and management;
- knowledge sources and connections;
- source state and synchronization;
- Ask;
- citations and opening the original source;
- freshness;
- activity and problems requiring attention;
- basic settings.

UI, bot, CLI, and future mobile clients are thin interfaces over the same
reusable application/API boundaries. No UI-only lifecycle or mutation path is
permitted. Product controls must preserve user-controlled access and must not
silently modify original source content.

## Product architecture invariants

- Product language leads with user outcomes, not platform internals.
- Answers are grounded in selected knowledge and expose usable provenance.
- The product distinguishes indexed knowledge, synchronization state, and
  source freshness; it does not imply live or current knowledge without
  evidence.
- A lack of sufficient grounding is a visible product result.
- Source connection, disablement, re-enablement, detachment, refresh, and
  recovery are product lifecycle actions, not client-specific tricks.
- Durable configuration and knowledge state survive restart and supported
  upgrades.
- Provider failures must not silently destroy previously valid indexed
  knowledge.
- Generated content, where offered, is separate from original sources and
  does not write back by default.

## Success metrics / acceptance gates

LKW 1.0 is product-acceptable only when all of the following are demonstrated:

1. A new supported installation reaches a running LKW without repository-
   internal knowledge.
2. A user can connect or provide their own data and obtain a grounded answer.
3. The answer exposes usable source evidence/citations.
4. Restart preserves required configuration and durable state.
5. Normal daily synchronization does not require repetitive manual
   intervention.
6. The user can see last/next synchronization or freshness state where
   relevant.
7. A provider outage does not silently destroy previously valid indexed
   knowledge.
8. Typical authentication and synchronization failures explain what happened
   and what action is available.
9. Product updates preserve supported persistent data/configuration according
   to the later upgrade contract.
10. Insufficient grounding is reported instead of fabricated confidence.
11. No normal product flow requires direct Mongo/Qdrant or manual internal
   operation manipulation.
12. A real-user acceptance session completes without assistance from the
   author.

## Explicit non-goals for the first public product

LKW 1.0 does not commit to:

- enterprise HA;
- multi-instance worker coordination;
- a SaaS multi-tenant control plane;
- dozens of vendors;
- broad write-back into connected systems;
- mobile application implementation;
- a plugin marketplace;
- enterprise compliance certification;
- speculative platform abstractions without an LKW use case;
- replacing accepted backend mechanisms merely to make them “newer”.

## Current starting-point evidence

The bounded evidence pass for this contract found:

- the supported quickstart describes a one-command-per-OS local evaluation
  path using managed sample intake, indexed Ask, a grounded answer, a source
  citation, and persisted Ask-run verification
  ([`QUICKSTART.md`](QUICKSTART.md));
- the quickstart explicitly limits that path to indexed behavior and does not
  claim Hybrid Ask, live-provider access, production readiness, or compliance;
- the application README describes the current local/Docker evaluation path
  and states that its documented platform proof is separate from the product
  quickstart
  ([`applications/local_workspace_application/README.md`](../../../../applications/local_workspace_application/README.md));
- the user journey labels several connection, freshness, and hybrid-access
  capabilities as target or planned, while documenting indexed citations and
  the bounded quickstart as the current evidence
  ([`USER_JOURNEY.md`](USER_JOURNEY.md)).

These observations are starting-point evidence only. Historical
`PLANNED`, `IMPLEMENTED`, `PARTIAL`, or `ACCEPTED` labels are not copied into
this contract as production-readiness claims.

## Roadmap implications: PRODUCT-2…PRODUCT-12

Future product tasks must prove the contract in user-journey order:

- PRODUCT-2: establish the daily-use workspace, source, Ask, evidence, state,
  and attention model;
- PRODUCT-3: prove installation and first-run usability for the installation
  persona;
- PRODUCT-4: prove durable configuration/state and restart/upgrade behavior;
- PRODUCT-5: verify Slack, Google Workspace, and Microsoft 365 end to end
  before claiming public vendor support;
- PRODUCT-6…PRODUCT-9: prove synchronization, freshness, source lifecycle,
  outage preservation, recovery, and failure guidance;
- PRODUCT-10…PRODUCT-12: prove thin-client/API reuse, real-user acceptance,
  and release readiness against these gates.

Exact implementation status and production readiness remain validation work.
When a gap is found, the next task must identify the smallest reusable
platform change required by a product requirement; it must not add mechanism
without a user need.

## Change control

Once accepted, this document is the canonical LKW daily-use product contract.
Later roadmap tasks may discover implementation gaps, but must not silently
expand or redefine the product. Material changes to the primary personas,
first-product vendor set, core workflows, product boundaries, or non-goals
require an explicit product/architecture decision.
