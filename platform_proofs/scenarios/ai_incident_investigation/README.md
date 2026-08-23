# AI Incident Investigation with Independent Verification

> **Can an AI investigate an operational incident without turning correlation into a confident false diagnosis?**

A regional logistics operator needs a defensible root-cause diagnosis when warehouse SLA collapses — not a fluent story that sounds right under time pressure.

> [!NOTE]
> **Scenario status:** ACCEPTED FOR IMPLEMENTATION — scenario concept passed human Scenario Quality Gate; implementation, executable proof, evidence, and report have not started yet. This is not published/accepted proof evidence.

## Abstract

A regional logistics operator watches on-time delivery through a key warehouse fall sharply during a peak routing window while heavy-parcel complaints spike. The first operational picture points to volume overload: shipment counts are up, delays rise in the same period, and the affected hub stands out. Acting on that story before the weekend peak would mean overtime, rerouting, and staffing moves — all costly if the diagnosis is wrong. The opening evidence is intentionally misleading; correlation looks like causation. This scenario asks whether an AI investigation system can challenge that first explanation, gather the evidence needed to distinguish competing causes, and either deliver a bounded defensible diagnosis or honestly refuse with UNRESOLVED when certainty is not justified.

## At a glance

| Field | Value |
| --- | --- |
| **Problem** | Warehouse SLA degradation during peak routing window |
| **Observed impact** | On-time rate ~94% → ~78%; heavy-parcel complaints spike |
| **Trap** | Volume and delay rise together — correlation presented as causation |
| **Decision risk** | Wrong routing, staffing, or capacity action before weekend peak |
| **Scenario outcome** | RESOLVED or UNRESOLVED |
| **Status** | ACCEPTED FOR IMPLEMENTATION |
| **Proof class** | SCENARIO |
| **Slug** | `ai_incident_investigation` |

## Visual proof story

<a href="assets/proof-story-light.svg">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/proof-story-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/proof-story-light.svg">
  <img alt="Incident investigation flows from operational incident through falsification to RESOLVED or UNRESOLVED outcomes" src="assets/proof-story-light.svg">
</picture>
</a>

[View full-size diagram](assets/proof-story-light.svg)

## The problem

North Central warehouse on-time delivery drops from roughly **94% to 78%** during a Tuesday–Thursday window while heavy-parcel complaints spike. Operations leadership needs a **defensible root-cause diagnosis** from fragmented telemetry, staffing, equipment, and shipment facts — the same sources a human incident lead would query.

## The risk

Wrong conclusions trigger real operational harm: overtime at the wrong facility, bad reroutes, missed equipment repair, and weekend peak failure. SLA misses affect contractual penalties, customer trust, and readiness for the next peak.

## The naive failure / trap

Initial facts look like overload: volume up ~22%, delays up, North Central disproportionately affected. A naive investigator concludes **warehouse overload** and recommends capacity responses. That story is plausible, leadership-aligned, and **wrong** given full admissible evidence.

<a href="assets/correlation-trap-light.svg">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/correlation-trap-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/correlation-trap-light.svg">
  <img alt="Unsupported volume-delay shortcut versus before-during-after evidence supporting sorter degradation diagnosis" src="assets/correlation-trap-light.svg">
</picture>
</a>

[View full-size diagram](assets/correlation-trap-light.svg)

## Adversarial challenge

The scenario embeds credible adversarial conditions: a volume–delay correlation trap, conflicting staffing sources, stale records, missing decisive telemetry, and competing hypotheses that must be distinguished. A skeptical engineer should be able to ask whether a simple investigator-plus-critic graph is sufficient — and the design should show why not.

Full adversarial conditions, skeptic challenge, and quality gate rationale: [Scenario Specification § A](SCENARIO_SPEC.md#a-scenario).

## What the proof claims

> **No material incident diagnosis is accepted unless its material claims are supported by auditable evidence and survive an independent falsification attempt.**

Correlation-only narratives do not qualify. The system must surface evidence defects, gather targeted follow-up when needed, and either emit a **bounded RESOLVED** diagnosis or an honest **UNRESOLVED** refusal.

Normative claim, guarantees, and excluded claims: [Scenario Specification § B](SCENARIO_SPEC.md#b-solution).

## PASS / FAIL (summary)

| PASS | FAIL |
| --- | --- |
| Concrete evidence defect identified | Generic disagreement |
| Conflict surfaced | Source silently discarded |
| Stale evidence rejected or qualified | Stale treated as current |
| Missing evidence requested or UNRESOLVED | Evidence invented |
| Supported RESOLVED or honest UNRESOLVED | Confident unsupported root cause |

Full normative PASS/FAIL contract: [Scenario Specification § B](SCENARIO_SPEC.md#pass).

## Outcomes

| Outcome | Meaning |
| --- | --- |
| **RESOLVED** | Best-supported bounded operational root-cause diagnosis with auditable evidence trail; competing hypotheses weakened |
| **UNRESOLVED** | Critical distinguishing evidence unavailable or hypotheses indistinguishable — no confident guessing |

## Latest verified run

> [!NOTE]
> **Not yet available.** This scenario is accepted for implementation; no executable proof run has been performed. After implementation this section will show verdict, proof version, Intergrax SHA, model/provider, run timestamp, RESOLVED/UNRESOLVED outcome, and key invariant results.

## Run / report / evidence / source

> [!NOTE]
> **Not yet available.** No report, evidence bundle, or reproduction path exists at design stage. Links appear here only after real execution and report acceptance.

## Limitations

Single bounded logistics incident fixture with designed adversarial conditions — not arbitrary enterprise data. Design stage only: no runtime, evidence, or report exists yet. Evaluator semantics are scoped to this scenario's claim.

Full limitations and excluded claims: [Scenario Specification § B](SCENARIO_SPEC.md#limitations).

## Go deeper

**[Read the full Scenario Specification](SCENARIO_SPEC.md)** — deep contract for scenario design, solution semantics, Intergrax fit, gap decision, and proof build (A/B/C/D/E).
