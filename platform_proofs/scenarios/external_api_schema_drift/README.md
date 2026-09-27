# External API / Schema Drift Mid-Execution

> **What happens when a critical external integration stops honoring its contract after a long-running fulfillment process has already created real-world commitments?**

> [!NOTE]
> **Scenario design:** ACCEPTED FOR IMPLEMENTATION · **Intergrax Fit:** COMPLETED · **Platform gaps:** CONFIRMED / resolution pending · **Implementation:** NOT INITIALIZED · **Proof/evidence:** NOT YET AVAILABLE

## Abstract

Asterion Industrial Systems is executing a cross-border fulfillment for a €1.8M industrial automation order. Inventory is allocated, export documentation is partially issued, and carrier capacity is reserved against a fixed departure window. Mid-process, the next required external operation—customs pre-clearance through a freight integrator—fails because the provider’s contract, protocol, or semantics no longer match what the integration was built for.

Retrying the same call does not restore service. Restarting the workflow from the beginning risks duplicating reservations, fees, and compliance artifacts. Blindly patching JSON or generating a new adapter without qualification can execute the wrong business meaning. Quietly requesting broader credentials changes who the system is allowed to act as.

This scenario asks whether an autonomous fulfillment application can recover **capability**—not theater—while preserving control of authority, qualification, governance, execution boundaries, and prior material business effects. The impressive part is not that the system generated an adapter. **The impressive part is that it knew when it should NOT generate one.** And when the capability really is missing, **the system must prove the gap before it is allowed to acquire anything.**

## At a glance

| Field | Value |
| --- | --- |
| **Problem** | External integration incompatibility mid-flight in a multi-day €1.8M fulfillment |
| **Observed impact** | Blocked shipment step, SLA/export window risk, stranded inventory and carrier slots |
| **Trap** | Treat every API failure as “build or buy a new adapter” |
| **Decision risk** | Wrong recovery duplicates side effects, breaches compliance, or expands authority |
| **Scenario outcome** | RESOLVED or UNRESOLVED (BLOCKED) per variant |
| **Status** | ACCEPTED FOR IMPLEMENTATION (FIT completed; init blocked by platform gaps) |
| **Proof class** | SCENARIO |

## Visual proof story

<!-- Add scenario-owned explanatory visual after Scenario Quality Gate.
     Use light/dark SVG per docs/project/technical/guides/DOCUMENTATION_DESIGN_SYSTEM.md.
     Do not use decorative imagery or fake execution results. -->

_Visual placeholder — enrich after Scenario Quality Gate._

## The problem

Long-running enterprise fulfillment cannot be naively restarted when a required external API drifts. Earlier steps already changed the real world: stock is reserved, documents exist, transport capacity is held, and correlation identifiers tie those effects together. The failure appears at a single integration step, but the business obligation spans the whole process.

Expand in [Scenario Specification § A](SCENARIO_SPEC.md#a-scenario).

## The risk

Mis-classifying the obstacle triggers duplicate carrier bookings, inconsistent export packs, penalty fees, compliance exposure, and customer SLA loss. Expanding OAuth scope or switching principals without governance can authorize actions the business never approved.

## The naive failure / trap

Common failures include infinite retry, full workflow restart, LLM-generated JSON mapping without semantic proof, direct provider bypass, executing an unqualified generated adapter, and silently widening credentials. Each can look like “recovery” while making the situation worse.

## Adversarial challenge

> “This is just try/catch, a fallback adapter, and an LLM that maps fields.”

The scenario requires distinguishing configuration, reuse, scoped adaptation, durable production change, and authority change—and proving when **no safe capability** exists. Syntactically valid HTTP 200 responses with changed semantics must not pass as “still compatible.” Details in [Scenario Specification § A](SCENARIO_SPEC.md#a-scenario).

## What the proof claims

When a required external integration becomes incompatible during an already-progressing business workflow, the application does not silently substitute behavior, broaden authority, execute an unqualified capability, or replay completed material effects. It resolves the need through canonical discovery, **acquisition only for a proven true capability gap**, qualification, binding, and governed execution—or stops with an auditable BLOCKED/UNRESOLVED outcome. **Production-change and authority-change outcomes are escalation boundaries, not executable acquisition successes.**

Full claim and contracts in [Scenario Specification § B](SCENARIO_SPEC.md#b-solution).

## PASS / FAIL (summary)

| PASS | FAIL |
| --- | --- |
| Per-variant correct disposition (reuse, configure, **canonical true-gap UCA success (C)**, scoped adaptation boundary, production/authority escalate, no safe capability, or H-routed evidence) with canonical evidence | Acquisition before discovery, true gap omitted, generic acquisition bypassed, **A3 treated as successful acquisition**, **A4 treated as authority grant**, unqualified execution, authority growth, duplicated business effects, or proof-owned business decisions |
| **Variant C:** complete discovery proves `MISSING_CAPABILITY`; CapabilityGap; acquisition only for true gap; qualification; binding ≠ execute; Execution Engine + ExecutionIdentityAuthority + ToolRuntime; business responsibility continues | Production change executed directly by UCA/AW; authority widened by UCA/AW |
| Prior material steps not replayed; ToolRuntime + Execution Engine boundaries preserved | Restart-from-step-1 replay, direct ToolRuntime bypass, fake application path |

Full normative PASS/FAIL contract in [Scenario Specification § B](SCENARIO_SPEC.md#pass).

## Outcomes

| Outcome | Meaning |
| --- | --- |
| **RESOLVED** | Original business responsibility continues after lawful capability restoration — **A**, **B**, **C** (canonical true-gap UCA path); **D** only if future FIT proves a qualified canonical scoped-adaptation path |
| **ESCALATED / UNRESOLVED** | **E** — durable production change required; current recovery cannot autonomously complete. **F** — authority change required; no autonomous authority growth. **G** — no safe capability / BLOCKED |
| **H (routing)** | Semantic false compatibility detected; routes to whichever **A–G** outcome is justified by evidence — not a separate acquisition type |

## Latest verified run

> [!NOTE]
> **Not yet available.** Populated only after a real proof run and report acceptance.

## Run / report / evidence / source

> [!NOTE]
> **Not yet available.** Links appear here after implementation and execution.

## Limitations

Does not prove universal API migration, zero downtime, automatic credential granting, or vendor certification. Synthetic external provider versions are allowed; fake application behavior is not. See [Scenario Specification § B](SCENARIO_SPEC.md#limitations).

## Go deeper

**[Read the full Scenario Specification](SCENARIO_SPEC.md)** — deep contract for scenario design, solution semantics, Intergrax fit, gap decision, and proof build (A/B/C/D/E).
