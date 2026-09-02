# Virtual Workforce

**Status:** Strategic product direction backed by canonical Autonomous Work architecture — **NOT a shipped product claim**  
**Canonical technical domain:** [`AUTONOMOUS_WORK`](../architecture/AUTONOMOUS_WORK.md)  
**Implementation plan:** [`AUTONOMOUS_WORK`](../maintainers/plans/AUTONOMOUS_WORK.md)

---

## What it is

Virtual Workforce is the product-facing way to describe Intergrax systems built from reusable **Virtual Workers**.

A Virtual Worker is not a bigger agent and not an agent running forever. It is a persistent unit of business responsibility that remains available over time, watches its goals, accepts or creates work, launches governed executions when needed, and returns to an idle state when there is nothing to do.

> **Agent executes work. Virtual Worker remains responsible for making sure the work gets done.**

Example:

```text
Order Operations Worker

Responsibility:
Process all incoming orders according to company policy.

Goals:
99% completed within 30 minutes
zero unauthorized actions
zero duplicate orders
```

The worker may use many executions and many agents over its lifetime.

---

## Long-running responsibility without full history

A Virtual Worker can remain responsible for work over long periods — days, weeks, or months — **without requiring its entire history to be loaded into the model**. It restores its current position, retrieves only the relevant information, performs bounded work, and persists what matters for the future.

Like an experienced employee who does not keep the entire company history in working memory: the worker knows **where to find the right information** and **remembers the state of its responsibility**.

---

## Why Intergrax is a strong fit

Intergrax already separates many of the mechanisms that enterprise autonomous work requires:

- governed execution and explicit authority,
- agents and orchestration,
- long-running execution primitives,
- memory and context,
- tools, skills and integrations,
- Collaborative Work identity/delegation,
- human approval,
- diagnostics,
- observability and evidence,
- CodeCraft for governed ephemeral code generation,
- sandbox execution,
- application hosting.

Autonomous Work adds the missing persistent business semantics above those mechanisms rather than rebuilding them inside a new super-agent.

---

## How a worker operates

```text
organization defines role + responsibility + goals + limits
                    ↓
               Virtual Worker
                    ↓
        waits in ACTIVE / IDLE state
                    ↓
 event / schedule / assignment / SLA / goal check
                    ↓
             work becomes necessary
                    ↓
            governed execution
                    ↓
          agents + tools + integrations
                    ↓
        success OR obstacle detected
                    ↓
       safe recovery / human escalation
                    ↓
             original work resumes
                    ↓
          evidence + KPI projection
                    ↓
             worker returns IDLE
```

The worker is persistent; the LLM is not continuously running.

---

## Safe adaptation to unexpected problems

A central Virtual Workforce direction is the ability to continue when the environment no longer matches what developers predicted.

Examples:

- an order arrives in an unknown file format,
- a vendor changes its API,
- a dependency is temporarily unavailable,
- an input is suspicious,
- a business decision becomes ambiguous,
- a previously available credential is revoked.

Intergrax should not treat all failures as retries and should not call CodeCraft on every error.

Target flow:

```text
obstacle
  ↓
diagnostics / classification
  ↓
retry? wait? alternate capability? human? quarantine?
  ↓
missing capability confirmed
  ↓
search approved Tool / Skill / Integration
  ↓
if none and policy allows:
CodeCraft → static gate → governance → hardened sandbox → tests → verification
  ↓
ephemeral capability
  ↓
resume original business goal
```

Core rule:

> **The worker may extend capability within policy. It may never extend its own authority.**

---

## Virtual Worker vs ordinary agent

| Ordinary agent | Virtual Worker |
|---|---|
| receives a task | owns a responsibility |
| usually exists for one execution | persists across many executions |
| returns a result | monitors goal completion over time |
| generally reacts to a prompt/task | can react to events and proactively evaluate goals |
| failure commonly ends the run | may classify the obstacle and recover safely |
| tools belong to one run/configuration | capabilities can be reused and conditionally acquired |
| run status is enough | needs worker lifecycle, goals, budgets and operator control |

---

## Reuse model

Virtual Worker is intended as a reusable platform primitive.

The same Autonomous Work domain can define very different workers only by changing configuration and assigned domain capabilities.

```text
Order Operations Worker
  → email + ERP + CRM + order agents

Incident Operations Worker
  → monitoring + GitHub + Kubernetes + incident agents

Accounts Receivable Worker
  → finance systems + email + finance agents

Supplier Integration Worker
  → vendor APIs + docs + CodeCraft + integration agents
```

The workers reuse shared Intergrax mechanisms rather than receiving private copies of governance, memory, execution, CodeCraft or observability.

---

## Product architecture

```text
VIRTUAL WORKFORCE APPLICATION
fleet / worker builder / goals / KPI / approvals / controls
                 ↓
AUTONOMOUS WORK DOMAIN
WorkerDefinition / WorkerInstance / Responsibility / Goal / Lifecycle / Recovery
                 ↓
          INTERGRAX PLATFORM
Execution · Agents · Governance · Collaborative Work · Memory · Tools
CodeCraft · Sandbox · Diagnostics · Observability · Hosting
```

The future Virtual Workforce application is a consumer of the domain, not its owner.

---

## Enterprise control model

A production-oriented Virtual Workforce requires explicit limits and operator controls:

- identity and workspace binding,
- authority scopes,
- policy profiles,
- budgets and concurrency limits,
- risk/autonomy tiers,
- network and secret restrictions,
- approval rules,
- pause / resume / stop,
- quarantine / kill switch,
- evidence drill-down,
- cost and SLA/KPI views.

Worker role or goal text never grants permissions.

---

## Adaptive capability tiers

| Tier | Meaning |
|---|---|
| A0 | use an existing approved capability |
| A1 | safely generate and use an ephemeral parser/helper |
| A2 | use a narrowly scoped adaptive integration with restricted network/secrets |
| A3 | prepare a durable production change through tests/shadow/canary/promotion |
| A4 | authority change — never self-authorized |

This makes autonomy configurable rather than binary.

---

## Reference flagship proof

The first recommended flagship is an **Autonomous Order Operations Worker**.

It should process normal orders while being deliberately challenged by:

- unknown and corrupted attachments,
- prompt-injection documents,
- missing or contradictory data,
- duplicate orders,
- API timeouts and rate limits,
- vendor API drift,
- revoked credentials,
- supplier outages,
- prohibited actions,
- malicious generated-code temptations,
- hardened sandbox unavailability,
- human-only business decisions,
- host restarts during active work/recovery.

The proof should measure business outcome and safety together: autonomous completion, human burden, recovery success/time, SLA, cost, evidence completeness, and zero unauthorized side effects / egress / policy violations / isolation downgrades.

---

## Maturity boundary

Today this is a **canonical architecture and implementation direction**, not a claim that Intergrax already ships a production Virtual Workforce.

Implemented platform primitives can be reused, but Autonomous Work-specific contracts, persistent worker runtime, recovery controller, worker control plane, reference application and end-to-end proof remain planned according to the canonical implementation plan.

For technical truth, read [`AUTONOMOUS_WORK.md`](../architecture/AUTONOMOUS_WORK.md). For implementation sequencing, read [`maintainers/plans/AUTONOMOUS_WORK.md`](../maintainers/plans/AUTONOMOUS_WORK.md).
