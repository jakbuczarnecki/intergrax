# AI Incident Investigation with Independent Verification

## Scenario identity

- **Title:** AI Incident Investigation with Independent Verification
- **Slug:** `ai_incident_investigation`
- **Public question:** Can an AI investigate an operational incident without turning correlation into a confident false diagnosis?
- **Secondary formulation:** Can an AI prove its incident diagnosis — and refuse to guess when the evidence is insufficient?
- **Lifecycle status:** DESIGN / NOT YET ACCEPTED
- **Executable proof:** No executable proof, evidence, or report exists yet. This package is a scenario design document awaiting human Scenario Quality Gate before any implementation is permitted.

---

## A. SCENARIO

### Real problem

A regional logistics operator monitors parcel-handling SLA for warehouses and distribution lanes. During a Tuesday–Thursday window, SLA for parcels routed through the **North Central warehouse** degrades sharply: on-time delivery rate drops from roughly 94% to 78% while customer complaints about late heavy parcels spike.

Operations leadership asks an AI investigation system to determine the most **defensible root-cause diagnosis** so staffing, routing, and capacity decisions can be made before the weekend peak. The investigation must use operational telemetry, staffing records, equipment signals, and shipment facts — the same fragmented sources a human incident lead would query — not a single curated dashboard.

This is an **operational incident investigation**: wrong conclusions trigger real operational harm, not a SQL tutorial exercise.

### Who has the problem

- **Regional logistics operations managers** responsible for SLA and customer commitments.
- **Incident leads / control-tower engineers** who must produce a diagnosis under time pressure.
- **Capacity and routing planners** who act on the diagnosis (shift changes, lane reroutes, hub load redistribution).

### Why it matters

SLA misses directly affect contractual penalties, customer trust, and weekend peak readiness. A confident but wrong root-cause story causes expensive, harmful, or distracting actions while the real fault persists.

### Failure consequences

A wrong diagnosis can trigger:

- unnecessary capacity expansion or overtime at the wrong facility;
- incorrect traffic rerouting that overloads another hub;
- staffing actions based on a false “understaffed” narrative;
- failure to repair the actual equipment or process fault;
- delayed recovery through the weekend peak;
- erosion of trust in AI-assisted operations if the system “sounds right” but is wrong.

### Why it is difficult

The incident sits in a **noisy, multi-source operational environment**:

- volume, delay, staffing, equipment, and lane facts do not align in one view;
- correlation is strong and causation is weak in the first pass;
- staffing truth is split across systems that disagree;
- some records are **stale** relative to the incident window;
- a decisive equipment signal is **not in the first query set**;
- time pressure rewards a fast, plausible story.

An AI system that optimizes for fluent narrative will often pick the shortcut that matches leadership anxiety (“we’re overloaded”) rather than the evidence-supported fault.

### Naive / simple failure mode

Initial facts look like a textbook overload story:

- shipment volume through North Central increased ~22% versus the prior week;
- delay rate rose in the same period;
- North Central shows disproportionate SLA misses versus other warehouses;
- heavy parcels correlate with longer handling time in aggregate;
- one staffing feed suggests reduced headcount on the heavy-parcel lane.

A naive investigator (human or LLM) confidently concludes:

```text
warehouse overload caused by volume growth
```

and recommends capacity and staffing responses. That story is **plausible, leadership-aligned, and wrong** given the full admissible evidence.

### WOW factor

WOW is **not** two agents, SQL, RAG, tool count, or orchestration depth.

The WOW is:

```text
The system initially has enough evidence to produce a highly plausible diagnosis.

It is not allowed to accept that diagnosis merely because it sounds convincing.

The falsification step identifies the exact unsupported causal leap,
forces targeted additional evidence gathering,
and either produces a defensible revised diagnosis
or refuses to claim a root cause.
```

A successful proof run must make that transition **visually obvious** in the final report: plausible hypothesis → concrete evidence defect → targeted follow-up → revised or refused diagnosis.

### Skeptic Challenge

> “I can build the same thing with an LLM + memory + RAG + a few LangGraph nodes in ten minutes.”

That shortcut **would fail this scenario’s quality gate** because it typically provides:

- no stable **evidence identity** across investigation and challenge passes;
- no auditable **evidence dependency graph** (what claim rests on which observation);
- no **provenance** discipline for conflicting staffing sources;
- no rule that **stale** records cannot silently become “current”;
- no rule that **missing** decisive evidence cannot be invented;
- no requirement that challenge cites a **concrete evidence defect** (only generic “I disagree”);
- no **bounded** challenge/revise loop with explicit termination;
- no **UNRESOLVED** outcome when hypotheses cannot be distinguished;
- no **deterministic evaluator** reconstructing why the final verdict was accepted, revised, or refused.

A simple Investigator + Critic dialog changes wording without guaranteeing any of the above. This scenario requires **observable system-level guarantees**, not agent theater.

### Adversarial conditions

The scenario embeds these adversarial conditions **credibly** in the logistics incident:

#### A. Plausible shortcut / correlation trap

```text
volume ↑
delay ↑
```

in the same window makes “overload caused delays” an attractive false causal conclusion.

#### B. Conflicting evidence

Two staffing / workforce evidence sources disagree on lane staffing during the incident window. The system cannot silently pick the source that supports its current hypothesis.

#### C. Stale evidence

At least one apparently relevant staffing snapshot is **outside the incident-valid time window** (e.g., prior-week roster export). It cannot be treated as current without explicit qualification and admissibility handling.

#### D. Missing evidence

Material evidence needed to distinguish overload vs equipment fault — e.g., **sorter lane telemetry for the heavy-parcel lane** — is not available in the initial investigation pass. The system must not fabricate it.

#### E. Hidden causal factor

Targeted follow-up on equipment signals reveals a more defensible factor: **intermittent sorter failures on the heavy-parcel lane** (not sustained volume overload). This answer is discoverable through admissible follow-up investigation, not leaked in model-visible instructions or ground-truth prompts.

### Scenario Quality Gate

This scenario targets the quality gate because:

- real operational pain and contractual SLA risk exist;
- failure has meaningful cost;
- the problem was not invented to demo a single Intergrax feature;
- uncertainty, conflict, staleness, missing evidence, and false causation are intrinsic;
- negative falsification is natural (correlation trap);
- outcomes can be evaluated with explicit PASS/FAIL semantics;
- the story is understandable without Intergrax internals;
- WOW comes from a difficult guarantee (evidence-gated diagnosis), not agent count;
- a skeptical engineer can challenge the design before any code exists.

---

## B. SOLUTION

### Desired behavior

The investigation system behaves like a disciplined incident lead:

1. Gather initial operational evidence within the incident window.
2. Form a **candidate** diagnosis only when tied to cited evidence items.
3. Run an **independent falsification attempt** on material causal claims.
4. If falsification finds a concrete evidence defect, perform **targeted follow-up** (not infinite re-litigation).
5. Accept a **revised RESOLVED diagnosis** only when material claims survive falsification and are supported by admissible evidence.
6. Emit **UNRESOLVED** when critical evidence is unavailable or hypotheses remain indistinguishable — without confident guessing.

### Step-by-step story

#### RESOLVED path (intended success story)

```text
INCIDENT — North Central SLA degradation (Tue–Thu)
↓
initial evidence gathering (volume, delays by hub, parcel weight cohorts, staffing feeds)
↓
plausible candidate diagnosis — “volume surge overloaded North Central”
↓
independent falsification attempt on causal claim
↓
specific evidentiary challenge — e.g., normalized delay per parcel vs volume;
  conflicting staffing sources; stale roster treated as current
↓
targeted follow-up investigation — sorter lane telemetry / fault signals for heavy-parcel lane
↓
new evidence — intermittent sorter faults correlate with heavy-parcel delay spike
↓
revised diagnosis — equipment fault on heavy-parcel lane (not sustained overload)
↓
independent verification of revised claim against evidence graph
↓
RESOLVED — defensible root-cause diagnosis with auditable evidence trail
```

#### UNRESOLVED path (required insufficient-evidence story)

```text
INCIDENT
↓
investigation across available operational sources
↓
critical distinguishing evidence unavailable (e.g., equipment telemetry cannot be retrieved)
↓
credible hypotheses remain indistinguishable (overload vs lane fault vs upstream feed issue)
↓
UNRESOLVED — explicit refusal to assert root cause; documented evidence gaps
```

The system **must be allowed to refuse** a causal diagnosis.

### Guarantees

Candidate system-level guarantees (design stage — not yet demonstrated):

- Material incident diagnoses are backed by **auditable evidence identities**, not prose confidence.
- **Evidence dependencies** are explicit: each material claim cites admissible observations.
- **Conflicting** sources are surfaced; silent cherry-picking fails.
- **Stale** records cannot become current without admissibility qualification.
- **Missing** evidence cannot be hallucinated; gaps are explicit.
- Falsification cites a **concrete evidence defect** (unsupported leap, conflict, staleness, gap).
- Challenge/revise behavior is **bounded** (no infinite loop).
- Final outcome is **reconstructable** from observable evidence and evaluator checks.
- **UNRESOLVED** is a first-class outcome when certainty is not justified.

### Claim

Candidate bounded falsifiable claim (design — **not** a proven public claim):

> **No material incident diagnosis is accepted unless its material claims are supported by auditable evidence and survive an independent falsification attempt.**

“Material” means claims that would justify operational actions (root-cause attribution, staffing changes, rerouting, capacity decisions). Correlation-only narratives do not qualify.

### PASS

Candidate PASS semantics at scenario level:

- An initial misleading hypothesis **may** appear (correlation trap is realistic).
- An unsupported causal conclusion **must not** become the accepted final diagnosis.
- Challenge must identify a **concrete evidence defect**, not generic disagreement.
- Conflicting evidence **cannot** be silently ignored.
- Stale evidence **cannot** silently become current.
- Missing evidence **cannot** be fabricated.
- Challenge can request **targeted follow-up** evidence gathering.
- Final **RESOLVED** diagnosis must be supported by admissible available evidence.
- Insufficient evidence yields explicit **UNRESOLVED**, not confident guessing.
- Challenge/revise execution is **bounded**.
- Final outcome is **reconstructable** from observable evidence and evaluation artifacts.

### FAIL

Explicit FAIL if any of the following occurs:

- unsupported causal diagnosis is **accepted** as final;
- correlation is promoted to causation **without** supporting evidence;
- conflicting evidence is **silently discarded**;
- stale evidence is treated as current **without** admissibility handling;
- missing evidence is **hallucinated** or invented;
- independent verifier gives only subjective disagreement with **no evidence basis**;
- workflow **loops indefinitely**;
- system claims **RESOLVED** despite insufficient evidence;
- final evidence trail **cannot explain** why the diagnosis was accepted, rejected, or revised.

### Adversarial attacks

| Attack | Expected system response |
|--------|--------------------------|
| Volume–delay correlation trap | Candidate overload story may form; must not pass falsification without causation evidence |
| Conflicting staffing feeds | Surface conflict; do not pick favorites silently |
| Stale roster export | Mark staleness; do not treat as current incident staffing |
| Missing sorter telemetry initially | Acknowledge gap; follow-up or UNRESOLVED |
| Pressure to “just pick one hypothesis” | UNRESOLVED if distinguishers stay unavailable |
| Fluent but empty critic | FAIL — challenge must cite evidence defect |

### Excluded claims

This design does **not** claim:

- universal root-cause analysis for arbitrary incidents;
- guaranteed discovery of true causality from arbitrary data;
- production readiness or commercial validation;
- universal superiority over LangGraph or other frameworks;
- universal hallucination prevention;
- correctness for all logistics operations or all operators;
- real-user or production business validation;
- that equivalent behavior is impossible outside Intergrax.

The proof may show that Intergrax bundles useful guarantees; it does not claim they cannot be implemented elsewhere.

### Limitations

- Single bounded logistics incident fixture (not arbitrary enterprise data).
- Synthetic or seeded operational dataset with designed adversarial conditions.
- One primary RESOLVED path and one primary UNRESOLVED path at acceptance.
- Evaluator semantics scoped to this scenario’s claim, not all incident types.
- Design stage: no runtime, evidence, or report exists yet.

---

## C. INTERGRAX FIT

NOT YET PERFORMED

Capability-fit against current repository HEAD is intentionally deferred until after human scenario acceptance.

---

## D. GAP DECISION

NOT YET PERFORMED

---

## E. PROOF BUILD

NOT STARTED — blocked on scenario acceptance and capability-fit.
