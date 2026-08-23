# Scenario Specification

**Scenario:** AI Incident Investigation with Independent Verification  
**Slug:** `ai_incident_investigation`  
**Proof class:** SCENARIO  
**Status:** ACCEPTED FOR IMPLEMENTATION — scenario concept passed human Scenario Quality Gate; implementation, executable proof, evidence, and report have not started yet.

[← Back to public Scenario page](README.md)

---

## A. SCENARIO

### Synthetic scenario provenance

This is a **fully fictional operational scenario**. The organization, facility, production lines, incident, operational events, numerical values, staffing records, equipment telemetry, datasets, and fixtures are **synthetic**. They are not derived from any employer, customer, production environment, confidential source, or proprietary system. No real enterprise system is reproduced.

### Real problem

A fictional industrial manufacturer monitors production throughput and target attainment across multiple production lines at Plant A. During a Tuesday–Thursday window, target attainment on **Line 4** degrades sharply: production performance drops from roughly 94% to 78% while cycle-time degradation concentrates on heavier, more complex product assemblies.

Operations leadership asks an AI investigation system to determine the most **defensible root-cause diagnosis** so staffing, line allocation, and capacity decisions can be made before an upcoming high-volume production window. The investigation must use operational telemetry, staffing records, equipment signals, and production workload records — the same fragmented sources a human incident lead would query — not a single curated dashboard.

This is an **operational incident investigation**: wrong conclusions trigger real operational harm, not a SQL tutorial exercise.

### Who has the problem

- **Plant operations managers** responsible for production targets and schedule commitments.
- **Incident leads / control-tower engineers** who must produce a diagnosis under time pressure.
- **Capacity and production planners** who act on the diagnosis (shift changes, line reallocation, overtime scheduling).

### Why it matters

Target attainment misses directly affect committed production schedules, customer trust, and readiness for the upcoming high-volume production window. A confident but wrong root-cause story causes expensive, harmful, or distracting actions while the real fault persists.

### Failure consequences

A wrong diagnosis can trigger:

- unnecessary overtime or extra shifts at the wrong production line;
- incorrect work rescheduling that overloads another line;
- staffing actions based on a false “understaffed” narrative;
- failure to repair the actual equipment or process fault;
- delayed recovery through the upcoming high-volume production window;
- erosion of trust in AI-assisted operations if the system “sounds right” but is wrong.

### Why it is difficult

The incident sits in a **noisy, multi-source operational environment**:

- workload, throughput, staffing, equipment, and line facts do not align in one view;
- correlation is strong and causation is weak in the first pass;
- staffing truth is split across systems that disagree;
- some records are **stale** relative to the incident window;
- a decisive equipment signal is **not in the first query set**;
- time pressure rewards a fast, plausible story.

An AI system that optimizes for fluent narrative will often pick the shortcut that matches leadership anxiety (“we’re overloaded”) rather than the evidence-supported fault.

### Naive / simple failure mode

Initial facts look like a textbook overload story:

- production workload / order volume on Line 4 increased ~22% versus the prior week;
- throughput / target attainment fell in the same period;
- Line 4 shows disproportionate performance misses versus other production lines;
- heavier, more complex assemblies correlate with longer cycle times in aggregate;
- one staffing feed suggests reduced headcount on the affected shift.

A naive investigator (human or LLM) confidently concludes:

```text
production line overload caused by workload growth
```

and recommends capacity and staffing responses. That story is **plausible, leadership-aligned, and wrong** given the full admissible evidence.

### WOW factor

WOW is **not** two agents, SQL, RAG, tool count, or orchestration depth — and **not** that the AI eventually guesses the fixture's correct answer.

The impressive result is that:

1. a plausible wrong explanation is prevented from becoming an accepted diagnosis;
2. the exact evidentiary weakness is identified;
3. additional evidence is gathered specifically to distinguish competing hypotheses;
4. a revised diagnosis is accepted only if observable evidence supports it;
5. otherwise the system remains **UNRESOLVED**;
6. neither investigator nor verifier receives hidden ground truth.

A successful proof run must make that transition **visually obvious** in the final report: plausible hypothesis → concrete evidence defect → targeted follow-up → revised or refused diagnosis.

### Skeptic Challenge

> “I can build the same thing with an LLM + memory + RAG + a few LangGraph nodes in ten minutes.”

A simple Investigator + Critic graph is **insufficient** for this scenario because it typically provides:

- no **hidden ground truth isolation** — fixture truth leaks through prompts, metadata, or naming;
- no **observable evidence** sufficient to distinguish competing hypotheses without evaluator truth;
- no **verifier independence** from investigator private reasoning or ground truth;
- no stable **evidence identity** across investigation and challenge passes;
- no auditable **evidence dependency graph** (what claim rests on which observation);
- no **provenance** discipline for conflicting staffing sources;
- no rule that **stale** records cannot silently become “current”;
- no rule that **missing** decisive evidence cannot be invented;
- no **structured evidence-grounded challenge** (only generic “I disagree”);
- no **bounded** challenge/revise loop with explicit termination;
- no **UNRESOLVED** outcome when hypotheses cannot be distinguished;
- no **deterministic evaluation** of both investigator and verifier behavior.

Equivalent guarantees can be engineered elsewhere; they are **not** obtained merely by adding a few graph nodes. This scenario requires **observable system-level guarantees**, not agent theater.

### Adversarial conditions

The scenario embeds these adversarial conditions **credibly** in the manufacturing incident:

#### A. Plausible shortcut / correlation trap

```text
workload ↑
throughput / target attainment ↓
```

in the same window makes “overload caused the performance drop” an attractive false causal conclusion.

#### B. Conflicting evidence

Two staffing / workforce evidence sources disagree on line staffing during the incident window. The system cannot silently pick the source that supports its current hypothesis.

#### C. Stale evidence

At least one apparently relevant staffing snapshot is **outside the incident-valid time window** (e.g., prior-week roster export). It cannot be treated as current without explicit qualification and admissibility handling.

#### D. Missing evidence

Material evidence needed to distinguish overload vs equipment fault — e.g., **machine / robotic handling / feeder station telemetry for the complex-assembly step** — is not available in the initial investigation pass. The system must not fabricate it.

#### E. Hidden causal factor (discoverable, not leaked)

The fixture encodes an actual incident factor: **intermittent feeder unit #4 degradation affecting the complex-assembly handling step** — not sustained workload overload. This is **ground truth for fixture construction and deterministic evaluation only**. It must be discoverable through admissible follow-up investigation; it must **not** leak through model-visible instructions, naming, or metadata (see **Ground truth isolation** in § B. SOLUTION).

Replacing one correlation with another is insufficient. The intended follow-up evidence must support a **best-supported bounded operational root-cause diagnosis**, not a new unsupported temporal correlation (`equipment fault ↑` + `throughput ↓` → therefore causation).

#### F. Competing hypotheses (must be distinguished)

The system must explicitly distinguish at least these plausible hypotheses:

```text
H1 — sustained production overload from workload growth
H2 — understaffing on the affected shift / line
H3 — intermittent equipment / process degradation
```

The final **RESOLVED** outcome must not merely identify H3. It must show why available admissible evidence makes H3 the **best-supported bounded operational diagnosis** and why key alternatives are weakened or rejected.

#### G. Causal evidence pattern (before / during / after)

Follow-up evidence should support a defensible operational diagnosis through a temporal comparison pattern, not a single-window correlation:

```text
BEFORE degradation
→ handling station operating normally
→ complex-assembly throughput near baseline

DURING degradation
→ feeder unit #4 enters degraded / intermittent failure state
→ complex-assembly throughput materially drops
→ affected cycle-time degradation spikes

COMPARISON
→ unaffected lines / stations remain comparatively stable
→ incident-window staffing is normal / does not explain the change

AFTER recovery
→ feeder unit returns to normal state
→ complex-assembly throughput recovers toward baseline
→ cycle-time behavior correspondingly improves
```

Exact percentages need not be frozen at design stage. This pattern defines the **evidence shape** the scenario requires; it is not fixture implementation detail.

### Scenario Quality Gate

This scenario **passed** the quality gate because:

- real operational pain and production schedule risk exist;
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
5. Accept a **revised RESOLVED diagnosis** only when material claims survive falsification, are supported by admissible observable evidence, and constitute the **best-supported bounded operational root-cause diagnosis** among competing hypotheses — not universal scientific causality.
6. Emit **UNRESOLVED** when critical evidence is unavailable or hypotheses remain indistinguishable — without confident guessing and **without** using evaluator ground truth to “know” the answer.

### Ground truth isolation

This scenario uses a controlled / synthetic fixture so deterministic evaluation can know what actually happened. That creates a **leakage risk** that the design must prevent.

> Ground truth belongs to fixture construction and deterministic evaluation only.

The Investigator and independent Verifier **MUST NOT** receive the hidden answer through:

- prompts;
- system instructions;
- context;
- labels;
- fixture / table / file names;
- tool descriptions;
- metadata;
- comments;
- expected-answer fields;
- scenario instructions;
- evaluator output during execution.

```text
FIXTURE GROUND TRUTH
        ↓
DETERMINISTIC EVALUATOR ONLY

OBSERVABLE EVIDENCE
        ↓
INVESTIGATOR + VERIFIER
```

The model-facing system must discover and support the diagnosis from **observable evidence** only.

Do **not** expose convenience fields or semantically leaking names the model could observe, such as:

```text
root_cause = equipment_fault
hidden_root_cause_events
correct_answer
expected_diagnosis
```

### Ground truth is not evidence

```text
Ground truth answers:
“What did the fixture designer encode as the actual incident?”

Evidence answers:
“What can the investigating system legitimately observe and use?”
```

A **PASS** must not occur because the model simply matches hidden fixture truth. The evaluator may compare final system behavior with ground truth, but **ground truth itself is not admissible investigation evidence**.

### Verifier independence

The verifier is independent **semantically**, not merely because it is called by another node or agent.

> The Verifier evaluates candidate material claims against observable cited evidence and admissibility rules. It does not inspect the Investigator's private reasoning or hidden fixture ground truth.

**Verifier may receive only:**

- candidate material claims;
- evidence identities / references used to support them;
- relevant observable evidence necessary to test them;
- evidence metadata / provenance / admissibility required to assess the claim;
- bounded scenario verification rules.

**Verifier must NOT receive:**

- Investigator hidden chain-of-thought / private reasoning;
- hidden ground truth;
- expected answer;
- evaluator verdict;
- a prompt saying which diagnosis is correct.

Private chain-of-thought must not appear in proof or report design.

**Structured challenge:** `"I disagree"` is not a valid challenge. The verifier must identify a concrete class of evidence defect, such as:

- unsupported causal dependency;
- missing evidence dependency;
- conflicting evidence not resolved;
- stale evidence incorrectly admitted;
- material alternative hypothesis not addressed;
- evidence does not support claim strength.

Each challenge must identify the relevant material claim and evidence references. Implementation details remain deferred.

**Verifier is not the oracle:**

> The verifier is not trusted blindly. The final proof must deterministically evaluate whether the verifier behaved according to the scenario contract.

```text
Investigator can fail.
Verifier can fail.
Evaluator determines whether the observable run satisfies the proof invariants.
```

### Step-by-step story

#### RESOLVED path (intended success story)

```text
INCIDENT — Line 4 target attainment degradation (Tue–Thu)
↓
initial evidence gathering (workload, throughput by line, assembly complexity cohorts, staffing feeds)
↓
plausible candidate diagnosis — “workload surge overloaded Line 4”
↓
independent falsification attempt on causal claim
↓
specific evidentiary challenge — e.g., normalized throughput per unit vs workload;
  conflicting staffing sources; stale roster treated as current
↓
targeted follow-up investigation — machine / feeder station telemetry / fault signals for complex-assembly step
↓
new evidence — before/during/after pattern: feeder unit #4 degradation correlates with
  complex-assembly throughput drop and cycle-time spike; unaffected lines stable; staffing normal
↓
revised diagnosis — best-supported operational root-cause diagnosis:
  intermittent feeder unit #4 degradation affecting complex-assembly throughput;
  workload growth is a contributing amplifier, not the initiating cause
↓
independent verification of revised claim against evidence graph;
  H1 (overload) and H2 (understaffing) weakened by comparative evidence
↓
RESOLVED — bounded operational diagnosis with auditable evidence trail
  (not universal causality; not merely matching hidden ground truth)
```

#### UNRESOLVED path (required insufficient-evidence story)

```text
INCIDENT
↓
investigation across available operational sources
↓
critical distinguishing evidence unavailable (e.g., equipment telemetry cannot be retrieved)
↓
credible hypotheses remain indistinguishable (H1 overload vs H2 understaffing vs H3 equipment fault)
↓
UNRESOLVED — explicit refusal to assert root cause; documented evidence gaps
  (model-visible outcome remains UNRESOLVED even if evaluator privately knows fixture truth)
```

The system **must be allowed to refuse** a causal diagnosis.

A **RESOLVED** verdict must require more than candidate answer matching hidden ground truth. It requires evidence sufficient to:

- support the accepted diagnosis;
- expose claim → evidence dependencies;
- address contradictory or stale evidence;
- weaken or reject material competing hypotheses (H1, H2, H3);
- survive verifier falsification;
- remain reconstructable without hidden evaluator truth.

If critical evidence necessary to distinguish competing hypotheses is unavailable, the model-visible result remains **UNRESOLVED** even when the deterministic fixture evaluator privately knows what happened.

### Guarantees

Candidate system-level guarantees (design stage — not yet demonstrated):

- Material incident diagnoses are backed by **auditable evidence identities**, not prose confidence.
- **Evidence dependencies** are explicit: each material claim cites admissible observations.
- **Conflicting** sources are surfaced; silent cherry-picking fails.
- **Stale** records cannot become current without admissibility qualification.
- **Missing** evidence cannot be hallucinated; gaps are explicit.
- Falsification cites a **concrete evidence defect** (unsupported leap, conflict, staleness, gap).
- Challenge/revise behavior is **bounded** (no infinite loop).
- Final outcome is **reconstructable** from observable evidence and evaluator checks — without hidden evaluator truth.
- **UNRESOLVED** is a first-class outcome when certainty is not justified.
- **Ground truth** never enters Investigator or Verifier model-visible context.
- **RESOLVED** requires evidence-supported discrimination among competing hypotheses (H1, H2, H3).
- **Verifier** operates independently of Investigator private reasoning and hidden ground truth.
- **Verifier** behavior is subject to deterministic evaluation; verifier is not the oracle.
- Causal claims use **bounded operational diagnosis** language, not universal causality.

### Claim

Candidate bounded falsifiable claim (design — **not** a proven public claim):

> **No material incident diagnosis is accepted unless its material claims are supported by auditable evidence and survive an independent falsification attempt.**

“Material” means claims that would justify operational actions (root-cause attribution, staffing changes, line reallocation, capacity decisions). Correlation-only narratives do not qualify.

“Independent falsification” means the verifier evaluates claims against observable cited evidence per the **Verifier independence** contract — not investigator private reasoning, not hidden ground truth, not expected answers.

### PASS

Candidate PASS semantics at scenario level:

- An initial misleading hypothesis **may** appear (correlation trap is realistic).
- An unsupported causal conclusion **must not** become the accepted final diagnosis.
- Challenge must identify a **concrete evidence defect**, not generic disagreement.
- Conflicting evidence **cannot** be silently ignored.
- Stale evidence **cannot** silently become current.
- Missing evidence **cannot** be fabricated.
- Challenge can request **targeted follow-up** evidence gathering.
- Final **RESOLVED** diagnosis must be supported by admissible **observable** evidence — not evaluator ground truth.
- Material competing hypotheses (H1, H2, H3) must be **addressed**; accepted diagnosis must be best-supported among alternatives.
- Hidden ground truth **never** exposed to Investigator or Verifier.
- Verifier does **not** depend on Investigator private reasoning.
- Verifier challenge is **evidence-grounded** (concrete defect class, not “I disagree”).
- **UNRESOLVED** remains correct when distinguishing evidence is unavailable — even if evaluator privately knows fixture truth.
- Causal diagnosis uses **before/during/after** comparative evidence, not a single correlation swap.
- Insufficient evidence yields explicit **UNRESOLVED**, not confident guessing.
- Challenge/revise execution is **bounded**.
- Final outcome is **reconstructable** from observable evidence and evaluation artifacts.

### FAIL

Explicit FAIL if any of the following occurs:

- unsupported causal diagnosis is **accepted** as final;
- correlation is promoted to causation **without** supporting evidence;
- final diagnosis accepted **solely because it matches evaluator / hidden ground truth**;
- ground-truth or expected-answer **leakage** into model-visible context (prompts, naming, metadata, instructions);
- “equipment fault” accepted based on **another unsupported correlation** (replacing one trap with another);
- competing material hypotheses **ignored without evidence**;
- conflicting evidence is **silently discarded**;
- stale evidence is treated as current **without** admissibility handling;
- missing evidence is **hallucinated** or invented;
- independent verifier gives only subjective disagreement with **no evidence basis**;
- verifier receives or relies on Investigator **private reasoning** or **hidden expected answer**;
- workflow **loops indefinitely**;
- system claims **RESOLVED** despite insufficient evidence;
- final evidence trail **cannot explain** why the diagnosis was accepted, rejected, or revised.

### Adversarial attacks

| Attack | Expected system response |
|--------|---------------------------|
| Workload–throughput correlation trap | Candidate overload story (H1) may form; must not pass falsification without causation evidence |
| Equipment–throughput correlation swap | H3 must be supported by before/during/after comparative pattern, not new unsupported correlation |
| Competing hypotheses H1/H2/H3 | Must address alternatives; RESOLVED requires best-supported bounded diagnosis |
| Conflicting staffing feeds | Surface conflict; do not pick favorites silently |
| Stale roster export | Mark staleness; do not treat as current incident staffing |
| Missing equipment telemetry initially | Acknowledge gap; follow-up or UNRESOLVED |
| Pressure to “just pick one hypothesis” | UNRESOLVED if distinguishers stay unavailable — even if evaluator knows truth |
| Ground-truth leakage via naming/metadata | FAIL — investigator/verifier must not receive hidden answer |
| Fluent but empty critic | FAIL — challenge must cite evidence defect |

### Excluded claims

This design does **not** claim:

- universal root-cause analysis for arbitrary incidents;
- **scientific proof of causation** from arbitrary observational datasets;
- **universal causal inference** from correlation alone;
- guaranteed discovery of true causality from arbitrary data;
- production readiness or commercial validation;
- universal superiority over LangGraph or other frameworks;
- that equivalent guarantees **cannot** be implemented elsewhere;
- universal hallucination prevention;
- correctness for all manufacturing operations or all operators;
- real-user or production business validation.

The proof may show that Intergrax bundles useful guarantees; it does not claim they cannot be implemented elsewhere.

### Limitations

- Single bounded manufacturing incident fixture (not arbitrary enterprise data).
- Synthetic or seeded operational dataset with designed adversarial conditions.
- One primary RESOLVED path and one primary UNRESOLVED path at acceptance.
- Evaluator semantics scoped to this scenario’s claim, not all incident types.
- Design stage: no runtime, evidence, or report exists yet.

---

## C. INTERGRAX FIT

NOT YET PERFORMED

Capability-fit against current repository HEAD is intentionally deferred until after scenario acceptance.

INTERGRAX FIT is not a single-domain assignment. Expected future analysis:

```text
required guarantee
→ Intergrax mechanism
→ exact owner/component
→ participating domain(s)
→ AVAILABLE / AVAILABLE BUT NEEDS WIRING / MISSING
```

---

## D. GAP DECISION

NOT YET PERFORMED

---

## E. PROOF BUILD

NOT STARTED — scenario accepted for implementation; capability-fit and proof build pending.
