<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Token Optimization Claim Guardrails

## Purpose

This document defines safe, bounded, non-marketing wording for public-facing references to Intergrax Token Optimization work. It is a claim-boundary control document, not a marketing brief. Use it before publishing blog posts, README excerpts, outreach copy, issue comments, or partner-facing summaries that mention token optimization.

**Main engine guide:** [`token_optimization/README.md`](token_optimization/README.md)

**Public proof dashboard:** [`../../PROOFS.md`](../proofs/PROOFS.md)

**README promotion gate:** performance badges, savings numbers, and broad proof-result promotion require accepted cross-provider evidence, checked-in public evidence, final claim review, and explicit limitation approval. Neutral capability discovery remains allowed when it uses qualified, non-numeric wording — see § README discovery and promotion boundary.

---

## Current claim boundary

A bounded durable in-cache compaction mechanism is implemented.

The supported bounded scope includes:

- durable artifact repository behavior;
- validation;
- CAS activation;
- explicit/default-off operation.

This does not establish:

- live provider-wide behavior;
- rollback execution;
- provider KV-cache mutation;
- production rollout;
- general availability;
- universal or production-proven savings.

Detailed implementation phases, dependencies and review state:
[`plan/TOKEN_OPTIMIZATION.md`](plan/TOKEN_OPTIMIZATION.md).

Stable evidence gates for broader public wording:

- accepted cross-provider proof;
- checked-in independently reviewable evidence;
- final public claim review;
- explicit limitations;
- promotion approval;
- real-customer evidence before real-customer claims.

Use platform-owned terminology: `cache-stable prompt assembly`, `stable prefix`, `dynamic tail`, `provider prefix-cache reuse`, `cache-aware optimization`, `in-cache compaction`, `universal Token Optimization proof`. Do not call the implementation “VIKTOR algorithm/cache/runtime.”

---

## Approved public wording

- Intergrax has a deterministic Token Optimization pipeline.
- Intergrax has policy-governed LLM configuration routing.
- Intergrax has protected-region validation, receipts, and fallback.
- Intergrax has cache-stable prompt assembly and exact-send integrity checks.
- Intergrax has a cache-aware orchestration gate in which only a deterministic `RUN` decision executes the selected optimization pipeline.
- Intergrax includes a bounded vLLM prefix-cache proof path for the documented version, model, and proof environment.
- Intergrax includes early token-optimization building blocks for deterministic exact deduplication and priority-aware context packing.
- The budget-aware packing layer is currently a character-budget prototype, not a provider-tokenizer-accurate budget engine.
- Existing synthetic or character-level evaluation results apply only to their documented workloads.
- Content reduction and provider prefix-cache reuse are measured and attributed separately.
- A bounded durable in-cache compaction mechanism is implemented with durable repository, validation, CAS activation, and explicit/default-off operation; it is not publicly claimable as complete live-provider or production behavior.

---

## Conditional wording

Claims in this section are allowed only with the stated qualifiers. Do not present them as universal or production-proven.

### vLLM prefix cache

Allowed only with:

- exact vLLM version
- exact model
- proof corpus
- cold/warm/changed-prefix controls
- environment limitations

### Content reduction

Allowed only with:

- corpus
- metric unit
- strategy
- fallback/no-op attribution
- quality validation

### Token metrics

Allowed only with:

- provider/tokenizer identified
- or char-level explicitly stated

### TOKEN-10D

Allowed:

- cache-aware orchestration gate is implemented
- the preferred `CacheAwareTokenOptimizationRuntime.run()` entrypoint normalizes explicit caller-provided cache and policy signals into `CacheAwareCompactionTimingInput` before orchestration
- the lower-level `CacheAwareTokenOptimizationOrchestrator` remains available for advanced callers that already possess a normalized timing input

Product boundary (preserve in public wording):

- unknown cache state is not a cache miss
- reported zero is not unknown
- TTL is never inferred
- global provider metrics are not per-request cache evidence
- provider prefix-cache reuse is not content reduction

Not allowed without further proof:

- cache-aware optimization automatically reduces cost in production

### Example conditional statements

- For the documented vLLM 0.23.0 proof environment and canonical tested model, Intergrax can report bounded provider prefix-cache reuse metrics.
- For the documented synthetic evaluation corpus, Intergrax can report character-level content reduction results when the metric unit and strategy are stated.
- TOKEN-10D implements controlled pipeline execution with provider-neutral cache signal normalization and a public runtime entrypoint; the preferred runtime normalizes explicit caller-provided signals before orchestration; it does not perform in-cache compaction.

### Allowed only after the evidence gates pass

- Reproducible prefix-cache proof on the tested vLLM version/model.
- Measured cached-prefix reuse for the tested proof corpus.
- Measured warm-vs-cold latency or prefill difference on the tested corpus.
- Measured content-reduction results for the tested cases.
- Measured combined behavior with attribution kept separate (content reduction vs provider cache reuse).

---

## Forbidden wording

Do not use the following without the required evidence, corpus, and qualifiers:

- Universal percentage savings.
- Guaranteed reduction for every model.
- Claude-equivalent billing reduction from vLLM or self-hosted runs.
- Global production readiness.
- Semantic equivalence for arbitrary lossy content.
- Guaranteed cache retention.
- Provider-independent cache behavior.
- Claims derived by mixing cached tokens with removed tokens.
- “automatic token reduction without tradeoffs”.
- “real customer workload reduction” (without real-customer proof).
- Prompt-cache proof has passed before the required evidence gates are satisfied.

Explicit forbidden examples:

- Do not say “Intergrax reduces token usage by X%” without a named corpus, provider/model, metric unit, strategy attribution, and checked-in evidence.
- Do not say “production-proven token savings” before the required production evidence exists.
- Do not call Intergrax a “token-accurate optimizer” while the general optimization path has no provider-aware tokenizer.
- Do not claim durable compaction is enabled by default, generally available, publicly proven, fully production-ready, or part of a completed production rollout.
- Do not claim rollback execution implemented, human-review UX implemented, provider KV-cache mutation, measured production savings from UCL, or long conversations are already cheaper.
- Do not claim production durable compaction, session revision store, semantic compression active in production, summary reuse implemented end-to-end, summary cache available in production, Nexus repository wiring, artifact reservations operational in production, internal summarizer recursion prevented in runtime, distributed single-flight creation, production artifact persistence, cross-process artifact sharing, restart durability for artifacts, single-flight creation implemented end-to-end, TOKEN-10E public rollout, or LKW UCL integration.

---

## Explicit metric and tokenizer boundaries

The general internal evaluation corpus is synthetic where documented, and parts of the content-reduction evaluation use char-level metrics.

The current general optimization evaluation path has no provider-aware tokenizer. Some provider integrations can report provider-side token or cache metrics for their bounded proof paths, but this does not make the general content-reduction pipeline provider-tokenizer-accurate.

Provider-reported token or cache metrics from a bounded vLLM proof do not turn the general optimization engine into a provider-tokenizer-accurate system.

Therefore, character-level evaluation must not be presented as a token-accurate savings claim. No token-accurate savings claim is supported for the general optimization path without a named provider/tokenizer and checked-in evidence.

---

## Required qualifiers

Any public mention of optimizer evaluation must include or remain consistent with:

- synthetic corpus (when applicable)
- internal or proof-corpus identification
- metric unit: chars or provider tokens
- tokenizer/provider if token-based
- separate dedupe, packing, truncation, and cache reuse
- exclude fallback/no-op cases or label them separately
- verify no raw/private content in report
- link to exact commit/report when publishing numbers
- include limitations

---

## Evidence checklist before publishing numbers

Before publishing any numeric claim:

- identify corpus
- identify whether corpus is synthetic or real
- identify metric unit: chars or provider tokens
- identify tokenizer/provider if token-based
- separate content reduction from provider prefix-cache reuse
- separate dedupe from packing from truncation
- exclude fallback/no-op cases or label them separately
- verify no raw/private content in report
- link to exact commit/report
- include limitations
- confirm accepted evidence, final claim review, and explicit limitations are recorded (for cache/prefix claims)

---

## Reviewer checklist

- Does the wording imply token-accurate savings without evidence?
- Does it imply production readiness?
- Does it mix dedupe, packing, truncation, and cache reuse?
- Does it hide that the current corpus is synthetic?
- Does it mention character-level metrics when numbers are shown?
- Does it avoid raw/private content?
- Does it conflate vLLM prefix-cache reuse with Claude billing discounts?
- Does it promote README proof links before the required public evidence is accepted?
- Does the wording imply that timing signals are automatically provider-derived?
- Does the wording imply that DEFER or BYPASS executed an optimization?
- Does the wording turn the bounded implemented mechanism into a claim of complete live-provider, provider-wide or production behavior?
- Does the wording describe TOKEN-10D sub-phases as incomplete while TOKEN-10D overall is accepted/closed?
- Does the wording conflate unknown cache state with a cache miss, reported zero with unknown, or global provider metrics with per-request cache evidence?

---

## README discovery and promotion boundary

### Neutral discovery allowed now

Allowed before performance evidence is accepted:

- a neutral Token Optimization capability name;
- classification as a featured platform capability;
- a link to the main guide ([`token_optimization/README.md`](token_optimization/README.md));
- non-numeric descriptions of implemented mechanisms;
- explicit bounded-proof wording;
- qualified current status with link to [`../../PROOFS.md`](../proofs/PROOFS.md).

### Performance promotion remains gated

Still requires accepted cross-provider evidence, checked-in public evidence, final claim review, and explicit limitation approval:

- savings percentages;
- performance badges;
- universal cost claims;
- broad proof-result promotion;
- claims of universal proof completion;
- production-proven savings.

Do not weaken the forbidden wording in § Forbidden wording above.
