<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization Claim Guardrails

## Purpose

This document defines safe, bounded, non-marketing wording for public-facing references to Intergrax Token Optimization work. It is a claim-boundary control document, not a marketing brief. Use it before publishing blog posts, README excerpts, outreach copy, issue comments, or partner-facing summaries that mention token optimization.

**Main engine guide:** [`../features/token_optimization/README.md`](../features/token_optimization/README.md)

**README promotion gate:** main README Token Optimization promotion (badge, savings numbers, proof links) is allowed only after **TOKEN-10H** when a checked-in proof has passed **TOKEN-10G** hard gates and been independently audited.

---

## What has been proven (foundation and execution engine)

The current proof is bounded to the following facts:

- Intergrax has a **deterministic Token Optimization pipeline** (layer registry, pipeline runner, built-in layer catalog, configuration evals, third-party plugin contract proof).
- Intergrax has **policy-governed LLM configuration routing** (`TokenOptimizationLLMRouter`, approved configuration catalog, deterministic compiler, live Ollama E2E on tested models).
- Intergrax has **protected-region validation**, **compression receipts**, and **safe fallback** metadata.
- Intergrax has implemented **deterministic exact deduplication**, **extractive filtering**, and a **budget-aware context packing prototype** (character-budget prototype).
- Intergrax has **cache-prefix contracts**, **append-only prefix validation helpers**, and a **provider-neutral cache-aware compaction timing policy** (helper-level — not yet wired into runtime request assembly).
- The **internal evaluation pack** uses a **synthetic evaluation corpus** with **char-level**, **strategy-separated** metrics.
- Reports are designed to avoid raw-content exposure.

Supporting vocabulary: deterministic pipeline, approved configuration routing, protected regions, receipts, synthetic evaluation corpus, char-level prototype, strategy-separated attribution, helper-level cache-prefix contracts.

---

## What is planned (TOKEN-10 — not yet proven)

Allowed to state as **planned** (not as passed proof):

- Cache-stable prompt assembly wired into the real request path.
- vLLM prefix-cache integration and universal TOML proof harness.
- Cache-aware router and pipeline orchestration with RUN/DEFER/BYPASS/REQUIRE_REVIEW gate.
- Policy-governed in-cache compaction.
- Reproducible universal platform proof separate from LKW product proof.

Use platform-owned terminology: `cache-stable prompt assembly`, `stable prefix`, `dynamic tail`, `provider prefix-cache reuse`, `cache-aware optimization`, `in-cache compaction`, `universal Token Optimization proof`. Do not call the implementation “VIKTOR algorithm/cache/runtime.”

---

## Allowed public wording now

- Intergrax has a deterministic Token Optimization pipeline.
- Intergrax has policy-governed LLM configuration routing.
- Intergrax has protected-region validation, receipts, and fallback.
- Cache-stable prompt and vLLM proof integration are **planned under TOKEN-10**.
- Existing character-level or synthetic results are bounded to their documented workloads.
- Intergrax includes early token-optimization building blocks for deterministic exact deduplication and priority-aware context packing.
- The budget-aware packing layer is currently a character-budget prototype, not a provider-tokenizer-accurate budget engine.

---

## Allowed only after TOKEN-10G proof passes

- Reproducible prefix-cache proof on the tested vLLM version/model.
- Measured cached-prefix reuse for the tested proof corpus.
- Measured warm-vs-cold latency or prefill difference on the tested corpus.
- Measured content-reduction results for the tested cases.
- Measured combined behavior with attribution kept separate (content reduction vs provider cache reuse).

---

## Forbidden without additional evidence

- Universal percentage savings.
- Guaranteed reduction for every model.
- Claude-equivalent billing reduction from vLLM or self-hosted runs.
- Global production readiness.
- Semantic equivalence for arbitrary lossy content.
- Guaranteed cache retention.
- Provider-independent cache behavior.
- Claims derived by mixing cached tokens with removed tokens.
- “Intergrax reduces token usage by X%” (without corpus, model, provider, and attribution qualifiers).
- “production-proven token savings” (without TOKEN-10G + LKW-PF6-C evidence as applicable).
- “token-accurate optimizer” (without provider-aware counting path for the claimed workload).
- “automatic token reduction without tradeoffs”.
- “real customer workload reduction” (without real-customer proof).
- Prompt-cache proof has passed (before TOKEN-10G).

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
- confirm TOKEN-10G hard gates passed (for cache/prefix claims)

---

## Reviewer checklist

- Does the wording imply token-accurate savings without evidence?
- Does it imply production readiness?
- Does it mix dedupe, packing, truncation, and cache reuse?
- Does it hide that the current corpus is synthetic?
- Does it mention character-level metrics when numbers are shown?
- Does it avoid raw/private content?
- Does it conflate vLLM prefix-cache reuse with Claude billing discounts?
- Does it promote README proof links before TOKEN-10H?
