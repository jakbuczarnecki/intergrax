<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization Claim Guardrails

## Purpose

This document defines safe, bounded, non-marketing wording for public-facing references to Intergrax token-optimization work completed in **TOKEN-OPT-3C-B** (deterministic exact deduplication layer), **TOKEN-OPT-3D** (budget-aware context packing prototype), and **TOKEN-OBS-3E-F** (stronger optimizer evaluation pack).

It is a claim-boundary control document, not a marketing brief. Use it before publishing blog posts, README excerpts, outreach copy, issue comments, or partner-facing summaries that mention token optimization.

## What has been proven

The current proof is bounded to the following facts:

- Intergrax has implemented **deterministic exact deduplication** as a standalone optimization layer.
- Intergrax has implemented a **budget-aware context packing prototype** using an **estimated character budget**.
- The packing prototype preserves **MUST_KEEP** fragments, prefers **HIGH_PRIORITY**, drops **DROPPABLE** under pressure, and only compacts **COMPRESSIBLE** fragments via whitespace normalization.
- The **internal evaluation pack** uses a **synthetic evaluation corpus**.
- Metrics are **char-level** and **strategy-separated**.
- Reports are designed to avoid raw-content exposure.

Supporting vocabulary: char-level, synthetic evaluation corpus, internal evaluation pack, prototype, strategy-separated attribution, deterministic exact dedupe, estimated character budget.

## What has not been proven

The following must not be implied from the current proof:

- No **provider-aware tokenizer** has been introduced for this proof.
- No **token-accurate savings claim** is made.
- No **production pipeline engine** has been introduced by **TOKEN-OBS-3E-F**.
- No **benchmark CLI** / **public benchmark artifact** is introduced by this step.
- No real customer or private data is used in the synthetic evaluation pack.
- No **semantic compression** or **LLM summarization** is included in this proof.
- No broad claim such as “reduces token usage by X%” is allowed from this proof alone.

## Approved public wording

Reusable approved statements:

- Intergrax includes early token-optimization building blocks for deterministic exact deduplication and priority-aware context packing.
- The current stronger-optimizer proof uses synthetic cases and character-level metrics to verify behavior such as duplicate removal, must-keep preservation, fallback handling, and strategy-separated attribution.
- The budget-aware packing layer is currently a character-budget prototype, not a provider-tokenizer-accurate budget engine.
- Evaluation results are intended to show mechanism behavior and attribution discipline, not general production savings claims.

## Conditional wording

Allowed only with explicit qualifiers:

- On synthetic evaluation cases, the internal evaluation pack can show character-level reductions from exact deduplication and priority-aware packing.
- When reporting numbers, numbers must be labeled as character-level, synthetic-corpus, case-specific, and strategy-separated.
- Any future token-savings statement requires a provider-aware tokenizer/counting adapter and a clearly identified evaluation corpus.

## Forbidden wording

Do not say:

- “Intergrax reduces token usage by X%”
- “production-proven token savings”
- “token-accurate optimizer”
- “model-aware token budget”
- “automatic token reduction without tradeoffs”
- “dedupe + packing benchmark proves general savings”
- “real customer workload reduction”
- “semantic compression”
- “LLM summarization”

Unless a future task explicitly proves those claims.

## Required qualifiers

Any public mention of the stronger-optimizer proof must include or remain consistent with:

- synthetic corpus
- internal evaluation
- character-level metrics
- strategy-separated attribution
- no provider-aware tokenizer yet
- no public benchmark claim yet

## Evidence checklist before publishing numbers

Before publishing any numeric claim:

- identify corpus
- identify whether corpus is synthetic or real
- identify metric unit: chars or provider tokens
- identify tokenizer/provider if token-based
- separate dedupe from packing from truncation
- exclude fallback/no-op cases or label them separately
- verify no raw/private content in report
- link to exact commit/report
- include limitations

## Reviewer checklist

- Does the wording imply token-accurate savings?
- Does it imply production readiness?
- Does it mix dedupe, packing, and truncation?
- Does it hide that the current corpus is synthetic?
- Does it mention character-level metrics when numbers are shown?
- Does it avoid raw/private content?
