<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization — Cache-prefix Stabilization Architecture and Contract Addendum

**Status:** Done / Closed architecture / contract addendum (`TOKEN-OPT-5A`)

**Parent feature architecture:** [`TOKEN_OPTIMIZATION.md`](TOKEN_OPTIMIZATION.md) (§8.3.1 Cache-prefix stabilization)

**Parent feature plan:** [`../plan/TOKEN_OPTIMIZATION.md`](../plan/TOKEN_OPTIMIZATION.md)

**Scope:** cache-aware prompt/thread architecture, provider cache attribution, and cache-safe optimization sequencing.

---

## 1. Why this addendum exists

Prompt caching is not just a cheaper way to count tokens. For long-running agent workloads it becomes an architectural constraint on prompt assembly, tool exposure, thread history, compaction timing, provider adapters, and observability.

This addendum captures the Token Optimization implications from cache-heavy agent-runtime designs such as Viktor's prompt-caching architecture research:

```text
https://viktor.com/research/how-we-built-viktor-around-prompt-caching
```

The key architectural conclusion for Intergrax:

```text
Cache-prefix stabilization is a first-class Token Optimization surface.
It must be measured separately from content-reduction strategies.
```

`TOKEN-OPT-5A` closes the architecture/contract slice for cache-prefix stabilization. The canonical feature architecture now cross-references this addendum as the detailed supporting contract. This addendum does not implement runtime behavior.

---

## 2. Core distinction

Prompt caching is a **cost and latency optimization**. It is not the same thing as source-content reduction.

The following must not be mixed in receipts, telemetry, benchmark summaries, or public claims:

```text
content removed by deduplication
content removed by budget-aware packing
content omitted by extractive filtering
content shortened by structural compaction
completion reduction from output policy
provider cached-input reuse
cache-hit cost discount
cache-hit latency reduction
```

Cache-related savings must have their own attribution bucket and confidence model.

Allowed cache attribution terms:

```text
cache_read_tokens
cache_creation_tokens
cached_input_tokens
uncached_input_tokens
cache_hit_ratio
prefix_stability_status
prefix_hash
cache_discount_estimate
cache_latency_delta_estimate
cache_invalidation_reason
```

Do not report cache reuse as `saved_tokens` unless a provider-specific measurement path supports a clear baseline/optimized comparison and the claim is explicitly marked as provider-cache attribution.

---

## 3. Stable prefix as a protected optimization surface

A provider-cacheable prompt prefix should be treated as a protected optimization surface.

The stable prefix may include, depending on provider and adapter shape:

```text
system policy
agent role and stable instructions
stable tool envelope / tool catalog view
stable safety policy
stable product/runtime conventions
long-lived workspace or thread context that is intentionally cacheable
```

The stable prefix must not include volatile values such as:

```text
wall-clock timestamps
random IDs
run IDs
trace IDs
request IDs
ephemeral tool lists
provider-generated metadata
per-step diagnostic counters
per-request user text
fresh retrieved evidence
dynamic source payloads
```

Dynamic content belongs in the prompt tail, not in the stable prefix.

---

## 4. Append-only thread invariant

When provider cache reuse is active, the prompt/thread shape should prefer an append-only model:

```text
stable prefix
  + append-only messages / observations / tool results
  + dynamic tail for the current step
```

Cache-safe behavior:

```text
append new information
preserve byte-stable already-sent prefix
preserve ordering of cacheable blocks
keep dynamic per-step data after the cacheable prefix
```

Cache-hostile behavior:

```text
rewriting old thread messages
reordering historical blocks
inserting new metadata into the prefix
regenerating tool catalog text differently per step
compacting the hot prefix while the cache is still valuable
adding timestamps or run IDs near the beginning of the prompt
```

This invariant is especially important before adding lossy summarization, semantic compression, adaptive rewriting, or runtime prompt assembly rewrites.

---

## 5. Tool surface implications

Tool optimization has two independent goals:

```text
1. reduce LLM-facing tool catalog size
2. keep the LLM-facing tool envelope cache-stable
```

A shorter tool catalog that changes frequently can be worse than a slightly longer stable tool catalog that gets high cache reuse.

Rules:

- `ToolSchemaOptimizer` must not mutate canonical `ToolContract` definitions.
- Future compact catalog views should be deterministic and cache-stable for the same effective tool set.
- Dynamic tool availability should not be injected into the stable prefix unless the effective set is intentionally part of the cache key.
- Per-request tool metadata, availability reasons, runtime IDs, and diagnostic annotations belong outside the stable prefix.
- Provider adapters may expose cache-safe tool-envelope capabilities, but policy and observability remain platform-owned.

---

## 6. Provider ownership boundary

Provider-specific prompt-cache behavior belongs to `LLM_ADAPTERS` and must not be reimplemented inside Token Optimization as a parallel provider abstraction.

Token Optimization may define shared contracts and attribution vocabulary. Provider adapters own provider-specific behavior such as:

```text
explicit cache breakpoints
implicit/automatic prompt caching
cache key support
cache retention / TTL support
cache read/write usage fields
cached-token accounting
session affinity requirements for edge/self-hosted providers
provider-specific price multipliers
```

Token Optimization consumes those signals through approved adapter usage/telemetry paths.

It must not create:

```text
a second tokenizer
a private provider-cache telemetry bus
a provider-specific prompt cache client outside LLM_ADAPTERS
hidden prompt-cache configuration
```

---

## 7. Cache-aware compaction timing

Content compaction is not always beneficial when prompt caching is active.

A compaction step that rewrites a hot cached prefix may destroy more value than it saves. Therefore future compaction policy must consider both:

```text
estimated content-reduction benefit
estimated cache invalidation cost
```

Cache-aware compaction should prefer:

```text
filtering dynamic tool/log output before it becomes stable prefix
packing/reducing dynamic tail content
compacting cold or idle thread history
compacting before cache expiry only when beneficial
preserving hot cacheable prefix until cache value decays
```

It should avoid:

```text
rewriting stable prefix during active multi-step runs
semantic summarization of hot cacheable history
repacking old thread history on every step
changing stable tool catalog formatting per request
```

### Cache-aware compaction timing policy (`TOKEN-OPT-5E`)

`TOKEN-OPT-5E` is **Done / Closed**. Cache-aware compaction timing policy is now documented and covered by synthetic policy tests.

Intergrax now has a provider-neutral cache-aware compaction timing policy. Runtime/provider integration remains deferred. The policy helps avoid rewriting hot cacheable prefixes when estimated cache invalidation cost outweighs estimated content-reduction benefit.

```text
dynamic tail reduction preferred over stable-prefix rewrite
stable-prefix / full-thread compaction remains conservative
protected/semantic risk → require manual review
near-expiry or cold-history may allow RUN
helper-level / provider-neutral only
no runtime provider caching in TOKEN-OPT-5E
in-cache compaction remains future work
```

---

## 8. In-cache compaction is future work

In-cache compaction means asking a model to compact a long history while the long history itself is read from provider cache.

This can be useful, but it is intentionally out of scope for `TOKEN-OPT-5A` because Intergrax has not yet enabled semantic compression, LLM summarization, or adaptive rewriting in the optimization path.

```text
No in-cache compaction in TOKEN-OPT-5A.
No LLM summarization in TOKEN-OPT-5A.
No semantic compression in TOKEN-OPT-5A.
No adaptive rewriting in TOKEN-OPT-5A.
```

Future work must require:

```text
explicit lossy/semantic opt-in
protected-region validation
quality/regression gates
receipts and rollback metadata where persistent
provider-cache attribution separated from content-reduction attribution
operator-visible fallback behavior
```

---

## 9. Observability and attribution

Cache-aware optimization requires distinct observability fields.

Recommended cache attribution fields:

```text
run_id
step_id
tenant_id
agent_id
provider
model
cache_mode
cache_key_scope
prefix_hash
prefix_stability_status
cache_read_tokens
cache_creation_tokens
cached_input_tokens
uncached_input_tokens
cache_hit_ratio
cache_invalidation_reason
cache_discount_estimate
cache_latency_delta_estimate
content_reduction_strategy
content_saved_tokens
content_saved_chars
```

Important rule:

```text
cache_* metrics describe provider cache behavior.
content_saved_* metrics describe content reduction.
They must remain separable in telemetry, receipts, benchmark summaries, and public proof claims.
```

Candidate domain signals / event kinds for later implementation:

```text
TOKEN_CACHE_PREFIX_STABLE
TOKEN_CACHE_PREFIX_CHANGED
TOKEN_CACHE_HIT_REPORTED
TOKEN_CACHE_MISS_REPORTED
TOKEN_CACHE_INVALIDATED
TOKEN_CACHE_ATTRIBUTION_RECORDED
```

These are conceptual names only. Actual event ownership remains with the existing HOS/domain-signal architecture.

---

## 10. Roadmap impact

`TOKEN-OPT-4B` is **Done / Closed**. `TOKEN-OPT-5A` is **Done / Closed**. `TOKEN-OPT-5B` is **Done / Closed** as a combined functional block covering provider cache contracts, append-only prefix invariant helpers, and synthetic prefix-stability evaluation. `TOKEN-OPT-5C` and `TOKEN-OPT-5D` are folded into `TOKEN-OPT-5B`. `TOKEN-OPT-5E` is **Done / Closed** — cache-aware compaction timing policy is documented and covered by synthetic policy tests. Runtime/provider integration remains deferred. In-cache compaction remains future work. Cache attribution remains separate from content-reduction savings. Next decision: choose the next functional Token Optimization block before returning to `TOKEN-7A` advisory recommendations.

Recommended order:

| Order | Task | Scope | Status | Runtime behavior |
|-------|------|-------|--------|------------------|
| 1 | `TOKEN-OPT-4B` | Extractive filtering evaluation cases / regression pack | **Done / Closed** | No new runtime algorithm |
| 2 | `TOKEN-OPT-5A` | Cache-prefix stabilization architecture / contract | **Done / Closed** | Docs/contracts only |
| 3 | `TOKEN-OPT-5B` | Prompt-cache contracts and cache-prefix stability proof | **Done / Closed** | Contracts + helpers + synthetic eval |
| 4 | `TOKEN-OPT-5C` | Folded into `TOKEN-OPT-5B` | Folded | — |
| 5 | `TOKEN-OPT-5D` | Folded into `TOKEN-OPT-5B` | Folded | — |
| 6 | `TOKEN-OPT-5E` | Cache-aware compaction timing policy | **Done / Closed** | Contracts + helper + synthetic policy tests |
| 7 | `TOKEN-7A` | Advisory recommendation contract | Deferred | Still deferred |

### TOKEN-OPT-5A — cache-prefix stabilization architecture / contract

**Status:** **Done / Closed**.

**Purpose:** Define prompt-cache-aware optimization boundaries, provider responsibilities, cache attribution, and cache-safe prompt/thread assembly invariants.

**Scope:**

```text
stable prefix definition
append-only prompt/thread invariant
provider-cache attribution vocabulary
cache-safe tool envelope rules
cache invalidation reasons
cache-vs-content-reduction attribution split
```

**Out of scope:**

```text
runtime prompt assembly changes
provider API calls
adapter wiring
semantic compression
LLM summarization
in-cache compaction
auto-apply
public marketing claims
```

**Closeout criteria:**

```text
prompt caching classified as cost/latency optimization, not content reduction
cache-prefix stability added as first-class optimization surface
append-only prompt/thread invariant documented
provider-specific cache behavior assigned to LLM_ADAPTERS
cache metrics separated from token/char savings
dynamic tool/schema injection risk documented
compaction timing must account for cache invalidation cost
no runtime behavior added
```

### TOKEN-OPT-5B — prompt-cache contracts and cache-prefix stability proof

**Status:** **Done / Closed**.

`TOKEN-OPT-5B` is Done / Closed as a combined functional block covering provider cache contracts, append-only prefix invariant helpers, and synthetic prefix-stability evaluation. `TOKEN-OPT-5C` and `TOKEN-OPT-5D` are folded into `TOKEN-OPT-5B`. `TOKEN-OPT-5E` is Done / Closed.

Delivered contracts and helpers (no provider API calls, no runtime prompt assembly):

```text
PromptCacheMode
PromptCacheInvalidationReason
PromptCachePolicy
PromptCacheProviderCapabilities
PromptCacheUsageSnapshot
PromptCacheAttribution
PromptCacheBlock / PromptCachePrefixSnapshot / PromptCachePrefixStabilityResult
build_prefix_snapshot / compute_prefix_hash / evaluate_prefix_stability
preserves_append_only_prefix
synthetic prompt_cache_prefix corpus
```

### TOKEN-OPT-5C — append-only prompt/thread invariant tests

**Status:** Folded into `TOKEN-OPT-5B`.

### TOKEN-OPT-5D — cache-prefix stability evaluation pack

**Status:** Folded into `TOKEN-OPT-5B`.

### TOKEN-OPT-5E — cache-aware compaction timing policy

**Status:** **Done / Closed**.

Purpose:

```text
Add a provider-neutral policy/helper layer deciding when compaction should run, defer, bypass, or require manual review based on cache prefix stability, cache hotness, TTL proximity, expected content-reduction benefit, expected cache invalidation cost, and safety risk.
```

Delivered (no provider API calls, no runtime prompt assembly):

```text
CacheAwareCompactionTarget
CacheAwareCompactionDecision
CacheAwareCompactionReason
CacheAwareCompactionTimingInput / CacheAwareCompactionTimingDecision
decide_cache_aware_compaction_timing
synthetic cache_aware_compaction corpus
focused policy tests
```

Runtime/provider integration remains deferred. In-cache compaction remains future work.

---

## 11. Public claim guardrails

Allowed claim shape:

```text
Measured provider-cache reuse reduced billed input cost on workflow X by Y%, with provider/model/cache policy Z and validation status V.
```

Forbidden claim shape:

```text
Intergrax saves Y% tokens everywhere through prompt caching.
```

Also forbidden:

```text
mixing cache discount with content-removal savings
mixing truncation savings with cache savings
using projected cache behavior as measured proof
omitting provider/model/cache policy from cache claims
```

Public proof must separately show:

```text
content-reduction savings
provider-cache attribution
quality/regression status
fallback status
known limitations
```

---

## 12. Current decision

`TOKEN-OPT-4B` is **Done / Closed**.

`TOKEN-OPT-5A` is **Done / Closed**.

`TOKEN-OPT-5B` is **Done / Closed**.

`TOKEN-OPT-5E` is **Done / Closed**.

Next decision:

```text
Choose the next functional Token Optimization block before returning to TOKEN-7A advisory recommendations.
```

Runtime/provider integration remains deferred. In-cache compaction remains future work. Do not return to `TOKEN-7A` advisory recommendations until Intergrax has enough real optimization mechanisms, provider/cache attribution contracts, and evaluation data to make recommendations useful and safe.
