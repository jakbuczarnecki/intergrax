<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Token Optimization — Plugin Engine Roadmap Reset

**Status:** Done / Closed (docs-only roadmap alignment).  
**Scope:** Token Optimization only.  
**Purpose:** Re-anchor the next Token Optimization phase around a plugin-based, layered optimization engine instead of adding isolated algorithms before the execution core exists.

---

## Current position

The Token Optimization track has already built the main foundation pieces:

- shared policy / source / mechanism / strategy vocabulary,
- protected-region validation and receipt foundations,
- telemetry and regression-reporting contracts,
- helper-level optimization layers and evaluation packs,
- prompt-cache / cache-prefix stabilization contracts,
- advisory recommendation, evaluation, policy gate, presets, and resolver.

The latest closed advisory block is:

```text
TOKEN-7A — Done / Closed
TOKEN-7B — Done / Closed
TOKEN-7C — Done / Closed
TOKEN-7D — Done / Closed
TOKEN-7D-R — Done / Closed
```

The important gap is not another single optimization algorithm. The important gap is the execution core:

```text
Layer Registry + Pipeline Runner + Plugin Boundary + Configurable Engine
```

---

## Roadmap decision

Do **not** continue by adding more standalone algorithms as disconnected helpers.

The next phase must first make Token Optimization executable as a configurable layered engine:

```text
registered layers
+ pipeline config
+ request
→ pipeline runner
→ per-layer results
→ final result / fallback
```

Only after that should new mechanisms such as near-deduplication, trim, schema minimization, or third-party optimizers be added as pipeline layers.

---

## Target capability

The target Token Optimization model is:

```text
Token Optimizer
= plugin-based layered engine
+ built-in optimization layer catalog
+ third-party plugin boundary
+ configuration/eval reports
+ later LLM optimization router
```

A developer should eventually be able to provide a custom optimizer through a thin adapter that implements the shared layer/plugin contract. The engine must then be configurable so that a pipeline can run only that plugin, or a selected combination of built-in and plugin layers.

Example target behavior:

```text
pipeline mode: REPLACE
layers:
  - plugin.vendor.custom_optimizer

result:
  only plugin.vendor.custom_optimizer is executed
  standard validation/fallback/reporting still applies
```

---

## Non-negotiable boundaries

The plugin engine must not allow plugins or built-in layers to bypass platform safety.

Required boundaries:

- no auto-apply outside explicit reviewed runtime integration,
- no plugin bypass of platform policy,
- no plugin bypass of protected-region validation,
- no private telemetry bus,
- no raw prompt / raw document / secret export in reports,
- no mutation of canonical tool contracts,
- no provider-specific tokenizer replacement,
- no semantic compression enabled by default,
- fallback to original content when validation fails.

The LLM-based router, when introduced later, must configure the deterministic engine. It must not replace the engine with free-form LLM summarization.

---

## Updated execution roadmap

### TOKEN-8A — Layer Registry and Pipeline Runner

**Goal:** Build the first real execution core for the layered optimizer.

Simple meaning:

```text
register layers
resolve pipeline config
run selected layers in order
collect layer results
return final pipeline result
```

Expected scope:

- layer registry,
- duplicate layer handling,
- built-in/custom/plugin descriptor compatibility,
- pipeline mode `DEFAULT` / `REPLACE`,
- disabled layer skipping,
- required-layer failure behavior,
- sequential `TokenOptimizationLayerRequest` / `TokenOptimizationLayerResult` flow,
- final `TokenOptimizationPipelineResult` aggregation.

Out of scope:

- dynamic package loading,
- LLM router,
- production runtime integration,
- provider calls,
- observability emission,
- new optimization algorithm.

---

### TOKEN-8B — Built-in Layer Catalog Wiring

**Goal:** Register existing built-in layers through the new registry instead of keeping them as disconnected helpers.

Simple meaning:

```text
make built-in layers discoverable and runnable by the pipeline runner
```

Initial candidates:

- extractive filtering,
- future trim layer,
- future dedup layer,
- future schema minimization layer.

Out of scope:

- new large algorithm design unless needed for a minimal catalog proof,
- LLM router,
- third-party package loading.

---

### TOKEN-8C — Pipeline Configuration Evals

**Goal:** Prove that different pipeline configurations produce measurable, safe, explainable outcomes.

Simple meaning:

```text
run the same cases through different engine configs and compare results
```

Example configurations:

```text
disabled
measure_only
only_trim
only_extract_filtering
trim_plus_extract_filtering
plugin_only_fake
```

Reports must show:

- original size / optimized size where measured,
- applied layers,
- bypassed layers,
- failed layers,
- fallback status,
- validation status,
- raw-content-safe metadata only.

---

### TOKEN-8D — Third-party Plugin Adapter Contract Proof

**Goal:** Prove that an external developer can implement a custom optimization layer and run it through the standard engine path.

Simple meaning:

```text
fake third-party layer
→ registry
→ pipeline mode REPLACE
→ only that layer runs
→ standard validation/fallback/reporting remains enforced
```

This is still a contract proof, not dynamic plugin loading from packages.

Out of scope:

- package manager integration,
- marketplace/distribution model,
- remote plugin execution,
- unsafe plugin sandboxing claims.

---

### TOKEN-9A — LLM Optimization Router Contract

**Goal:** Define the intelligent router that decides how to configure the deterministic engine for a specific input.

Simple meaning:

```text
LLM router does not optimize content directly.
LLM router chooses the engine configuration.
```

Router output should be a structured routing decision, for example:

```text
optimize: true
pipeline_mode: REPLACE
layers:
  - builtin.trim
reason: short terminal output only needs trimming
risk: low
review_required: false
```

Out of scope:

- free-form LLM compression,
- bypassing deterministic validation,
- production auto-apply.

---

### TOKEN-9B — LLM Router Evals

**Goal:** Test whether the router chooses safe engine configurations for representative cases.

Example expectations:

```text
short clean text → no optimization
long noisy log → trim / extractive filtering
code-heavy content → preserve / no mutation / measure only
repeated context → dedup candidate
high-risk evidence → review or bypass
```

Reports must remain redaction-safe and must evaluate routing quality, not just token savings.

---

### TOKEN-9C — Safe Router → Engine Integration

**Goal:** Connect the LLM router to the deterministic engine under strict policy gates.

Simple flow:

```text
input
→ router recommends pipeline config
→ policy gate validates allowed config
→ engine runs deterministic layers
→ validation / receipts / fallback
→ final result
```

Out of scope until explicitly reviewed:

- production auto-apply,
- silent lossy compression,
- router bypassing policy,
- router bypassing validation.

---

## Deferred algorithm work

Near-deduplication, schema minimization, trim, and other algorithms remain important, but should now be introduced as engine layers after the registry/pipeline runner exists.

Recommended placement:

```text
TOKEN-8A — engine core first
TOKEN-8B — built-in catalog wiring
TOKEN-8C — config evals
then add/expand algorithms as layer-specific tasks
```

---

## Immediate next task

The next implementation task should be:

```text
TOKEN-8A — Layer Registry and Pipeline Runner
```

One-sentence summary:

```text
TOKEN-8A builds the first executable core of Token Optimization: a registry and pipeline runner that can run selected optimization layers in a configurable order while preserving policy, validation, fallback, and reporting boundaries.
```
