# `harness.modality_smoke`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

**Modality plane smoke** (Phase W-ML): vision detection, ML predict, and batch predict. Validates Plane C registry wiring on lab hosts with `ModalityProfile` enabled.

## How it works

Resolves `vision.detect`, `ml.predict`, `ml.batch_predict` via `model_inference` registry and `ModalityInferenceExecutor`.

## How to use

```python
from intergrax.skills.providers.harness.manifests import HARNESS_MODALITY_SMOKE

AgentContract(id="modality_lab", skills=[HARNESS_MODALITY_SMOKE], ...)
```

Wire `modality_profile` on environment.

## What you get

Single skill to exercise CV + classical ML tools in gate tests.

## Tools unlocked

`vision.detect`, `ml.predict`, `ml.batch_predict`
