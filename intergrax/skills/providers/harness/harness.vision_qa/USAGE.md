# `harness.vision_qa`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

**Vision + RAG QA smoke**: object detection on media plus retrieval for grounded answers about visual content. Demonstrates multimodal harness path (W-ML) without full modality smoke breadth.

## How it works

Pairs `vision.detect` with `rag.retrieve` - agent can describe image regions and cross-check against indexed docs.

## How to use

```python
from intergrax.skills.providers.harness.manifests import HARNESS_VISION_QA

AgentContract(id="vision_qa_lab", skills=[HARNESS_VISION_QA], ...)
```

## What you get

Focused two-tool multimodal Q&A pack for lab demos.

## Tools unlocked

`vision.detect`, `rag.retrieve`
