# `modality.image_analyst`

**Bundle:** `modality` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Image analysis with detect, OCR, and ingest path.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `modality` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `vision.detect`, `vision.ocr_regions`, `rag.ingest_document`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `vision.detect` | Catalog tool |
| `vision.ocr_regions` | Catalog tool |
| `rag.ingest_document` | Catalog tool |

## Related skills

- Other `modality` bundle skills - see bundle [USAGE.md](../USAGE.md)
