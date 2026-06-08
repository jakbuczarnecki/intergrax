# `modality.vision_ocr`

**Bundle:** `modality` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Vision OCR pipeline for document images.

## How it works

vision.* + document.parse_preview via model_inference.

## How to use

modality_skill_profile(); enable modality on lab host.

## What you get

Multimodal path before rag.document_ingest.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `vision.ocr_regions` | OCR text regions |
| `vision.detect` | Region detection |
| `document.parse_preview` | Parse structure preview |

## Related skills

- `rag.document_ingest`
- `harness.vision_qa`
