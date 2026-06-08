# `modality.vision_segment`

**Bundle:** `modality` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Vision segmentation pipeline: segment regions, detect, and OCR.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `modality` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `vision.segment`, `vision.detect`, `vision.ocr_regions`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `vision.segment` | Catalog tool |
| `vision.detect` | Catalog tool |
| `vision.ocr_regions` | Catalog tool |

## Related skills

-
 
`
m
o
d
a
l
i
t
y
.
*
`
 
p
e
e
r
s
 
i
n
 
s
a
m
e
 
b
u
n
d
l
e
