---
id: IJ-2026-06-10-006
date: 2026-06-10
tiers:
  - tier-0
scope: EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE
plan_ref:
  - FAUDIT-32
status: completed
commit: 69d7adce
adr: none — operator prompts; no runtime contract change
---

# Per-domain Harness audit prompts (21 copy-paste instructions)

## Operator request

Replace ad-hoc per-iteration audit instructions with reusable, domain-scoped audit prompts aligned to the 21 architecture/plan pairs and the Harness audit map.

## Summary

Created `docs/audit/<DOMAIN>.md` for all 21 domains with copy-paste prompt blocks and `scripts/generate_domain_audit_prompts.py` for regeneration after canon changes. Linked from architecture hub and guides index.

## Project impact

Domain audits become repeatable and comparable — operators paste one prompt per layer, get evidence-based maturity scores, and feed remediation back into plan rows without reinventing audit scope each session.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/PLATFORM_FOUNDATION.md` (doc governance) |
| Plan | `docs/plan/PLATFORM_FOUNDATION.md` §6.1ah FAUDIT-32 |
| Guides | `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` |

## Changed artifacts

- `docs/audit/*.md` — 21 domain prompts
- `scripts/generate_domain_audit_prompts.py` — generator

## Verification

```bash
python scripts/check_docs_domain_pairs.py
```

Result: pass.

## Risks and follow-ups

- Prompts drift if canon changes without regenerating — run generator after domain contract updates.
- `audit/README.md` index was added in this commit but later removed; domain prompt files remain self-contained.
