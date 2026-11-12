# SFX generalization set seal (SFX-GEN-SEAL-01)

| Field | Value |
|-------|-------|
| Record ID | `SFX-GEN-SEAL-01` |
| Date | 2026-11-12 |
| Author session | `sfx-gen-author` |
| Capability | SCENARIO-FACTORY-X (SFX) |
| Reveal unit | `SFX-P12-04` |

## Independence statement

The GEN set was written by the session `sfx-gen-author`, which is independent of the SFX implementer, architect and orchestrator roles. The author did not read the SFX master plan, ADRs or design files, and wrote the problems only from a short brief (reference-scenario domains to avoid, required fields, format). No SFX implementer, architect or orchestrator has seen the problem content.

## Set summary (no problem text)

- **Problem count:** 6 (`GEN-01` .. `GEN-06`).
- **Domains:**
  1. `GEN-01` - HR onboarding
  2. `GEN-02` - Logistics exception management
  3. `GEN-03` - Customer service triage (telecommunications)
  4. `GEN-04` - Privacy compliance (GDPR data subject requests)
  5. `GEN-05` - Insurance claims (property first notice of loss)
  6. `GEN-06` - Utilities field service (water network incidents and crew dispatch)
- None of the domains duplicates the reference scenarios (invoice automation, API-to-API order sync, RAG policy decision support with human approval).
- Each problem has the fields `id`, `domain`, `problem_statement`, `stakeholders`, `systems_involved`, `constraints` (at least one HITL constraint), `success_criteria` (3-5 measurable items), `edge_cases` (at least 3, including at least one adversarial / prompt-injection input) and `data_available` (synthetic data only; all organisations and persons are fictitious).

## Canonicalization rule

`GEN_SET.json` is one JSON array sorted by `id`. Every object has its keys sorted. The file is UTF-8 without BOM, with no insignificant whitespace (separators `,` and `:`), no trailing newline, and non-ASCII characters written literally (not `\u` escaped), following RFC 8785 (JCS) style. All values are strings or arrays of strings, so there are no number-formatting differences. Reference implementation:

```python
json.dumps(items_sorted_by_id, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
```

The author confirmed that re-parsing the file and re-canonicalizing it gives identical bytes.

## Digest and storage

| Item | Value |
|------|-------|
| Algorithm | SHA-256 |
| Digest of `GEN_SET.json` | `8b6ff8ebbc1a1eb839ffe80b502a6ad7896d38214836077803934ac37d3e6e0d` |
| Size | 21,892 bytes |
| Cross-check | Python `hashlib` and Windows `certutil -hashfile ... SHA256` gave the same digest |
| Storage (outside the repo) | `D:\Projekty\_wt\_scratch\sfx-gen-set\GEN_SET.json` |
| Readable copy (not hashed) | `D:\Projekty\_wt\_scratch\sfx-gen-set\GEN_SET.md` |

`GEN_SET.json` is the only sealed artefact. `GEN_SET.md` is a convenience copy; if the two differ, the JSON wins.

## Reveal rule

- The GEN set content stays outside the repository and is revealed only at unit **`SFX-P12-04`**.
- At reveal, the first step is to recompute SHA-256 of `GEN_SET.json` and compare it with the digest above. A mismatch invalidates the GEN set evaluation.
- After the reveal, any change to the factory (code, templates, prompts, configuration or composition) requires re-running the **full GEN set plus REF-1..3** at the new SHA. Partial re-runs (only the problems that failed) are not valid evidence.

## No-tuning rule

- Before `SFX-P12-04`, nobody in the SFX implementer, architect or orchestrator roles may read, request, summarise or derive the GEN set content, or design, tune or test the factory against it.
- The factory must not contain GEN-specific templates, prompts, fixtures, keyword lists or special cases, before or after the reveal. Fixes made after the reveal must be general and are subject to the full re-run in the reveal rule.
- Any accidental exposure before `SFX-P12-04` must be recorded, and the operator decides whether a new GEN set must be written by a new independent author.
