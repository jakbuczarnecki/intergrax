# Implementation Journal Entry Template

Copy this file to `entries/YYYY-MM-DD-<scope>-<slug>.md` and fill every section.
**Language:** English only. **Do not** duplicate plan tables or architecture canon — link instead.

---

```markdown
---
id: IJ-YYYY-MM-DD-NNN
date: YYYY-MM-DD
tier: tier-0 | tier-1 | tier-2 | tier-3
scope: RAG | TOOLS | agents/vendor_discovery | applications/local_workspace_application | ...
plan_ref: M-RAG.23 | TOOL-ENG-4 | AUDIT-IDEAL-14.3 | K.1 | ...
status: completed
commit: <short-hash or pending>
adr: docs/adr/ADR-XXX-NNN.md | none — <one-line rationale>
---

# <Short title — imperative, matches plan_ref>

## Operator request

Paraphrase the architect/operator question or instruction (1–3 sentences).
What problem were we trying to solve in this iteration?

## Summary

What was implemented or delivered? Reference symbols, profiles, or wiring — do not paste large code blocks.

## Project impact

What did this unlock for the Harness, agents, or product hosts?
One short paragraph; focus on capability, not file list.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/<DOMAIN>.md` or `agents/<slug>/ARCHITECTURE.md` |
| Plan | `docs/plan/<DOMAIN>.md` row / phase, or agent `IMPLEMENTATION_PLAN.md` |
| Audit / gap | GAP-* or AUDIT-IDEAL-* if applicable |

## Changed artifacts

- `path/to/file.py` — brief role
- `docs/plan/RAG.md` — status row update

## Verification

```bash
# commands run
uv run pytest path/to/test.py -q
```

Result: pass / fail / not run — reason.

## Risks and follow-ups

- Remaining risk or deferred item (link plan row).
- Suggested next step aligned with Harness AI vision.
```

---

## Filename convention

```text
entries/YYYY-MM-DD-<scope>-<slug>.md
```

Examples:

- `2026-06-10-rag-m-rag-23.md`
- `2026-06-10-tools-tool-eng-4.md`
- `2026-06-10-agents-vendor-discovery-k1.md`
- `2026-06-10-applications-lkw-hybrid-daemon.md`

## Index row (append to `INDEX.md`)

```markdown
| IJ-YYYY-MM-DD-NNN | YYYY-MM-DD | tier-0 RAG | M-RAG.23 | [rag query expansion wiring](entries/2026-06-10-rag-m-rag-23.md) | `94bea682` |
```
