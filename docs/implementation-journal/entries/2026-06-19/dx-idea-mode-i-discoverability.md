---
id: IJ-2026-06-19-002
date: 2026-06-19
tiers:
  - tier-0
scope: EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE
plan_ref:
  - DX-IDEA-01
  - DX-IDEA-02
status: completed
commit: 5385d518
adr: none — DX documentation and bootstrap gate only; no runtime contract change
---

# DX-IDEA — Mode I idea intake discoverability and bootstrap gate

## Operator request

Close discoverability gaps in the Mode I idea-audit template mechanism (hub, audit map, plan register, bootstrap README) and add a consistency gate script.

## Summary

Indexed **Mode I** idea intake in `intergrax_runtime_architecture.md`, `INTEGRAX_HARNESS_AUDIT_MAP.md`, and `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` architecture §43.2. Registered **DX-IDEA-01** and **DX-IDEA-02** in the DX plan. Clarified `MODE=` semantics in bootstrap and orchestrator (Step 7 always requires explicit operator confirmation). Added illustrative USER CONFIG example (WhatsApp integration). Created `scripts/audit/check_idea_audit_bootstrap.py` and wired it into `AGENTS.md` verification. Updated `.cursor/rules/intergrax-iteration.mdc` documentation model and task routing.

## Project impact

Operators can find the idea-intake workflow from hub, audit map, AGENTS, and cursor rules without ad-hoc prompts. Bootstrap ↔ orchestrator drift is guarded by a lightweight script.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` §43.2 |
| Plan | `docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` — DX-IDEA-01, DX-IDEA-02 |
| ADR | none — documentation and DX workflow only |
| Audit / gap | Mode I mechanism meta-audit (discoverability P2) |

## Changed artifacts

- `docs/bootstrap/idea_audit.txt` — MODE semantics; Step 7 confirmation wording
- `docs/audit/IDEA_AUDIT_ORCHESTRATOR.md` — example USER CONFIG; journal guidance; MODE table
- `docs/bootstrap/README.md` — Mode I excludes init/resume
- `docs/intergrax_runtime_architecture.md` — hub links to Mode I
- `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — Mode I index
- `docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` — DX-IDEA phase register
- `docs/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` — §43.2 surface row
- `scripts/audit/check_idea_audit_bootstrap.py` — new consistency gate
- `scripts/audit/architecture_audit_common.py` — `idea_audit` map entries
- `AGENTS.md`, `.cursor/rules/intergrax-iteration.mdc` — routing and verification

## Verification

```bash
python scripts/audit/check_idea_audit_bootstrap.py
python scripts/audit/check_docs_domain_pairs.py
```

Result: pass.

## Risks and follow-ups

- `check_implementation_journal.py` still fails on pre-existing 2026-06-18 entries (out of scope).
- Concrete product ideas (e.g. WhatsApp) should run Mode I and update INTEGRATIONS pair after approval.
