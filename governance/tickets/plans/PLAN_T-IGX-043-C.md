# PLAN T-IGX-043-C — ARCH-FREEZE node T-AF2-DS-EXC-PACKAGE: PDC-X residual qualification package

Status: DRAFT for Operator review. Drafted 2026-11-16 by an implementer session of the ARCH-FREEZE program (plan MEP-AF2 v2.1, node T-AF2-DS-EXC-PACKAGE, wave W0). Nothing here is approved, applied or audited.

## 1. Purpose

Publish one documentation file, `docs/project/maintainers/qualification/PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md`, that qualifies each residual left by PDC-X for admissibility in a frozen architecture. It is the evidence package that the W3 product train and the W7 independent auditor use. It is a qualification of residual admissibility, not a re-certification of PDC-X, and it accepts or closes no row. Final acceptance of every row stays with the W7 independent audit (T-B7-ECAP-2C).

## 2. Scope

- Table A: the 60 declared exceptions of `PDC_EXCEPTION_REGISTER.md` (EX-G-01..07, EX-L-01..14, EX-D1-01..07, EX-D2-01..07, EX-D3-01..04, EX-D4-01..05, EX-D5-01..05, EX-D6-01..04, EX-D7-01..03, EX-D8-01..04), each exactly once, with the 13 columns required by plan section E.
- Table B: the 13 `PARTIAL` rows of `PDC_DEBT_REGISTER.md` section 13 (D1-R009, D1-R015, D1-R017, D2-R004, D3-R005, D3-R018, D3-R019, D5-R002, D6-R007, D6-R010, D6-R015, D7-R015, D7-R022), each exactly once, with the plan node of section E.
- Table C: the other handoff residuals (MCP-SERVER-X R-P7-01..06, R-P8-01, F-P8-01, F-MCP-P5/P6-GOV, P8-PROC-01, LIM-RESEARCH-STRICT-FACTORY-MEMORY; ENT-AUDIT-X F-W3-001, OPEN-BP-01; PDC handoff sections 5.9 and 6), each with its plan node.
- Authority: DS-EXC as decided in Operator packet 1 (recorded 2026-11-16, an English rendering of the Operator's chat answer), per-item dispositions of plan section E, SEC-05/06 handling of section D and OD-AF-18.

## 3. Explicitly out of scope

- Every path under `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`, `pyproject.toml` and `uv.lock`.
- The PDC-X record directories, the roadmap, the FRZ checklist and the IIAC records. No PDC-X record file is edited.
- Any PASS, closure, certification, FRZ promotion or independent-verification claim. Dispositions of plan section E are quoted with attribution, not re-asserted.
- Reachability measurement: that is T-AF2-CLOSED-WORLD (T-IGX-043-E). This package marks each reachability statement as `PDC_STATED`, `STATIC_VERIFIED @7960ae7f6 path:line` or `TO_BE_MEASURED (T-AF2-CLOSED-WORLD)`.

## 4. Write set (the only paths the PR may touch)

```text
docs/project/maintainers/qualification/PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md
governance/tickets/T-IGX-043-C.md
governance/tickets/plans/PLAN_T-IGX-043-C.md
```

## 5. Method

1. Worktree outside the repository on branch `af2/w0-dsexc`, cut from `origin/development` at `7960ae7f6c82da60ca9acdc0ceebf63810002a42`.
2. Row data come from the PDC registers and handoffs at that SHA. Every row has a source line in `FACT_CHECK.md` (kept under `.tmp/session/arch-freeze/W0-dsexc/`, not committed).
3. Static caller checks use `git grep` at the baseline SHA; a check states only what it shows.
4. The exactly-once check is a `python -I` script kept in the output folder; its output goes into the PR body.

## 6. Test contract

No product test is run. One docs-only verification step, one process at a time:

- the exactly-once script over the new file, `PDC_EXCEPTION_REGISTER.md` and `PDC_DEBT_REGISTER.md` (expected: 60/60 EX ids and 13/13 `PARTIAL` ids);
- `uv run python scripts/docs/check_docs_domain_pairs.py`, only if the file location triggers a pair rule;
- reviewer check: `git diff --stat origin/development...HEAD` lists exactly the three paths of section 4.

## 7. Risks

- **Stale PDC statements.** Several `path:line` cites in the PDC register have drifted. Mitigation: each affected row says so; `PDC_STATED` cells are quotations, not facts of this package.
- **Over-reading a static check.** A grep that finds a definition does not prove behaviour. Mitigation: each `STATIC_VERIFIED` sentence names exactly what was found.
- **Pin validity.** The file lives under `docs/`; it does not change `intergrax/`, `agents/` or `applications/`, so it does not affect the IIAC activation pin (ACT-N11). Whether the docs-only change needs an EBH-7 re-pin is decided by the orchestrator under DP-61 (one EBH-7 pin as the last event).
