# Plan review — T-IGX-033-A (BND-03 CLI half / command-set seam; r2 independent review)

| Field | Value |
|---|---|
| Reviewer session id | `hist-rev-033a` |
| Assurance | `process_enforced` |
| Review SHA | `fe09375177bd0fa0d9ebb661b7f9bf64e2f1ed25` |
| Plan | `governance/tickets/plans/PLAN_T-IGX-033-A.md` |
| Ticket | `governance/tickets/T-IGX-033-A.md` |
| Date | 2026-11-08 |

## Digest verification

Recomputed digest:

`ae167672d4075e3e4dc190ac93c78a937d7ad0c435dfc3971e615143e4273775`

Matches ticket.

## Defect confirmation at review SHA

`intergrax/cli/external_work.py` contains Tier-0 upward static/dynamic edges into Tier-3 (`importlib.import_module`, `from applications....`), consistent with HIST02-F-001 / F-002 CLI half. Plan section 1 documents F-002 sites.

`intergrax/applications/_shared/product_manifest_registry.py` uses `importlib.import_module` for Tier-3 manifest modules — **explicitly out of scope** for this ticket (master plan A.2 / plan Q4).

## Allowed files

Sections 3.1 and 3.2 enumerate product, test, matrix, and doc paths. No `product_manifest_registry.py`. No `AGENTS.md`. No IIAC material-path edits.

New gate `test_bnd03_tier0_upward_edges_gate.py` removes tolerances (ebh5 CLI skip, EBH-7 `TIER_IMPORT_EXCEPTIONS` → empty) without weakening assertions.

## Gate weakening

None proposed. RED-first on new BND-03 gate and dependent EBH-5/EBH-7 evaluations. Historical EBH-7 audit report not edited.

## ACT-N11

Seven product paths under `intergrax/` and `applications/` per plan §6; expected multi-line ACT-N11 red until re-activation.

## File-disjoint overlap

Disjoint from 030-B, 030-C, 032-A. No shared product files.

## Adopted decision (b2)

Command-set seam; launcher `python -m applications.governed_contractor_application`. Old CLI names become invalid on Tier-0 CLI (argparse fail-closed). `product_manifest_registry.py` excluded → blocker `HIST-F-BND-03B`; FRZ-BND-03 not closed by 033-A alone.

## Verdict

**PASS**
