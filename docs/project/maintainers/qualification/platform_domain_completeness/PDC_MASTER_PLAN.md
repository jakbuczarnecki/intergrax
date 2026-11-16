# PDC-X — master plan (pointer)

The canonical PDC-X master plan (PDC-P0 through PDC-P9) is the ticket plan [`governance/tickets/plans/PLAN_T-IGX-036-A.md`](../../../../governance/tickets/plans/PLAN_T-IGX-036-A.md). Its plan digest is recorded in [`T-IGX-036-A.md`](../../../../governance/tickets/T-IGX-036-A.md).

The plan contains:

- **A** — program baseline
- **B** — domain inventory
- **C–E** — preliminary requirement inventory (164 IDs `PDC-Dn-Rnnn`), gap classification and certification coverage
- **F** — necessary ADRs only
- **G** — WBS P0–P9
- **H** — cross-domain test matrix XD-01..12
- **I** — dependency DAG
- **J** — certification architecture and SHA chain
- **K** — publication plan
- **L** — terminal closure matrix
- **M** — first executable batch

Requirement counts and classifications in the plan are **preliminary**. PDC-P2 finalizes them in `PDC_DOMAIN_REQUIREMENTS_MATRIX`. Every row still marked `TRACKED_UNKNOWN` blocks progress until P2 resolves it; none of them is a PASS.

Later phases update this campaign through their own tickets (`T-IGX-036-B` onward). This pointer is not a second plan.
