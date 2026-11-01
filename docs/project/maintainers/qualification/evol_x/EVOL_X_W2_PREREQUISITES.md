# EVOL-X W2 — Execution prerequisites

## Mandatory operator decisions (signed)

- [ ] **OD-EVOLX-01** — stage entry
- [ ] **OD-EVOL-01** — placement Option C (or documented alternative)
- [ ] **OD-EVOL-02** — workflow policy (i)
- [ ] **OD-EVOL-05** — assessor scope + Option 1 vs 2 for production port
- [ ] **OD-EVOL-06** — revision state facts source
- [ ] **OD-EVOL-10** — CI profile, collect policy, workflow sync to `main`

## AF gate

- [ ] **AF-8** — independent architecture audit PASS or operator waiver on record

## Resolved blockers from W1

| Blocker | Resolution owner |
|---------|------------------|
| N-18 gate collect errors | OD-EVOL-10 defines CI-equivalent collect manifest |
| N-04 graph_version gap | Class B decision before workflow migration claims |
| X-03, X-06, X-07 exclusions | W2-05 proofs |

## First W2 tasks (order)

1. W2-01 contracts per `EVOL_X_CONTRACT_SPEC.md`
2. W2-02 import boundaries
3. W2-03 owner/frozen JSON data
4. W2-04..07 discovery + graph
5. W2-09 CI profile (after OD-EVOL-10)

## Explicit non-starters

- No W3 selection until W2 graph digest stable
- No W4-04 production port until OD-EVOL-05b Option 2 approved
- No pytest VL3 until W6
