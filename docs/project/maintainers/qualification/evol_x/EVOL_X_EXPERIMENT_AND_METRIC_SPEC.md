# EVOL-X — Experiment S1–S8 & metric oracle spec (Annex F, W1)

## Oracles

| Oracle | Role | Independence |
|--------|------|--------------|
| **O-F** | Fixture tree causal truth (known affected surfaces/tests) | Ground truth for M-03 precision/recall on P-S |
| **O-D** | Declared oracle JSON per fixture (`oracle_declared.json`) | Second method for consumer lists |
| **O-X** | Platform synthetic `ChangeProposal` on HEAD inventory | Non-causal; documented limitation |

## S1–S8 summary

| S | Fixture dir | Positive | Negative arm |
|---|-------------|----------|--------------|
| S1 | `tests/fixtures/evol_x/s1_local_agent/` | Blast radius app A only | Cross-app escape |
| S2 | `s2_shared_contract/` | All consumers listed | Hidden consumer |
| S3 | `s3_workflow_inflight/` | Pin preserved on restart | Unsafe in-flight adoption |
| S4 | `s4_provider_replace/` | Provider swap | Consumer rewrite required |
| S5 | `s5_state_migration/` | Typed MIGRATE/INCOMPATIBLE | Silent break |
| S6 | `s6_negative_regression/` | Gate FAIL on injections | Bypass attempt |
| S7 | `s7_rollback/` | DM matrix + AD harness | DM-04 durable incompatible |
| S8 | `s8_new_application/` | Scaffold extension only | Unauthorized core edit |

Fixtures **not created in W1** — paths and oracle IDs frozen here for W5.

## Metrics (formulas frozen)

- **M-03P** = |Sel ∩ R| / |Sel|; **M-03R** = |Sel ∩ R*| / |R*|; **M-03C** mandatory coverage; **M-03E** escape count.
- **M-09 SLA proposal:** median operator attribution ≤15 min; tool ≤120 s; 100% seed match — **OD-EVOL-08**.

## Populations

- **P-S:** fixture arcs (deterministic).
- **P-H:** historical commits — **informative only** unless OD-EVOL-07 promotes.

Independent reviewer for oracle assignment: **OD-EVOL-11** (name TBD) — W1 records requirement AF-8.
